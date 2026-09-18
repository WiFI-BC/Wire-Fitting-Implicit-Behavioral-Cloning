"""Gymnasium wrapper around the vendored IBC BlockPush(PUSH) env.

The WiFI-BC pipeline (training scripts, evaluation)
talks to environments through the standard gymnasium 5-tuple API. The
vendored IBC env (`envs.ibc_block_pushing.block_pushing.BlockPush`) is
an *old-style* gym env: returns 4-tuples from `step` and a plain obs from
`reset`. This wrapper:

  * adapts the API (5-tuple step, 2-tuple reset),
  * flattens the dict observation into a single 1-D vector with a stable
    ordering, so it slots into the existing ObservationNormalizer/DataLoader
    pipeline the same way ParticleEnv / DummyEnv do,
  * exposes `block_translation`, `target_translation`, etc. via `info` for
    evaluation (success rate, distance metrics).

State layout (10D, before frame-stacking):
    [block_translation (2), block_orientation (1),
     effector_translation (2), effector_target_translation (2),
     target_translation (2), target_orientation (1)]

The ordering matches the canonical sorted feature order used by the IBC
TFRecord dataset — see `envs.datasets.BlockPushDataset` for the loader-side
ordering. KEEP IN SYNC: if you reorder here you must reorder the dataset
loader too.

Action: 2D, in env's native action space (~[-0.025, 0.029]×[-0.021, 0.043]).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import envs.ibc_block_pushing  # noqa: F401  — registers gin stub
from envs.ibc_block_pushing.block_pushing import (
    ACTION_MAX,
    ACTION_MIN,
    BlockPush,
    BlockTaskVariant,
    WORKSPACE_BOUNDS,
)


# Canonical, stable feature ordering — the same one the dataset loader uses.
OBS_KEYS_AND_DIMS = (
    ("block_translation", 2),
    ("block_orientation", 1),
    ("effector_translation", 2),
    ("effector_target_translation", 2),
    ("target_translation", 2),
    ("target_orientation", 1),
)
OBS_DIM = sum(d for _, d in OBS_KEYS_AND_DIMS)  # 10


def flatten_pushing_obs(obs: dict) -> np.ndarray:
    """Dict obs → flat numpy vector using the canonical key order."""
    return np.concatenate([
        np.atleast_1d(np.asarray(obs[k], dtype=np.float32)).flatten()
        for k, _ in OBS_KEYS_AND_DIMS
    ]).astype(np.float32)


class PushingEnv(gym.Env):
    """Gymnasium-style 1-block / 1-target pushing env, backed by IBC's BlockPush.

    Reference: Florence et al. 2021 (Implicit BC), §5 / Table 3.

    Parameters mirror IBC's BlockPush defaults; we don't expose them all to
    avoid configuration drift away from the published task.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        n_steps: int = 100,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # IBC paper's `pushing_states/mlp_ebm_langevin.gin` sets
        # `train_eval.goal_tolerance = 0.02`. The env class default is the
        # stricter 0.01 (`block_pushing.py` line 178). We surface this here
        # and default to the paper value so WiFI-BC eval matches Table 3.
        goal_dist_tolerance: float = 0.02,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.render_mode = render_mode

        # Underlying IBC env. control_frequency / step_frequency / abs_action
        # are at their published defaults.
        self._env = BlockPush(
            task=BlockTaskVariant.PUSH,
            goal_dist_tolerance=goal_dist_tolerance,
        )
        if seed is not None:
            self._env.seed(seed)

        # Action space: IBC env uses Box(-0.1, 0.1, (2,)) but the *data-driven*
        # min/max from the oracle (block_pushing.ACTION_MIN/MAX) are tighter.
        # Expose the data-driven range so the policy's output matches what the
        # training distribution actually contains. The env clips internally.
        self.action_space = spaces.Box(
            low=ACTION_MIN.astype(np.float32),
            high=ACTION_MAX.astype(np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Flat observation space. Bounds: position/translation channels are
        # bounded by the workspace; orientations live in [-π, π].
        ws_min = WORKSPACE_BOUNDS[0]  # (x_min, y_min)
        ws_max = WORKSPACE_BOUNDS[1]
        # Generous bounds — block can drift outside workspace; we let the
        # ObservationNormalizer clamp downstream.
        lows: list[float] = []
        highs: list[float] = []
        for k, d in OBS_KEYS_AND_DIMS:
            if k == "block_translation" or k == "target_translation":
                lows.extend([-2.0] * d)
                highs.extend([2.0] * d)
            elif k == "effector_translation" or k == "effector_target_translation":
                lows.extend([float(ws_min[0]), float(ws_min[1])])
                highs.extend([float(ws_max[0]), float(ws_max[1])])
            elif k.endswith("_orientation"):
                lows.extend([-np.pi] * d)
                highs.extend([np.pi] * d)
            else:
                lows.extend([-np.inf] * d)
                highs.extend([np.inf] * d)
        self.observation_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

        self.steps = 0
        self._last_obs_dict: Optional[dict] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        obs_dict = self._env.reset()
        self.steps = 0
        self._last_obs_dict = obs_dict
        return flatten_pushing_obs(obs_dict), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        # Clip to the data-driven action box so we don't hand the env values
        # the oracle dataset never demonstrated.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs_dict, reward, ibc_done, _ = self._env.step(action)
        self._last_obs_dict = obs_dict

        self.steps += 1
        terminated = bool(ibc_done) or self.steps >= self.n_steps

        block = np.asarray(obs_dict["block_translation"], dtype=np.float32)
        target = np.asarray(obs_dict["target_translation"], dtype=np.float32)
        dist = float(np.linalg.norm(block - target))
        success = bool(getattr(self._env, "succeeded", False))

        info = {
            "block_to_target_distance": dist,
            "success": success,
            "block_translation": block,
            "target_translation": target,
        }
        return flatten_pushing_obs(obs_dict), float(reward), terminated, False, info

    def render(self):
        # IBC env supports rgb_array rendering via image_size kwarg but we
        # skipped that path to avoid the camera setup. The training pipeline
        # doesn't need rendering.
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


gym.register(
    id="Pushing-v0",
    entry_point="envs.pushing_env:PushingEnv",
    max_episode_steps=100,
)


if __name__ == "__main__":
    print("Smoke-testing vendored IBC BlockPush wrapper ...")
    env = PushingEnv()
    obs, _ = env.reset(seed=0)
    print(f"  obs shape: {obs.shape}  dtype: {obs.dtype}")
    print(f"  action_space: {env.action_space}")
    print(f"  initial block={obs[0:2]} target={obs[7:9]}")
    total = 0.0
    for step in range(env.n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"  episode ended at step {step}, success={info['success']}, "
          f"final block-target dist={info['block_to_target_distance']:.3f}")
    env.close()
