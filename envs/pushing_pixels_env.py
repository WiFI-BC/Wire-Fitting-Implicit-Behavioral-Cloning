"""Gymnasium wrapper around the vendored IBC BlockPush env, RGB obs mode.

Sibling of `pushing_env.py` for the IBC paper's "Block Pushing — Single
target, Images" task (Florence et al. 2021, Table 3 row matching the
`pushing_pixels/pixel_ebm_langevin.gin` config). Same single-block /
single-target physics as `PushingEnv`, but the observation is the rendered
camera image at the env's native 240×320×3 uint8 resolution rather than a
flat state vector.

Action: 2D effector planar delta — identical scale to PushingEnv (we reuse
ACTION_MIN/ACTION_MAX from the vendored block_pushing module).

Image resolution is the env native (`block_pushing.IMAGE_HEIGHT=240,
block_pushing.IMAGE_WIDTH=320`). The trained encoder
(`wifi_bc.models.ConvMaxpoolEncoder`) does its own bilinear resize to
(180, 240) internally, matching IBC's `image_prepro.preprocess`.
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
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


class PushingPixelsEnv(gym.Env):
    """Gymnasium-style 1-block / 1-target pushing env with RGB observations.

    Reference: Florence et al. 2021 (Implicit BC), §5 / Table 3
    ("Block Pushing — Single target, Images").

    Observation: (H, W, 3) uint8 image, channels-last to match IBC's TFRecord
    layout. The training pipeline transposes to (3, H, W) before the conv
    encoder consumes it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        n_steps: int = 100,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # IBC's `pushing_pixels/pixel_ebm_langevin.gin` sets
        # `train_eval.goal_tolerance = 0.02` — same as states.
        goal_dist_tolerance: float = 0.02,
        image_height: int = IMAGE_HEIGHT,
        image_width: int = IMAGE_WIDTH,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.render_mode = render_mode
        self.image_height = image_height
        self.image_width = image_width

        # `image_size=(h, w)` flips IBC's BlockPush into camera-render mode:
        # `_compute_state` adds an 'rgb' key and the obs dict carries the
        # camera buffer alongside the low-dim features. We pull rgb from
        # there in step()/reset() and return it directly.
        self._env = BlockPush(
            task=BlockTaskVariant.PUSH,
            goal_dist_tolerance=goal_dist_tolerance,
            image_size=(image_height, image_width),
        )
        if seed is not None:
            self._env.seed(seed)

        self.action_space = spaces.Box(
            low=ACTION_MIN.astype(np.float32),
            high=ACTION_MAX.astype(np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # uint8 in [0, 255], channels-last (same as the published TFRecord
        # bytes after JPEG decode → matches PushingPixelsDataset).
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_height, image_width, 3),
            dtype=np.uint8,
        )

        self.steps = 0
        self._last_obs_dict: Optional[dict] = None

    def _extract_rgb(self, obs_dict: dict) -> np.ndarray:
        return np.asarray(obs_dict["rgb"], dtype=np.uint8)

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
        return self._extract_rgb(obs_dict), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
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
        return self._extract_rgb(obs_dict), float(reward), terminated, False, info

    def render(self):
        if self._last_obs_dict is not None and "rgb" in self._last_obs_dict:
            return np.asarray(self._last_obs_dict["rgb"], dtype=np.uint8)
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


gym.register(
    id="PushingPixels-v0",
    entry_point="envs.pushing_pixels_env:PushingPixelsEnv",
    max_episode_steps=100,
)
