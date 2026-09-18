"""Shared LIBERO evaluation plumbing.

One trained policy is evaluated across all 10 `libero_goal` tasks. Each episode
is assigned a task (round-robin over `seed`) and an initial state, the task's
language embedding conditions the policy, and success is counted per episode;
the headline metric is the success rate across episodes, which is LIBERO's
canonical number.

`LiberoSimulation` holds what every LIBERO evaluation needs — task lookup,
initial states, env construction, settling, action denormalization and the
episode loop. `envs/libero_goal_pixels_simulation.py` subclasses it with the
camera-based observation the paper reports.

Points worth knowing when adapting this:
  1. Live observation key names. The live robosuite env may name low-dim keys
     differently from the HDF5 demos; `envs.libero.resolve_live_obs` bridges
     them through `LIVE_KEY_ALIASES`.
  2. Success signal. Reward > 0 OR `env.check_success()` counts as success.
     LIBERO's OffScreenRenderEnv uses the OLD gym 4-tuple step API.
  3. Action sign and scale. Actions are denormalized from the model's
     `action_norm_range` back to raw via the saved act_min/act_max, then
     clipped to [-1, 1], which is LIBERO/OSC's native range.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from wifi_bc.normalizations import ObservationNormalizer
from envs.libero import resolve_live_obs, get_task_infos

from .base_simulation import BaseSimulation


class LiberoSimulation(BaseSimulation):
    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 300,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        camera_height: int = 128,
        camera_width: int = 128,
    ) -> None:
        # NOTE: we deliberately DON'T call super().__init__ — BaseSimulation
        # builds a JSON-bounds minmax ObservationNormalizer keyed by env_id,
        # which has no entry for LIBERO. We set the needed attrs directly and
        # build a standardize normalizer from the training-time stats instead.
        if norm_stats is None or "state_shape" not in norm_stats:
            raise ValueError(
                "LiberoSimulation requires norm_stats with the libero_goal "
                "schema (obs keys, goal embeddings, state_shape). Was the model "
                "trained with active_env=libero_goal?"
            )
        self.control_point_generator = control_point_generator
        self.q_estimator = q_estimator
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.render_mode = render_mode
        self.results = []
        self.env = None
        self._env_task_idx: int | None = None
        self.camera_height = camera_height
        self.camera_width = camera_width

        # ── Schema + goal embeddings from training ─────────────────────────
        self.obs_keys = list(norm_stats["libero_obs_keys"])
        self.goal_embeddings = np.asarray(norm_stats["goal_embeddings"], dtype=np.float32)
        self.goal_task_names = list(norm_stats["goal_task_names"])
        self.n_tasks = int(self.goal_embeddings.shape[0])
        # state_shape is an int for the flat state variant, a [C,H,W] list for the
        # pixel subclass — keep whichever type it is.
        _ss = norm_stats["state_shape"]
        self.state_shape = tuple(_ss) if isinstance(_ss, (list, tuple)) else int(_ss)

        # Standardize normalizer over the FULL state vector (no frame tiling).
        # Only for the flat variant (obs_mean present); the pixel subclass has no
        # obs_mean (conv encoder preprocesses) and sets obs_normalizer=None itself.
        if "obs_mean" in norm_stats:
            self.obs_normalizer = ObservationNormalizer(
                env_id="libero_goal",
                device=device,
                frame_stack=1,
                obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
                obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
            )
        else:
            self.obs_normalizer = None

        # Action denorm (model space -> raw env action), same scheme as pen.
        self._raw_act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
        self._raw_act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
        lo_hi = norm_stats.get("action_norm_range", (-1.0, 1.0))
        self._act_lo, self._act_hi = float(lo_hi[0]), float(lo_hi[1])

        # Per-task metadata (bddl files etc.) + init states are loaded lazily.
        self._task_infos: list[dict] | None = None
        self._init_states: dict[int, np.ndarray] = {}
        self._frame_buf: deque[np.ndarray] = deque(maxlen=frame_stack)

    # ── env lifecycle ────────────────────────────────────────────────────
    def _lazy_task_infos(self) -> list[dict]:
        if self._task_infos is None:
            self._task_infos = get_task_infos()
        return self._task_infos

    def _get_init_states(self, task_idx: int) -> np.ndarray:
        if task_idx not in self._init_states:
            from libero.libero import benchmark

            bench = benchmark.get_benchmark_dict()["libero_goal"]()
            self._init_states[task_idx] = bench.get_task_init_states(task_idx)
        return self._init_states[task_idx]

    def create_env(self, task_idx: int = 0):
        """Get a RENDERLESS LIBERO env for *task_idx* (single env, recreated).

        State-based eval only needs `object-state` (from object sensors), not
        pixels — so the base ControlEnv runs with the offscreen renderer and
        camera obs OFF. No EGL → no MUJOCO_GL dependency, and recreating it on
        task change is safe (no GL context to leak/segfault). We hold ONE env at
        a time (close the previous before opening the next) so eval memory stays
        bounded — caching all 10 task envs OOM-killed the job.
        """
        if self.env is not None and self._env_task_idx == task_idx:
            return self.env
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
            self.env = None
        from libero.libero.envs.env_wrapper import ControlEnv

        info = self._lazy_task_infos()[task_idx]
        self.env = ControlEnv(
            bddl_file_name=info["bddl_file"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        self._env_task_idx = task_idx
        return self.env

    # ── action selection (appends goal embedding) ────────────────────────
    def _build_state(self, live_obs: dict) -> np.ndarray:
        """Live obs dict -> [frame_stacked low-dim obs | goal embedding]."""
        obs_vec = resolve_live_obs(live_obs, self.obs_keys)
        self._frame_buf.append(obs_vec)
        if self.frame_stack > 1:
            while len(self._frame_buf) < self.frame_stack:
                self._frame_buf.appendleft(obs_vec)
            stacked = np.concatenate(list(self._frame_buf))
        else:
            stacked = obs_vec
        return np.concatenate([stacked, self._current_goal_emb]).astype(np.float32)

    def select_action(self, state_vec: np.ndarray) -> np.ndarray:
        obs_tensor = (
            torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)
        with torch.no_grad():
            cps = self.control_point_generator(obs_tensor)  # (1, N, A)
            obs_exp = obs_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = self.q_estimator(obs_exp, cps).squeeze(-1)  # (1, N)
            best = q.argmax(dim=1)
            action_norm = cps[0, best[0], :].cpu().numpy()
        return self._denormalize_action(action_norm)

    def _denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        raw = self._raw_act_min + (np.asarray(action_norm, dtype=np.float32) - self._act_lo) * scale
        return np.clip(raw, -1.0, 1.0).astype(np.float32)

    # Standard LIBERO eval protocol (e.g. OpenVLA's script): after restoring
    # the init state, step ~10 no-op actions so physics settle (objects placed
    # by set_init_state may still be moving) before the policy takes over.
    # No-op = zero OSC delta + gripper open (-1). Not counted toward the horizon.
    SETTLE_STEPS = 10
    _NOOP_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    def _settle(self, env, live_obs):
        for _ in range(self.SETTLE_STEPS):
            live_obs, _, _, _ = env.step(self._NOOP_ACTION)
        return live_obs

    # ── episode loop (multi-task) ────────────────────────────────────────
    def run_episode(self, seed: int | None = None) -> dict:
        s = int(seed) if seed is not None else 0
        task_idx = s % self.n_tasks
        round_idx = s // self.n_tasks

        env = self.create_env(task_idx)
        self._current_goal_emb = self.goal_embeddings[task_idx]
        self._frame_buf.clear()

        init_states = self._get_init_states(task_idx)
        env.seed(s)
        env.reset()
        init_idx = round_idx % len(init_states)
        live_obs = env.set_init_state(init_states[init_idx])
        live_obs = self._settle(env, live_obs)

        total_reward = 0.0
        episode_length = 0
        success = False
        terminated = False

        for _ in range(self.max_episode_steps):
            state_vec = self._build_state(live_obs)
            action = self.select_action(state_vec)
            live_obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            episode_length += 1
            step_success = bool(reward > 0) or bool(getattr(env, "check_success", lambda: False)())
            if isinstance(info, dict):
                step_success = step_success or bool(info.get("success", False))
            if step_success:
                success = True
            if done:
                terminated = True
                break

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": not terminated,
            "success": success,
            "task_idx": task_idx,
            "task_name": self.goal_task_names[task_idx],
        }

    def get_summary(self) -> dict[str, float]:
        if not self.results:
            return {}
        successes = [bool(r.get("success", False)) for r in self.results]
        rewards = [float(r.get("total_reward", 0.0)) for r in self.results]
        return {
            "num_episodes": len(self.results),
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
            self.env = None
            self._env_task_idx = None
