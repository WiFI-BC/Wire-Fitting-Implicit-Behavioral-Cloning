"""Eval sim for LIBERO-Goal PIXEL (standard protocol), goal-conditioned.

Reuses LiberoSimulation's task metadata and action denormalization, but:
  - the env RENDERS (need camera images): OffScreenRenderEnv.
  - episodes are GROUPED by task (task-major seed mapping) so each task's render
    env is created ONCE — round-robin per-episode env churn segfaults the EGL
    context (learned on the state-based variant).
  - action selection is pixel CP-argmax: encode image once, score the CP cloud,
    with the proprio+goal conditioning fed via the pixel nets' `_cond` attr.

Obs built at eval:
  image = channel-stack [agentview, wrist] (x frame_stack) uint8 (1, C, H, W)
  cond  = [frame_stacked proprio | goal embedding]  (1, cond_dim)

Needs MUJOCO_GL=egl on a GPU compute node. UNVERIFIED until run there — confirm
the live image keys / orientation match the demos (see notes).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from envs.libero import resolve_live_obs
from .libero_simulation import LiberoSimulation

# Live robosuite image keys (demos store agentview_rgb / eye_in_hand_rgb).
_LIVE_IMAGE_CANDIDATES = {
    "agentview": ("agentview_image", "agentview_rgb"),
    "wrist": ("robot0_eye_in_hand_image", "eye_in_hand_image", "eye_in_hand_rgb"),
}


class LiberoGoalPixelsSimulation(LiberoSimulation):
    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 300,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        num_eval_seeds: int = 50,
        camera_height: int = 128,
        camera_width: int = 128,
    ) -> None:
        super().__init__(
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            frame_stack=frame_stack,
            norm_stats=norm_stats,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        # The base built a standardize obs_normalizer; pixels don't use it (the
        # conv encoder preprocesses). Drop it so it can't be misapplied.
        self.obs_normalizer = None
        self.cond_dim = int(norm_stats["cond_dim"])
        # Cloud-selection rule (previously a hard argmax here, see
        # simulations/cp_selection.py for why that is the fragile choice).
        self.cp_selection = str(norm_stats.get("cp_selection", "argmax"))
        self.cp_selection_temperature = float(norm_stats.get("cp_selection_temperature", 1.0) or 1.0)
        self.cp_score_norm = str(norm_stats.get("cp_score_norm", "none"))
        self.proprio_keys = list(norm_stats["libero_obs_keys"])
        # Train-time random-crop aug ⇒ eval-time CENTER crop to the same size,
        # so the encoder sees the same input scale/coverage it trained on.
        self.crop_size = int(norm_stats.get("image_crop_size", 0) or 0)
        # Action chunking: the model emits K*7 per CP; we execute the K steps
        # open-loop (re-planning after each chunk), success checked every step.
        self.action_chunk = int(norm_stats.get("action_chunk", 1) or 1)
        # Cameras the checkpoint was trained with; older checkpoints predate the
        # key and used both. Eval must stack exactly these, in this order.
        self.cameras = tuple(norm_stats.get("libero_cameras", ("agentview", "wrist")))
        self.num_eval_seeds = int(num_eval_seeds)
        self._eps_per_task = max(1, (self.num_eval_seeds + self.n_tasks - 1) // self.n_tasks)
        self._img_buf: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._proprio_buf: deque[np.ndarray] = deque(maxlen=frame_stack)

    # Render-ON env (override the renderless state-based one), cached single env.
    def create_env(self, task_idx: int = 0):
        if self.env is not None and self._env_task_idx == task_idx:
            return self.env
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
            self.env = None
        from libero.libero.envs import OffScreenRenderEnv

        info = self._lazy_task_infos()[task_idx]
        self.env = OffScreenRenderEnv(
            bddl_file_name=info["bddl_file"],
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
        )
        self._env_task_idx = task_idx
        return self.env

    @staticmethod
    def _get_live_image(live_obs: dict, which: str) -> np.ndarray:
        for cand in _LIVE_IMAGE_CANDIDATES[which]:
            if cand in live_obs:
                return np.asarray(live_obs[cand], dtype=np.uint8)  # (H, W, 3)
        raise KeyError(
            f"No live image key for {which!r}; tried {_LIVE_IMAGE_CANDIDATES[which]}. "
            f"Available: {sorted(live_obs.keys())}"
        )

    def _build_inputs(self, live_obs: dict):
        """Live obs -> (image (1,C,H,W) uint8 tensor, cond (1,cond_dim) tensor)."""
        frame = np.concatenate([self._get_live_image(live_obs, c) for c in self.cameras],
                               axis=-1)  # (H,W,3*n_cams)
        # A checkpoint trained without proprio records libero_obs_keys == [].
        proprio = (resolve_live_obs(live_obs, self.proprio_keys) if self.proprio_keys
                   else np.zeros(0, dtype=np.float32))
        if self.frame_stack > 1:
            self._img_buf.append(frame)
            self._proprio_buf.append(proprio)
            while len(self._img_buf) < self.frame_stack:
                self._img_buf.appendleft(frame)
                self._proprio_buf.appendleft(proprio)
            img = np.concatenate(list(self._img_buf), axis=-1)
            pr = np.concatenate(list(self._proprio_buf))
        else:
            img, pr = frame, proprio
        if self.crop_size:
            s = self.crop_size
            oy = (img.shape[0] - s) // 2
            ox = (img.shape[1] - s) // 2
            img = img[oy:oy + s, ox:ox + s]
        img = np.transpose(img, (2, 0, 1)).copy()  # (C,S,S)
        cond = np.concatenate([pr, self._current_goal_emb]).astype(np.float32)
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)            # uint8 (1,C,H,W)
        cond_t = torch.from_numpy(cond).unsqueeze(0).float().to(self.device)  # (1,cond_dim)
        return img_t, cond_t

    def select_action(self, live_obs: dict) -> np.ndarray:
        img_t, cond_t = self._build_inputs(live_obs)
        self.control_point_generator._cond = cond_t
        self.q_estimator._cond = cond_t
        with torch.no_grad():
            cps = self.control_point_generator(img_t)            # (1, N, A)
            feats = self.q_estimator.encode(img_t)               # (1, F)
            q = self.q_estimator.score(feats, cps).squeeze(-1)   # (1, N)
            from envs.cp_selection import select_from_cloud
            idx = select_from_cloud(q, cps, self.cp_selection,
                                    self.cp_selection_temperature, self.cp_score_norm)
            action_norm = cps[0, idx, :].cpu().numpy()
        return self._denormalize_action(action_norm)

    # Task-major episode loop (group by task → 1 env build per task).
    def run_episode(self, seed: int | None = None) -> dict:
        s = int(seed) if seed is not None else 0
        task_idx = (s // self._eps_per_task) % self.n_tasks
        init_round = s % self._eps_per_task

        env = self.create_env(task_idx)
        self._current_goal_emb = self.goal_embeddings[task_idx]
        self._img_buf.clear()
        self._proprio_buf.clear()

        init_states = self._get_init_states(task_idx)
        env.seed(s)
        env.reset()
        live_obs = env.set_init_state(init_states[init_round % len(init_states)])
        live_obs = self._settle(env, live_obs)  # standard ~10 no-op settle steps

        total_reward = 0.0
        episode_length = 0
        success = False
        terminated = False
        while episode_length < self.max_episode_steps:
            chunk = self.select_action(live_obs)              # (K*7,) denormalized
            steps = chunk.reshape(self.action_chunk, -1)
            for k in range(self.action_chunk):
                live_obs, reward, done, info = env.step(steps[k])
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
                if episode_length >= self.max_episode_steps:
                    break
            if terminated:
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


class LiberoGoalPixelsIBCSimulation(LiberoGoalPixelsSimulation):
    """Original-IBC (pure pixel EBM) eval on LIBERO-Goal, standard protocol.

    Reuses LiberoGoalPixelsSimulation's render env, obs/cond building, crop,
    settle-wait and grouped-by-task loop, but selects actions by Langevin MCMC
    on the pixel energy net (no control points): encode the image ONCE per env
    step, run the chain against the value head (late fusion), take argmin energy.
    """

    def __init__(
        self,
        energy_net,
        device: str,
        max_episode_steps: int,
        frame_stack: int,
        norm_stats: dict,
        langevin_cfg: dict,
        num_eval_seeds: int = 50,
        action_in_model_range: tuple[float, float] = (-1.0, 1.0),
        uniform_boundary_buffer: float = 0.05,
    ) -> None:
        super().__init__(
            control_point_generator=None,
            q_estimator=energy_net,
            device=device,
            max_episode_steps=max_episode_steps,
            render_mode=None,
            frame_stack=frame_stack,
            norm_stats=norm_stats,
            num_eval_seeds=num_eval_seeds,
        )
        self.energy_net = energy_net
        self.langevin_cfg = langevin_cfg
        lo, hi = float(action_in_model_range[0]), float(action_in_model_range[1])
        buf = float(uniform_boundary_buffer)
        adim = int(np.asarray(self._raw_act_min).shape[0])
        self._amin = torch.full((adim,), lo - buf, device=device)
        self._amax = torch.full((adim,), hi + buf, device=device)
        self._act_lo, self._act_hi = lo, hi

    def select_action(self, live_obs: dict) -> np.ndarray:
        from wifi_bc.sampling import sample_langevin

        img_t, cond_t = self._build_inputs(live_obs)
        self.energy_net._cond = cond_t
        c = self.langevin_cfg
        with torch.no_grad():
            feats = self.energy_net.encode(img_t)  # (1, F) — once per env step
        samples = sample_langevin(
            energy_function=lambda o, a: self.energy_net.score(feats, a).squeeze(-1),
            observations=img_t,  # batch-size/device carrier only
            num_samples=int(c["num_samples"]),
            action_min=self._amin,
            action_max=self._amax,
            num_iterations=int(c["num_iterations"]),
            lr_init=float(c["lr_init"]),
            lr_final=float(c["lr_final"]),
            polynomial_decay_power=float(c.get("polynomial_decay_power", 2.0)),
            delta_action_clip=float(c.get("delta_action_clip", 0.1)),
            noise_scale=float(c["noise_scale"]),
            device=self.device,
        )
        with torch.no_grad():
            e = self.energy_net.score(feats, samples).squeeze(-1)  # (1, N)
            best = samples[0, e.argmin(dim=-1)[0]].cpu().numpy()
        return self._denormalize_action(best)
