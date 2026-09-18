"""Simulation class for the AdroitHandPen-v1 (D4RL/pen/human-v2) environment.

Paper-faithful normalization (IBC App. B.1 / B.3):
  - Observations: per-dim zero-mean unit-variance (standardize) using stats
    saved at training time in `norm_stats.pt`.
  - Actions: model output lives in `action_norm_range` (default [-1, 1]);
    we linearly map to the dataset's per-dim [act_min, act_max] before
    `env.step`. Identical scheme to PushingSimulation.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import mujoco

from wifi_bc.normalizations import ObservationNormalizer

from .base_simulation import BaseSimulation


class PenHumanV2Simulation(BaseSimulation):
    """Evaluator for trained WiFI-BC policies on AdroitHandPen-v1 (D4RL pen-human).

    IBC Table 2 reports avg return (their EBM: 2586 ± 65). We surface both
    `total_reward` (paper-comparable) and `success` (binary, from env info).

    When `norm_stats` is provided, this simulation runs in the IBC-paper
    normalization regime: obs standardized per-dim, model emits actions in
    `action_norm_range` and they're denormalized to the env's native scale at
    every step. When `norm_stats` is None, falls back to legacy minmax obs
    normalization + raw action passthrough (for compat with old checkpoints).
    """

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        execute_horizon: int = 0,
    ) -> None:
        super().__init__(
            env_id="AdroitHandPen-v1",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats
        # Action chunking: model emits K*24 per CP. execute_horizon R is an
        # eval-time receding-horizon knob: execute the first R actions and
        # then replan. 0/>=K preserves pure open-loop chunk execution.
        self.action_chunk = int((norm_stats or {}).get("action_chunk", 1) or 1)
        eh = int(execute_horizon or 0)
        self.execute_horizon = (
            self.action_chunk if eh <= 0 else min(eh, self.action_chunk)
        )
        # CP selection: "argmax" (default) picks the highest-Q control point;
        # "sample" draws from softmax(Q / temperature) over the CP cloud — a
        # generative, multimodal-friendly selection (closer to how a diffusion
        # policy samples a mode, vs argmax's mode-averaging). Read from norm_stats
        # so every eval path (envs/evaluate.py, best-checkpoint eval) shares the setting.
        _ns = norm_stats or {}
        self.cp_selection = str(_ns.get("cp_selection", "argmax"))
        self.cp_selection_temperature = float(_ns.get("cp_selection_temperature", 1.0) or 1.0)

        # Replace the base ObservationNormalizer (built from JSON bounds in
        # minmax mode) with the dataset-derived standardize one when
        # norm_stats is available — this is what makes eval paper-faithful.
        if norm_stats is not None and "obs_mean" in norm_stats and "obs_std" in norm_stats:
            self.obs_normalizer = ObservationNormalizer(
                env_id="AdroitHandPen-v1",
                device=device,
                frame_stack=frame_stack,
                obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
                obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
            )

        # Action denormalization stats. Model output ∈ action_norm_range;
        # env wants [act_min, act_max] (which for pen is naturally [-1, 1] but
        # the dataset may not span the full range per-dim, so the min-max
        # rescale is still meaningful). Same convention as PushingSimulation:
        # `_act_min_t`/`_act_rng_t` are RESERVED for the legacy ibc_with_cps
        # semantics (CP outputs raw, Q sees [0, 1]) — we set them to None so
        # the Langevin/DFO refinement wrappers in envs/evaluate.py
        # short-circuit to `cp_for_q = cps` (correct: Q and model both live
        # in action_norm_range here).
        self._act_min_t = None
        self._act_rng_t = None
        if norm_stats is not None and "act_min" in norm_stats and "act_max" in norm_stats:
            self._raw_act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
            self._raw_act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
            lo_hi = norm_stats.get("action_norm_range", (-1.0, 1.0))
            self._act_lo = float(lo_hi[0])
            self._act_hi = float(lo_hi[1])
        else:
            self._raw_act_min = None
            self._raw_act_max = None
            self._act_lo = None
            self._act_hi = None

    def create_env(self) -> gym.Env:
        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            self.env_id,
            reward_type="dense",
            max_episode_steps=self.max_episode_steps,
            render_mode=self.render_mode,
        )
        return env

    def _render_callback(self, reward: float, total_reward: float) -> None:
        if self.render_mode == "human" and hasattr(self.env.unwrapped, "mujoco_renderer"):
            renderer = self.env.unwrapped.mujoco_renderer
            if hasattr(renderer, "viewer") and renderer.viewer:
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Reward",
                    f"{reward:.4f}",
                )
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Total Reward",
                    f"{total_reward:.4f}",
                )

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        """Linear map from [act_lo, act_hi] back to [act_min, act_max].

        No-op when norm_stats is None (legacy path) — falls back to identity.
        """
        if self._raw_act_min is None:
            return action_normalized
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        return (
            self._raw_act_min
            + (np.asarray(action_normalized, dtype=np.float32) - self._act_lo) * scale
        ).astype(np.float32)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        obs_tensor = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)
            if self.cp_selection == "sample":
                probs = torch.softmax(q_values[0] / self.cp_selection_temperature, dim=-1)
                sel = int(torch.multinomial(probs, 1)[0].item())
            else:
                sel = int(q_values.argmax(dim=1)[0].item())
            action_normalized = control_points[0, sel, :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        action = self._denormalize_action(action_normalized)
        # Defensive clip — env action box is [-1, 1] for AdroitHandPen-v1.
        action = np.clip(action, -1.0, 1.0)
        if return_q_range:
            return action, q_range
        return action

    def run_episode(self, seed: int | None = None) -> dict:
        if self.env is None:
            self.env = self.create_env()

        obs, _ = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)

        total_reward = 0.0
        episode_length = 0
        done = False
        info: dict = {}
        success = False

        while not done:
            chunk = self.select_action(stacked_obs)  # (K*24,) denormalized
            steps = np.asarray(chunk).reshape(self.action_chunk, -1)
            for k in range(self.execute_horizon):
                obs, reward, terminated, truncated, info = self.env.step(steps[k])
                stacked_obs = self._update_frame_buffer(obs)
                total_reward += float(reward)
                episode_length += 1
                step_success = bool(info.get("success", info.get("is_success", False)))
                if step_success:
                    success = True
                self._render_callback(reward, total_reward)
                done = terminated or truncated
                if done:
                    break

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [bool(r.get("success", False)) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        return summary
