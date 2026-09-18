"""Simulation runner for the IBC BlockPush(PUSH) environment.

Mirrors `ParticleSimulation` so the rest of the WiFI-BC pipeline
(`envs.evaluate`, multi-seed evaluation) can swap in
this env by env-id.

Paper-faithful normalization (matches `get_normalizers.py` /
`ibc/train/stats.py` in google-research/ibc):
  - Observations: standardize (x - mean) / std using stats saved at
    training time in `norm_stats.pt`.
  - Actions: model output lives in [-1, 1]; we linearly map to the env's
    native effector-delta range (`act_min`, `act_max`) before `env.step`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from wifi_bc.normalizations import ObservationNormalizer

from .base_simulation import BaseSimulation
from .pushing_env import PushingEnv


class PushingSimulation(BaseSimulation):
    """Evaluator for trained WiFI-BC policies on Pushing-v0.

    `succeeded` comes from the IBC env's internal success flag (block within
    its goal tolerance of the target after enough settling time). We report
    final and minimum block→target distances so trial logs surface partial
    progress even when the strict success bit doesn't flip.

    When `norm_stats` is provided, this simulation runs in the IBC-paper
    normalization regime: obs is standardized per-dim, actions are produced
    by the model in [-1, 1] and denormalized to the env's native scale at
    every step. When `norm_stats` is None, it falls back to the legacy
    min-max obs normalization + raw action passthrough (for compat with
    older checkpoints).
    """

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        # Surface so envs/evaluate.py can pass the published value (0.02)
        # — env default falls back to it anyway, but explicit > implicit.
        goal_dist_tolerance: float = 0.02,
    ) -> None:
        super().__init__(
            env_id="Pushing-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.goal_dist_tolerance = goal_dist_tolerance
        self.norm_stats = norm_stats

        # Replace the base ObservationNormalizer (built from observation_bounds
        # JSON in minmax mode) with the dataset-derived standardize one when
        # norm_stats is available. This is what makes the eval pipeline
        # paper-faithful end-to-end.
        if norm_stats is not None and "obs_mean" in norm_stats and "obs_std" in norm_stats:
            self.obs_normalizer = ObservationNormalizer(
                env_id="Pushing-v0",
                device=device,
                frame_stack=frame_stack,
                obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
                obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
            )

        # Action denormalization stats. Model output ∈ [-1, 1]; env wants
        # [act_min, act_max]. action_norm_range stays configurable but
        # defaults to (-1, 1) — what PushingDataset writes.
        #
        # Naming carefully: `_act_min_t` / `_act_rng_t` are reserved for the
        # legacy ibc_with_cps semantics (CP outputs raw, Q sees [0, 1]). We
        # set them to None so the LangevinRefinedParticleSimulation wrapper
        # in envs/evaluate.py — which still checks those names —
        # short-circuits to `cp_for_q = cps`, which is exactly right for the
        # paper-faithful design (Q and model both live in [-1, 1]).
        self._act_min_t = None
        self._act_rng_t = None
        if norm_stats is not None:
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

    def create_env(self) -> PushingEnv:
        return PushingEnv(
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
            goal_dist_tolerance=self.goal_dist_tolerance,
        )

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        """Linear map from [act_lo, act_hi] back to [act_min, act_max].

        No-op when norm_stats is None (legacy path) — falls back to the base
        class's identity behavior.
        """
        if self._raw_act_min is None:
            return action_normalized
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        return (self._raw_act_min + (np.asarray(action_normalized, dtype=np.float32) - self._act_lo) * scale).astype(np.float32)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, A) in [-1, 1]
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)

            # Q estimator sees actions in the SAME space the model emits and
            # was trained with: [-1, 1] when norm_stats is present.
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)
            best_idx = q_values.argmax(dim=1)
            action_normalized = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        # Denormalize before env.step. PushingEnv also clips internally so any
        # numerical overshoot at the action box edges is harmless.
        action = self._denormalize_action(action_normalized)
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
        min_dist = float("inf")
        done = False
        info: dict = {}

        while not done:
            action = self.select_action(stacked_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += float(reward)
            episode_length += 1
            min_dist = min(min_dist, float(info.get("block_to_target_distance", np.inf)))
            done = terminated or truncated

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(info.get("success", False)),
            "min_dist_to_target": min_dist,
            "final_dist_to_target": float(info.get("block_to_target_distance", np.inf)),
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [r.get("success", False) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        dists = [r.get("min_dist_to_target", np.inf) for r in self.results]
        finite_dists = [d for d in dists if np.isfinite(d)]
        summary["avg_min_dist_to_target"] = float(np.mean(finite_dists)) if finite_dists else float("inf")
        summary["std_min_dist_to_target"] = float(np.std(finite_dists)) if finite_dists else float("inf")
        return summary
