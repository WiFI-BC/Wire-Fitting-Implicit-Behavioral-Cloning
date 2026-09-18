"""Simulation class for the FrankaKitchen-v1 (D4RL/kitchen/*-v2) environment.

Why kitchen (vs the door/hammer/relocate dead-ends):
  The Adroit ports emit a DIFFERENT dense reward than the legacy-d4rl env the
  IBC paper measured, so raw-return comparison is invalid. kitchen's reward is
  a threshold-based subtask COUNT (0..N), which survives the
  mujoco_py->gymnasium-robotics port (verified: replaying dataset expert
  actions reproduces the dataset reward to <2% and completes all 4 subtasks).

Headline metric: tasks_completed (0..N) — matches IBC Table 2
  (kitchen-complete = 3.37/4). success = ALL target subtasks solved.

Normalization (IBC App. B.1 / B.3, identical scheme to door/pen):
  - Observations: per-dim standardize from `norm_stats.pt` (the 59-D
    'observation' field of the Dict obs; desired_goal is fixed for -complete).
  - Actions: model output in `action_norm_range` ([-1, 1]) linearly mapped to
    the dataset's per-dim [act_min, act_max] before `env.step`.
"""

from __future__ import annotations

import numpy as np
import torch

from wifi_bc.normalizations import ObservationNormalizer

from .base_simulation import BaseSimulation


class KitchenSimulation(BaseSimulation):
    """Evaluator for trained WiFI-BC policies on FrankaKitchen-v1."""

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 280,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        dataset_name: str = "D4RL/kitchen/complete-v2",
        execute_horizon: int = 0,
    ) -> None:
        super().__init__(
            env_id="FrankaKitchen-v1",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats
        self.dataset_name = dataset_name
        # Column selection saved at train time (kitchen_qpos_only) — the
        # policy must see exactly the dims it was trained on.
        self._obs_indices = (
            norm_stats.get("obs_indices") if isinstance(norm_stats, dict) else None
        )
        # Action chunking: K comes from training (norm_stats — the model's
        # output IS a K*A vector). execute_horizon R is eval-time: execute the
        # first R steps of each chunk then replan; 0/>=K = execute all K
        # (pure chunking, pen-style).
        self.action_chunk = int((norm_stats or {}).get("action_chunk", 1) or 1)
        eh = int(execute_horizon or 0)
        self.execute_horizon = (
            self.action_chunk if eh <= 0 else min(eh, self.action_chunk)
        )

        if norm_stats is not None and "obs_mean" in norm_stats and "obs_std" in norm_stats:
            self.obs_normalizer = ObservationNormalizer(
                env_id="FrankaKitchen-v1",
                device=device,
                frame_stack=frame_stack,
                obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
                obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
            )

        # CP outputs already live in model space ([-1,1]) for kitchen; the
        # eval-time DFO/Langevin refinement wrappers in envs/evaluate.py
        # read _act_min_t (None => candidates need no extra normalization
        # before the Q net). Mirrors DoorHumanV2Simulation.
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

    def create_env(self):
        # Recover the exact FrankaKitchen-v1 the dataset was recorded with
        # (correct tasks_to_complete + obs layout). Guarantees the eval target
        # set matches the demos, and avoids the reward_type kwarg AdroitHand
        # accepts but FrankaKitchen does not.
        import minari
        ds = minari.load_dataset(self.dataset_name, download=True)
        return ds.recover_environment(eval_env=True)

    def _obs_vec(self, obs) -> np.ndarray:
        """FrankaKitchen returns a Dict obs; the policy uses 'observation'."""
        if isinstance(obs, dict):
            obs = obs["observation"]
        v = np.asarray(obs, dtype=np.float32)
        if self._obs_indices is not None:
            v = v[self._obs_indices]
        return v

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        """Linear map from [act_lo, act_hi] back to [act_min, act_max]."""
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
            best_idx = q_values.argmax(dim=1)
            action_normalized = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        action = self._denormalize_action(action_normalized)
        action = np.clip(action, -1.0, 1.0)
        if return_q_range:
            return action, q_range
        return action

    def run_episode(self, seed: int | None = None) -> dict:
        if self.env is None:
            self.env = self.create_env()

        obs, _ = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(self._obs_vec(obs))

        total_reward = 0.0
        episode_length = 0
        done = False
        info: dict = {}
        tasks_completed = 0

        while not done:
            # select_action returns a K*A chunk when action_chunk > 1 (the
            # model's native output); execute the first `execute_horizon`
            # steps, then replan. K=1 degenerates to the single-step loop.
            chunk = np.asarray(self.select_action(stacked_obs)).reshape(
                self.action_chunk, -1
            )
            for k in range(self.execute_horizon):
                obs, reward, terminated, truncated, info = self.env.step(chunk[k])
                stacked_obs = self._update_frame_buffer(self._obs_vec(obs))
                total_reward += float(reward)
                episode_length += 1
                # episode_task_completions accumulates the target subtasks solved.
                tasks_completed = len(info.get("episode_task_completions", []))
                done = terminated or truncated
                if done:
                    break

        # success = ALL target subtasks solved. Count the targets from the env's goal
        # set: the D4RL kitchen datasets recover FrankaKitchen-v1 with
        # remove_task_when_completed=False, so info["tasks_to_complete"] never
        # shrinks and "completed + remaining" would demand twice the targets.
        goal = getattr(getattr(self.env, "unwrapped", self.env), "goal", None)
        n_targets = len(goal) if goal else tasks_completed + len(info.get("tasks_to_complete", []))
        success = n_targets > 0 and tasks_completed >= n_targets

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
            "tasks_completed": tasks_completed,
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [bool(r.get("success", False)) for r in self.results]
        tasks = [int(r.get("tasks_completed", 0)) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        summary["avg_tasks_completed"] = float(np.mean(tasks))
        return summary
