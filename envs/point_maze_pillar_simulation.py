"""Simulation classes for PointMazePillar-v0.

`PointMazePillarSimulation` follows the BaseSimulation contract (same as
every other env) for envs/evaluate.py's generic evaluate dispatch.
`PointMazePillarDiagnosticSimulation` drives plot_point_maze_pillar_debug.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .point_maze_pillar_env import PointMazePillarEnv
from envs.vis_point_maze_pillar import plot_point_maze_pillar_debug
from wifi_bc.normalizations import ObservationNormalizer


class PointMazePillarSimulation(BaseSimulation):
    """Simulation for testing trained policies on PointMazePillar-v0."""

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 400,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
    ) -> None:
        super().__init__(
            env_id="PointMazePillar-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats
        self._act_min_t = None
        self._act_rng_t = None

    def create_env(self) -> PointMazePillarEnv:
        return PointMazePillarEnv(
            render_mode=self.render_mode, max_episode_steps=self.max_episode_steps,
        )

    def run_episode(self, seed: int | None = None) -> dict:
        result = super().run_episode(seed=seed)
        result["success"] = result["terminated"]
        return result


class PointMazePillarDiagnosticSimulation:
    """Diagnostic runner for PointMazePillar-v0 — drives
    plot_point_maze_pillar_debug at configurable snapshot steps."""

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device="cpu",
        save_dir="plots/point_maze_pillar",
        max_episode_steps=400,
        frame_stack=1,
        snapshot_steps=None,
        **kwargs,
    ):
        self.model = control_point_generator
        self.estimator = q_estimator
        self.device = device
        self.save_dir = save_dir
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.snapshot_steps = snapshot_steps or [1, 5, 20, 50, 100, 200]

        self.obs_normalizer = ObservationNormalizer(
            env_id="PointMazePillar-v0", device=self.device, frame_stack=self.frame_stack
        )
        self.env = PointMazePillarEnv(max_episode_steps=max_episode_steps)
        self.all_results = []

    def run_simulation(self, num_episodes=3, seed=None):
        print(f"Running PointMazePillar diagnostic ({num_episodes} episodes)...")
        os.makedirs(self.save_dir, exist_ok=True)
        self.all_results = []

        for ep in range(num_episodes):
            ep_seed = (seed or 0) * 1000 + ep
            obs, _ = self.env.reset(seed=ep_seed)
            start_pos = obs[:2].copy()
            goal = obs[4:6].copy()
            trajectory = [obs[:2].copy()]

            total_reward = 0.0
            done = False
            step = 0

            while not done:
                step += 1
                state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                state_tensor = self.obs_normalizer.normalize(state_tensor)

                if step in self.snapshot_steps:
                    save_path = os.path.join(
                        self.save_dir, f"ep{ep}_step{step:03d}_seed{seed}.png"
                    )
                    plot_point_maze_pillar_debug(
                        model=self.model, estimator=self.estimator, device=self.device,
                        save_path=save_path, state=state_tensor, trajectory=trajectory,
                        goal=goal, agent_pos=obs[:2].copy(), agent_vel=obs[2:4].copy(),
                        start_pos=start_pos, step_idx=step, episode_idx=ep,
                        title=f"PointMazePillar (Seed {seed})",
                    )

                with torch.no_grad():
                    cps = self.model(state_tensor)
                    state_exp = state_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                    q_vals = self.estimator(state_exp, cps).squeeze(-1)
                    best_idx = q_vals.argmax(dim=1)
                    action = cps[0, best_idx[0]].cpu().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                trajectory.append(obs[:2].copy())
                done = terminated or truncated

            success = terminated
            self.all_results.append({
                "total_reward": total_reward, "episode_length": step, "success": success,
            })
            print(f"  Episode {ep}: steps={step}, reward={total_reward:.2f}, success={success}")

        return self.all_results

    def close(self):
        self.env.close()

    def get_summary(self):
        if not self.all_results:
            return {"num_episodes": 0, "reward_mean": 0.0, "success_rate": 0.0}
        rewards = [r["total_reward"] for r in self.all_results]
        successes = [r["success"] for r in self.all_results]
        return {
            "num_episodes": len(self.all_results),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "success_rate": np.mean(successes),
        }
