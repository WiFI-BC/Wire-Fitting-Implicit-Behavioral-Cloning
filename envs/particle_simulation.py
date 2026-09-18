"""Simulation class for the Particle-v0 environment."""

import os

import numpy as np
import torch
from PIL import Image

from .base_simulation import BaseSimulation
from .particle_env import ParticleEnv


class ParticleSimulation(BaseSimulation):
    """Simulation for testing trained policies on the Particle environment.
    
    This class evaluates the trained control point generator by running
    episodes in the particle environment and collecting metrics.
    """

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        n_dim: int = 2,
        device: str = "cpu",
        max_episode_steps: int = 50,
        render_mode: str | None = None,
        frame_stack: int = 1,
        save_gif_dir: str | None = None,
        norm_stats: dict | None = None,
    ) -> None:
        """Initialize the Particle simulation.

        Args:
            norm_stats: Optional {"act_min","act_max"} arrays. When provided,
                control points are normalized to [0,1] before the estimator
                call (matches estimator training distribution for ibc_with_cps).
        """
        super().__init__(
            env_id="Particle-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
            particle_n_dim=n_dim,
        )
        self.n_dim = n_dim
        self.render_mode = render_mode
        self.save_gif_dir = save_gif_dir
        self._episode_counter = 0
        if norm_stats is not None:
            act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
            act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
            rng = act_max - act_min
            rng = np.where(rng == 0, np.ones_like(rng), rng)
            self._act_min_t = torch.from_numpy(act_min).to(device)
            self._act_rng_t = torch.from_numpy(rng).to(device)
        else:
            self._act_min_t = None
            self._act_rng_t = None

    def create_env(self) -> ParticleEnv:
        """Create the Particle gymnasium environment."""
        env = ParticleEnv(
            n_dim=self.n_dim,
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
        )
        return env

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        """Select action from control points based on Q-values.
        
        Override to handle particle action bounds [0, 1].
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)
        
        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, action_dim)

            # Expand state to match control points
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)

            # Optionally normalize control points into the estimator's training space
            if self._act_min_t is not None:
                cp_for_q = (control_points - self._act_min_t) / self._act_rng_t
            else:
                cp_for_q = control_points

            q_values = self.q_estimator(obs_expanded, cp_for_q).squeeze(-1)

            best_idx = q_values.argmax(dim=1)
            action = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())
        
        # Clip to valid range [0, 1] for particle env
        ret_action = np.clip(action, 0.0, 1.0)
        
        if return_q_range:
            return ret_action, q_range
        return ret_action

    def run_episode(self, seed: int | None = None) -> dict:
        """Run a single episode and return metrics."""
        if self.env is None:
            self.env = self.create_env()
        
        obs, info = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        frames = []
        
        while not done:
            action = self.select_action(stacked_obs)


            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += reward
            episode_length += 1
            
            # Render and capture frame
            if self.render_mode:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
            
            done = terminated or truncated
        
        # Save GIF if we have frames
        self._episode_counter += 1
        if frames and self.save_gif_dir:
            os.makedirs(self.save_gif_dir, exist_ok=True)
            gif_path = os.path.join(self.save_gif_dir, f"seed_{seed}_episode_{self._episode_counter:03d}.gif")
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=100, loop=0
            )
        
        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": info.get("success", False),
            "min_dist_to_first_goal": info.get("min_dist_to_first_goal", np.inf),
            "min_dist_to_second_goal": info.get("min_dist_to_second_goal", np.inf),
        }

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics including particle-specific metrics."""
        summary = super().get_summary()
        
        if not self.results:
            return summary
        
        # Add success rate
        successes = [r.get("success", False) for r in self.results]
        summary["success_rate"] = np.mean(successes)
        
        # Add average goal distances
        first_dists = [r.get("min_dist_to_first_goal", 0) for r in self.results]
        second_dists = [r.get("min_dist_to_second_goal", 0) for r in self.results]
        summary["avg_min_dist_first_goal"] = np.mean(first_dists)
        summary["avg_min_dist_second_goal"] = np.mean(second_dists)
        
        return summary
