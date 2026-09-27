"""Base simulation class for testing trained policies on gym environments."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import Any
import gymnasium as gym

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from wifi_bc.normalizations import ObservationNormalizer


class BaseSimulation(ABC):
    """Abstract base class for running simulations with trained policies.
    
    This class provides the foundation for testing trained control point generators
    on gymnasium environments. Subclasses should implement environment-specific
    setup and action selection logic.
    """

    def __init__(
        self,
        env_id: str,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 400,
        frame_stack: int = 1,
        particle_n_dim: int | None = None,
    ) -> None:
        """Initialize the simulation.
        
        Args:
            env_id: The gymnasium environment ID (e.g., 'AdroitHandPen-v1').
            control_point_generator: The trained policy model that generates control points.
            q_estimator: The trained Q-value estimator.
            device: The device to run computations on ('cpu' or 'cuda').
            max_episode_steps: Maximum steps per episode.
        """
        self.env_id = env_id
        self.control_point_generator = control_point_generator
        self.q_estimator = q_estimator
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.env = None
        self.results: list[dict[str, Any]] = []
        self.frame_stack = frame_stack
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=frame_stack)
        
        # Observation normalizer (uses official bounds from JSON file)
        self.obs_normalizer = ObservationNormalizer(
            env_id=env_id,
            device=device,
            frame_stack=frame_stack,
            particle_n_dim=particle_n_dim,
        )

    @abstractmethod
    def create_env(self) -> gym.Env:
        """Create and return the gymnasium environment."""
        pass

    def _render_callback(self, reward: float, total_reward: float) -> None:
        """Callback for rendering custom overlays (optional)."""
        pass

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Linearly map a model-space action to the env's native action box.

        Default: no-op (the model already emits raw env actions). Subclasses
        that train in a normalized action space (e.g. PushingSimulation when
        norm_stats are present) override this to apply the inverse map.

        Called by paths that produce actions outside `select_action`,
        notably the Langevin-refined eval wrapper.
        """
        return action

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Select action from control points based on Q-values.
        
        Uses the Q-estimator to evaluate each control point and selects
        the one with the maximum Q-value.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The selected action as a numpy array.
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)  # Normalize to [0, 1]
        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, action_dim)
            
            # Expand state to match control points: (1, state_dim) -> (1, N, state_dim)
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            
            # Get Q-values for all control points (state-conditioned)
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)  # (1, N)
            
            # Select control point with maximum Q-value
            best_idx = q_values.argmax(dim=1)  # (1,)
            action = control_points[0, best_idx[0], :].cpu().numpy()
        return action

    def _reset_frame_buffer(self, obs: np.ndarray) -> np.ndarray:
        """Reset the frame buffer and return the stacked initial observation."""
        self._frame_buffer.clear()
        for _ in range(self.frame_stack):
            self._frame_buffer.append(obs.copy())
        return self._get_stacked_obs()

    def _update_frame_buffer(self, obs: np.ndarray) -> np.ndarray:
        """Add a new frame and return the stacked observation."""
        self._frame_buffer.append(obs.copy())
        return self._get_stacked_obs()

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate buffered frames into a single observation vector."""
        if self.frame_stack <= 1:
            return self._frame_buffer[-1]
        return np.concatenate(list(self._frame_buffer))

    def run_episode(self, seed: int | None = None) -> dict[str, Any]:
        """Run a single episode and return metrics.
        
        Args:
            seed: Optional random seed for this episode.
            
        Returns:
            A dictionary containing episode metrics.
        """
        if self.env is None:
            self.env = self.create_env()
        
        obs, info = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = self.select_action(stacked_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += reward
            episode_length += 1
            
            # Render callback
            self._render_callback(reward, total_reward)

            done = terminated or truncated
        
        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            # Gymnasium's convention: the env reports task success in `info`.
            # Reported here so an environment added later gets a correct success
            # rate without overriding this method — every environment shipped in
            # the paper overrides it, which used to hide the fact that the base
            # class silently returned no success at all (so a new env scored 0%
            # however well it did).
            "success": bool(info.get("success", info.get("is_success", False))),
        }

    def run_simulation(
        self, 
        num_episodes: int = 100, 
        seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Run the simulation for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to run.
            seed: Base random seed. Each episode uses seed + episode_idx.
            
        Returns:
            A list of dictionaries containing per-episode metrics.
        """
        self.results = []
        self.control_point_generator.eval()
        self.q_estimator.eval()

        for idx in range(num_episodes):
            episode_seed = (seed + idx) if seed is not None else None
            result = self.run_episode(seed=episode_seed)
            result["episode_index"] = idx
            self.results.append(result)
                
        return self.results

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics from the simulation results.
        
        Returns:
            A dictionary with summary statistics.
        """
        if not self.results:
            return {}
        
        episode_lengths = [r.get("episode_length", 0) for r in self.results]
        total_rewards = [r.get("total_reward", 0.0) for r in self.results]
        
        return {
            "num_episodes": len(self.results),
            "reward_mean": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "episode_length_mean": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
        }

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
