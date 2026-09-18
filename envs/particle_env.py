"""Simple particle environment with gymnasium wrapper.

Based on the Google IBC particle environment:
https://github.com/google-research/ibc/blob/master/environments/particle/particle.py

The task is to go to the first goal (green), then the second goal (blue).
"""

import copy
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np


class ParticleEnv(gym.Env):
    """Simple particle environment with gymnasium wrapper.
    
    The env is configurable but the default is:
    "go to the green goal, then the blue goal"
    
    Observation space (4N dimensions):
        - position of the agent (N dimensions)
        - velocity of the agent (N dimensions)
        - position of the first goal (N dimensions)
        - position of the second goal (N dimensions)
    
    Action space (N dimensions):
        - position setpoint for the agent (N dimensions)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    
    def __init__(
        self,
        n_steps: int = 50,
        n_dim: int = 2,
        hide_velocity: bool = False,
        seed: Optional[int] = None,
        dt: float = 0.005,  # 200 Hz internal simulation
        repeat_actions: int = 10,  # 10 makes control 20 Hz
        k_p: float = 10.0,
        k_v: float = 5.0,
        goal_distance: float = 0.05,
        render_mode: Optional[str] = None,
    ):
        """Creates an env instance.
        
        Args:
            n_steps: Number of steps until done.
            n_dim: Number of dimensions.
            hide_velocity: Whether to hide velocity info from agent.
            seed: Random seed.
            dt: Timestep for internal simulation.
            repeat_actions: Repeat the action this many times, each for dt.
            k_p: P gain in PD controller.
            k_v: D gain in PD controller.
            goal_distance: Acceptable distance to goals for success.
            render_mode: Render mode ('human', 'rgb_array', or None).
        """
        super().__init__()
        
        self.n_steps = n_steps
        self.n_dim = n_dim
        self.hide_velocity = hide_velocity
        self.dt = dt
        self.repeat_actions = repeat_actions
        self.k_p = k_p
        self.k_v = k_v
        self.goal_distance = goal_distance
        self.render_mode = render_mode
        
        # Random state
        self._rng = np.random.RandomState(seed=seed)
        
        # Action space: position setpoint in [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_dim,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = self._create_observation_space()
        
        # Internal state
        self.steps = 0
        self.obs_log: list[dict] = []
        self.act_log: list[dict] = []
        self.min_dist_to_first_goal = np.inf
        self.min_dist_to_second_goal = np.inf
        self.img_save_dir = None
        self.reset_counter = 0
        
        # Render
        self.fig = None
        self.ax = None
        
    def _create_observation_space(self) -> spaces.Box:
        """Create the observation space."""
        # Full observation: pos_agent, vel_agent, pos_first_goal, pos_second_goal
        # Position in [0, 1], velocity unbounded but typically small
        low = np.concatenate([
            np.zeros(self.n_dim),  # pos_agent
            np.full(self.n_dim, -np.inf) if not self.hide_velocity else np.array([]),
            np.zeros(self.n_dim),  # pos_first_goal
            np.zeros(self.n_dim),  # pos_second_goal
        ]).astype(np.float32)
        
        high = np.concatenate([
            np.ones(self.n_dim),   # pos_agent
            np.full(self.n_dim, np.inf) if not self.hide_velocity else np.array([]),
            np.ones(self.n_dim),   # pos_first_goal
            np.ones(self.n_dim),   # pos_second_goal
        ]).astype(np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def seed(self, seed: Optional[int] = None):
        """Set the random seed."""
        self._rng = np.random.RandomState(seed=seed)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed=seed)
        
        self.reset_counter += 1
        self.steps = 0
        self.obs_log = []
        self.act_log = []
        
        # Initialize state
        obs_dict = {
            'pos_agent': self._rng.rand(self.n_dim).astype(np.float32),
            'vel_agent': np.zeros(self.n_dim, dtype=np.float32),
            'pos_first_goal': self._rng.rand(self.n_dim).astype(np.float32),
            'pos_second_goal': self._rng.rand(self.n_dim).astype(np.float32),
        }
        
        self.obs_log.append(obs_dict)
        self.min_dist_to_first_goal = np.inf
        self.min_dist_to_second_goal = np.inf
        
        return self._get_flat_observation(), {}
    
    def _get_flat_observation(self) -> np.ndarray:
        """Get flattened observation vector."""
        obs = self.obs_log[-1]
        if self.hide_velocity:
            return np.concatenate([
                obs['pos_agent'],
                obs['pos_first_goal'],
                obs['pos_second_goal'],
            ])
        else:
            return np.concatenate([
                obs['pos_agent'],
                obs['vel_agent'],
                obs['pos_first_goal'],
                obs['pos_second_goal'],
            ])
    
    def _internal_step(self, action: np.ndarray):
        """Perform one internal simulation step."""
        self.act_log.append({'pos_setpoint': action.copy()})
        obs = self.obs_log[-1]
        
        # PD control: u = k_p * (x_desired - x) - k_v * x_dot
        u_agent = self.k_p * (action - obs['pos_agent']) - self.k_v * obs['vel_agent']
        
        # Euler integration
        new_pos = obs['pos_agent'] + obs['vel_agent'] * self.dt
        new_vel = obs['vel_agent'] + u_agent * self.dt
        
        new_obs = copy.deepcopy(obs)
        new_obs['pos_agent'] = new_pos.astype(np.float32)
        new_obs['vel_agent'] = new_vel.astype(np.float32)
        self.obs_log.append(new_obs)
    
    def _dist(self, goal: np.ndarray) -> float:
        """Distance from current position to goal."""
        return np.linalg.norm(self.obs_log[-1]['pos_agent'] - goal)
    
    def _get_reward(self, done: bool) -> float:
        """Compute reward. Returns 1.0 if agent hits both goals and stays at second."""
        # Update minimum distances
        self.min_dist_to_first_goal = min(
            self._dist(self.obs_log[0]['pos_first_goal']),
            self.min_dist_to_first_goal
        )
        self.min_dist_to_second_goal = min(
            self._dist(self.obs_log[0]['pos_second_goal']),
            self.min_dist_to_second_goal
        )
        
        if done:
            hit_first = self.min_dist_to_first_goal < self.goal_distance
            hit_second = self.min_dist_to_second_goal < self.goal_distance
            return 1.0 if (hit_first and hit_second) else 0.0
        return 0.0
    
    @property
    def succeeded(self) -> bool:
        """Check if the task was successful."""
        hit_first = self.min_dist_to_first_goal < self.goal_distance
        hit_second = self.min_dist_to_second_goal < self.goal_distance
        current_distance = self._dist(self.obs_log[0]['pos_second_goal'])
        at_second = current_distance < self.goal_distance
        return hit_first and hit_second and at_second
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        self.steps += 1
        
        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        
        # Internal simulation steps
        for _ in range(self.repeat_actions):
            self._internal_step(action)
        
        observation = self._get_flat_observation()
        terminated = self.steps >= self.n_steps
        reward = self._get_reward(terminated)
        
        info = {
            'min_dist_to_first_goal': self.min_dist_to_first_goal,
            'min_dist_to_second_goal': self.min_dist_to_second_goal,
            'success': self.succeeded if terminated else False,
        }
        
        return observation, reward, terminated, False, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.n_dim == 2:
            return self._render_2d()
        else:
            return self._render_nd()
    
    def _render_2d(self) -> np.ndarray:
        """Render 2D visualization."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            if self.render_mode == "human":
                plt.ion()
        
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Step {self.steps}/{self.n_steps}')
        
        # Draw goals
        first_goal = self.obs_log[0]['pos_first_goal']
        second_goal = self.obs_log[0]['pos_second_goal']
        
        self.ax.scatter(*first_goal, c='green', s=200, marker='*', label='First Goal', zorder=5)
        self.ax.scatter(*second_goal, c='blue', s=200, marker='*', label='Second Goal', zorder=5)
        
        # Draw goal circles
        circle1 = plt.Circle(first_goal, self.goal_distance, color='green', fill=False, linestyle='--')
        circle2 = plt.Circle(second_goal, self.goal_distance, color='blue', fill=False, linestyle='--')
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)
        
        # Draw trajectory
        if len(self.obs_log) > 1:
            trajectory = np.array([obs['pos_agent'] for obs in self.obs_log])
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.5, linewidth=1)
        
        # Draw current position
        current_pos = self.obs_log[-1]['pos_agent']
        self.ax.scatter(*current_pos, c='red', s=100, marker='o', label='Agent', zorder=10)
        
        self.ax.legend(loc='upper right')
        
        # Render canvas
        self.fig.canvas.draw()
        if self.render_mode == "human":
            self.fig.canvas.flush_events()
        
        # Convert to RGB array (buffer_rgba returns RGBA, slice to RGB)
        buf = self.fig.canvas.buffer_rgba()
        data = np.asarray(buf, dtype=np.uint8)
        return data[:, :, :3].copy()
    
    def _render_nd(self) -> np.ndarray:
        """Render N-D visualization as 1D projections."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(self.n_dim, 1, figsize=(10, 2 * self.n_dim))
            if self.n_dim == 1:
                self.ax = [self.ax]
            if self.render_mode == "human":
                plt.ion()
        
        for d in range(self.n_dim):
            self.ax[d].clear()
            self.ax[d].set_xlim(0, self.n_steps)
            self.ax[d].set_ylim(0, 1)
            self.ax[d].set_ylabel(f'Dim {d}')
            
            # Plot trajectory
            if len(self.obs_log) > 1:
                trajectory = [obs['pos_agent'][d] for obs in self.obs_log]
                self.ax[d].plot(range(len(trajectory)), trajectory, 'r-')
            
            # Plot goals as horizontal lines
            first_goal = self.obs_log[0]['pos_first_goal'][d]
            second_goal = self.obs_log[0]['pos_second_goal'][d]
            self.ax[d].axhline(y=first_goal, color='green', linestyle='--', label='First Goal')
            self.ax[d].axhline(y=second_goal, color='blue', linestyle='--', label='Second Goal')
        
        self.ax[-1].set_xlabel('Step')
        self.ax[0].legend()
        
        # Render canvas
        self.fig.canvas.draw()
        if self.render_mode == "human":
            self.fig.canvas.flush_events()
        
        # Convert to RGB array (buffer_rgba returns RGBA, slice to RGB)
        buf = self.fig.canvas.buffer_rgba()
        data = np.asarray(buf, dtype=np.uint8)
        return data[:, :, :3].copy()
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Register the environment
gym.register(
    id='Particle-v0',
    entry_point='envs.particle_env:ParticleEnv',
    max_episode_steps=50,
)


if __name__ == "__main__":
    # Test the environment
    print("Testing ParticleEnv...")
    
    env = ParticleEnv(n_dim=2, render_mode="human")
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    total_reward = 0
    for step in range(50):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            break
    
    print(f"Episode finished after {step + 1} steps")
    print(f"Total reward: {total_reward}")
    print(f"Success: {info.get('success', False)}")
    
    env.close()
