"""Reach-v0 — a deliberately tiny task used to exercise the "bring your own
environment" path end to end.

A point agent in the unit square is pushed toward a randomly placed goal. The
action is a velocity setpoint, the reward is the negative distance to the goal,
and an episode succeeds when the agent gets within `goal_radius`.

It exists so the steps needed to train WiFI-BC on an environment that is not one
of the paper's seven can be demonstrated (and counted) without downloading
anything.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBS_DIM = 4      # agent xy + goal xy
ACTION_DIM = 2   # velocity setpoint


class ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_steps: int = 50, goal_radius: float = 0.05,
                 step_size: float = 0.1, render_mode: str | None = None):
        super().__init__()
        self.n_steps = int(n_steps)
        self.goal_radius = float(goal_radius)
        self.step_size = float(step_size)
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self._agent = np.zeros(2, np.float32)
        self._goal = np.zeros(2, np.float32)
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._agent, self._goal]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent = self.np_random.uniform(0, 1, 2).astype(np.float32)
        self._goal = self.np_random.uniform(0, 1, 2).astype(np.float32)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self._agent = np.clip(self._agent + a * self.step_size, 0.0, 1.0).astype(np.float32)
        self._t += 1
        dist = float(np.linalg.norm(self._agent - self._goal))
        success = dist < self.goal_radius
        terminated = bool(success)
        truncated = self._t >= self.n_steps
        return self._obs(), -dist, terminated, truncated, {"success": success, "distance": dist}
