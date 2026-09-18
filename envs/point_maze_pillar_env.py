"""PointMazePillar — a "more realistic" complement to dummy_bimodal's toy
obstacle-detour task, built on Gymnasium-Robotics' MuJoCo PointMaze
(already a dependency of this repo, used for Kitchen/Adroit).

Same concept as dummy_bimodal (agent must detour left or right around a
central obstacle, both sides equally valid), but:
  - real rigid-body physics (MuJoCo point-mass with actuated 2D force,
    momentum, friction) instead of a hand-coded PD point-mass
  - native Gymnasium-Robotics rendering (isometric top-down, works out of
    the box via render_mode="rgb_array")
  - fixed single start/goal (confirmed with user): every episode is the
    EXACT SAME ambiguous decision, so there's no "hunting for the
    interesting state" and the oracle only needs tuning once

State (6D, flat — no Dict obs): [x, y, vx, vy, goal_x, goal_y]. goal_x/y are
constant across episodes (fixed goal) but kept in the state vector for
consistency with how a real conditional policy would be built.
Action: 2D force in [-1, 1]^2 (native PointMaze action space, passthrough).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
from gymnasium import spaces

gym.register_envs(gymnasium_robotics)

# 2-corridor maze: a central pillar (the 3x3 block of '1's) with open
# corridors on either side connecting a fixed start ('R') to a fixed goal
# ('G'), symmetric left/right around the pillar.
PILLAR_MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, "R", 1, 1, 1, "G", 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

# Fixed start/goal in maze XY (world coordinates the underlying env already
# uses, NOT the pos-noise-free minimum a manual override needs — see reset()).
FIXED_START_XY = np.array([-2.0, 0.0], dtype=np.float64)
FIXED_GOAL_XY = np.array([2.0, 0.0], dtype=np.float64)
SUCCESS_DISTANCE = 0.45  # matches PointMaze's default distance_threshold

# PILLAR_MAZE_MAP's outer ring of '1's IS a real wall, not just an implicit
# boundary — cell_rowcol_to_xy confirms wall cells centered at x,y = +-3
# (maze_size_scaling=1, so each cell is 1 unit wide -> wall inner face sits
# at 3 - 0.5 = 2.5). Diagnostic plots draw the pillar but were silently
# omitting this outer wall — same geometry, just also worth showing.
WALL_INNER_HALF_EXTENT = 2.5  # inner (walkable-side) face of the outer wall
WALL_OUTER_HALF_EXTENT = 3.5  # outer face (matches wall cell centers +- 0.5)


def draw_maze_walls(ax, color, alpha=0.35, zorder=2, thickness=None) -> None:
    """Draw the maze's outer boundary wall as a 4-piece frame (avoids
    needing a rectangle-with-a-hole patch). Caller draws the pillar itself
    separately — same color/alpha keeps both walls visually identical."""
    import matplotlib.pyplot as plt

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    o, i = WALL_OUTER_HALF_EXTENT, WALL_INNER_HALF_EXTENT
    if thickness is not None:
        # Draw a band of exactly this width outside the inner boundary, instead of
        # filling all the way to the outer wall extent / axes limits.
        o = i + float(thickness)
    # Clip the outer edge to the axes limits so the frame isn't cut off by
    # whatever xlim/ylim the caller already set.
    left, right = max(xmin, -o), min(xmax, o)
    bottom, top = max(ymin, -o), min(ymax, o)
    # Each tuple is (bottom-left-x, bottom-left-y, width, height). Top/bottom
    # strips span the full clipped x-range (so they also cover the corners);
    # left/right strips only need the middle y-band to avoid double-drawing
    # those corners (harmless overlap either way, same color/alpha).
    pieces = [
        (left, i, right - left, top - i),        # top strip
        (left, bottom, right - left, -i - bottom),  # bottom strip
        (left, -i, -i - left, 2 * i),             # left strip
        (i, -i, right - i, 2 * i),                # right strip
    ]
    for x, y, w, h in pieces:
        if w > 0 and h > 0:
            ax.add_patch(plt.Rectangle((x, y), w, h, color=color, alpha=alpha, zorder=zorder))


class PointMazePillarEnv(gym.Env):
    """Thin wrapper around PointMaze_UMaze-v3 with a fixed start/goal pillar
    layout — the underlying env's own random reset-cell sampling is
    bypassed by directly overriding qpos/goal after its reset() runs (needed
    for the MuJoCo model/data structures to exist first)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 300):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._inner = gym.make(
            "PointMaze_UMaze-v3",
            maze_map=PILLAR_MAZE_MAP,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        self.action_space = self._inner.action_space  # Box(-1, 1, (2,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self._steps = 0

    def _flatten_obs(self, dict_obs) -> np.ndarray:
        return np.concatenate([
            dict_obs["observation"].astype(np.float32),
            dict_obs["desired_goal"].astype(np.float32),
        ])

    def reset(self, seed=None, options=None):
        # Let the inner env's reset() run first (builds/resets the MuJoCo
        # sim state) — its randomly-chosen start/goal cell gets overwritten
        # immediately after.
        dict_obs, info = self._inner.reset(seed=seed)
        u = self._inner.unwrapped
        u.data.qpos[:2] = FIXED_START_XY
        u.data.qvel[:2] = 0.0
        u.goal = FIXED_GOAL_XY.copy()
        u.update_target_site_pos()
        mujoco.mj_forward(u.model, u.data)

        self._steps = 0
        dict_obs = {
            "observation": np.concatenate([FIXED_START_XY, [0.0, 0.0]]),
            "achieved_goal": FIXED_START_XY.copy(),
            "desired_goal": FIXED_GOAL_XY.copy(),
        }
        return self._flatten_obs(dict_obs), info

    def step(self, action):
        self._steps += 1
        dict_obs, reward, terminated, truncated, info = self._inner.step(action)
        pos = dict_obs["observation"][:2]
        dist = float(np.linalg.norm(pos - FIXED_GOAL_XY))
        success = dist < SUCCESS_DISTANCE
        terminated = bool(success)
        truncated = bool(self._steps >= self.max_episode_steps)
        info = dict(info)
        info["success"] = success
        info["distance_to_goal"] = dist
        return self._flatten_obs(dict_obs), float(reward), terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()


gym.register(
    id="PointMazePillar-v0",
    entry_point="envs.point_maze_pillar_env:PointMazePillarEnv",
    max_episode_steps=300,
)
