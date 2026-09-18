"""Visualization for PointMazePillar-v0.

Actions are 2D forces, not an angle, so this uses Cartesian scatter/heatmap
panels instead of dummy_bimodal's polar ones. 3 panels:
1. CP cloud in force space (fx, fy), colored by score, with the single
   highest-scoring CP ring-highlighted.
2. Score heatmap over the unit force disk, with the two analytically-
   computed expert forces (top/bottom corridor) marked for reference.
3. 2D top-down map: pillar (square), start, goal, trajectory so far.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from envs.plot_style import apply_house_style, style_axes, SURFACE, TEXT, TEXT2, GRID, C1, C2
from envs.point_maze_pillar_env import draw_maze_walls

PILLAR_HALF_EXTENT = 1.5  # matches the maze's 3x3 wall block, world coords
SCORE_CMAP = LinearSegmentedColormap.from_list("score_cmap", [C1, C2])  # blue -> orange, matches goal/agent


def _oracle_force(pos, vel, goal, start, side, detour_y=2.0, kp=6.0, kv=3.0):
    """Recompute the 3-stage waypoint oracle's force for one side, at the
    given (pos, vel) — same logic as PointMazePillarDataset's generation
    loop, just called for a single state instead of a full rollout."""
    waypoints = [
        np.array([start[0], side * detour_y]),
        np.array([goal[0], side * detour_y]),
        goal,
    ]
    target = waypoints[0]
    for i, wp in enumerate(waypoints):
        if np.linalg.norm(pos - wp) < 0.35 and i < len(waypoints) - 1:
            continue
        target = wp
        break
    force = kp * (target - pos) - kv * vel
    return np.clip(force, -1.0, 1.0)


def plot_point_maze_pillar_debug(
    model,
    estimator,
    device,
    save_path,
    state,
    trajectory,
    goal,
    agent_pos,
    agent_vel,
    start_pos,
    step_idx,
    episode_idx,
    title="PointMazePillar Diagnostic",
):
    model.eval()
    estimator.eval()
    apply_house_style()

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    expert_force_a = _oracle_force(agent_pos, agent_vel, goal, start_pos, side=1.0)
    expert_force_b = _oracle_force(agent_pos, agent_vel, goal, start_pos, side=-1.0)
    expert_forces = np.stack([expert_force_a, expert_force_b])  # (2, 2)

    with torch.no_grad():
        control_points = model(state)  # (1, N, 2)
        cp_actions = control_points.squeeze(0).cpu().numpy()  # (N, 2)

        state_expanded = state.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_cps = estimator(state_expanded, control_points).squeeze(-1).squeeze(0).cpu().numpy()
        if q_cps.ndim == 0:
            q_cps = np.array([q_cps.item()])

    fig = plt.figure(figsize=(18, 6.5), facecolor=SURFACE)
    fig.suptitle(f"{title} | Episode {episode_idx}, Step {step_idx}", fontsize=20, color=TEXT)

    expert_colors = [C1, C2]

    # ========== Plot 1: CP cloud in force space ==========
    ax1 = fig.add_subplot(1, 3, 1)
    style_axes(ax1, hide_spines=("top", "right"))
    ax1.set_title("1. Control Points\n(color = score, ring = highest)", fontsize=14, color=TEXT)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.add_patch(plt.Circle((0, 0), 1.0, fill=False, linestyle='--', color=GRID, linewidth=1.2))
    norm1 = plt.Normalize(q_cps.min(), q_cps.max())
    sc1 = ax1.scatter(cp_actions[:, 0], cp_actions[:, 1], c=q_cps, cmap=SCORE_CMAP, norm=norm1,
                       s=90, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    cb1 = plt.colorbar(sc1, ax=ax1)
    cb1.set_label('score', fontsize=13, color=TEXT2)
    cb1.ax.tick_params(colors=TEXT2, labelsize=11)
    best_idx = int(np.argmax(q_cps))
    ax1.scatter([cp_actions[best_idx, 0]], [cp_actions[best_idx, 1]], s=260, facecolors='none',
                edgecolors=TEXT, linewidths=1.8, zorder=6, label='Highest-score CP')
    ax1.axhline(0, color=GRID, linewidth=1)
    ax1.axvline(0, color=GRID, linewidth=1)
    ax1.set_xlabel('force_x', fontsize=13)
    ax1.set_ylabel('force_y', fontsize=13)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12, labelcolor=TEXT)

    # ========== Plot 2: score heatmap over force disk ==========
    ax2 = fig.add_subplot(1, 3, 2)
    style_axes(ax2, hide_spines=("top", "right"))
    ax2.set_title("2. Score Heatmap\n(force disk)", fontsize=14, color=TEXT)
    n = 60
    fx = np.linspace(-1, 1, n)
    fy = np.linspace(-1, 1, n)
    FX, FY = np.meshgrid(fx, fy)
    grid = np.stack([FX.ravel(), FY.ravel()], axis=1).astype(np.float32)
    inside = (grid[:, 0] ** 2 + grid[:, 1] ** 2) <= 1.0
    grid_t = torch.from_numpy(grid).to(device)
    state_grid = state.repeat(grid.shape[0], 1)
    with torch.no_grad():
        q_grid = estimator(state_grid, grid_t).squeeze(-1).cpu().numpy()
    q_grid_masked = np.where(inside, q_grid, np.nan).reshape(n, n)
    im = ax2.pcolormesh(FX, FY, q_grid_masked, cmap=SCORE_CMAP, shading='auto')
    cb = plt.colorbar(im, ax=ax2)
    cb.set_label('score', fontsize=13, color=TEXT2)
    cb.ax.tick_params(colors=TEXT2, labelsize=11)
    for i in range(2):
        ax2.scatter([expert_forces[i, 0]], [expert_forces[i, 1]], color=expert_colors[i],
                    marker='*', s=280, edgecolors=SURFACE, linewidths=1.2, zorder=5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('force_x', fontsize=13)
    ax2.set_ylabel('force_y', fontsize=13)
    ax2.tick_params(axis='both', labelsize=11)

    # ========== Plot 3: 2D top-down map ==========
    ax3 = fig.add_subplot(1, 3, 3)
    style_axes(ax3, hide_spines=("top", "right"))
    ax3.set_title("3. Top-down Map", fontsize=14, color=TEXT)
    ax3.set_xlim(-3.2, 3.2)
    ax3.set_ylim(-3.2, 3.2)
    ax3.set_aspect('equal')
    ax3.grid(True, color=GRID, linewidth=1)
    draw_maze_walls(ax3, color=TEXT2, alpha=0.35)
    pillar = plt.Rectangle(
        (-PILLAR_HALF_EXTENT, -PILLAR_HALF_EXTENT), 2 * PILLAR_HALF_EXTENT, 2 * PILLAR_HALF_EXTENT,
        color=TEXT2, alpha=0.35, zorder=2,
    )
    ax3.add_patch(pillar)
    traj_arr = np.array(trajectory)
    if len(traj_arr) > 1:
        ax3.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=TEXT, alpha=0.7, linewidth=1.5, label='Path')
    ax3.scatter([start_pos[0]], [start_pos[1]], color=TEXT, s=80, marker='s', zorder=4, label='Start')
    ax3.scatter([goal[0]], [goal[1]], color=C1, s=220, marker='*', edgecolors=SURFACE,
                linewidths=1.0, zorder=5, label='Goal')
    ax3.scatter([agent_pos[0]], [agent_pos[1]], color=C2, s=110, marker='o', edgecolors=SURFACE,
                linewidths=1.0, zorder=5, label='Agent')
    ax3.legend(loc='upper left', frameon=False, fontsize=12, labelcolor=TEXT)
    ax3.set_xlabel('X', fontsize=13)
    ax3.set_ylabel('Y', fontsize=13)
    ax3.tick_params(axis='both', labelsize=11)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.22, wspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
