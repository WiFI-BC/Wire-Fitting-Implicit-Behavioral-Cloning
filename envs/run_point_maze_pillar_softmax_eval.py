"""Stochastic rollout eval for PointMazePillar-v0: instead of the usual
argmax-over-CPs action selection, sample the action from the control-point
cloud with probability = softmax(Q-values). Runs N episodes, plots every
resulting trajectory overlaid on the maze, and reports how many actually
routed through the top vs bottom corridor.

This directly tests whether the CP cloud's "two modes" (visible in the
snapshot Q-heatmap) reflect genuine ALTERNATIVE behaviors the policy will
actually execute when not forced through greedy argmax, vs. a cloud that
LOOKS bimodal in a static plot but collapses to one path every time under
any reasonable action rule.

Rollouts are saved to trajectories.npz next to the plot, so the figure can be
restyled without re-running the policy (--from-npz).

Usage:
    python -m envs.run_point_maze_pillar_softmax_eval \\
        --checkpoint checkpoints/control_point_generator.pt \\
        --episodes 100 --temperature 1.0 --output-dir plots/point_maze_pillar_softmax

    # Re-plot saved rollouts only:
    python -m envs.run_point_maze_pillar_softmax_eval \\
        --from-npz plots/point_maze_pillar_softmax/trajectories.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.point_maze_pillar_env import PointMazePillarEnv
from wifi_bc.models import ControlPointGenerator, QEstimator
from wifi_bc.normalizations import ObservationNormalizer

CONFIG_PATH = Path(
    os.environ.get("WIFI_BC_CONFIG_PATH")
    or (Path(__file__).resolve().parent.parent / "config" / "config.json")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to control_point_generator.pt (required unless --from-npz)")
    parser.add_argument("--from-npz", type=str, default=None,
                         help="Skip rollouts and re-plot a saved trajectories.npz")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="softmax(Q / temperature) — 1.0 = plain softmax(Q)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--output-dir", type=str, default="plots/point_maze_pillar_softmax")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--continue-past-success", action="store_true",
                         help="keep acting after entering the success radius (success is still scored on "
                              "first entry) until within --arrive-dist of the goal or out of steps")
    parser.add_argument("--goal-region", action="store_true",
                         help="draw the success radius as a shaded disc under the goal marker")
    parser.add_argument("--arrive-dist", type=float, default=0.1,
                         help="with --continue-past-success, stop once this close to the goal")
    args = parser.parse_args()

    if args.from_npz:
        trajectories, successes, corridors = _load_rollouts(args.from_npz)
        _plot(trajectories, successes, corridors, args.output_dir, args.temperature, args.goal_region)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --from-npz is given")

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    env_config = config["environments"]["point_maze_pillar"]
    em = env_config["model"]

    device = args.device
    cp_gen = ControlPointGenerator(
        input_dim=6, output_dim=2, control_points=em["control_points"],
        hidden_dims=[em["num_neurons"]] * em["num_hidden_layers"], action_bounds=(-1.0, 1.0),
    )
    cp_gen.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    cp_gen.to(device).eval()

    ckpt_dir = os.path.dirname(args.checkpoint)
    q_est = QEstimator(
        state_dim=6, action_dim=2,
        hidden_dims=[em["num_neurons"]] * em["num_hidden_layers"],
    )
    q_est.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "q_estimator.pt"), map_location=device, weights_only=True)
    )
    q_est.to(device).eval()

    obs_normalizer = ObservationNormalizer(env_id="PointMazePillar-v0", device=device, frame_stack=1)
    rng = np.random.default_rng(args.seed)

    def select_action_softmax(obs: np.ndarray) -> np.ndarray:
        st = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        st = obs_normalizer.normalize(st)
        with torch.no_grad():
            cps = cp_gen(st)  # (1, N, 2)
            st_exp = st.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_est(st_exp, cps).squeeze(-1).squeeze(0).cpu().numpy()  # (N,)
        logits = q / args.temperature
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        idx = rng.choice(len(probs), p=probs)
        return cps[0, idx].cpu().numpy()

    env = PointMazePillarEnv(max_episode_steps=args.max_steps)
    trajectories = []
    successes = []
    corridors = []  # "top" | "bottom" | "none" (never left the pillar's y-band)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        traj = [obs[:2].copy()]
        max_y = obs[1]
        min_y = obs[1]
        succeeded = False
        for t in range(args.max_steps):
            action = select_action_softmax(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs[:2].copy())
            max_y = max(max_y, obs[1])
            min_y = min(min_y, obs[1])
            succeeded = succeeded or bool(terminated)
            if args.continue_past_success:
                # Success is still scored on first entry into the goal radius; the
                # rollout keeps going so the plotted path shows the approach to the goal.
                if (succeeded and info["distance_to_goal"] < args.arrive_dist) or truncated:
                    break
            elif terminated or truncated:
                break
        trajectories.append(np.array(traj))
        successes.append(succeeded)
        if max_y > 1.6:
            corridors.append("top")
        elif min_y < -1.6:
            corridors.append("bottom")
        else:
            corridors.append("none")
        print(f"  ep {ep}: steps={len(traj)-1} success={succeeded} corridor={corridors[-1]}")

    env.close()

    successes = np.array(successes)
    corridors = np.array(corridors)
    print(f"\nSuccess rate: {successes.mean():.1%} ({successes.sum()}/{args.episodes})")
    for c in ("top", "bottom", "none"):
        n = (corridors == c).sum()
        n_succ = (successes & (corridors == c)).sum()
        print(f"  corridor={c}: {n}/{args.episodes} ({n/args.episodes:.1%}), "
              f"success within: {n_succ}/{n if n else 1}")

    _save_rollouts(trajectories, successes, corridors, args.output_dir)
    _plot(trajectories, successes, corridors, args.output_dir, args.temperature, args.goal_region)


def _save_rollouts(trajectories, successes, corridors, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "trajectories.npz")
    arrays = {f"traj_{i}": t for i, t in enumerate(trajectories)}
    np.savez(path, successes=np.asarray(successes), corridors=np.asarray(corridors), **arrays)
    print(f"Rollouts saved to {path}")


def _load_rollouts(path):
    data = np.load(path)
    n = sum(1 for k in data.files if k.startswith("traj_"))
    trajectories = [data[f"traj_{i}"] for i in range(n)]
    return trajectories, data["successes"], data["corridors"]


def _plot(trajectories, successes, corridors, output_dir, temperature, goal_region=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from envs.plot_style import apply_house_style, style_axes, SURFACE, TEXT, TEXT2, C_WIFI
    from envs.point_maze_pillar_env import draw_maze_walls, WALL_INNER_HALF_EXTENT

    apply_house_style()
    os.makedirs(output_dir, exist_ok=True)
    PILLAR_HALF_EXTENT = 1.5

    # Transparent canvas, no grid or spines: the maze walls frame the figure.
    # Drawn at 3.5 in for a half-column (~1.75 in) placement: everything prints at half size.
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="none")
    style_axes(ax, hide_spines=("top", "right", "left", "bottom"))
    ax.set_facecolor("none")
    WALL = 0.1  # thin outer wall; axes stop at its outer edge
    lim = WALL_INNER_HALF_EXTENT + WALL
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(False)

    draw_maze_walls(ax, color=TEXT2, alpha=0.35, thickness=WALL)
    pillar = plt.Rectangle(
        (-PILLAR_HALF_EXTENT, -PILLAR_HALF_EXTENT), 2 * PILLAR_HALF_EXTENT, 2 * PILLAR_HALF_EXTENT,
        color=TEXT2, alpha=0.35, zorder=2,
    )
    ax.add_patch(pillar)

    # Every rollout is WiFI-BC, so all take its paper colour; the corridor (mode) is the variant,
    # shown by line style. Failed rollouts are drawn fainter.
    corridor_style = {"top": "-", "bottom": (0, (2.5, 1.2)), "none": ":"}
    counted = {"top": False, "bottom": False, "none": False}
    for traj, success, corridor in zip(trajectories, successes, corridors):
        corridor = str(corridor)
        label = None
        if not counted[corridor]:
            label = corridor.capitalize()
            counted[corridor] = True
        ax.plot(traj[:, 0], traj[:, 1], linestyle=corridor_style[corridor], color=C_WIFI,
                alpha=0.6 if success else 0.25, linewidth=0.6, label=label, zorder=3)

    if goal_region:
        # Episodes end on entering this radius, so every successful trajectory stops on its edge.
        from envs.point_maze_pillar_env import SUCCESS_DISTANCE
        ax.add_patch(plt.Circle((2.0, 0.0), SUCCESS_DISTANCE, facecolor=TEXT2, alpha=0.18, edgecolor="none", zorder=2.5))
        ax.add_patch(plt.Circle((2.0, 0.0), SUCCESS_DISTANCE, fill=False, edgecolor=TEXT2, linewidth=0.6,
                                linestyle="--", zorder=4))
    ax.scatter([-2.0], [0.0], color=TEXT, s=30, marker="s", zorder=5, label="Start")
    ax.scatter([2.0], [0.0], color=TEXT, s=240, marker="*", edgecolors=SURFACE,
               linewidths=0.6, zorder=5, label="Goal")
    # The trajectory bundle fills the entire frame, so the legend goes below the
    # maze as a single row: Top, Bottom, Start, Goal.
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("Top", "Bottom", "Start", "Goal") if k in labels]
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(order), frameon=False,
                       fontsize=13, labelcolor=TEXT, handlelength=1.1, handletextpad=0.4,
                       columnspacing=1.0, borderaxespad=0.2)
    # Legend handles: thick opaque corridor lines, and Start/Goal markers set
    # explicitly (markerscale would shrink the square).
    marker_sizes = {"Start": 45, "Goal": 150}
    handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
    for handle, text in zip(handles, legend.get_texts()):
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([marker_sizes.get(text.get_text(), 260)])
        else:
            handle.set_linewidth(2.2)
            handle.set_alpha(1.0)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    path = os.path.join(output_dir, "trajectories.png")
    fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
