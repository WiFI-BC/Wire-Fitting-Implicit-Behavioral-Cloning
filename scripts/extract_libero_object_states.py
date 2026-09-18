"""Solve the object-blind caveat: add `obs/object-state` to the LIBERO-Goal demos.

The downloaded libero_goal demos store only PROPRIO low-dim obs (ee_*, gripper,
joint) — no object-state — so a state-based policy can't see object positions.
But each demo also stores the full mujoco sim `states`, and the env can
regenerate any observation from a state (`regenerate_obs_from_state`). We replay
every step, pull `object-state`, and write it back into each demo's `obs/` group.

After this runs, `LiberoGoalDataset` auto-discovers `object-state` (it's in the
preferred key order), the obs vector grows, and the LIVE eval env emits the same
key — train/eval stay consistent.

Needs a GPU, because it replays every demo through MuJoCo offscreen:

    MUJOCO_GL=egl uv run python scripts/extract_libero_object_states.py

Idempotent: demos that already have `obs/object-state` are skipped unless
--overwrite. Small cameras speed the (one-time) replay; images are discarded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make `utils` importable when run as `python scripts/...` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--camera-size", type=int, default=84,
                    help="Render size during replay (images discarded; small=fast).")
    ap.add_argument("--object-key", default="object-state")
    args = ap.parse_args()

    if "MUJOCO_GL" not in os.environ:
        print("[warn] MUJOCO_GL not set; offscreen render may fail. Try MUJOCO_GL=egl.")

    import h5py
    from envs.libero import get_task_infos
    from libero.libero.envs import OffScreenRenderEnv

    infos = get_task_infos()
    object_dims: list[int] = []

    for t in infos:
        demo_file = t["demo_file"]
        print(f"\n=== task[{t['index']}] {t['name']} ===\n  {demo_file}")
        if not os.path.exists(demo_file):
            raise FileNotFoundError(demo_file)

        # Decide whether any work is needed before booting the (heavy) env.
        with h5py.File(demo_file, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            need = args.overwrite or any(
                args.object_key not in f["data"][dk]["obs"] for dk in demo_keys
            )
        if not need:
            with h5py.File(demo_file, "r") as f:
                d = int(f["data"][demo_keys[0]]["obs"][args.object_key].shape[1])
            object_dims.append(d)
            print(f"  already has {args.object_key} (dim {d}); skipping.")
            continue

        env = OffScreenRenderEnv(
            bddl_file_name=t["bddl_file"],
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
        )
        try:
            with h5py.File(demo_file, "r+") as f:
                data = f["data"]
                for dk in demo_keys:
                    grp = data[dk]
                    states = np.asarray(grp["states"])
                    obs_grp = grp["obs"]
                    if args.object_key in obs_grp:
                        if not args.overwrite:
                            continue
                        del obs_grp[args.object_key]

                    env.reset()
                    obj = np.empty((len(states), 0), dtype=np.float32)
                    rows = []
                    for s in states:
                        o = env.regenerate_obs_from_state(s)
                        if args.object_key not in o:
                            raise KeyError(
                                f"{args.object_key!r} not in regenerated obs. "
                                f"Available: {sorted(o.keys())}"
                            )
                        rows.append(np.asarray(o[args.object_key], dtype=np.float32))
                    obj = np.stack(rows)
                    obs_grp.create_dataset(args.object_key, data=obj)
                    object_dims.append(obj.shape[1])
                # Report this task's object dim (last written).
            d = object_dims[-1]
            print(f"  wrote {args.object_key} dim {d} to {len(demo_keys)} demos.")
        finally:
            env.close()

    uniq = sorted(set(object_dims))
    print(f"\nobject-state dims across tasks: {uniq}")
    if len(uniq) != 1:
        raise SystemExit(
            "object-state dim differs across tasks — concatenation into one "
            "multi-task vector requires a constant dim. Inspect the tasks above."
        )
    obj_dim = uniq[0]
    # Report the proprio dim from the ACTUAL selected keys (after exclusions),
    # so this stays correct if the schema changes.
    from envs.libero import select_lowdim_obs_keys
    import h5py as _h5

    with _h5.File(infos[0]["demo_file"], "r") as f:
        og = f["data"][sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[0]]["obs"]
        keys = select_lowdim_obs_keys(list(og.keys()))
        proprio = sum(int(og[k].shape[1]) for k in keys if k != "object-state")
        print(f"\nselected low-dim keys: {keys}")
    print(
        f"DONE. New low-dim obs = {proprio} proprio + {obj_dim} object = "
        f"{proprio + obj_dim}.  With MiniLM goal (384): "
        f"state_dim = {proprio + obj_dim + 384}."
    )
    print("Set environments.libero_goal.state_dim in config/config.json to that.")


if __name__ == "__main__":
    main()
