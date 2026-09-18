"""Shared helpers for the LIBERO-Goal environment (state-based, multi-task,
language-goal-conditioned).

This module is the single source of truth shared by:
  - envs.datasets.LiberoGoalDataset          (builds training transitions)
  - envs.libero_simulation              (steps the live env at eval)
  - scripts.precompute_libero_goal_embs         (caches language embeddings)

Design notes
------------
* Observation modality is LOW-DIM state. LIBERO demo HDF5s store both RGB and
  low-dim obs under `data/demo_*/obs/<key>`. We use only the non-image keys.

* The exact set / names of low-dim obs keys differs across LIBERO releases
  (e.g. `ee_states` vs `ee_pos`+`ee_ori`, presence of `joint_states`). Rather
  than hard-code a brittle list, the dataset DISCOVERS the available low-dim
  keys from the first demo file and records them. The selected key list + the
  per-key dims are persisted in `norm_stats.pt`, and the eval simulation reads
  them back so the obs vector is byte-identical between train and eval.

* The live robosuite/LIBERO env returns obs under robosuite-style names which
  may differ from the HDF5 demo key names. `LIVE_KEY_ALIASES` maps each canonical
  (HDF5) key to the list of live-env keys to try. Confirm/extend this map once
  the LIBERO package + demos are installed (see README of this batch).

* All LIBERO package imports are LAZY (inside functions) so this module can be
  imported on machines where LIBERO is not installed (e.g. an analysis box).

LIBERO: Liu et al., NeurIPS 2023. Suite "libero_goal": 10 tasks, identical
scene/objects, the language GOAL differs per task — the canonical multimodal
test for goal-conditioned BC.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIBERO_PKG = os.path.join(_REPO_ROOT, "third_party", "LIBERO", "libero", "libero")


def _ensure_libero_config() -> None:
    """Keep LIBERO self-contained inside the repo and independent of $HOME.

    LIBERO stores absolute paths (to demos/bddl/assets) in a config file and,
    on first import, PROMPTS interactively to create it. We:
      - point LIBERO_CONFIG_PATH at <repo>/.libero (not ~/.libero), and
      - (re)write config.yaml every import with paths derived from THIS
        module's __file__, so they always point at this checkout wherever it
        has been moved or however its parent path is spelled.
    Must run before any `import libero`; all our entry points import this module
    first, so importing envs.libero is enough.
    """
    cfg_dir = os.environ.setdefault(
        "LIBERO_CONFIG_PATH", os.path.join(_REPO_ROOT, ".libero")
    )
    os.makedirs(cfg_dir, exist_ok=True)
    cfg = {
        "benchmark_root": _LIBERO_PKG,
        "bddl_files": os.path.join(_LIBERO_PKG, "bddl_files"),
        "init_states": os.path.join(_LIBERO_PKG, "init_files"),
        "datasets": os.path.normpath(os.path.join(_LIBERO_PKG, "..", "datasets")),
        "assets": os.path.join(_LIBERO_PKG, "assets"),
    }
    import yaml

    cfg_path = os.path.join(cfg_dir, "config.yaml")
    payload = yaml.dump(cfg)
    # Skip the write when content is already correct, and write ATOMICALLY
    # (temp file + os.replace) otherwise. This module rewrites the config on
    # every import; with many jobs importing simultaneously, a plain
    # open("w") truncates the file while a sibling job reads it → yaml.load
    # returns None → "'NoneType' object is not iterable" in get_libero_path
    # (killed Dstandardlibero_007 at startup). os.replace is atomic on POSIX:
    # readers see the old or the new file, never a partial one.
    try:
        with open(cfg_path, "r") as f:
            if f.read() == payload:
                return
    except FileNotFoundError:
        pass
    tmp_path = f"{cfg_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(payload)
    os.replace(tmp_path, cfg_path)


_ensure_libero_config()

# Benchmark suite name in LIBERO's registry.
BENCHMARK_NAME = "libero_goal"
NUM_TASKS = 10
ACTION_DIM = 7  # OSC position+orientation delta (6) + gripper (1), all in [-1, 1].

# Substrings that mark an obs key as an image / non-low-dim modality.
_IMAGE_KEY_MARKERS = ("rgb", "image", "depth", "segmentation", "seg")

# LIBERO demo-only keys we DROP from the obs vector because the LIVE robosuite
# env has no matching key (so they can't be reconstructed identically at eval):
#   - `ee_ori`    : demos store euler(3); the live env only exposes the EEF
#                   orientation as a quaternion (robot0_eef_quat, dim 4).
#   - `ee_states` : demos store [eef_pos, eef_euler] (6) — redundant with
#                   ee_pos + ee_ori, and again euler-based with no live key.
# EEF orientation is still implied by the 7 joint positions (joint_states),
# which DO match the live env. Keeping these would make train/eval obs differ
# in both dim and meaning. Drop them for a consistent, bridgeable schema.
_EXCLUDE_LOWDIM_KEYS = ("ee_ori", "ee_states")

# Preferred ordering for known low-dim keys. Keys not in this list are appended
# afterwards in sorted order, so discovery stays deterministic and stable.
_PREFERRED_KEY_ORDER = (
    "ee_pos",
    "ee_ori",
    "ee_states",
    "gripper_states",
    "joint_states",
    "object-state",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
)

# Map canonical (HDF5 demo) key -> ordered live-env key candidates to try.
# The live env (robosuite) often uses `robot0_*` prefixes. The first candidate
# present in the live obs dict wins. CONFIRM once LIBERO is installed.
LIVE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "ee_pos": ("robot0_eef_pos", "ee_pos"),
    "ee_ori": ("robot0_eef_quat", "ee_ori"),
    "ee_states": ("ee_states", "robot0_eef_pos"),
    "gripper_states": ("robot0_gripper_qpos", "gripper_states"),
    "joint_states": ("robot0_joint_pos", "joint_states"),
    "object-state": ("object-state",),
}


def is_lowdim_key(key: str) -> bool:
    """True when *key* is a low-dim (non-image) obs modality."""
    k = key.lower()
    return not any(marker in k for marker in _IMAGE_KEY_MARKERS)


def select_lowdim_obs_keys(available_keys: Sequence[str]) -> list[str]:
    """Deterministically order the low-dim obs keys present in a demo file.

    Preferred keys first (in `_PREFERRED_KEY_ORDER`), then any remaining
    low-dim keys in sorted order. Image keys are dropped.
    """
    low = [
        k for k in available_keys
        if is_lowdim_key(k) and k not in _EXCLUDE_LOWDIM_KEYS
    ]
    preferred = [k for k in _PREFERRED_KEY_ORDER if k in low]
    rest = sorted(k for k in low if k not in preferred)
    return preferred + rest


def build_lowdim_vector(obs: dict, keys: Sequence[str]) -> np.ndarray:
    """Concatenate `obs[key]` over *keys* into one float32 vector.

    Each entry must be a 1-D per-timestep vector (already indexed to a single
    step). Missing keys raise KeyError with the available keys listed.
    """
    chunks = []
    for key in keys:
        if key not in obs:
            raise KeyError(
                f"LIBERO obs missing key {key!r}. Available: {sorted(obs.keys())}"
            )
        chunks.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    return np.concatenate(chunks)


def resolve_live_obs(live_obs: dict, canonical_keys: Sequence[str]) -> np.ndarray:
    """Build the model obs vector from a LIVE env obs dict.

    For each canonical (training-time) key, try its `LIVE_KEY_ALIASES`
    candidates and use the first present. Falls back to the canonical key name
    itself. Raises KeyError if none resolve.
    """
    chunks = []
    for key in canonical_keys:
        # Try the EXACT canonical (HDF5 demo) key first — LIBERO's live env
        # post-processes robosuite obs into the same named keys it stored in
        # the demos, so this is the common case and guarantees a dim match.
        # Aliases are only a fallback for releases that rename keys.
        candidates = (key,) + tuple(
            c for c in LIVE_KEY_ALIASES.get(key, ()) if c != key
        )
        for cand in candidates:
            if cand in live_obs:
                chunks.append(np.asarray(live_obs[cand], dtype=np.float32).reshape(-1))
                break
        else:
            raise KeyError(
                f"None of {candidates} found in live LIBERO obs for canonical "
                f"key {key!r}. Available: {sorted(live_obs.keys())}. "
                f"Update LIVE_KEY_ALIASES in utils/libero.py."
            )
    return np.concatenate(chunks)


# ─── Benchmark / task metadata (lazy LIBERO import) ──────────────────────────

def get_task_infos() -> list[dict]:
    """Return per-task metadata for libero_goal, in canonical benchmark order.

    Each entry: {index, name, language, bddl_file, demo_file}. Requires the
    LIBERO package + the libero_goal demo dataset to be installed.
    """
    from libero.libero import benchmark, get_libero_path

    bench = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    dataset_root = get_libero_path("datasets")
    bddl_root = get_libero_path("bddl_files")

    infos = []
    for i in range(bench.n_tasks):
        task = bench.get_task(i)
        infos.append(
            {
                "index": i,
                "name": task.name,
                "language": task.language,
                "bddl_file": os.path.join(bddl_root, task.problem_folder, task.bddl_file),
                "demo_file": os.path.join(
                    dataset_root, bench.get_task_demonstration(i)
                ),
            }
        )
    return infos


# ─── Goal embedding cache I/O ────────────────────────────────────────────────

def save_goal_embeddings(
    path: str,
    names: Sequence[str],
    embeddings: np.ndarray,
    instructions: Sequence[str],
) -> None:
    """Persist the per-task language embeddings to *path* (.npz).

    embeddings: (NUM_TASKS, D) float32, row i = embedding of task i.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n_tasks, D); got {embeddings.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        names=np.asarray(list(names)),
        embeddings=embeddings,
        instructions=np.asarray(list(instructions)),
    )


def load_goal_embeddings(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Load (names, embeddings (n_tasks, D) float32, instructions) from *path*."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Goal-embedding cache not found at {path!r}. Run "
            f"scripts/precompute_libero_goal_embs.py first."
        )
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in data["names"].tolist()]
    embeddings = data["embeddings"].astype(np.float32)
    instructions = [str(x) for x in data["instructions"].tolist()]
    return names, embeddings, instructions
