#!/usr/bin/env bash
# One-time setup for the LIBERO-Goal task.
#
# LIBERO is not vendored here: this script clones it into third_party/ (which is
# gitignored), registers it as an editable dependency of the `libero` extra, and
# fetches the demonstrations and language-goal embeddings that training needs.
#
# Run it from anywhere; it works from the repo root. Requires `uv`, `git` and
# network access. It is idempotent — re-running skips whatever is already done.
#
# The demo download is about 6 GB.

set -euo pipefail
cd "$(dirname "$0")/.."
echo "repo: $PWD"

# 1. Clone LIBERO and add the package marker upstream is missing. Without
#    `libero/__init__.py` the editable build installs only dist-info and every
#    `import libero` fails.
if [[ ! -f third_party/LIBERO/setup.py ]]; then
  echo "[1/4] cloning LIBERO into third_party/LIBERO ..."
  mkdir -p third_party
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
else
  echo "[1/4] third_party/LIBERO already present."
fi
touch third_party/LIBERO/libero/__init__.py

# 2. Register the clone as an editable dependency of the `libero` extra, then
#    sync. It is added here rather than shipped in pyproject.toml so that a
#    plain `uv sync` works for everyone who never touches LIBERO — uv resolves
#    every extra's sources, and a path source that does not exist yet is a hard
#    error. Declaring it (rather than `uv pip install`) is what keeps uv's
#    auto-sync from pruning it on the next `uv run`.
echo "[2/4] registering LIBERO and syncing the environment ..."
if ! grep -q '^libero = { path = "third_party/LIBERO"' pyproject.toml; then
  uv add --editable third_party/LIBERO --optional libero
fi
uv sync --extra libero

# Sanity check. Do NOT import robosuite here: it initializes a GL context at
# import time and fails on a machine without a GPU. It is only exercised later,
# during evaluation, with MUJOCO_GL set.
uv run python -c "import libero; from libero.libero import benchmark; benchmark.get_benchmark_dict(); print('libero OK:', libero.__file__)"

# 3. Download the libero_goal demonstrations. LIBERO's own downloader prompts
#    interactively; scripts/download_libero_goal.py answers for it and forces
#    the config to <repo>/.libero so every path stays inside the project.
DEMO_DIR="third_party/LIBERO/libero/datasets/libero_goal"
if ! ls "$DEMO_DIR"/*.hdf5 >/dev/null 2>&1; then
  echo "[3/4] downloading libero_goal demos (~6 GB) ..."
  uv run python scripts/download_libero_goal.py
else
  echo "[3/4] demos already present ($(ls "$DEMO_DIR"/*.hdf5 | wc -l) files)."
fi

# 4. Precompute the language-goal embeddings the policy conditions on (CPU only).
EMB=datasets/libero/libero_goal_goal_embs.npz
if [[ ! -f "$EMB" ]]; then
  echo "[4/4] precomputing goal embeddings ..."
  uv run python scripts/precompute_libero_goal_embs.py --out "$EMB" --encoder minilm
else
  echo "[4/4] embeddings already present: $EMB"
fi

cat <<'NEXT'

=============================================================================
Setup done. The remaining step renders with MuJoCo and needs a GPU:

  export MUJOCO_GL=egl        # use osmesa if egl is unavailable

  # Add the object-state observations to the demos:
  uv run python scripts/extract_libero_object_states.py

Then train and evaluate:

  uv run python -m training.wifi_bc_training --env libero_goal_pixels
  uv run python -m envs.evaluate --checkpoint checkpoints --env libero_goal_pixels
=============================================================================
NEXT
