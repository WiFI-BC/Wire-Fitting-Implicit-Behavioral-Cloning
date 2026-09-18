"""Non-interactive libero_goal demo download (HuggingFace source).

LIBERO's own benchmark_scripts/download_libero_datasets.py is interactive AND
doesn't import envs.libero, so it uses ~/.libero instead of our in-repo config
and prompts for paths. This wrapper imports envs.libero first (sets
LIBERO_CONFIG_PATH to <repo>/.libero and writes config.yaml), then calls
LIBERO's download function directly with use_huggingface=True (no prompts, more
reliable than the original links).

    uv run --extra libero python scripts/download_libero_goal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envs.libero  # noqa: F401  (import side effect: sets LIBERO_CONFIG_PATH)
from libero.libero import get_libero_path
import libero.libero.utils.download_utils as download_utils


def main() -> None:
    download_dir = get_libero_path("datasets")
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading libero_goal demos (HuggingFace) -> {download_dir}")
    download_utils.libero_dataset_download(
        datasets="libero_goal",
        download_dir=download_dir,
        use_huggingface=True,
    )
    download_utils.check_libero_dataset(download_dir=download_dir)
    print("Done.")


if __name__ == "__main__":
    main()
