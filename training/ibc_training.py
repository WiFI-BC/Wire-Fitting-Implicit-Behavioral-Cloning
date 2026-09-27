"""Train the IBC (Implicit Behavioral Cloning) baseline.

Same contract as every other training script here: the environment and all
hyperparameters come from `config/config.json` (override the file with
WIFI_BC_CONFIG_PATH), so a run is fully described by the config plus the
environment name.

    uv run python -m training.ibc_training --env pen

IBC's hyperparameters live under each environment's `methods.ibc.training` block,
named the same way as every other method's. Anything the block does not set
falls back to `baselines.ibc.DEFAULT_IBC_HPARAMS`, the paper-faithful defaults.

Writes `q_estimator.pt` and `hparams.json` to
`checkpoints/ibc/<env>` unless `--save-dir` says otherwise. Evaluate the result
with `uv run python -m envs.evaluate --checkpoint <dir>`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve_hparams(config: dict, active_env: str) -> dict:
    """IBC hyperparameters for this environment, merged onto the defaults."""
    from baselines.ibc import hparams_from_env_config
    from wifi_bc.config import resolve_env_config

    return hparams_from_env_config(resolve_env_config(config, active_env, "ibc"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--env", default=None,
                        help="Environment to train on (default: config's active_env).")
    parser.add_argument("--config", default=None,
                        help="Config file to read (default: config/config.json).")
    parser.add_argument("--save-dir", default=None,
                        help="Where to write the checkpoint (default: checkpoints/ibc/<env>).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed; overrides the config's trial_seed.")
    args = parser.parse_args()

    config_path = Path(
        args.config
        or os.environ.get("WIFI_BC_CONFIG_PATH")
        or (Path(__file__).resolve().parent.parent / "config" / "config.json")
    )
    # baselines.ibc re-reads the config for the env block, so point it at the
    # same file before importing it.
    os.environ["WIFI_BC_CONFIG_PATH"] = str(config_path)
    from baselines.ibc import train_ibc

    with open(config_path) as f:
        config = json.load(f)

    active_env = args.env or config.get("active_env", "particle")
    if active_env not in config["environments"]:
        parser.error(f"unknown env {active_env!r}; "
                     f"choose from {sorted(config['environments'])}")

    hparams = resolve_hparams(config, active_env)
    if args.seed is not None:
        hparams["trial_seed"] = args.seed

    print(f"IBC training on {active_env}")
    print(json.dumps(hparams, indent=2, default=str))

    meta = train_ibc(hparams, active_env=active_env, save_dir=args.save_dir)
    print(f"\nCheckpoint: {meta['checkpoint_path']}")
    print(f"Evaluate with: uv run python -m envs.evaluate "
          f"--checkpoint {meta['checkpoint_dir']} --env {active_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
