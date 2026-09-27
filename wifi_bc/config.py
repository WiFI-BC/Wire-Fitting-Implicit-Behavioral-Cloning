"""Config loading and per-method resolution.

`config/config.json` holds one block per environment. Its `training` and `model`
sub-blocks carry WiFI-BC's best-found hyperparameters — the paper's method, and
so the default. Every baseline needs different values for the same environment
(a diffusion policy has no control points; BC trains for a different number of
steps), which live under `methods.<name>` and are merged on top:

    environments.pushing.training          WiFI-BC's values
    environments.pushing.methods.diffusion_policy.training   what DP overrides

`resolve_env_config(config, "pushing", "diffusion_policy")` returns the merged
environment block, so every training and evaluation entry point reads its own
method's numbers from one shipped file and reproduces the reported result.

Method names match the training scripts: `wifi_bc`, `ibc`, `diffusion_policy`,
`consistency_policy`, `bc_mse`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.json"

METHODS = ("wifi_bc", "ibc", "diffusion_policy", "consistency_policy", "bc_mse")


def config_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the config file: explicit path, then WIFI_BC_CONFIG_PATH, then
    the shipped `config/config.json`. Read at call time so a caller can point
    every entry point at an alternative config."""
    return Path(path or os.environ.get("WIFI_BC_CONFIG_PATH") or DEFAULT_CONFIG_PATH)


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(config_path(path)) as f:
        return json.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins. Neither input is mutated."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_env_config(config: dict, active_env: str, method: str = "wifi_bc") -> dict:
    """Environment block with `methods.<method>` merged over the defaults.

    The `methods` key itself is dropped from the result, so consumers see one
    flat environment block and never have to know which method they are.
    """
    if active_env not in config.get("environments", {}):
        raise KeyError(
            f"unknown environment {active_env!r}; "
            f"choose from {sorted(config.get('environments', {}))}"
        )
    env_config = dict(config["environments"][active_env])
    overrides = env_config.pop("methods", {}).get(method, {})
    return _deep_merge(env_config, overrides)


def default_checkpoint_dir(method: str, active_env: str) -> str:
    """Where a run writes its weights when the config does not say.

    Namespaced by method and environment, because every method used to default
    to a bare `checkpoints/` and their files have different names. Training two
    methods in a row left both sets side by side, and evaluation — which infers
    the method from the files present — then picked whichever its checks
    happened to test first.
    """
    return str(Path("checkpoints") / method / active_env)


def resolve_config_path(argv: list[str] | None = None) -> Path:
    """Config file for this run: `--config <path>` on the command line, else
    WIFI_BC_CONFIG_PATH, else the shipped `config/config.json`.

    The training scripts build their module state at import time, before any
    argparse runs, so `--config` is read here. Without this they silently
    ignored the flag and trained on the shipped config instead — a sweep that
    believed it had set 60 steps ran 300,000.
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return config_path()


def resolve_active_env(config: dict, argv: list[str] | None = None) -> str:
    """Which environment to run: `--env <name>` on the command line, else
    WIFI_BC_ENV, else the config's `active_env`.

    The training scripts build their whole module state from the config at
    import time, so they read the environment here rather than through argparse.
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--env" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--env="):
            return arg.split("=", 1)[1]
    env = os.environ.get("WIFI_BC_ENV") or config.get("active_env", "particle")
    if env not in config.get("environments", {}):
        raise KeyError(
            f"unknown environment {env!r}; "
            f"choose from {sorted(config.get('environments', {}))}"
        )
    return env
