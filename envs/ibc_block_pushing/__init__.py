"""Vendored from google-research/ibc (Apache 2.0).

We only need the *environment* code (block_pushing.py and
the pybullet utilities it depends on).
The original IBC env imports `gin` for config injection and `tf_agents`
heavily in the metrics/oracles modules. We stub gin to no-ops and skip
the metrics + oracles modules entirely — none of them are needed when the
training data already comes from the official TFRecord oracle dataset.

Asset (.urdf, .obj) paths in the originals point at internal Google paths
(`third_party/py/ibc/...`); `utils.utils_pybullet.load_urdf` is patched to
resolve them against this package's `assets/` directory instead.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ASSETS_DIR = (Path(__file__).parent / "assets").resolve()


# ─── gin stub ────────────────────────────────────────────────────────────────
# IBC env decorates classes with @gin.configurable / @gin.constants_from_enum.
# We don't run gin config files (we set params via Python directly), so all
# decorators become identity functions.
def _make_gin_stub() -> types.ModuleType:
    gin = types.ModuleType("gin")

    def _identity(*args, **kwargs):
        # Support both bare `@gin.configurable` and `@gin.configurable(...)` usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def wrapper(cls_or_fn):
            return cls_or_fn
        return wrapper

    gin.configurable = _identity
    gin.constants_from_enum = _identity
    gin.register = _identity
    gin.external_configurable = _identity
    gin.parse_config = lambda *a, **kw: None
    gin.parse_config_file = lambda *a, **kw: None
    gin.config_str = lambda *a, **kw: ""
    return gin


if "gin" not in sys.modules:
    sys.modules["gin"] = _make_gin_stub()


def asset_path(relative: str) -> str:
    """Resolve a vendored asset name (e.g. 'block.urdf') to absolute path."""
    return str(ASSETS_DIR / relative)
