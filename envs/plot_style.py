"""Shared house style for paper-quality matplotlib figures, matching
scripts/plot_ablation_separation.py's convention (Palatino Linotype, warm
off-white surface, muted text, gridline-as-baseline instead of spines).
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Reference palette (dataviz skill, light mode): surface & text tokens.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Two categorical colors, reused across this repo's figures for a 2-way split.
C1, C2 = "#2a78d6", "#eb6834"

# Paper method palette: one fixed colour per method in every figure; variants of a method
# differ by marker and line style, never by colour. RGB (234,155,86), (90,128,184),
# (161,186,102), (127,127,127), plus a brown for Consistency Policy.
# Accessibility: WiFI-BC orange and IBC green are close under deuteranopia (validator
# dE 2.6), so every figure must also give each family its own marker shape.
C_WIFI, C_DP, C_IBC, C_BC, C_CP = "#ea9b56", "#5a80b8", "#a1ba66", "#7f7f7f", "#8c564b"
METHOD_COLORS = {"wifi": C_WIFI, "dp": C_DP, "ibc": C_IBC, "bc": C_BC, "cp": C_CP}


def use_palatino() -> str:
    """Register Palatino Linotype and return its family name; raise if unavailable."""
    from matplotlib import font_manager as fm

    name = "Palatino Linotype"
    if not any(f.name == name for f in fm.fontManager.ttflist):
        for path in sorted(Path("/mnt/c/Windows/Fonts").glob("pala*.ttf")) + sorted(Path.home().glob(".fonts/pala*.ttf")):
            fm.fontManager.addfont(str(path))
    if not any(f.name == name for f in fm.fontManager.ttflist):
        raise SystemExit("Palatino Linotype not found (looked in matplotlib, /mnt/c/Windows/Fonts, ~/.fonts)")
    styles = {f.style for f in fm.fontManager.ttflist if f.name == name}
    if "italic" not in styles:
        raise SystemExit(f"Palatino Linotype italic not found (styles registered: {sorted(styles)})")
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": f"{name}:italic",
                         "mathtext.bf": f"{name}:bold", "mathtext.default": "it"})
    return name


def apply_house_style() -> str:
    """Register Palatino + set the shared rcParams block. Returns the family name."""
    family = use_palatino()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT, "ytick.color": TEXT2,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return family


def style_axes(ax, hide_spines=("top", "right", "left", "bottom")) -> None:
    """Apply the shared spine/grid/tick treatment to one axes."""
    ax.set_facecolor(SURFACE)
    for side in hide_spines:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
