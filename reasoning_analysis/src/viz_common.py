"""
Shared style/palette for all figures. Palette values and the assignment rule (fixed
hue per category, never cycled; sequential = one hue light->dark; diverging =
two hues + neutral midpoint) come from the project's dataviz skill reference
palette, validated with its scripts/validate_palette.js before use here:
  validate_palette.js "#2a78d6,#eb6834,#4a3aa7" --mode light -> ALL CHECKS PASS

This is a static-figure (PNG/SVG, matplotlib) context, not an interactive web
artifact -- the skill's dark-mode/hover/filter-control machinery doesn't apply here
and is intentionally not used; the parts that DO apply (color-by-job, one axis,
legend/label rules, mark specs) are followed throughout.

Brand = hue family (blue=Anthropic, orange=OpenAI, violet=Human -- a third,
visually distinct reference category, not a competing "brand"). Size tier = a
one-hue lightness ramp within that family (small=light, mid=base, large=dark),
mirroring the reference palette's blue sequential ramp (steps 250/450/650) and
extending the same lightness-step logic to orange since only blue's ramp is given
verbatim in the reference.
"""
import colorsys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]

FIGURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- Chart chrome (light mode only -- this is a static report, not a themed page) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# --- Categorical anchors (validated) ---
BLUE = "#2a78d6"     # Anthropic
ORANGE = "#eb6834"   # OpenAI
VIOLET = "#4a3aa7"   # Human (reference, not a brand)
RED = "#e34948"      # diverging pole / negative
GRAY_MID = "#f0efec"  # diverging neutral midpoint

BRAND_COLORS = {"anthropic": BLUE, "openai": ORANGE, "human": VIOLET}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def make_tint_shade_ramp(base_hex, n=3):
    """A monotonic-lightness, fixed-hue ramp around `base_hex` (small=light,
    mid=base color itself, large=dark), the same construction the reference
    palette's blue sequential ramp uses -- extended here to orange, since the
    reference only gives blue's ramp verbatim ("the second [sequential context]
    takes the next categorical slot's hue, each as its own one-hue ramp")."""
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n == 1:
        return [base_hex]
    lightness_steps = np.linspace(l + 0.28, l - 0.20, n)
    lightness_steps = np.clip(lightness_steps, 0.15, 0.88)
    out = []
    for lt in lightness_steps:
        rr, gg, bb = colorsys.hls_to_rgb(h, lt, min(s * 1.05, 1.0))
        out.append(_rgb_to_hex((rr, gg, bb)))
    return out


ANTHROPIC_TIERS = make_tint_shade_ramp(BLUE, 3)   # [small, mid, large]
OPENAI_TIERS = make_tint_shade_ramp(ORANGE, 3)

TIER_ORDER = {"small": 0, "mid": 1, "large": 2}


def model_color(brand, size_tier):
    ramp = ANTHROPIC_TIERS if brand == "anthropic" else OPENAI_TIERS
    idx = TIER_ORDER.get(size_tier, 1)
    return ramp[idx]


def style_axes(ax, grid_axis="y"):
    """Recessive grid/axes, muted ticks -- consistent look across all 14 figures."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def savefig(fig, name, dpi=160):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    png_path = FIGURES_DIR / f"{name}.png"
    svg_path = FIGURES_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path.relative_to(FIGURES_DIR.parents[1])} (+ .svg)")
