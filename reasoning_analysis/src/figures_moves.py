"""Figures 8-9: reasoning-move composition, transition heatmaps, JS divergence."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz_common as vc
from . import move_taxonomy as mt

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"

# The 8-hue reference categorical palette, validated for adjacent pairs in stacks/bars
# (exactly this figure form) -- used here for MOVES, a taxonomy shared across all
# models, not for brand/model identity (which uses the blue/orange ramps elsewhere).
MOVE_COLORS = {
    "restate": "#2a78d6", "interpret_ambiguity": "#eb6834", "plan": "#1baf7a",
    "hypothesize": "#eda100", "verify": "#e87ba4", "self_correct": "#008300",
    "meta_format": "#4a3aa7", "execute": "#e34948",
}


def _model_order_with_human(models):
    meta_order = {"anthropic": 0, "openai": 1, "human": 2}
    brand_of = {}
    for m in models:
        if m == "human":
            brand_of[m] = "human"
        elif m.startswith("claude"):
            brand_of[m] = "anthropic"
        else:
            brand_of[m] = "openai"
    return sorted(models, key=lambda m: meta_order.get(brand_of[m], 9))


def fig8_move_composition():
    """Q: what mix of reasoning moves does each model (and the human) actually use?"""
    dist = pd.read_csv(TABLES / "move_distribution_by_model.csv")
    order = _model_order_with_human(dist["model"].tolist())
    dist = dist.set_index("model").loc[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for move in mt.MOVES:
        vals = dist[move].values * 100
        ax.bar(x, vals, bottom=bottom, color=MOVE_COLORS[move], width=0.6, label=move,
               edgecolor=vc.SURFACE, linewidth=1.2, zorder=2)
        bottom += vals

    vc.style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("share of sentences (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Figure 8. Reasoning-move composition per model, human included as reference",
                 loc="left", fontsize=10.5, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig08_move_composition")


def fig9_move_transitions():
    """Q: does one move tend to follow another in a model-characteristic pattern, and
    how far apart are models in their overall move usage (JS divergence)?"""
    trans = pd.read_csv(TABLES / "move_transition_matrices.csv")
    models = sorted(trans["model"].unique(), key=lambda m: (0 if m.startswith("claude") else 1, m))

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    cmap = plt.get_cmap("Blues")
    for i, model in enumerate(models):
        ax = fig.add_subplot(gs[i // 3, i % 3]) if i < 6 else None
        if ax is None:
            continue
        sub = trans[trans["model"] == model].set_index("from_move").loc[mt.MOVES, mt.MOVES]
        im = ax.imshow(sub.values, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(mt.MOVES)))
        ax.set_yticks(range(len(mt.MOVES)))
        short = [m[:4] for m in mt.MOVES]
        ax.set_xticklabels(short, rotation=90, fontsize=6.5)
        ax.set_yticklabels(short, fontsize=6.5)
        ax.set_title(model, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_js = fig.add_subplot(gs[:, 3])
    js = pd.read_csv(TABLES / "move_js_divergence.csv").set_index("model_a")
    js_models = [m for m in js.index if m != "human"] + (["human"] if "human" in js.index else [])
    js_ordered = js.loc[js_models, js_models]
    im = ax_js.imshow(js_ordered.values, cmap="Blues")
    ax_js.set_xticks(range(len(js_models)))
    ax_js.set_yticks(range(len(js_models)))
    ax_js.set_xticklabels(js_models, rotation=90, fontsize=7.5)
    ax_js.set_yticklabels(js_models, fontsize=7.5)
    ax_js.set_title("JS divergence\n(move distributions)", fontsize=9)
    for spine in ax_js.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax_js, fraction=0.046, pad=0.06)

    fig.suptitle("Figure 9. Per-model move-transition matrices (P(next move | current move)) "
                 "and pairwise Jensen-Shannon divergence", fontsize=11, color=vc.INK_SECONDARY, y=0.98)
    fig.tight_layout()
    vc.savefig(fig, "fig09_move_transitions")
