"""Figures 13-14: human step-alignment heatmaps and coverage-vs-extra scatter."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz_common as vc

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"
EMBED_STORE = Path(__file__).resolve().parents[1] / "outputs" / "embeddings"

# The most- and least-human-aligned models by average coverage (Phase 4 finding) --
# the "2-3 contrasting models" the spec asks for.
CONTRAST_MODELS = ["claude-haiku-4-5", "gpt-5.6-terra"]


def fig13_human_alignment_heatmaps():
    """Q: concretely, which parts of a model's reasoning correspond to the human's
    steps, and which don't -- for the best- and worst-aligned models?"""
    data = np.load(EMBED_STORE / "human_step_alignment_matrices.npz")
    align = pd.read_csv(TABLES / "human_step_alignment_summary.csv")
    instance_ids = sorted(align["instance_id"].unique())

    fig, axes = plt.subplots(len(instance_ids), len(CONTRAST_MODELS), figsize=(8, 3 * len(instance_ids)))
    for row, iid in enumerate(instance_ids):
        for col, model in enumerate(CONTRAST_MODELS):
            ax = axes[row, col]
            key = f"{iid}__{model}"
            if key not in data:
                ax.axis("off")
                continue
            mat = data[key]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xlabel("model sentence idx", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{iid}\nhuman sentence idx", fontsize=7.5)
            ax.set_title(model, fontsize=8.5)
            ax.tick_params(labelsize=6)

    fig.suptitle("Figure 13. Human-vs-model step alignment: highest-coverage model (left) vs. "
                 "lowest-coverage model (right), per example", fontsize=10.5, color=vc.INK_SECONDARY, y=1.0)
    fig.tight_layout()
    vc.savefig(fig, "fig13_human_alignment_heatmaps")


def fig14_human_coverage_extra():
    """Q: across the 5 examples, which models cover the most of what the human said,
    and which introduce the most content the human never touched?"""
    align = pd.read_csv(TABLES / "human_step_alignment_summary.csv")
    lex = pd.read_csv(Path(TABLES) / "lexical_features.csv")
    meta = lex[["model", "brand", "size_tier"]].drop_duplicates().set_index("model")

    fig, ax = plt.subplots(figsize=(7, 6))
    for model, grp in align.groupby("model"):
        c = vc.model_color(meta.loc[model, "brand"], meta.loc[model, "size_tier"])
        ax.scatter(grp["coverage"], grp["extra"], s=26, alpha=0.35, color=c, zorder=2)
        ax.scatter(grp["coverage"].mean(), grp["extra"].mean(), s=140, color=c, edgecolor=vc.INK_PRIMARY,
                   linewidth=1.2, zorder=4, label=model)

    vc.style_axes(ax, grid_axis="both")
    ax.set_xlabel("coverage (share of human steps matched)")
    ax.set_ylabel("extra (share of model steps with no human match)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Figure 14. Coverage vs. extra per model (small = one of 5 examples, large = mean)",
                 loc="left", fontsize=10, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig14_human_coverage_extra")
