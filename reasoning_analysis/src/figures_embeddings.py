"""Figures 3-5: model similarity + dendrogram, UMAP, probe accuracy."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from . import viz_common as vc

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"
EMBED_STORE = Path(__file__).resolve().parents[1] / "outputs" / "embeddings"

_BLUE_SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])


def _model_meta():
    lex = pd.read_csv(TABLES / "lexical_features.csv")
    return lex[["model", "brand", "size_tier"]].drop_duplicates().set_index("model")


def fig3_similarity_heatmap():
    """Q: which models' reasoning content is most similar to which others', and does
    that clustering follow brand?"""
    sim = pd.read_csv(TABLES / "model_similarity_matrix.csv").set_index("model_a")
    models = list(sim.index)
    linkage = np.load(EMBED_STORE / "similarity_linkage.npy")
    meta = _model_meta()

    fig = plt.figure(figsize=(8.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 0.3], hspace=0.02, wspace=0.03)

    ax_dendro = fig.add_subplot(gs[0, 0])
    dd = dendrogram(linkage, labels=models, ax=ax_dendro, color_threshold=0, above_threshold_color=vc.INK_SECONDARY)
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.set_facecolor(vc.SURFACE)

    leaf_order = dd["ivl"]
    ordered = sim.loc[leaf_order, leaf_order]

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(ordered.values, cmap=_BLUE_SEQ, vmin=ordered.values.min(), vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(leaf_order)))
    ax.set_xticklabels(leaf_order, rotation=40, ha="right")
    ax.set_yticks(range(len(leaf_order)))
    ax.set_yticklabels(leaf_order)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(leaf_order)):
        for j in range(len(leaf_order)):
            ax.text(j, i, f"{ordered.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if ordered.values[i, j] > (ordered.values.min() + 1) / 2 else vc.INK_PRIMARY)
    # brand/tier tick color bars
    for i, m in enumerate(leaf_order):
        c = vc.model_color(meta.loc[m, "brand"], meta.loc[m, "size_tier"])
        ax.add_patch(plt.Rectangle((-1.4, i - 0.5), 0.7, 1, color=c, clip_on=False, transform=ax.transData))
        ax.add_patch(plt.Rectangle((i - 0.5, len(leaf_order) + 0.15), 1, 0.7, color=c, clip_on=False, transform=ax.transData))

    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=cax, label="avg. cosine similarity\n(shared instances)")

    fig.suptitle("Figure 3. Model-by-model reasoning similarity clusters by brand, not by size tier",
                  fontsize=10, color=vc.INK_SECONDARY, y=0.98)
    vc.savefig(fig, "fig03_similarity_heatmap")


def fig4_umap():
    """Q: do models separate into visually distinct clusters once instance-level
    content is removed (residual/style embeddings)?"""
    umap_df = pd.read_csv(TABLES / "umap_residual_embeddings.csv")
    markers = {"small": "o", "mid": "s", "large": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (brand, tier), grp in umap_df.groupby(["brand", "size_tier"]):
        ax.scatter(grp["umap_x"], grp["umap_y"], s=32, alpha=0.75,
                   color=vc.BRAND_COLORS.get(brand, vc.INK_MUTED), marker=markers.get(tier, "o"),
                   edgecolor="white", linewidth=0.4, label=f"{brand} / {tier}")
    vc.style_axes(ax, grid_axis="both")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)
    ax.set_title("Figure 4. UMAP of residual (style) embeddings -- color=brand, shape=size tier",
                 loc="left", fontsize=10, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig04_umap_residual")


def fig5_probe_accuracy():
    """Q: how well can a simple classifier tell brand or size apart from style
    embeddings vs. lexical features alone, vs. a majority-class baseline?"""
    probes = pd.read_csv(TABLES / "probe_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    labels = []
    accs = []
    baselines = []
    colors = []
    pvals = []
    color_map = {("brand", "residual_embeddings"): vc.BLUE, ("brand", "lexical_features"): "#86b6ef",
                 ("size_tier", "residual_embeddings"): vc.ORANGE, ("size_tier", "lexical_features"): "#f3ac85"}
    for _, r in probes.iterrows():
        labels.append(f"{r['target']}\n({r['input'].replace('_', ' ')})")
        accs.append(r["cv_accuracy"] * 100)
        baselines.append(r["majority_baseline"] * 100)
        pvals.append(r["permutation_pvalue"])
        colors.append(color_map.get((r["target"], r["input"]), vc.INK_MUTED))

    x = np.arange(len(labels))
    ax.bar(x, accs, color=colors, width=0.55, zorder=2)
    for i, b in enumerate(baselines):
        ax.plot([x[i] - 0.3, x[i] + 0.3], [b, b], color=vc.INK_PRIMARY, linewidth=1.5, zorder=3)
    ax.plot([], [], color=vc.INK_PRIMARY, linewidth=1.5, label="majority-class baseline")
    for i, p in enumerate(pvals):
        ax.text(x[i], accs[i] + 2, f"p={p:.3f}", ha="center", fontsize=7.5, color=vc.INK_SECONDARY)

    vc.style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("cross-validated accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Figure 5. Brand is near-perfectly separable; size tier is a real but weaker signal "
                  "(permutation-test p-values shown; no CI computed for accuracy itself)",
                 loc="left", fontsize=9.5, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig05_probe_accuracy")
