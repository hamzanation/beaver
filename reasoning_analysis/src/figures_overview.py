"""Figures 1-2: data overview (traces + accuracy per model) and length distributions."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz_common as vc
from . import data_loading as dl

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def _wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (better behaved than the normal
    approximation at n=100 and proportions near 0 or 1)."""
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def _model_order(models_df):
    """Anthropic small->large, then OpenAI small->large -- the fixed order used
    consistently across every figure (color follows the entity, never its rank)."""
    order_key = models_df.apply(
        lambda r: (0 if r["brand"] == "anthropic" else 1, vc.TIER_ORDER.get(r["size_tier"], 1)), axis=1
    )
    return models_df.assign(_k=order_key).sort_values("_k")["model"].tolist()


def fig1_data_overview():
    """Q: how many traces does each model contribute, and how accurate is it?"""
    coverage = pd.read_csv(TABLES / "trace_coverage.csv")
    rows, models = dl.load_model_instances(sample=None)
    df = pd.DataFrame(rows)
    acc = df.groupby("model")["correct"].agg(k="sum", n="count").reset_index()
    acc["accuracy"] = acc["k"] / acc["n"]
    acc[["lo", "hi"]] = [_wilson(k, n) for k, n in zip(acc["k"], acc["n"])]

    meta = df[["model", "brand", "size_tier"]].drop_duplicates()
    order = _model_order(meta)
    coverage = coverage.set_index("model").loc[order].reset_index()
    acc = acc.set_index("model").loc[order].reset_index()
    colors = [vc.model_color(meta.set_index("model").loc[m, "brand"], meta.set_index("model").loc[m, "size_tier"])
              for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.bar(order, coverage["has_trace"], color=colors, width=0.62, zorder=2)
    ax.bar(order, coverage["attempted"] - coverage["has_trace"], bottom=coverage["has_trace"],
           color=vc.GRIDLINE, width=0.62, zorder=2, label="attempted, no trace captured")
    vc.style_axes(ax)
    ax.set_ylabel("instances (dataset: dw, setting 2)")
    ax.set_title("Traces per model", loc="left", fontsize=11)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.bar(order, acc["accuracy"] * 100, color=colors, width=0.62, zorder=2,
           yerr=[(acc["accuracy"] - acc["lo"]) * 100, (acc["hi"] - acc["accuracy"]) * 100],
           error_kw={"ecolor": vc.INK_SECONDARY, "elinewidth": 1.2, "capsize": 3})
    vc.style_axes(ax)
    ax.set_ylabel("execution accuracy (%), 95% Wilson CI (n=100 each)")
    ax.set_title("Accuracy per model", loc="left", fontsize=11)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")

    fig.suptitle("Figure 1. Data volume and accuracy, by model (blue=Anthropic, orange=OpenAI, "
                  "light→dark = small→large)", fontsize=10, color=vc.INK_SECONDARY, y=1.03)
    fig.tight_layout()
    vc.savefig(fig, "fig01_data_overview")


def fig2_length_distributions():
    """Q: how does reasoning-trace length differ by model, and does it track with
    whether the trace is a raw thinking trace or a narrated summary?"""
    lex = pd.read_csv(TABLES / "lexical_features.csv")
    order = _model_order(lex[["model", "brand", "size_tier"]].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})

    ax = axes[0]
    data = [lex.loc[lex["model"] == m, "n_words"].dropna().values for m in order]
    colors = [vc.model_color(lex.loc[lex["model"] == m, "brand"].iloc[0], lex.loc[lex["model"] == m, "size_tier"].iloc[0])
              for m in order]
    parts = ax.violinplot(data, showmedians=True, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(vc.INK_SECONDARY)
        pc.set_alpha(0.85)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(vc.INK_SECONDARY)
        parts[key].set_linewidth(1)
    vc.style_axes(ax)
    ax.set_xticks(range(1, len(order) + 1))  # violinplot positions bodies at 1..n, not 0..n-1
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("words per trace")
    ax.set_title("Length distribution per model", loc="left", fontsize=11)

    ax = axes[1]
    pct_summary = lex.groupby("model")["is_summary"].mean().loc[order] * 100
    ax.barh(order[::-1], pct_summary.loc[order[::-1]], color=[colors[order.index(m)] for m in order[::-1]], height=0.6, zorder=2)
    vc.style_axes(ax, grid_axis="x")
    ax.set_xlabel("% of traces flagged as summarized\n(bold-header format, Phase 0/1 finding)")
    ax.set_title("Raw vs. summarized", loc="left", fontsize=11)
    ax.set_xlim(0, 100)

    fig.suptitle("Figure 2. Trace length varies enormously by model, and is confounded with format "
                  "(summary vs. raw)", fontsize=10, color=vc.INK_SECONDARY, y=1.03)
    fig.tight_layout()
    vc.savefig(fig, "fig02_length_distributions")
