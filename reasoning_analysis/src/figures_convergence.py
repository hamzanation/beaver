"""
Figures 15-19: answer convergence (see src/convergence.py and outputs/convergence_report.md).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import data_loading as dl
from . import move_taxonomy as mt
from . import viz_common as vc
from .figures_moves import MOVE_COLORS

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"

CORRECT_GREEN = "#1baf7a"
# Sequential (one hue, light -> dark) for "how many models share this wrong result".
SHARED_RAMP = {1: "#dcdad3", 2: "#b9b2e3", 3: "#8a7fd0", 4: "#6455bd", 5: "#4a3aa7", 6: "#2e2378"}
FAILED_FILL = vc.SURFACE


def _model_order():
    models = sorted(pd.read_csv(TABLES / "convergence_prediction_outcomes_unescaped.csv")["model"].unique())
    tier_of, _ = dl.derive_size_tiers(models, dl.load_pricing())
    key = lambda m: ({"anthropic": 0, "openai": 1}[dl.brand_of(m)], vc.TIER_ORDER[tier_of[m]])
    order = sorted(models, key=key)
    return order, {m: vc.model_color(dl.brand_of(m), tier_of[m]) for m in order}


def fig15_unescape_accuracy():
    """Q: how much accuracy do the GPT models lose to HTML-escaped operators?"""
    order, color = _model_order()
    out = pd.read_csv(TABLES / "convergence_prediction_outcomes_unescaped.csv")
    acc = out.groupby("model").agg(official=("official_correct", "mean"), unescaped=("correct", "mean"),
                                   n=("correct", "size"), n_entities=("had_html_entities", "sum")).loc[order] * [100, 100, 1, 1]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        a, b = acc.loc[m, "official"], acc.loc[m, "unescaped"]
        ax.plot([a, b], [yi, yi], color=vc.AXIS, linewidth=2, zorder=2)
        ax.scatter([a], [yi], s=70, facecolor=vc.SURFACE, edgecolor=color[m], linewidth=2, zorder=3)
        ax.scatter([b], [yi], s=70, color=color[m], edgecolor=vc.SURFACE, linewidth=1.5, zorder=4)
        label = f"{b:.0f}%" if abs(b - a) < 0.5 else f"{a:.0f}% → {b:.0f}%"
        note = f"   ({int(acc.loc[m, 'n_entities'])} of {int(acc.loc[m, 'n'])} queries had &gt;/&lt;)" if acc.loc[m, "n_entities"] else ""
        ax.annotate(label + note, (max(a, b), yi), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=vc.INK_PRIMARY)
    vc.style_axes(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 60)
    ax.set_xlabel("execution accuracy (%), all 100 attempted instances")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor=vc.SURFACE, markeredgecolor=vc.INK_SECONDARY,
               markersize=8, label="as scored"),
        Line2D([], [], marker="o", linestyle="", color=vc.INK_SECONDARY, markersize=8,
               label="after html-unescaping the predicted SQL"),
    ], frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Figure 15. Accuracy before and after unescaping HTML entities in the predicted SQL",
                 loc="left", fontsize=10.5, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig15_unescape_accuracy")


def fig16_answer_groups():
    """Q: on each problem, which models returned the same result -- right or wrong?

    One column per problem, one row per model. Correct = green. A wrong result is
    shaded by how many of the 6 models returned that exact same (wrong) result set;
    a query that failed to run (or no prediction) is an outlined empty cell."""
    order, _ = _model_order()
    cl = pd.read_csv(TABLES / "convergence_answer_clusters_unescaped.csv")
    out = pd.read_csv(TABLES / "convergence_prediction_outcomes_unescaped.csv")
    cl = cl.merge(out[["model", "instance_id", "error_type"]], on=["model", "instance_id"])
    cl["group_size"] = cl.groupby(["instance_id", "cluster"])["model"].transform("size")
    failed = cl["error_type"].isin(["exec_error", "timeout", "no_prediction"])
    cl.loc[failed, "group_size"] = 0

    per = cl.groupby("instance_id").agg(
        n_correct=("correct", "sum"),
        max_shared_wrong=("group_size", lambda s: s[~cl.loc[s.index, "correct"]].max() if (~cl.loc[s.index, "correct"]).any() else 0))
    cols = per.sort_values(["n_correct", "max_shared_wrong"], ascending=[False, False]).index.tolist()

    fig, ax = plt.subplots(figsize=(15, 3.6))
    cell = cl.set_index(["model", "instance_id"])
    for xi, iid in enumerate(cols):
        for yi, m in enumerate(order[::-1]):
            if (m, iid) not in cell.index:
                continue
            r = cell.loc[(m, iid)]
            if r["correct"]:
                fc, ec = CORRECT_GREEN, vc.SURFACE
            elif r["group_size"] == 0:
                fc, ec = FAILED_FILL, vc.INK_MUTED
            else:
                fc, ec = SHARED_RAMP[int(r["group_size"])], vc.SURFACE
            ax.add_patch(plt.Rectangle((xi + 0.06, yi + 0.06), 0.88, 0.88, facecolor=fc, edgecolor=ec,
                                       linewidth=0.6 if ec == vc.INK_MUTED else 1.0))
    # Boundaries between n_correct blocks, labeled on top.
    nc = per.loc[cols, "n_correct"].values
    starts = [0] + [i for i in range(1, len(nc)) if nc[i] != nc[i - 1]] + [len(nc)]
    for s, e in zip(starts[:-1], starts[1:]):
        if s:
            ax.axvline(s, color=vc.INK_SECONDARY, linewidth=0.8)
        ax.text((s + e) / 2, len(order) + 0.25, f"{nc[s]}/6 correct\n({e - s})", ha="center", va="bottom",
                fontsize=7.5, color=vc.INK_SECONDARY)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(order))
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels(order[::-1], fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(cols)} problems, grouped by how many models solved them; within a group, "
                  "problems with the largest shared wrong answer first", fontsize=9, color=vc.INK_SECONDARY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors=vc.INK_SECONDARY)
    handles = [Patch(facecolor=CORRECT_GREEN, label="correct")]
    handles += [Patch(facecolor=SHARED_RAMP[k], label=f"wrong, same result as {k - 1} other model{'s' if k > 2 else ''}"
                      if k > 1 else "wrong, unique result") for k in range(1, 6)]
    handles += [Patch(facecolor=FAILED_FILL, edgecolor=vc.INK_MUTED, label="query failed / no prediction")]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Figure 16. Which models returned the same result on each problem (html-unescaped)",
                 loc="left", fontsize=10.5, color=vc.INK_SECONDARY, pad=34)
    vc.savefig(fig, "fig16_answer_groups")


def fig17_shared_wrong_matrix():
    """Q: which model pairs make the SAME mistake? Rate = (# problems where both were
    wrong and returned an equivalent result set) / (# problems where both were wrong
    and both queries ran)."""
    order, _ = _model_order()
    pairs = pd.read_csv(TABLES / "convergence_pairs_unescaped.csv")
    out = pd.read_csv(TABLES / "convergence_prediction_outcomes_unescaped.csv")
    ran = set(map(tuple, out.loc[~out.error_type.isin(["exec_error", "timeout", "no_prediction"]),
                                 ["model", "instance_id"]].values))
    wrong = pairs[pairs.pair_kind.str.startswith("both_wrong")].copy()
    wrong = wrong[[(a, i) in ran and (b, i) in ran for a, b, i in zip(wrong.model_a, wrong.model_b, wrong.instance_id)]]

    n = len(order)
    rate = np.full((n, n), np.nan)
    text = {}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j >= i:
                continue
            g = wrong[((wrong.model_a == a) & (wrong.model_b == b)) | ((wrong.model_a == b) & (wrong.model_b == a))]
            k = int(g.same_answer.sum())
            rate[i, j] = k / len(g) if len(g) else np.nan
            text[(i, j)] = f"{100 * rate[i, j]:.0f}%\n{k}/{len(g)}"

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("v", ["#f3f1fb", "#4a3aa7"])
    # Lower triangle only: drop the all-empty first row and last column.
    im = ax.imshow(np.ma.masked_invalid(rate[1:, :-1]), cmap=cmap, vmin=0, vmax=0.5)
    for (i, j), t in text.items():
        ax.text(j, i - 1, t, ha="center", va="center", fontsize=8.5,
                color="white" if rate[i, j] > 0.3 else vc.INK_PRIMARY)
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels(order[:-1], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n - 1))
    ax.set_yticklabels(order[1:], fontsize=9)
    ax.tick_params(length=0, colors=vc.INK_SECONDARY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(vc.SURFACE)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("share of jointly-failed problems with the same wrong result", fontsize=8.5, color=vc.INK_SECONDARY)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=vc.INK_SECONDARY)
    ax.set_title("Figure 17. How often two models make the same mistake", loc="left",
                 fontsize=10.5, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig17_shared_wrong_matrix")


def fig18_similarity_contrasts():
    """Q: are traces more alike when two models reach the same answer?

    Each point = mean within-problem difference (same-answer pairs minus
    different-answer pairs), averaged over problems that have both kinds; bars =
    95% bootstrap CI over problems. One panel per metric, since they don't share a
    scale."""
    panels = [("trace_cosine", "trace embedding cosine\n(higher = more similar)"),
              ("move_js", "move-mix JS divergence\n(lower = more similar)"),
              ("table_jaccard", "tables used in final SQL, Jaccard\n(higher = more similar)")]
    comps = ["same_answer vs diff_answer (any)", "same_vs_diff_answer (both wrong)", "both_correct vs both_wrong"]
    comp_labels = ["same answer − different answer\n(all pairs)", "same wrong − different wrong\n(both wrong)",
                   "both correct − both wrong"]
    data = {v: pd.read_csv(TABLES / f"convergence_contrasts_{v}.csv").set_index(["metric", "comparison"])
            for v in ("as_scored", "unescaped")}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(comps))[::-1]
    for ax, (metric, xlabel) in zip(axes, panels):
        for v, off, filled in (("unescaped", 0.1, True), ("as_scored", -0.1, False)):
            d = data[v].loc[[(metric, c) for c in comps]]
            ax.errorbar(d["mean_diff"], y + off, xerr=[d["mean_diff"] - d["ci_low"], d["ci_high"] - d["mean_diff"]],
                        fmt="none", ecolor=vc.INK_MUTED, elinewidth=1.4, capsize=2, zorder=3)
            ax.scatter(d["mean_diff"], y + off, s=42, zorder=4, linewidth=1.6,
                       facecolor=vc.INK_PRIMARY if filled else vc.SURFACE, edgecolor=vc.INK_PRIMARY)
            if filled:
                for yi, (_, r) in zip(y, d.iterrows()):
                    ax.annotate(f"n={int(r['n_instances'])}", (r["ci_high"], yi + off), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=7.5, color=vc.INK_MUTED)
        ax.axvline(0, color=vc.AXIS, linewidth=1.2, zorder=1)
        vc.style_axes(ax, grid_axis="x")
        ax.set_xlabel(xlabel, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(comp_labels, fontsize=8.5)
    fig.suptitle("Figure 18. Within-problem difference in similarity: pairs that agree on the answer vs. pairs that don't "
                 "(point = mean over problems, bar = 95% bootstrap CI)", fontsize=10, color=vc.INK_SECONDARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=vc.INK_PRIMARY, markersize=7, label="html-unescaped"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=vc.SURFACE, markeredgecolor=vc.INK_PRIMARY,
               markersize=7, label="as scored")], frameon=False, fontsize=8.5, ncol=2, loc="lower center")
    vc.savefig(fig, "fig18_similarity_contrasts")


def fig19_case_studies(cases=(("dw_1191", "all 6 correct"), ("dw_1737", "all 6 wrong, 5 share one wrong result"))):
    """Q: what does 'same answer, different reasoning' look like on single problems?
    Move composition of each model's trace on two problems."""
    order, _ = _model_order()
    labels = pd.read_csv(TABLES / "move_labels_per_sentence.csv")
    cl = pd.read_csv(TABLES / "convergence_answer_clusters_unescaped.csv").set_index(["instance_id", "model"])

    fig, axes = plt.subplots(1, len(cases), figsize=(13, 3.8), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (iid, desc) in zip(axes, cases):
        sub = labels[labels.instance_id == iid]
        left = np.zeros(len(order))
        counts = sub.groupby("model")["predicted_move"].value_counts().unstack(fill_value=0).reindex(order).fillna(0)
        totals = counts.sum(axis=1).values
        for mv in mt.MOVES:
            share = (counts[mv].values / np.where(totals == 0, 1, totals)) * 100 if mv in counts else np.zeros(len(order))
            ax.barh(y, share, left=left, color=MOVE_COLORS[mv], height=0.62, edgecolor=vc.SURFACE,
                    linewidth=1.2, label=mv, zorder=2)
            left += share
        for yi, m, t in zip(y, order, totals):
            r = cl.loc[(iid, m)]
            same = (cl.loc[iid]["cluster"] == r["cluster"]).sum()
            tag = "correct" if r["correct"] else (f"wrong, shared by {same}" if same > 1 else "wrong, unique")
            ax.annotate(f"{int(t)} sent. · {tag}", (100, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=vc.INK_SECONDARY)
        vc.style_axes(ax, grid_axis="x")
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of trace sentences (%)", fontsize=9)
        ax.set_title(f"{iid}: {desc}", loc="left", fontsize=9.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].legend(frameon=False, fontsize=8, ncol=8, loc="upper left", bbox_to_anchor=(0, -0.22))
    fig.suptitle("Figure 19. Reasoning-move mix per model on two problems where models converged",
                 fontsize=10.5, color=vc.INK_SECONDARY, x=0.01, ha="left")
    fig.tight_layout(w_pad=1)
    vc.savefig(fig, "fig19_case_studies")


def all_figures():
    fig15_unescape_accuracy()
    fig16_answer_groups()
    fig17_shared_wrong_matrix()
    fig18_similarity_contrasts()
    fig19_case_studies()
