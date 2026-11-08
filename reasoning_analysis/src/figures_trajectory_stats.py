"""Figures 10-12: trajectory drift/step-size, revisit rate vs correctness, coefficient forest plot."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz_common as vc
from . import data_loading as dl
from . import embeddings as emb
from . import text_normalize as tn

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"
N_BINS = 10


def _model_order(meta):
    order_key = meta.apply(lambda r: (0 if r["brand"] == "anthropic" else 1, vc.TIER_ORDER.get(r["size_tier"], 1)), axis=1)
    return meta.assign(_k=order_key).sort_values("_k")["model"].tolist()


def _compute_step_size_by_position(device="cuda"):
    """Not precomputed by Phase 2E (which only kept a per-trace scalar mean_step_size)
    -- built here for the figure, reusing the already-cached sentence embeddings so
    this costs no fresh GPU inference beyond whatever wasn't already cached."""
    rows, _models = dl.load_model_instances(sample=None)
    records = []
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        sentences = tn.split_sentences(tn.normalize_text(r["raw_text"]))
        if len(sentences) < 2:
            continue
        sent_embs = emb.embed_texts(sentences, device=device)
        for i in range(len(sentences) - 1):
            pos = i / (len(sentences) - 1) if len(sentences) > 1 else 0.0
            b = min(int(pos * N_BINS), N_BINS - 1)
            step = 1.0 - float(np.dot(sent_embs[i], sent_embs[i + 1]))
            records.append({"model": r["model"], "position_bin": b, "step_size": step})
    return pd.DataFrame(records)


def fig10_trajectory(device="cuda"):
    """Q: does reasoning drift away from the prompt over the course of a trace, and
    does the pace of that drift (step size) differ by model?"""
    drift = pd.read_csv(TABLES / "trajectory_prompt_drift.csv")
    meta = drift[["model", "brand", "size_tier"]].drop_duplicates()
    order = _model_order(meta)
    colors = {m: vc.model_color(meta.set_index("model").loc[m, "brand"], meta.set_index("model").loc[m, "size_tier"])
              for m in order}

    print("  (fig 10) computing step-size-by-position -- reuses cached sentence embeddings...")
    step_df = _compute_step_size_by_position(device=device)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for m in order:
        sub = drift[drift["model"] == m].groupby("position_bin")["mean_prompt_similarity"].mean()
        sub = sub.reindex(range(N_BINS))
        ax.plot((np.arange(N_BINS) + 0.5) / N_BINS, sub.values, color=colors[m], linewidth=2, label=m, zorder=3)
    vc.style_axes(ax, grid_axis="both")
    ax.set_xlabel("normalized position in trace")
    ax.set_ylabel("mean cosine similarity to prompt")
    ax.set_title("Prompt-similarity drift", loc="left", fontsize=10.5)

    ax = axes[1]
    for m in order:
        sub = step_df[step_df["model"] == m].groupby("position_bin")["step_size"].mean()
        sub = sub.reindex(range(N_BINS))
        ax.plot((np.arange(N_BINS) + 0.5) / N_BINS, sub.values, color=colors[m], linewidth=2, label=m, zorder=3)
    vc.style_axes(ax, grid_axis="both")
    ax.set_xlabel("normalized position in trace")
    ax.set_ylabel("mean step size (1 - cosine sim. to next sentence)")
    ax.set_title("Step size over the trace", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)

    fig.suptitle("Figure 10. Reasoning trajectory: drift from the prompt and step size, by normalized position",
                 fontsize=10.5, color=vc.INK_SECONDARY, y=1.03)
    fig.tight_layout()
    vc.savefig(fig, "fig10_trajectory")


def fig11_revisit_vs_correctness():
    """Q: do traces that loop back to earlier points tend to be more or less correct,
    and does that differ by brand?

    Revisit rate is exactly 0 for most traces, which makes violins misleading (the
    smoothed density invents mass below zero and hides the pile-up at 0). So every trace
    is drawn as a point, the mean is a black diamond, and the labels give n and the share
    of traces with any revisit at all."""
    master = pd.read_csv(TABLES / "master_analysis_table.csv")
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    ymax = master["revisit_rate"].max() * 1.08
    for ax, brand in zip(axes, ["anthropic", "openai"]):
        sub = master[master["brand"] == brand]
        color = vc.BRAND_COLORS[brand]
        labels = []
        for j, (name, val) in enumerate([("incorrect", False), ("correct", True)]):
            g = sub.loc[sub["correct"] == val, "revisit_rate"].dropna()
            ax.scatter(j + rng.uniform(-0.2, 0.2, len(g)), g, s=16, alpha=0.35, color=color,
                       edgecolor="none", zorder=2)
            ax.scatter([j], [g.mean()], marker="D", s=80, color=vc.INK_PRIMARY, edgecolor="white",
                       linewidth=0.8, zorder=4)
            ax.annotate(f"mean {g.mean():.3f}", (j, g.mean()), xytext=(14, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=vc.INK_PRIMARY)
            labels.append(f"{name}\nn={len(g)}, {100 * (g > 0).mean():.0f}% have any revisit")
        vc.style_axes(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.6, 1.75)
        ax.set_ylim(-0.02, ymax)
        ax.set_title(brand, loc="left", fontsize=10.5)
    axes[0].set_ylabel("revisit rate per trace")
    fig.suptitle("Figure 11. Revisit rate by correctness (each dot = one trace, black diamond = mean; "
                 "traces under 3 sentences have no revisit rate and are excluded)", fontsize=10, color=vc.INK_SECONDARY, y=1.03)
    fig.tight_layout()
    vc.savefig(fig, "fig11_revisit_vs_correctness")


def fig12_coefficient_forest():
    """Q: which style features predict correctness, once brand/size are controlled for?

    Coefficients are converted to an ODDS RATIO PER +1 STANDARD DEVIATION of each feature
    so the rows are comparable: a raw coefficient is "per +1 unit", and the features have
    very different natural ranges (hedging SD ~1.4 words/100; move rates SD ~0.06-0.3).
    Log-odds coefficient beta -> OR = exp(beta * SD); the bootstrap CI endpoints transform
    the same way (the mapping is monotone, so percentiles carry over)."""
    corr = pd.read_csv(TABLES / "correctness_results.csv")
    master = pd.read_csv(TABLES / "master_analysis_table.csv")
    sub = corr[corr["subset"] == "all"].copy()
    sub["sd"] = sub["feature"].map(lambda f: master[f].std())
    dropped = sub[sub[["coef", "ci_low", "ci_high"]].isna().any(axis=1)]["feature"].tolist()
    sub = sub.dropna(subset=["coef", "ci_low", "ci_high"])
    for c in ("coef", "ci_low", "ci_high"):
        sub[c + "_or"] = np.exp(sub[c] * sub["sd"])
    sub = sub.sort_values("coef_or").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    y = np.arange(len(sub))
    sig = (sub["pvalue_bh"] < 0.05).values
    colors = [vc.RED if s_ else vc.INK_MUTED for s_ in sig]
    ax.errorbar(sub["coef_or"], y, xerr=[sub["coef_or"] - sub["ci_low_or"], sub["ci_high_or"] - sub["coef_or"]],
                fmt="none", ecolor=vc.INK_MUTED, elinewidth=1.6, capsize=2, zorder=3)
    ax.scatter(sub["coef_or"], y, c=colors, s=34, zorder=4, edgecolor="white", linewidth=0.4)
    ax.axvline(1, color=vc.AXIS, linewidth=1.2, zorder=1)
    ax.set_xscale("log")
    ticks = [0.3, 0.5, 0.7, 1, 1.4, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    vc.style_axes(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}  (1 SD = {sd:.3g})" for f, sd in zip(sub["feature"], sub["sd"])], fontsize=8.5)
    ax.set_xlabel("odds ratio of a correct answer per +1 SD of the feature  (log scale; 1 = no effect)")
    ax.annotate("lower odds of correct ←", (1, len(sub) - 0.2), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8, color=vc.INK_SECONDARY)
    ax.annotate("→ higher odds of correct", (1, len(sub) - 0.2), xytext=(6, 0), textcoords="offset points",
                ha="left", fontsize=8, color=vc.INK_SECONDARY)
    note = "red = BH-adjusted p < 0.05 (only hedging_per_100w)" + (
        f"; not shown (CI failed to compute): {', '.join(dropped)}" if dropped else "")
    ax.set_title("Figure 12. Which style features predict correctness?\n" + note,
                 loc="left", fontsize=9.5, color=vc.INK_SECONDARY)
    fig.tight_layout()
    vc.savefig(fig, "fig12_coefficient_forest")
