"""Figures 6-7: lexical feature profiles (bootstrap CI) and Fightin' Words scatter."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz_common as vc

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"

PROFILE_FEATURES = [
    "hedging_per_100w", "contrast_self_correction_per_100w", "commitment_action_per_100w",
    "meta_format_per_100w", "grounding_rate_per_100w", "first_person_per_100w",
    "question_rate_per_100w", "bullets_per_100w",
]


def _model_order(meta):
    order_key = meta.apply(lambda r: (0 if r["brand"] == "anthropic" else 1, vc.TIER_ORDER.get(r["size_tier"], 1)), axis=1)
    return meta.assign(_k=order_key).sort_values("_k")["model"].tolist()


def _bootstrap_mean_ci(values, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return values.mean(), *np.percentile(boots, [2.5, 97.5])


def fig6_lexical_profiles():
    """Q: how do specific reasoning-style features differ by model, with uncertainty?"""
    lex = pd.read_csv(TABLES / "lexical_features.csv")
    meta = lex[["model", "brand", "size_tier"]].drop_duplicates()
    order = _model_order(meta)
    colors = {m: vc.model_color(meta.set_index("model").loc[m, "brand"], meta.set_index("model").loc[m, "size_tier"])
              for m in order}

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, feat in zip(axes.flat, PROFILE_FEATURES):
        means, los, his = [], [], []
        for m in order:
            vals = lex.loc[lex["model"] == m, feat].dropna().values
            mean, lo, hi = _bootstrap_mean_ci(vals)
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)
        x = np.arange(len(order))
        ax.bar(x, means, color=[colors[m] for m in order], width=0.6, zorder=2,
               yerr=[los, his], error_kw={"ecolor": vc.INK_SECONDARY, "elinewidth": 1, "capsize": 2})
        vc.style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("claude-", "c-").replace("gpt-5.6-", "g-") for m in order],
                            rotation=45, ha="right", fontsize=7.5)
        ax.set_title(feat.replace("_per_100w", "").replace("_", " "), fontsize=9.5, loc="left")

    fig.suptitle("Figure 6. Lexical feature profiles by model (bars = mean per 100 words, "
                  "whiskers = 95% bootstrap CI resampling instances)", fontsize=10.5, color=vc.INK_SECONDARY, y=1.02)
    fig.tight_layout()
    vc.savefig(fig, "fig06_lexical_profiles")


def _place_labels(ax, xs, ys, words, colors_edge=None, fontsize=7):
    """Label points without letting the text pile on top of each other: each label sits
    beside its point, and is nudged up/down in steps until it clears every label already
    placed whose horizontal extent overlaps it. A thin leader line joins a nudged label
    to its point."""
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    gap = (yhi - ylo) * 0.038
    char_w = (xhi - xlo) * 0.0105
    dx = (xhi - xlo) * 0.008
    placed = []  # (x0, x1, y)
    order = sorted(range(len(xs)), key=lambda i: -abs(ys[i]))  # most extreme first keep their spot
    for i in order:
        x, y, w = xs[i], ys[i], words[i]
        x0, x1 = x + dx, x + dx + char_w * len(w)
        ly = y
        for step in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
            cand = y + step * gap
            if all(not (x0 < b1 and x1 > b0 and abs(cand - by) < gap * 0.95) for b0, b1, by in placed):
                ly = cand
                break
        placed.append((x0, x1, ly))
        if abs(ly - y) > gap * 0.4:
            ax.plot([x, x + dx], [y, ly], color=vc.AXIS, linewidth=0.6, zorder=2)
        ax.text(x + dx, ly, w, fontsize=fontsize, color=vc.INK_PRIMARY, va="center", zorder=5)


def _fightin_scatter(ax, df, label_a, label_b, color_a, color_b, top_n=12):
    freq = np.log10(df["count_a"] + df["count_b"] + 1)
    ax.scatter(freq, df["z"], s=10, alpha=0.35, color=vc.INK_MUTED, zorder=2)
    top = pd.concat([df.nlargest(top_n, "z"), df.nsmallest(top_n, "z")])
    top_freq = np.log10(top["count_a"] + top["count_b"] + 1)
    colors = [color_a if z > 0 else color_b for z in top["z"]]
    ax.scatter(top_freq, top["z"], s=30, color=colors, zorder=4, edgecolor=vc.INK_SECONDARY, linewidth=0.5)
    ax.axhline(0, color=vc.AXIS, linewidth=1, zorder=1)
    vc.style_axes(ax, grid_axis="y")
    ax.margins(x=0.06, y=0.08)
    ax.set_xlabel("log10(word frequency): 1 = ~10 uses, 3 = ~1,000 uses")
    ax.set_ylabel("Fightin' Words z-score (evidence of over-use)")
    ax.set_title(f"{label_a}  (z > 0)   vs.   {label_b}  (z < 0)", loc="left", fontsize=9.5)
    fig = ax.figure
    fig.canvas.draw()  # fix axis limits before placing labels
    _place_labels(ax, list(top_freq), list(top["z"]), list(top["word"]))


def fig7_fightin_words():
    """Q: which words are most distinctively over-used by one brand vs. the other,
    and by small vs. large models within a brand?"""
    brand_df = pd.read_csv(TABLES / "fightin_words_brand_anthropic_vs_openai.csv")
    size_df = pd.read_csv(TABLES / "fightin_words_size_anthropic_small_vs_large.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    _fightin_scatter(axes[0], brand_df, "Anthropic", "OpenAI", vc.BLUE, vc.ORANGE)
    # Right panel compares two ANTHROPIC models, so it uses those models' own tier colors
    # (light blue = haiku/small, dark blue = opus/large) -- orange means OpenAI everywhere else.
    _fightin_scatter(axes[1], size_df, "claude-haiku-4-5 (small)", "claude-opus-5 (large)",
                     vc.ANTHROPIC_TIERS[0], vc.ANTHROPIC_TIERS[2])

    fig.suptitle("Figure 7. Distinctive vocabulary: brand (left) and haiku vs. opus within Anthropic (right; "
                  "the other 5 size comparisons are in outputs/tables)",
                 fontsize=10, color=vc.INK_SECONDARY, y=1.02)
    fig.tight_layout()
    vc.savefig(fig, "fig07_fightin_words")
