"""
Phase 3: correctness and statistics.

  - Mixed-effects: feature ~ brand + size_tier + log(completion_tokens) + (1 | instance_id)
  - Correctness: correct ~ feature + brand + size_tier + (1 | instance_id) -- fit via GEE
    with an exchangeable working correlation grouped by instance_id (statsmodels doesn't
    have a simple, robust frequentist mixed logit; GEE is the spec's own suggested
    fallback for exactly this situation).
  - Bootstrap CIs resample INSTANCES (not trace rows) -- an instance's rows (its multiple
    models) move together as a block, or not at all.
  - Benjamini-Hochberg FDR correction across the feature-test family.
  - Robustness: rerun on is_summary==False (raw traces) only. NOTE: is_summary is ~
    collinear with brand in this dataset (Phase 0/1 finding: 100% of OpenAI traces vs.
    0-9% of Anthropic are flagged is_summary), so "raw-only" is, in practice, very close
    to "Anthropic-only" here -- this check can't cleanly separate the brand effect from
    the summary-format effect, and that limitation is reported alongside the numbers,
    not hidden.

`setting` is NOT included as a covariate: every row in this dataset has setting=2 (Phase 0
scope decision), so it's a constant with zero variance and can't be estimated.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.stats.multitest import multipletests

from . import data_loading as dl
from . import move_taxonomy as mt

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "outputs" / "tables"

LEXICAL_FEATURES = [
    "hedging_per_100w", "contrast_self_correction_per_100w", "commitment_action_per_100w",
    "meta_format_per_100w", "grounding_rate_per_100w", "question_rate_per_100w",
    "first_person_per_100w", "bullets_per_100w", "code_blocks_per_100w",
]
TRAJECTORY_FEATURES = ["mean_step_size", "revisit_rate", "final_sentence_similarity_to_final_answer"]
MOVE_FEATURES = [f"move_{m}_rate" for m in mt.MOVES]

ALL_FEATURES = LEXICAL_FEATURES + TRAJECTORY_FEATURES + MOVE_FEATURES


def build_master_table(sample=None):
    """One row per (model, instance_id) with every Phase 1/2 feature joined together."""
    rows, _models = dl.load_model_instances(sample=sample)
    lex_path = TABLES / "lexical_features.csv"
    traj_path = TABLES / "trajectory_summary.csv"
    moves_path = TABLES / "move_labels_per_sentence.csv"
    for p in (lex_path, traj_path, moves_path):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run --stage lexical, embeddings, trajectory, and moves first.")

    lex = pd.read_csv(lex_path)
    traj = pd.read_csv(traj_path)
    moves = pd.read_csv(moves_path)

    move_props = (
        moves.groupby(["model", "instance_id"])["predicted_move"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for m in mt.MOVES:
        if m not in move_props.columns:
            move_props[m] = 0.0
    move_props = move_props.rename(columns={m: f"move_{m}_rate" for m in mt.MOVES})
    move_props = move_props[["model", "instance_id"] + MOVE_FEATURES]

    base = pd.DataFrame([r for r in rows if r["has_reasoning_trace"]])
    base = base[["model", "instance_id", "brand", "size_tier", "correct", "completion_tokens", "run_name"]]

    df = base.merge(lex, on=["model", "instance_id"], how="left", suffixes=("", "_lex"))
    df = df.merge(traj[["model", "instance_id"] + TRAJECTORY_FEATURES + ["n_sentences"]],
                  on=["model", "instance_id"], how="left")
    df = df.merge(move_props, on=["model", "instance_id"], how="left")

    df["log_completion_tokens"] = np.log(df["completion_tokens"].clip(lower=1))
    df["correct_int"] = df["correct"].astype(int)
    df["is_summary"] = df["is_summary"].astype("boolean")

    # size_tier is nested within brand (a "small" Anthropic model and a "small" OpenAI
    # model aren't the same absolute size) -- fine for the additive formulas the spec
    # asks for, but noted here since it's a real modeling simplification.
    df["brand"] = pd.Categorical(df["brand"], categories=["anthropic", "openai"])
    df["size_tier"] = pd.Categorical(df["size_tier"], categories=["small", "mid", "large"], ordered=False)

    return df


def _brand_size_terms(sub):
    """Only include brand/size_tier as covariates when they actually vary in `sub` --
    e.g. the is_summary==False subset is ~100% Anthropic (Phase 0/1 confound), so brand
    is a constant there and can't be estimated, while size_tier still varies fine.
    Dropping only the degenerate term (not skipping the whole feature) is what lets the
    raw-only robustness check produce anything at all."""
    terms = []
    used = []
    if sub["brand"].nunique() >= 2:
        terms.append("C(brand, Treatment('anthropic'))")
        used.append("brand")
    if sub["size_tier"].nunique() >= 2:
        terms.append("C(size_tier, Treatment('small'))")
        used.append("size_tier")
    return terms, used


def _fit_mixedlm(df, feature):
    sub = df.dropna(subset=[feature, "brand", "size_tier", "log_completion_tokens", "instance_id"])
    terms, used = _brand_size_terms(sub)
    if not terms:
        return None, sub, used
    formula = f"{feature} ~ " + " + ".join(terms + ["log_completion_tokens"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, sub, groups=sub["instance_id"])
        result = model.fit(reml=True)
    return result, sub, used


def _fit_correctness_gee(df, feature):
    sub = df.dropna(subset=[feature, "brand", "size_tier", "correct_int", "instance_id"])
    terms, used = _brand_size_terms(sub)
    formula = f"correct_int ~ {feature}" + ("".join(f" + {t}" for t in terms))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GEE.from_formula(formula, groups="instance_id", data=sub,
                                  family=Binomial(), cov_struct=Exchangeable())
        result = model.fit()
    return result, sub, used


def _bootstrap_ci(df, feature, fit_fn, term, n_boot=200, random_state=42):
    """Resamples UNIQUE instance_ids with replacement; each resampled instance_id
    contributes ALL of its rows (every model's trace for that instance) as one block."""
    rng = np.random.RandomState(random_state)
    instance_ids = df["instance_id"].dropna().unique()
    if len(instance_ids) == 0:
        return None, None

    estimates = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(instance_ids, size=len(instance_ids), replace=True)
        # Re-key instance_id per draw (draw index suffix) so two draws of the same
        # original instance aren't silently merged into one group by the model's
        # grouping variable -- each drawn block stays its own cluster.
        parts = []
        for draw_idx, iid in enumerate(sampled_ids):
            block = df[df["instance_id"] == iid].copy()
            block["instance_id"] = f"{iid}__draw{draw_idx}"
            parts.append(block)
        boot_df = pd.concat(parts, ignore_index=True)

        try:
            result, _sub, _used = fit_fn(boot_df, feature)
            if result is not None and term in result.params.index:
                estimates.append(result.params[term])
        except Exception:
            continue

    if len(estimates) < max(10, n_boot // 4):
        return None, None
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def _apply_bh_correction(result_df):
    """A single NaN p-value poisons statsmodels' multipletests() for the WHOLE array
    (returns all-NaN, not just the offending entry) -- exclude NaNs before correcting,
    then write the adjusted values back only for the rows that had a real p-value."""
    result_df["pvalue_bh"] = np.nan
    valid = result_df["pvalue"].notna()
    if valid.any():
        result_df.loc[valid, "pvalue_bh"] = multipletests(result_df.loc[valid, "pvalue"], method="fdr_bh")[1]
    return result_df


def run_mixed_effects_battery(df, features=None, n_boot=200):
    features = features or ALL_FEATURES
    rows = []
    for feat in features:
        if feat not in df.columns:
            print(f"  (skip) {feat}: not in master table")
            continue
        result, sub, used = _fit_mixedlm(df, feat)
        if result is None:
            print(f"  (skip) {feat}: no variation in brand or size_tier after dropping NaNs -- nothing estimable")
            continue
        if "brand" not in used or "size_tier" not in used:
            print(f"  NOTE: {feat} fit with covariates={used} (one of brand/size_tier was dropped -- degenerate in this subset)")
        for term_label, term_name in [
            ("brand[openai_vs_anthropic]", "C(brand, Treatment('anthropic'))[T.openai]"),
            ("size_tier[mid_vs_small]", "C(size_tier, Treatment('small'))[T.mid]"),
            ("size_tier[large_vs_small]", "C(size_tier, Treatment('small'))[T.large]"),
        ]:
            if term_name not in result.params.index:
                continue
            coef = result.params[term_name]
            pval = result.pvalues[term_name]
            lo, hi = _bootstrap_ci(df, feat, _fit_mixedlm, term_name, n_boot=n_boot)
            rows.append({"feature": feat, "term": term_label, "coef": coef, "pvalue": pval,
                         "ci_low": lo, "ci_high": hi, "n_obs": int(result.nobs)})
        print(f"  fit: {feat} (n={int(result.nobs)})")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = _apply_bh_correction(result_df)
    return result_df


def run_correctness_battery(df, features=None, n_boot=200, label=""):
    features = features or ALL_FEATURES
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        result, sub, used = _fit_correctness_gee(df, feat)
        if result is None or feat not in result.params.index:
            print(f"  (skip{' ' + label if label else ''}) {feat}: not estimable")
            continue
        if "brand" not in used or "size_tier" not in used:
            print(f"  NOTE{' ' + label if label else ''}: {feat} fit with covariates={used} "
                  f"(one of brand/size_tier was dropped -- degenerate in this subset)")
        coef = result.params[feat]
        pval = result.pvalues[feat]
        if pd.isna(coef):
            print(f"  WARNING{' ' + label if label else ''}: {feat} did not converge to a usable "
                  f"coefficient (NaN) -- likely too little variation in this subset; kept as NaN, not dropped")
        lo, hi = _bootstrap_ci(df, feat, _fit_correctness_gee, feat, n_boot=n_boot)
        rows.append({"feature": feat, "coef": coef, "pvalue": pval, "ci_low": lo, "ci_high": hi,
                     "n_obs": int(result.nobs), "subset": label or "all", "covariates_used": ",".join(used)})
        print(f"  fit{' ' + label if label else ''}: {feat} (n={int(result.nobs)})")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = _apply_bh_correction(result_df)
    return result_df
