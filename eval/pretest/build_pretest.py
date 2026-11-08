"""
Build the BEAVER dw pre-test: pick 15 of the 100 dev_sampled questions whose results
predict a model's full-100 execution accuracy, and validate that prediction on runs
that played no part in picking them. See README.md for the what and the why.

Run by hand (only needed again when new full runs should be folded in):
    cd eval/pretest && ../../beaver-eval/Scripts/python.exe build_pretest.py

Inputs: every fewshot dw run under eval/unified-output/fewshot/ (and its dated
subfolders) that has a summary_ex_acc_sanitized.json, i.e. was scored with
`evaluate_ex_acc.py --sanitize --output_name summary_ex_acc_sanitized.json`.

Writes to eval/pretest/:
    pretest_ids.json   the 15 question ids (fewshot/run.sh --ids_file pretest/pretest_ids.json)
    item_bank.csv      band, solve counts and gold-SQL flags for all 100 questions
    calibration.json   scale factor + readiness thresholds read by score_pretest.py
    validation.csv     predicted vs actual full-100 accuracy for every run
"""
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
PRETEST_DIR = Path(__file__).resolve().parent
FEWSHOT_UNIFIED = EVAL_DIR / "unified-output" / "fewshot"
DATASET = "dw"

# Current-generation setting-2 runs: these define question difficulty and pick the set.
CALIBRATION_MODELS = {"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5",
                      "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}
CALIBRATION_MIN_DATE = "20260901"
SUPERSEDED_RUNS = {
    # 28/100 answers cut off at max_tokens=5120; replaced by the -maxtok16k rerun
    "claude-haiku-4-5-beaver-dw-setting2-log-20260901-140932",
}
SKIP_SUBDIRS = {"20260708", "20260716"}  # earlier setting-1 attempts, redone in 20260731

# Band = how many of the 6 calibration models solved the question at setting 2.
# Weighted toward easy/medium: at settings 0 and 1 only easy questions ever get solved,
# so they carry the signal there; medium/hard separate the stronger models at setting 2.
BAND_ALLOCATION = {"easy": 6, "medium": 5, "hard": 3, "unsolved": 1}
MIN_FORMAT_CANARIES = 3   # gold SQL uses < or > -- exposes HTML-escaped output (&lt; &gt;)
MIN_LONG_QUERIES = 2      # gold SQL in the longest quartile -- exposes token-cap truncation
N_DRAWS = 3000
SEED = 0

READINESS = {
    "min_predicted_accuracy": 0.10,        # below this a full run sits at the floor (old models at setting 0: 2-6%)
    "max_execution_error_fraction": 0.20,  # SQL that fails to execute even after sanitizing
    "max_truncated_fraction": 0.0,         # any answer cut off at the token cap -> raise the cap first
}


def band_of(k):
    if k >= 4:
        return "easy"
    if k >= 2:
        return "medium"
    if k == 1:
        return "hard"
    return "unsolved"


def discover_runs():
    candidates = [d for d in FEWSHOT_UNIFIED.iterdir() if d.is_dir()]
    for sub in FEWSHOT_UNIFIED.iterdir():
        if sub.is_dir() and re.fullmatch(r"\d{8}.*", sub.name) and sub.name not in SKIP_SUBDIRS:
            candidates += [d for d in sub.iterdir() if d.is_dir()]
    runs = []
    for d in candidates:
        m = re.match(r"(.+?)-beaver-(\w+?)-setting(\d)(?:-([\w-]+?))?-log-(\d{8})", d.name)
        if not m or m.group(2) != DATASET or d.name in SUPERSEDED_RUNS:
            continue
        if m.group(4) and "pretest" in m.group(4):
            continue
        summary = d / "summary_ex_acc_sanitized.json"
        if not summary.exists():
            print(f"  skip {d.name}: no summary_ex_acc_sanitized.json")
            continue
        details = json.load(open(summary))["details"]
        if len(details) != 100:
            continue
        model, setting, run_date = m.group(1), int(m.group(3)), m.group(5)
        is_cal = model in CALIBRATION_MODELS and setting == 2 and run_date >= CALIBRATION_MIN_DATE
        runs.append({"run": d.name, "path": d, "model": model, "setting": setting,
                     "role": "calibration" if is_cal else "holdout",
                     "scores": {x["file"][:-4]: x["score"] for x in details}})
    return sorted(runs, key=lambda r: (r["role"], r["setting"], r["run"]))


def gold_features(gold_dir):
    rows = {}
    for f in sorted(gold_dir.glob("*.sql")):
        q = f.read_text(encoding="utf-8").upper()
        rows[f.stem] = {
            "gold_chars": len(q),
            "gold_joins": q.count(" JOIN "),
            "gold_subqueries": q.count("SELECT") - 1,
            "format_canary": bool(re.search(r"[<>]", q)),
        }
    df = pd.DataFrame(rows).T
    df["long_query"] = df.gold_chars >= df.gold_chars.quantile(0.75)
    return df


def scale_factor(frac, actual):
    """Least-squares c in actual ~= c * pretest_fraction (line through the origin)."""
    return float(frac @ actual / (frac @ frac))


def main():
    rng = np.random.default_rng(SEED)
    runs = discover_runs()
    items = sorted(runs[0]["scores"])
    Y = np.array([[r["scores"][i] for i in items] for r in runs], dtype=float)
    meta = pd.DataFrame([{k: r[k] for k in ("run", "model", "setting", "role")} for r in runs])
    cal = (meta.role == "calibration").to_numpy()
    hold = ~cal
    actual = Y.mean(1)
    print(f"{cal.sum()} calibration runs, {hold.sum()} holdout runs, {len(items)} questions")

    feats = gold_features(next(r["path"] for r in runs if r["role"] == "calibration") / "gold")
    canary = feats.loc[items, "format_canary"].to_numpy(bool)
    long_q = feats.loc[items, "long_query"].to_numpy(bool)
    solved_cal = Y[cal].sum(0).astype(int)
    bands = np.array([band_of(k) for k in solved_cal])
    pools = {name: np.where(bands == name)[0] for name in BAND_ALLOCATION}
    print("band sizes:", {n: len(p) for n, p in pools.items()})

    # Constrained stratified random draws; keep the one whose (calibrated) prediction best
    # reproduces the 6 calibration runs' full-100 accuracy.
    best, best_err = None, np.inf
    tried = 0
    while tried < N_DRAWS:
        idx = np.concatenate([rng.choice(pools[n], c, replace=False) for n, c in BAND_ALLOCATION.items()])
        if canary[idx].sum() < MIN_FORMAT_CANARIES or long_q[idx].sum() < MIN_LONG_QUERIES:
            continue
        tried += 1
        frac = Y[:, idx].mean(1)
        c = scale_factor(frac[cal], actual[cal])
        err = np.abs(c * frac[cal] - actual[cal]).mean()
        if err < best_err:
            best, best_err = np.sort(idx), err

    frac = Y[:, best].mean(1)
    c = scale_factor(frac[cal], actual[cal])
    meta["actual_acc"] = actual
    meta["pretest_correct"] = Y[:, best].sum(1).astype(int)
    meta["predicted_acc"] = c * frac
    meta["abs_err"] = (meta.predicted_acc - meta.actual_acc).abs()
    thr = READINESS["min_predicted_accuracy"]
    meta["ready_predicted"] = meta.predicted_acc >= thr
    meta["ready_actual"] = meta.actual_acc >= thr

    # Baseline: the same estimator on purely random 15-question subsets.
    rand_err = []
    for _ in range(2000):
        idx = rng.choice(len(items), len(best), replace=False)
        f = Y[:, idx].mean(1)
        if f[cal].sum() == 0:
            continue
        rand_err.append(np.abs(scale_factor(f[cal], actual[cal]) * f - actual)[hold].mean())

    pd.set_option("display.width", 200)
    print(meta.drop(columns=["model"]).round(3).to_string(index=False))
    print(f"scale factor c = {c:.3f}  (each pre-test question ~ {100 * c / len(best):.1f} points of full-100 accuracy)")
    print(f"MAE calibration={meta.abs_err[cal].mean():.3f}  holdout={meta.abs_err[hold].mean():.3f}  "
          f"random-15 holdout median={np.median(rand_err):.3f} (p90 {np.percentile(rand_err, 90):.3f})")
    agree = (meta.ready_predicted == meta.ready_actual)[hold]
    print(f"readiness verdict (>= {thr:.0%}) agrees with the full run on {agree.sum()}/{hold.sum()} holdout runs")

    bank = feats.copy()
    bank["band"] = bands
    bank["solved_by_calibration_runs"] = solved_cal
    bank["solved_by_all_runs"] = Y.sum(0).astype(int)
    for s in sorted(meta.setting.unique()):
        bank[f"solve_rate_setting{s}"] = Y[(meta.setting == s).to_numpy()].mean(0).round(3)
    bank["in_pretest"] = False
    bank.loc[[items[i] for i in best], "in_pretest"] = True
    bank.index.name = "instance_id"
    bank.to_csv(PRETEST_DIR / "item_bank.csv")

    ids = [items[i] for i in best]
    json.dump({
        "dataset": DATASET,
        "built": date.today().isoformat(),
        "ids": ids,
        "bands": {i: bank.loc[i, "band"] for i in ids},
        "note": "See eval/pretest/README.md. Run: fewshot/run.sh --ids_file ../pretest/pretest_ids.json --tag pretest",
    }, open(PRETEST_DIR / "pretest_ids.json", "w"), indent=2)
    json.dump({
        "built": date.today().isoformat(),
        "n_questions": len(ids),
        "scale_factor": c,
        "readiness": READINESS,
        "validation": {
            "calibration_mae": float(meta.abs_err[cal].mean()),
            "holdout_mae": float(meta.abs_err[hold].mean()),
            "holdout_max_abs_err": float(meta.abs_err[hold].max()),
            "random15_holdout_mae_median": float(np.median(rand_err)),
            "holdout_verdict_agreement": f"{int(agree.sum())}/{int(hold.sum())}",
        },
    }, open(PRETEST_DIR / "calibration.json", "w"), indent=2)
    meta.to_csv(PRETEST_DIR / "validation.csv", index=False)
    print("pretest ids:", ids)


if __name__ == "__main__":
    main()
