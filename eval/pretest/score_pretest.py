"""
Score a finished pre-test run and say whether the model is ready for a full run at
that setting. run_pretest.sh calls this; to run it by hand:

    cd eval/pretest
    ../../beaver-eval/Scripts/python.exe score_pretest.py \
        ../unified-output/fewshot/pretest/<model>-beaver-dw-setting<S>-pretest-log-<ts>

Needs summary_ex_acc.json in the run dir scored with `evaluate_ex_acc.py --sanitize`,
plus pretest_ids.json / item_bank.csv / calibration.json from this directory.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

PRETEST_DIR = Path(__file__).resolve().parent
TRUNCATION_STOP_REASONS = {"max_tokens", "length", "incomplete", "max_output_tokens"}
Z80 = 1.2816


def wilson(k, n, z=Z80):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="unified-output/fewshot/pretest/<pretest run>")
    ap.add_argument("--summary", default="summary_ex_acc.json")
    ap.add_argument("--setting", type=int, default=None, help="Override the setting parsed from the run dir name")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    m = re.search(r"-setting(\d)", run_dir.name)
    setting = args.setting if args.setting is not None else (int(m.group(1)) if m else None)
    if setting is None:
        sys.exit("Could not infer the setting from the run dir name; pass --setting")

    bank = pd.read_csv(PRETEST_DIR / "item_bank.csv", index_col="instance_id")
    calib = json.load(open(PRETEST_DIR / "calibration.json"))
    ids = json.load(open(PRETEST_DIR / "pretest_ids.json"))["ids"]
    thr = calib["readiness"]
    c = calib["scale_factor"]

    summary = json.load(open(run_dir / args.summary))
    if not summary["metrics"].get("sanitized"):
        print("WARNING: summary was not scored with --sanitize; HTML-escaped SQL counts as wrong "
              "and the format check below cannot see it.\n")
    details = {d["file"][:-4]: d for d in summary["details"]}
    missing = [i for i in ids if i not in details]
    if missing:
        sys.exit(f"Run is missing pre-test questions {missing} -- was it run with --ids_file pretest_ids.json?")

    rows = []
    for iid in ids:
        d = details[iid]
        usage = d.get("usage") or {}
        stops = {call.get("stop_reason") for call in usage.get("calls", [])}
        rows.append({
            "instance_id": iid,
            "band": bank.loc[iid, "band"],
            "correct": int(d["score"]),
            "html_escaped": bool(d.get("html_unescaped")),
            "abstained": bool(d.get("abstained")),
            "exec_error": bool(d.get("pred_error")),
            "truncated": bool(stops & TRUNCATION_STOP_REASONS),
            "api_error": bool(usage.get("errors")),
            "out_tokens": (usage.get("usage") or {}).get("completion_tokens"),
        })
    r = pd.DataFrame(rows).set_index("instance_id")
    n, k = len(r), int(r.correct.sum())

    pred = c * k / n
    lo, hi = (c * x for x in wilson(k, n))
    need = math.ceil(thr["min_predicted_accuracy"] * n / c - 1e-9)

    print(f"Pre-test: {run_dir.name}  (setting {setting}, {n} questions)\n")
    print("Correct by band (band = how many of the 6 calibration models solved it at setting 2):")
    for band in ["easy", "medium", "hard", "unsolved"]:
        sub = r[r.band == band]
        if len(sub):
            print(f"  {band:9s} {int(sub.correct.sum())}/{len(sub)}")
    print(f"  total     {k}/{n}\n")
    print(f"Predicted full-100 accuracy at setting {setting}: {pred:.0%}  "
          f"(= {c:.3f} x {k}/{n}; 80% interval {lo:.0%}-{hi:.0%})")
    v = calib["validation"]
    print(f"  validation on runs not used to build the pre-test: mean abs error {v['holdout_mae']:.1%}, "
          f"worst {v['holdout_max_abs_err']:.1%}")
    if k >= n - 2:
        print("  NOTE: near the pre-test ceiling -- this model is stronger than any calibration model, so the "
              "prediction is a lower bound. Fold its full run into build_pretest.py afterwards.")

    print("\nOutput health (problems a full run would score as wrong answers):")
    health = [
        ("SQL fails to execute", r.exec_error.mean(), thr["max_execution_error_fraction"]),
        ("answer cut off at token cap", r.truncated.mean(), thr["max_truncated_fraction"]),
        ("API errors", r.api_error.mean(), 0.0),
    ]
    health_ok = True
    for name, val, limit in health:
        ok = val <= limit + 1e-9
        health_ok &= ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {val:.0%} (limit {limit:.0%})")
    # Informational only: already handled by the harness (sql_sanitize.py)
    print(f"  info HTML-escaped SQL, auto-unescaped: {r.html_escaped.mean():.0%}")
    print(f"  info answered 'None' (abstained, scored wrong): {r.abstained.mean():.0%}")

    print()
    if not health_ok:
        print("VERDICT: FIX HARNESS FIRST -- the failures above are output/format problems, not reasoning.")
        print("A full run now would under-measure this model. See README.md 'If a health check fails'.")
    elif k >= need:
        print(f"VERDICT: READY for setting {setting} ({k}/{n} correct; ready needs >= {need}, "
              f"i.e. predicted >= {thr['min_predicted_accuracy']:.0%}).")
        if setting > 0:
            print(f"Next: run the pre-test at setting {setting - 1} to see if it is also ready for fewer hints.")
    else:
        print(f"VERDICT: NOT READY for setting {setting} ({k}/{n} correct; ready needs >= {need}). "
              f"A full run would land near the floor, where models can't be told apart.")
        if setting < 2:
            print(f"Use setting {setting + 1} (more hints) for this model.")

    print("\nPer question:")
    print(r[["band", "correct", "exec_error", "truncated", "abstained", "out_tokens"]].to_string())


if __name__ == "__main__":
    main()
