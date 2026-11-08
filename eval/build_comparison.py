"""
Aggregates unified-output/<baseline>/<run_name>/{summary_ex_acc.json,summary_subtasks.json}
across every run of a baseline into one comparison CSV/JSON, joined against
model_registry.py for a size tier. This replaces the ad-hoc, hand-built
comparison-*-summary.csv files that used to accompany runs (eval/unified-output/ is
gitignored, so there was previously no committed script that produced them).

Usage:
    python build_comparison.py --baseline fewshot [--dataset dw] [--setting 1]
"""
import argparse
import csv
import json
import re
from pathlib import Path

from model_registry import get_model_info

RUN_NAME_RE = re.compile(r"^(?P<model>.+)-beaver-(?P<dataset>[^-]+)-setting(?P<setting>\d+)-log-(?P<timestamp>[\d-]+)$")

COLUMNS = [
    "model", "run_name", "run_dir", "dataset", "setting", "size_tier", "provider",
    "total_evaluated", "exact_matches", "accuracy_including_empty", "accuracy_excluding_empty",
    "avg_table_retrieval_f1", "avg_column_mapping_f1", "avg_join_key_f1",
    "avg_domain_knowledge_f1", "avg_query_decomposition_score",
    "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens", "avg_latency_sec",
    "avg_cost_usd", "total_cost_usd",
    "judge_avg_tokens", "judge_total_cost_usd",
]


def parse_run_name(run_name):
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return None
    return m.group("model"), m.group("dataset"), int(m.group("setting"))


def load_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_row(baseline, run_dir):
    run_name = run_dir.name
    parsed = parse_run_name(run_name)
    if parsed is None:
        return None
    model, dataset, setting = parsed

    ex_acc = load_json(run_dir / "summary_ex_acc.json").get("metrics", {})
    subtasks = load_json(run_dir / "summary_subtasks.json")
    model_info = get_model_info(model)

    return {
        "model": model,
        "run_name": run_name,
        "run_dir": str(Path("unified-output") / baseline / run_name),
        "dataset": dataset,
        "setting": setting,
        "size_tier": model_info["size_tier"],
        "provider": model_info["provider"],
        "total_evaluated": ex_acc.get("total_evaluated"),
        "exact_matches": ex_acc.get("exact_matches"),
        "accuracy_including_empty": ex_acc.get("accuracy_including_empty"),
        "accuracy_excluding_empty": ex_acc.get("accuracy_excluding_empty"),
        "avg_table_retrieval_f1": subtasks.get("avg_table_retrieval_f1"),
        "avg_column_mapping_f1": subtasks.get("avg_column_mapping_f1"),
        "avg_join_key_f1": subtasks.get("avg_join_key_f1"),
        "avg_domain_knowledge_f1": subtasks.get("avg_domain_knowledge_f1"),
        "avg_query_decomposition_score": subtasks.get("avg_query_decomposition_score"),
        "avg_prompt_tokens": ex_acc.get("avg_prompt_tokens"),
        "avg_completion_tokens": ex_acc.get("avg_completion_tokens"),
        "avg_total_tokens": ex_acc.get("avg_total_tokens"),
        "avg_latency_sec": ex_acc.get("avg_latency_sec"),
        "avg_cost_usd": ex_acc.get("avg_cost_usd"),
        "total_cost_usd": ex_acc.get("total_cost_usd"),
        "judge_avg_tokens": subtasks.get("judge_avg_tokens"),
        "judge_total_cost_usd": subtasks.get("judge_total_cost_usd"),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a per-model comparison summary CSV/JSON from unified-output run directories")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline name, e.g. fewshot, dinsql, dailsql, reforce")
    parser.add_argument("--dataset", type=str, default=None, help="Restrict to a single dataset (e.g. dw); default: include all found")
    parser.add_argument("--setting", type=int, default=None, help="Restrict to a single hint setting (0/1/2); default: include all found")
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    baseline_dir = eval_dir / "unified-output" / args.baseline
    if not baseline_dir.exists():
        print(f"No unified-output directory for baseline '{args.baseline}' at {baseline_dir}")
        return

    rows = []
    for run_dir in sorted(baseline_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        row = build_row(args.baseline, run_dir)
        if row is None:
            continue
        if args.dataset is not None and row["dataset"] != args.dataset:
            continue
        if args.setting is not None and row["setting"] != args.setting:
            continue
        rows.append(row)

    if not rows:
        print("No matching runs found.")
        return

    datasets = sorted({row["dataset"] for row in rows})
    settings = sorted({row["setting"] for row in rows})
    dataset_label = args.dataset or (datasets[0] if len(datasets) == 1 else "-".join(datasets))
    setting_label = (
        f"setting{args.setting}" if args.setting is not None
        else (f"setting{settings[0]}" if len(settings) == 1 else "all-settings")
    )

    out_stem = baseline_dir / f"comparison-{dataset_label}-{setting_label}-all-summary"

    with open(f"{out_stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(f"{out_stem}.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_stem}.csv and {out_stem}.json")


if __name__ == "__main__":
    main()
