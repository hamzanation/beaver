import json
import argparse
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / '.env')

from utils.ex_acc import get_mysql_credentials, execute_sql_with_timeout, compare_results
from utils.utils import write_json
from usage_utils import read_usage_json
from sql_sanitize import sanitize_sql, is_abstention, has_html_entities

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for text-to-SQL baselines")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for MySQL credentials, e.g. dw")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the unified output directory containing generated/ and gold/ subdirectories")
    parser.add_argument("--creds_path", type=str, default=None, help="Optional JSON file containing MySQL connection fields")
    parser.add_argument("--sanitize", action="store_true", help="Unescape HTML entities (&lt; &gt;) and treat a bare 'None' as an empty prediction before executing (see sql_sanitize.py)")
    parser.add_argument("--output_name", type=str, default="summary_ex_acc.json", help="Summary filename inside input_dir, e.g. summary_ex_acc_sanitized.json to keep the original")
    args = parser.parse_args()
    
    generated_dir =  Path(args.input_dir) / "generated"
    gold_dir = Path(args.input_dir) / "gold"
    usage_dir = Path(args.input_dir) / "usage"
    
    if not generated_dir.exists() or not gold_dir.exists():
        print(f"Error: Could not find 'generated' or 'gold' directories inside {args.input_dir}")
        sys.exit(1)
        
    mysql_creds = get_mysql_credentials(args.dataset, args.creds_path)
    if not mysql_creds:
        print(f"Error: Could not load MySQL credentials for dataset {args.dataset}")
        print("Set MYSQL_HOST, MYSQL_USER, and MYSQL_PASSWORD in .env at the repo root, or pass --creds_path path/to/mysql_creds.json")
        sys.exit(1)

    _, conn_err = execute_sql_with_timeout("SELECT 1", mysql_creds)
    if conn_err:
        print(f"Error: Could not connect to MySQL database '{mysql_creds.get('database')}'.")
        print(conn_err)
        print("Start MySQL, verify MYSQL_HOST/MYSQL_USER/MYSQL_PASSWORD, and make sure the BEAVER database is loaded before scoring.")
        sys.exit(1)
        
    gold_files = sorted(gold_dir.glob("*.sql"))
    
    total_queries = len(gold_files)
    total_attempted = total_queries
    
    total_score = 0
    nonempty_gold_total = 0
    nonempty_gold_score = 0
    
    results = []
    
    print(f"Evaluating {total_queries} queries from {args.input_dir}...")
    
    for gold_sql_path in tqdm(gold_files):
        filename = gold_sql_path.name
        pred_sql_path = generated_dir / filename
        usage_record = read_usage_json(usage_dir / f"{gold_sql_path.stem}.json")

        # 1. Execute Gold SQL
        with open(gold_sql_path, "r") as f:
            gold_sql = f.read().strip()

        gold_df, gold_err = execute_sql_with_timeout(gold_sql, mysql_creds)
        if gold_err:
            results.append({
                "file": filename,
                "match": False,
                "score": 0,
                "message": "Gold SQL execution failed",
                "gold_empty": None,
                "pred_empty": None,
                "gold_error": gold_err,
                "pred_error": None,
                "usage": usage_record,
            })
            continue
            
        # 2. Execute Predicted SQL
        pred_sql = ""
        if pred_sql_path.exists():
            with open(pred_sql_path, "r") as f:
                pred_sql = f.read().strip()
        html_unescaped = has_html_entities(pred_sql)
        abstained = is_abstention(pred_sql)
        if args.sanitize:
            pred_sql = sanitize_sql(pred_sql)
        
        pred_df = None
        pred_err = None
        if pred_sql:
            pred_df, pred_err = execute_sql_with_timeout(pred_sql, mysql_creds)
            
        if pred_err:
            pred_df = pd.DataFrame()
        elif pred_df is None:
            pred_df = pd.DataFrame()
            
        # Compare
        match, msg = compare_results(pred_df, gold_df)
        score = 1 if match else 0
        
        total_score += score
        gold_is_empty = gold_df.empty
        
        if not gold_is_empty:
            nonempty_gold_total += 1
            nonempty_gold_score += score
            
        results.append({
            "file": filename,
            "match": match,
            "score": score,
            "message": msg,
            "gold_empty": gold_is_empty,
            "pred_empty": pred_df.empty,
            "gold_error": gold_err,
            "pred_error": pred_err,
            "html_unescaped": html_unescaped,
            "abstained": abstained,
            "usage": usage_record,
        })
    
    acc_including_empty = (100 * total_score / total_attempted) if total_attempted > 0 else 0.0
    if nonempty_gold_total > 0:
        acc_excluding_empty = 100 * nonempty_gold_score / nonempty_gold_total

    # Aggregate resource telemetry across examples that recorded usage.json
    usage_records = [r["usage"] for r in results if r.get("usage")]
    total_prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in usage_records)
    total_completion_tokens = sum(r["usage"]["completion_tokens"] for r in usage_records)
    total_tokens = sum(r["usage"]["total_tokens"] for r in usage_records)
    total_latency_sec = sum(r.get("latency_sec", 0.0) for r in usage_records)
    priced = [r["cost_usd"] for r in usage_records if r.get("cost_usd") is not None]
    usage_metrics = {
        "examples_with_usage": len(usage_records),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": (total_prompt_tokens / len(usage_records)) if usage_records else None,
        "avg_completion_tokens": (total_completion_tokens / len(usage_records)) if usage_records else None,
        "avg_total_tokens": (total_tokens / len(usage_records)) if usage_records else None,
        "avg_latency_sec": (total_latency_sec / len(usage_records)) if usage_records else None,
        "total_cost_usd": sum(priced) if priced else None,
        "avg_cost_usd": (sum(priced) / len(priced)) if priced else None,
        "examples_priced": len(priced),
    }

    # Save results summary
    summary_data = {
        "metrics": {
            "total_evaluated": total_attempted,
            "exact_matches": total_score,
            "accuracy_including_empty": acc_including_empty,
            "nonempty_gold_total": nonempty_gold_total,
            "nonempty_gold_score": nonempty_gold_score,
            "accuracy_excluding_empty": acc_excluding_empty if nonempty_gold_total > 0 else None,
            "sanitized": args.sanitize,
            "predictions_html_escaped": sum(1 for r in results if r.get("html_unescaped")),
            "predictions_abstained": sum(1 for r in results if r.get("abstained")),
            **usage_metrics,
        },
        "details": sorted(results, key=lambda x: x["file"])
    }
    
    summary_path = Path(args.input_dir) / args.output_name
    write_json(summary_data, summary_path)
    
    print(f"\nEvaluation Complete.")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary_data['metrics'], indent=2))

if __name__ == "__main__":
    main()
