"""
Data loading & joining for the reasoning-trace analysis pipeline.

Builds one row per (model, instance_id) for every ATTEMPTED instance (from
summary_ex_acc.json's `details`), not just instances that produced a visible
reasoning trace. `has_reasoning_trace` is an explicit column, never a silent
filter -- callers decide per-analysis whether to filter on it (Phase 0 decision
with the user: keep attempted-but-traceless rows in the master table so
trace-coverage can be measured, filter them out only for analyses that need
actual reasoning text).

Scope, per Phase 0 findings/decisions with the user (2026-09):
  - dataset=dw, setting=2 only (the only data that exists right now).
  - 6 models in scope: claude-haiku-4-5, claude-sonnet-5, claude-opus-5 (Anthropic)
    and gpt-5.6-luna, gpt-5.6-terra, gpt-5.6-sol (OpenAI) -- each has both a
    reasoning/ dir and a summary_ex_acc.json. gpt-5.4-mini/gpt-5.5/claude-sonnet-4-5
    are pre-reasoning-instrumentation runs, confirmed out of scope.
  - Run directories are discovered by pattern, not hardcoded by timestamp, so this
    stays reproducible as new runs land under eval/unified-output/fewshot/.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FEWSHOT_UNIFIED = REPO_ROOT / "eval" / "unified-output" / "fewshot"
FEWSHOT_RAW = REPO_ROOT / "eval" / "fewshot" / "output"
PRICING_CSV = REPO_ROOT / "model_pricing.csv"
HUMAN_DIR = REPO_ROOT / "example_prompts" / "manual_reasoning.json"

# Pre-reasoning-instrumentation runs -- confirmed out of scope for this analysis.
EXCLUDED_MODELS = {"gpt-5.4-mini", "gpt-5.5", "claude-sonnet-4-5"}


def discover_run_dirs():
    """Every fewshot run dir that has both summary_ex_acc.json and a reasoning/ dir."""
    dirs = []
    if not FEWSHOT_UNIFIED.exists():
        return dirs
    for d in sorted(FEWSHOT_UNIFIED.iterdir()):
        if not d.is_dir():
            continue
        if (d / "summary_ex_acc.json").exists() and (d / "reasoning").exists():
            dirs.append(d)
    return dirs


def brand_of(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    return "unknown"


def load_pricing():
    """model_name -> output-token $/1M, from model_pricing.csv."""
    rows = {}
    with open(PRICING_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model_name"]] = float(r["cost_per_million_output_tokens_usd"])
    return rows


def derive_size_tiers(models, pricing):
    """Rank by output price WITHIN brand (cross-brand prices aren't comparable) ->
    small/mid/large. Returns (tier_of: model->str, log_output_price_of: model->float)."""
    by_brand = defaultdict(list)
    for m in models:
        by_brand[brand_of(m)].append(m)

    tier_of = {}
    log_price_of = {}
    for _brand, ms in by_brand.items():
        ranked = sorted(ms, key=lambda m: pricing.get(m, float("inf")))
        n = len(ranked)
        if n == 3:
            labels = ["small", "mid", "large"]
        elif n == 2:
            labels = ["small", "large"]
        elif n == 1:
            labels = ["only"]
        else:
            labels = [f"tier{i}_of_{n}" for i in range(n)]
        for i, m in enumerate(ranked):
            tier_of[m] = labels[i]
            price = pricing.get(m)
            log_price_of[m] = math.log(price) if price and price > 0 else float("nan")
    return tier_of, log_price_of


def _find_raw_generation_log(run_name, instance_id):
    path = FEWSHOT_RAW / run_name / instance_id / "generation.log"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


_GOLD_SQL_CACHE = {}


def find_gold_sql(instance_id):
    """Gold SQL is identical across every run of a given dataset -- grab it from
    whichever discovered run directory has it. Used as the "final answer" analog for
    human rationales (Phase 4), since a human trace has no predicted_0.sql of its own;
    per the spec, human rationales are likely after-the-fact explanations of the
    correct (gold) query, not a think-aloud toward the human's own answer."""
    if instance_id in _GOLD_SQL_CACHE:
        return _GOLD_SQL_CACHE[instance_id]
    for d in discover_run_dirs():
        path = d / "gold" / f"{instance_id}.sql"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            _GOLD_SQL_CACHE[instance_id] = text or None
            return _GOLD_SQL_CACHE[instance_id]
    _GOLD_SQL_CACHE[instance_id] = None
    return None


def find_predicted_sql(run_name, instance_id):
    """The final answer text for E's final-sentence-similarity-to-final-answer metric."""
    path = FEWSHOT_RAW / run_name / instance_id / "predicted_0.sql"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _extract_prompt_from_log(log_text):
    """generation.log is written as 'Prompt:\\n<prompt>\\n\\nResponse:\\n<response>' by
    execute.py -- pull just the prompt half for grounding-rate computation."""
    if log_text is None:
        return None
    marker = "Prompt:\n"
    resp_marker = "\n\nResponse:\n"
    if marker not in log_text:
        return None
    start = log_text.index(marker) + len(marker)
    end = log_text.find(resp_marker, start)
    return log_text[start:end] if end != -1 else log_text[start:]


def _model_name_for_run(run_dir):
    reasoning_files = list((run_dir / "reasoning").glob("*.json"))
    if reasoning_files:
        return json.loads(reasoning_files[0].read_text(encoding="utf-8")).get("model")
    summary = json.load(open(run_dir / "summary_ex_acc.json"))
    if summary["details"]:
        return summary["details"][0].get("usage", {}).get("model")
    return None


def load_model_instances(sample=None):
    """Returns (rows, models). One row per (model, instance_id) for every attempted
    instance across all in-scope runs. `sample` caps rows PER MODEL (dev runs)."""
    run_dirs = discover_run_dirs()
    pricing = load_pricing()

    run_model = {d: _model_name_for_run(d) for d in run_dirs}
    in_scope_dirs = [d for d in run_dirs if run_model[d] not in EXCLUDED_MODELS and run_model[d] is not None]
    models = sorted({run_model[d] for d in in_scope_dirs})
    tier_of, log_price_of = derive_size_tiers(models, pricing)

    rows = []
    for d in in_scope_dirs:
        model = run_model[d]
        run_name = d.name
        summary = json.load(open(d / "summary_ex_acc.json"))
        reasoning_dir = d / "reasoning"

        details = summary["details"]
        if sample:
            details = details[:sample]

        for det in details:
            iid = det["file"].replace(".sql", "")
            trace_path = reasoning_dir / f"{iid}.json"
            raw_text = None
            if trace_path.exists():
                tdata = json.loads(trace_path.read_text(encoding="utf-8"))
                gen_texts = [
                    e["text"] for e in tdata.get("traces", [])
                    if e.get("stage") == "generation" and e.get("text")
                ]
                raw_text = "\n\n".join(gen_texts) if gen_texts else None
            has_trace = raw_text is not None

            usage_record = det.get("usage") or {}
            calls = usage_record.get("calls") or [{}]
            call = calls[0]

            prompt_text = _extract_prompt_from_log(_find_raw_generation_log(run_name, iid))

            rows.append({
                "instance_id": iid,
                "model": model,
                "brand": brand_of(model),
                "size_tier": tier_of.get(model),
                "log_output_price": log_price_of.get(model),
                "run_name": run_name,
                "dataset": usage_record.get("dataset"),
                "setting": usage_record.get("setting"),
                "correct": bool(det.get("match")),
                "score": det.get("score"),
                "gold_empty": det.get("gold_empty"),
                "gold_error": det.get("gold_error"),
                "has_reasoning_trace": has_trace,
                "raw_text": raw_text,
                "prompt_text": prompt_text,
                "reasoning_effort": usage_record.get("reasoning_effort"),
                "thinking_budget_tokens": usage_record.get("thinking_budget_tokens"),
                "thinking_effort": usage_record.get("thinking_effort"),
                "prompt_tokens": call.get("prompt_tokens"),
                "completion_tokens": call.get("completion_tokens"),
                "reasoning_tokens": call.get("reasoning_tokens"),
                "latency_sec": call.get("latency_sec"),
                "stop_reason": call.get("stop_reason"),
                "cost_usd": usage_record.get("cost_usd"),
                "errors": usage_record.get("errors"),
            })
    return rows, models


def load_human_instances():
    """example_prompts/manual_reasoning.json/*.json -> rows compatible with
    load_model_instances()'s schema (model="human", brand="human"), for Phase 4."""
    rows = []
    for f in sorted(HUMAN_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        iid = data.get("instance_id")
        if iid != f.stem:
            raise ValueError(
                f"Human example {f} has instance_id={iid!r} but filename implies {f.stem!r} "
                "-- re-run Phase 0's check, this was previously found and fixed once already."
            )
        gen_texts = [
            e["text"] for e in data.get("traces", [])
            if e.get("stage") == "generation" and e.get("text")
        ]
        raw_text = "\n\n".join(gen_texts) if gen_texts else None

        prompt_path = REPO_ROOT / "example_prompts" / f"{iid}.md"
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace") if prompt_path.exists() else None

        rows.append({
            "instance_id": iid,
            "model": "human",
            "brand": "human",
            "size_tier": None,
            "raw_text": raw_text,
            "prompt_text": prompt_text,
            "gold_sql": find_gold_sql(iid),
            "has_reasoning_trace": raw_text is not None,
        })
    return rows


def shared_instance_ids(rows, models=None):
    """Instance IDs with a reasoning trace present across ALL given models (the
    fully-overlapping core used for paired cross-model analyses, per Phase 0
    decision: restrict cross-model comparisons to this set for now)."""
    by_model = defaultdict(set)
    for r in rows:
        if r["has_reasoning_trace"]:
            by_model[r["model"]].add(r["instance_id"])
    model_list = models if models is not None else list(by_model.keys())
    sets = [by_model[m] for m in model_list if m in by_model]
    if not sets:
        return set()
    result = set(sets[0])
    for s in sets[1:]:
        result &= s
    return result


def trace_coverage_table(rows):
    """Per-model: attempted, has_trace count/rate, split by correctness -- the
    diagnostic the user asked to keep front-and-center (Phase 0 finding: missingness
    is correctness-correlated for gpt-5.6-luna/terra, not missing-completely-at-random)."""
    by_model = defaultdict(lambda: {"attempted": 0, "has_trace": 0,
                                     "correct_attempted": 0, "correct_has_trace": 0,
                                     "incorrect_attempted": 0, "incorrect_has_trace": 0})
    for r in rows:
        b = by_model[r["model"]]
        b["attempted"] += 1
        b["has_trace"] += int(r["has_reasoning_trace"])
        if r["correct"]:
            b["correct_attempted"] += 1
            b["correct_has_trace"] += int(r["has_reasoning_trace"])
        else:
            b["incorrect_attempted"] += 1
            b["incorrect_has_trace"] += int(r["has_reasoning_trace"])

    table = []
    for model, b in sorted(by_model.items()):
        table.append({
            "model": model,
            "attempted": b["attempted"],
            "has_trace": b["has_trace"],
            "trace_coverage_rate": b["has_trace"] / b["attempted"] if b["attempted"] else None,
            "trace_rate_given_correct": (b["correct_has_trace"] / b["correct_attempted"]) if b["correct_attempted"] else None,
            "trace_rate_given_incorrect": (b["incorrect_has_trace"] / b["incorrect_attempted"]) if b["incorrect_attempted"] else None,
        })
    return table
