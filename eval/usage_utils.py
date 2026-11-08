import json
import os
from pathlib import Path


def new_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Numeric fields beyond the base three that get summed across calls when present.
# Both are exact API-reported counts (OpenAI reasoning_tokens, Anthropic cache tokens) --
# never fabricated/estimated.
_OPTIONAL_NUMERIC_USAGE_FIELDS = ("reasoning_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

# Per-call metadata fields that describe one call rather than summing across calls.
# block_types_seen / raw_non_text_blocks_debug are temporary diagnostics for the new
# Anthropic adaptive-thinking scheme, whose exact response block shape isn't documented
# yet -- remove both once the real field name is confirmed and extraction is fixed.
_OPTIONAL_CALL_FIELDS = ("stop_reason", "block_types_seen", "raw_non_text_blocks_debug")


def add_usage(acc, usage):
    """In-place accumulate `usage` (any dict-like with token keys) into `acc`."""
    if not usage:
        return acc
    acc["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
    acc["completion_tokens"] += usage.get("completion_tokens", 0) or 0
    acc["total_tokens"] += usage.get("total_tokens", 0) or 0
    for key in _OPTIONAL_NUMERIC_USAGE_FIELDS:
        if key in usage:
            acc[key] = acc.get(key, 0) + (usage.get(key, 0) or 0)
    return acc


def estimate_cost(model, usage, pricing_table):
    """Returns estimated USD cost for `usage` under `model`'s $/1M token pricing, or None if unpriced."""
    prices = pricing_table.get(model)
    if not prices:
        return None
    input_price = prices.get("input_per_1m")
    output_price = prices.get("output_per_1m")
    if input_price is None or output_price is None:
        return None
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def new_usage_record(instance_id, model, dataset=None, setting=None, reasoning_effort=None, thinking_budget_tokens=None, thinking_effort=None):
    """
    reasoning_effort / thinking_budget_tokens / thinking_effort default to the
    BEAVER_REASONING_EFFORT / BEAVER_THINKING_BUDGET_TOKENS / BEAVER_THINKING_EFFORT env
    vars when not passed explicitly, so every existing call site (dailsql/dinsql/
    fewshot/reforce) records the condition a run was executed under without needing its
    own plumbing. thinking_budget_tokens and thinking_effort are two different Anthropic
    extended-thinking control schemes used by different model generations (see
    anthropic_llm.anthropic_chat_once) -- a given run should only ever set one of them.
    """
    if reasoning_effort is None:
        reasoning_effort = os.environ.get("BEAVER_REASONING_EFFORT")
    if thinking_budget_tokens is None:
        env_budget = os.environ.get("BEAVER_THINKING_BUDGET_TOKENS")
        thinking_budget_tokens = int(env_budget) if env_budget else None
    if thinking_effort is None:
        thinking_effort = os.environ.get("BEAVER_THINKING_EFFORT")

    return {
        "instance_id": instance_id,
        "model": model,
        "dataset": dataset,
        "setting": setting,
        "reasoning_effort": reasoning_effort,
        "thinking_budget_tokens": thinking_budget_tokens,
        "thinking_effort": thinking_effort,
        "calls": [],
        "usage": new_usage(),
        "latency_sec": 0.0,
        "cost_usd": None,
        "num_calls": 0,
        "errors": 0,
    }


def record_call(record, stage, usage, latency_sec, error=False):
    """Append one LLM call's telemetry into a usage record built by new_usage_record()."""
    usage = usage or {}
    call = {
        "stage": stage,
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "latency_sec": latency_sec,
    }
    for key in _OPTIONAL_NUMERIC_USAGE_FIELDS + _OPTIONAL_CALL_FIELDS:
        if key in usage:
            call[key] = usage[key]
    record["calls"].append(call)
    add_usage(record["usage"], usage)
    record["latency_sec"] += latency_sec or 0.0
    record["num_calls"] += 1
    if error:
        record["errors"] += 1
    return record


def finalize_cost(record, pricing_table):
    record["cost_usd"] = estimate_cost(record["model"], record["usage"], pricing_table)
    return record


def write_usage_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_usage_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def new_reasoning_trace(instance_id, model, dataset=None, setting=None):
    """Holds the raw reasoning/thinking text for one instance, kept separate from
    usage.json so per-example token accounting stays small and the qualitative text
    (for failure-taxonomy coding) is easy to mine independently."""
    return {"instance_id": instance_id, "model": model, "dataset": dataset, "setting": setting, "traces": []}


def record_reasoning_trace(trace_record, stage, text):
    if text:
        trace_record["traces"].append({"stage": stage, "text": text})
    return trace_record


def write_reasoning_trace_json(path, payload):
    """No-op when nothing was captured, so runs without reasoning enabled don't litter
    the output tree with empty reasoning_trace.json files."""
    if not payload.get("traces"):
        return
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
