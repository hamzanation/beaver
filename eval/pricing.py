"""
User-maintained $/1M-token pricing table, keyed by the same model name used in
--model / eval/model_registry.py. Confirmed prices for the four currently-analyzed
models are loaded from model_pricing.csv at import time, so that CSV stays the single
source of truth and can't silently drift out of sync with this file. Models without a
CSV row (new/unconfirmed additions) fall back to the None/None placeholders below.

Any model missing from this table (or missing either price) makes
eval.utils.usage.estimate_cost() return None for that model -- cost then shows up as
null/excluded in per-example and aggregate telemetry instead of silently being 0.
Local LM Studio models have no API cost, so they're seeded at $0/$0 rather than left
out (leaving them out would report an unhelpful "unpriced" instead of the true "free").
"""

import csv
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent.parent / "model_pricing.csv"


def _load_from_csv():
    prices = {}
    if not _CSV_PATH.exists():
        return prices
    with open(_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            prices[row["model_name"]] = {
                "input_per_1m": float(row["cost_per_million_input_tokens_usd"]),
                "output_per_1m": float(row["cost_per_million_output_tokens_usd"]),
            }
    return prices


PRICING = {
    "qwen2.5-coder-14b-instruct": {"input_per_1m": 0.0, "output_per_1m": 0.0},

    # TODO: confirm exact OpenRouter pricing (and exact model slug) before trusting these for cost analysis.
    "claude-opus-4-8": {"input_per_1m": None, "output_per_1m": None},
    # TODO: fill in real $/1M pricing for gpt-5.6-luna (the new OpenAI small-tier model) -- unpriced until then.
    "gpt-5.6-luna": {"input_per_1m": None, "output_per_1m": None},
    "qwen/qwen2.5-coder-7b-instruct": {"input_per_1m": None, "output_per_1m": None},
    "qwen/qwen2.5-coder-32b-instruct": {"input_per_1m": None, "output_per_1m": None},
}

# Confirmed rows from model_pricing.csv win over the placeholders above.
PRICING.update(_load_from_csv())
