"""
User-maintained registry mapping a model name (whatever you pass via --model) to a
relative size tier and provider. There is no reliable API to get parameter counts for
closed models, so `size_tier` is just your own judgment call about relative size within
a model family (e.g. "mini" variants are smaller than the base model; "haiku" is smaller
than "sonnet" which is smaller than "opus"). Add an entry here whenever you benchmark a
new model name -- unknown models fall back to size_tier "unknown" rather than erroring.

size_tier is free text, but for it to sort sensibly in comparisons, stick to a small
fixed vocabulary: "small", "medium", "large", "unknown".

param_count_b (billions of parameters) is only set for open-weight models, where the
count is actually published. It's the one place a real size-vs-accuracy scaling claim
can be made -- for closed models, size_tier stays a label, never a measured quantity.
"""

MODEL_REGISTRY = {
    "qwen2.5-coder-14b-instruct": {"provider": "lmstudio", "size_tier": "small", "param_count_b": 14},

    "gpt-5.4-mini": {"provider": "openai", "size_tier": "small"},  # superseded by gpt-5.6-luna for new small-tier runs; kept so old data/CSVs still resolve
    "gpt-5.6-luna": {"provider": "openai", "size_tier": "small"},
    "gpt-5.5": {"provider": "openai", "size_tier": "large"},

    "claude-haiku-4-5": {"provider": "anthropic", "size_tier": "small"},
    "claude-sonnet-4-5": {"provider": "anthropic", "size_tier": "medium"},
    "claude-opus-4-8": {"provider": "anthropic", "size_tier": "large"},

    # OpenRouter-hosted open-weight anchor for the matched-tier scaling study --
    # confirm exact slugs against OpenRouter's current catalog before running.
    "qwen/qwen2.5-coder-7b-instruct": {"provider": "open-weight", "size_tier": "small", "param_count_b": 7},
    "qwen/qwen2.5-coder-32b-instruct": {"provider": "open-weight", "size_tier": "large", "param_count_b": 32},
}

_DEFAULT_ENTRY = {"provider": "unknown", "size_tier": "unknown"}


def get_model_info(model):
    return MODEL_REGISTRY.get(model, _DEFAULT_ENTRY)
