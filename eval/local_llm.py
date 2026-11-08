import os

from openai import OpenAI


LOCAL_MODEL_NAMES = {
    "qwen2.5-coder-14b-instruct",
}


def is_lmstudio_model(model: str) -> bool:
    provider = os.environ.get("BEAVER_LLM_PROVIDER", "").lower()
    if provider in {"lmstudio", "lm-studio", "local"}:
        return True
    return (
        model in LOCAL_MODEL_NAMES
        or model.startswith("lmstudio/")
        or model.startswith("lmstudio:")
        or model.startswith("local/")
    )


def lmstudio_model_name(model: str) -> str:
    override = os.environ.get("LMSTUDIO_MODEL")
    if override:
        return override
    for prefix in ("lmstudio/", "lmstudio:", "local/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def lmstudio_base_url() -> str:
    base_url = (
        os.environ.get("LMSTUDIO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://localhost:1234/v1"
    ).rstrip("/")

    if base_url.endswith("/api/v1/chat"):
        return base_url[: -len("/api/v1/chat")] + "/v1"
    if base_url.endswith("/chat/completions"):
        return base_url[: -len("/chat/completions")]
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def lmstudio_client() -> OpenAI:
    return OpenAI(
        base_url=lmstudio_base_url(),
        api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
    )


def response_usage(response):
    usage = getattr(response, "usage", None)
    out = {
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    # gpt-5/o-series reasoning models report reasoning tokens as a sub-count of
    # completion_tokens via completion_tokens_details -- exact, never estimated.
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    if reasoning_tokens is not None:
        out["reasoning_tokens"] = reasoning_tokens

    choices = getattr(response, "choices", None)
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason is not None:
            out["stop_reason"] = finish_reason
    return out


def response_usage_responses_api(response):
    """Same as response_usage(), but for the Responses API's usage object, which uses
    input_tokens/output_tokens/output_tokens_details instead of Chat Completions'
    prompt_tokens/completion_tokens/completion_tokens_details."""
    usage = getattr(response, "usage", None)
    out = {
        "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    output_details = getattr(usage, "output_tokens_details", None) if usage else None
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None) if output_details else None
    if reasoning_tokens is not None:
        out["reasoning_tokens"] = reasoning_tokens

    status = getattr(response, "status", None)
    if status is not None:
        out["stop_reason"] = status
    return out


def extract_reasoning_summary_text(response):
    """Pulls the reasoning *summary* text out of a Responses API response, when the
    call requested one (reasoning={"summary": "auto"}). This is a model-generated
    abstraction of the reasoning, not the raw chain-of-thought -- OpenAI does not
    expose the latter for any current reasoning model, unlike Anthropic's extended
    thinking, which does return the real thinking text. Field names here follow the
    documented Responses API shape; verify against the actual SDK version in use if
    this ever comes back empty for a run where reasoning was clearly requested.
    """
    texts = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for part in getattr(item, "summary", None) or []:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return "\n\n".join(texts) or None
