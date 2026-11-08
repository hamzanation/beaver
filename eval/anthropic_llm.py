import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# The adaptive thinking scheme has no explicit numeric budget to size max_tokens
# against (unlike thinking_budget_tokens), so these are a conservative floor per
# effort level instead -- picked to avoid truncation, not a documented API minimum.
# If a run still shows stop_reason="max_tokens" at a given effort level, raise the
# corresponding floor here (or set BEAVER_MAX_TOKENS above it directly).
_ADAPTIVE_THINKING_MIN_MAX_TOKENS = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}
_ADAPTIVE_THINKING_DEFAULT_MIN_MAX_TOKENS = 8192

MODEL_ALIASES = {
    "claude": DEFAULT_CLAUDE_MODEL,
    "anthropic/claude": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet-4.5": DEFAULT_CLAUDE_MODEL,
    "anthropic/claude-sonnet-4.5": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet-4-5": DEFAULT_CLAUDE_MODEL,
    "anthropic/claude-sonnet-4-5": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-4.8": "claude-opus-4-8",
    "anthropic/claude-opus-4.8": "claude-opus-4-8",
    "claude-opus-4-8": "claude-opus-4-8",
    "anthropic/claude-opus-4-8": "claude-opus-4-8",
}


def is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


def anthropic_model_name(model: str) -> str:
    override = os.environ.get("ANTHROPIC_MODEL")
    if override:
        return override
    return MODEL_ALIASES.get(model.lower(), model.removeprefix("anthropic/"))


def _anthropic_url() -> str:
    return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")


def _split_messages(messages):
    system_parts = []
    anthropic_messages = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            anthropic_messages.append({"role": role, "content": content})
    return "\n\n".join(system_parts) or None, anthropic_messages


def anthropic_chat_once(model, messages, temperature=0, max_tokens=None, thinking_budget_tokens=None, thinking_effort=None):
    """
    thinking_budget_tokens and thinking_effort control extended thinking under two
    different, mutually exclusive API schemes depending on model generation:
      - Older models (claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-8): explicit
        thinking.type=enabled + a numeric budget_tokens.
      - Newer models (e.g. claude-sonnet-5): thinking.type=adaptive + a qualitative
        output_config.effort level, mirroring OpenAI's reasoning_effort. Confirmed by
        a live 400 from claude-sonnet-5 rejecting the older scheme outright and naming
        this one -- pass thinking_effort (e.g. "low"/"medium"/"high") for these models
        instead of thinking_budget_tokens.
    If both are passed, thinking_effort (the adaptive scheme) wins.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Claude models")

    system, anthropic_messages = _split_messages(messages)
    requested_max_tokens = int(max_tokens or os.environ.get("BEAVER_MAX_TOKENS", "1024"))
    thinking_budget_tokens = int(thinking_budget_tokens) if thinking_budget_tokens else None
    thinking_enabled = bool(thinking_effort or thinking_budget_tokens)

    payload = {
        "model": anthropic_model_name(model),
        "messages": anthropic_messages,
    }
    if system:
        payload["system"] = system

    if thinking_effort:
        # display defaults to "omitted" (thinking blocks returned with an empty
        # "thinking" field, only the signature populated) on claude-sonnet-5,
        # claude-opus-5, claude-opus-4-8, and others -- must opt in to "summarized"
        # explicitly to get readable text back at all. Confirmed via Anthropic's docs
        # after seeing exactly this empty-thinking/populated-signature shape live.
        payload["thinking"] = {"type": "adaptive", "display": "summarized"}
        payload["output_config"] = {"effort": thinking_effort}
        min_max_tokens = _ADAPTIVE_THINKING_MIN_MAX_TOKENS.get(thinking_effort, _ADAPTIVE_THINKING_DEFAULT_MIN_MAX_TOKENS)
        if requested_max_tokens < min_max_tokens:
            # Without this, adaptive thinking can burn the entire (small) default
            # max_tokens before producing a final answer -- or even a visible thinking
            # block -- leaving both the SQL and the reasoning trace empty.
            requested_max_tokens = min_max_tokens
    elif thinking_budget_tokens:
        if requested_max_tokens <= thinking_budget_tokens:
            # The API requires max_tokens > thinking.budget_tokens (max_tokens covers
            # thinking + the final answer combined). Add the originally-requested amount
            # on top of the budget as headroom for the actual answer, instead of sending
            # an invalid request.
            requested_max_tokens = thinking_budget_tokens + requested_max_tokens
        # display also defaults to "omitted" on claude-opus-4-8 even though it uses this
        # scheme (not just the adaptive one) -- set it explicitly here too rather than
        # relying on a default that varies by model.
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens, "display": "summarized"}

    payload["max_tokens"] = requested_max_tokens

    if not thinking_enabled and temperature is not None:
        # Extended thinking (either scheme) requires temperature=1 (or unset) -- never
        # send an explicit temperature alongside it, since the API rejects any other value.
        payload["temperature"] = temperature

    data = json.dumps(payload).encode("utf-8")
    request = Request(
        _anthropic_url(),
        data=data,
        headers={
            "content-type": "application/json",
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            "x-api-key": api_key,
        },
        method="POST",
    )

    retries = int(os.environ.get("ANTHROPIC_RETRIES", "3"))
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=int(os.environ.get("ANTHROPIC_TIMEOUT", "120"))) as response:
                body = json.loads(response.read().decode("utf-8"))
            text_blocks = []
            thinking_blocks = []
            block_types_seen = []
            non_text_blocks_raw = []
            for block in body.get("content", []):
                block_type = block.get("type")
                block_types_seen.append(block_type)
                if block_type == "text":
                    text_blocks.append(block.get("text", ""))
                else:
                    non_text_blocks_raw.append(block)
                    # Any non-text block is treated as reasoning content, regardless of
                    # its type name -- check every plausible field rather than assuming
                    # one shape, since this has already been wrong twice for this model.
                    candidate = block.get("thinking") or block.get("summary") or block.get("text")
                    if isinstance(candidate, list):
                        candidate = "\n".join(
                            (c.get("text") if isinstance(c, dict) else str(c)) for c in candidate if c
                        )
                    if candidate:
                        thinking_blocks.append(str(candidate))
            usage = body.get("usage", {})
            thinking_text = "\n\n".join(thinking_blocks) or None
            result = {
                "content": "".join(text_blocks),
                "thinking_text": thinking_text,
                "block_types_seen": block_types_seen,
                "stop_reason": body.get("stop_reason"),
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
                },
            }
            if not thinking_text and non_text_blocks_raw:
                # We saw non-text block(s) but couldn't extract text from any known
                # field name -- surface the raw block(s) so the actual shape can be
                # read directly instead of guessing field names again.
                result["raw_non_text_blocks_debug"] = non_text_blocks_raw
            return result
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(f"Anthropic API error {e.code}: {detail}") from e
            time.sleep(2 ** attempt)
        except URLError as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Anthropic API connection error: {e}") from e
            time.sleep(2 ** attempt)


def anthropic_chat(model, messages, temperature=0, n=1, max_tokens=None, thinking_budget_tokens=None, thinking_effort=None):
    responses = []
    thinking_texts = []
    stop_reasons = []
    block_types_seen_all = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw_debug_all = []
    for _ in range(n):
        result = anthropic_chat_once(
            model, messages, temperature=temperature, max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens, thinking_effort=thinking_effort,
        )
        responses.append(result["content"])
        if result.get("thinking_text"):
            thinking_texts.append(result["thinking_text"])
        stop_reasons.append(result.get("stop_reason"))
        block_types_seen_all.append(result.get("block_types_seen"))
        if result.get("raw_non_text_blocks_debug"):
            raw_debug_all.append(result["raw_non_text_blocks_debug"])
        for key, value in result["usage"].items():
            usage[key] = usage.get(key, 0) + (value or 0)
    out = {"response": responses, **usage}
    if thinking_texts:
        out["thinking_text"] = "\n\n---\n\n".join(thinking_texts)
    out["stop_reason"] = stop_reasons[0] if len(stop_reasons) == 1 else stop_reasons
    out["block_types_seen"] = block_types_seen_all[0] if len(block_types_seen_all) == 1 else block_types_seen_all
    if raw_debug_all:
        out["raw_non_text_blocks_debug"] = raw_debug_all[0] if len(raw_debug_all) == 1 else raw_debug_all
    return out
