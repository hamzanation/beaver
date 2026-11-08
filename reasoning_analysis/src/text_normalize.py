"""
Text normalization for reasoning traces.

Ordering constraint (per spec, and Phase 0 finding that OpenAI summaries are
~100% bold-header-formatted vs 0-9% for Anthropic): is_summary / header counts /
truncated must be computed on the RAW text, BEFORE normalize_text() strips
headers/progress markers. lexical_features.py enforces this order -- don't call
normalize_text() before detect_pre_normalization_flags() on the same text.
"""
import re
import unicodedata

_TYPOGRAPHIC_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "′": "'",
    "“": '"', "”": '"', "″": '"',
})

BOLD_HEADER_RE =re.compile(r"^[ \t]*\*\*[^*\n]{3,80}\*\*[ \t]*$", re.MULTILINE)
ATX_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+.*$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
BULLET_RE = re.compile(r"^[ \t]*[-*•][ \t]+", re.MULTILINE)

PROGRESS_MARKER_RE = re.compile(
    r"^[ \t]*(writing the query|writing the sql|let me write|i'?ll write|"
    r"generating (the )?sql|now writing|working on the (query|sql))[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def detect_pre_normalization_flags(text: str, stop_reason: str = None) -> dict:
    """Structural flags that must be measured before any normalization touches the text."""
    if not text:
        return {
            "is_summary": False, "n_bold_headers": 0, "n_atx_headers": 0,
            "n_code_blocks": 0, "n_bullets": 0, "truncated": False,
        }
    n_bold = len(BOLD_HEADER_RE.findall(text))
    n_atx = len(ATX_HEADER_RE.findall(text))
    n_code = len(CODE_FENCE_RE.findall(text))
    n_bullets = len(BULLET_RE.findall(text))
    return {
        # Bold-header blocks are the observed signature of OpenAI's Responses-API
        # reasoning *summary* (Phase 0: 100% of gpt-5.6-{luna,terra,sol} traces vs.
        # 0% opus-5/sonnet-5, 9% haiku-4-5) -- used as the is_summary proxy per spec.
        "is_summary": n_bold > 0,
        "n_bold_headers": n_bold,
        "n_atx_headers": n_atx,
        "n_code_blocks": n_code,
        "n_bullets": n_bullets,
        # stop_reason comes from usage.json, not the text itself -- pass it in rather
        # than guessing truncation from content.
        "truncated": stop_reason in {"max_tokens", "length"},
    }


def normalize_text(text: str) -> str:
    """Unicode-normalize, strip markdown headers/code fences/progress markers.
    Call detect_pre_normalization_flags() on the RAW text first if you need those flags."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # NFKC leaves typographic quotes alone. OpenAI's summaries use curly apostrophes
    # (I’ll) for ~40% of their apostrophes while Anthropic's traces never do, so without
    # this the same contraction tokenizes differently by brand (i + ll vs. i'll) and
    # lexicon phrases containing an apostrophe miss the curly form.
    text = text.translate(_TYPOGRAPHIC_QUOTES)
    text = CODE_FENCE_RE.sub(" ", text)
    text = BOLD_HEADER_RE.sub(" ", text)
    text = ATX_HEADER_RE.sub(" ", text)
    text = PROGRESS_MARKER_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SEGMENTER = None


def split_sentences(text: str):
    """pysbd sentence splitter, lazily initialized (avoids import cost when unused)."""
    global _SEGMENTER
    if not text:
        return []
    if _SEGMENTER is None:
        import pysbd
        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in _SEGMENTER.segment(text) if s.strip()]
