"""
Phase 1: per-trace lexical features, normalized per 100 words unless noted.

Feature computation order matters: structural flags (is_summary, header/bullet/code
counts, truncated) are measured on the RAW text via text_normalize.detect_pre_normalization_flags(),
THEN the text is normalized (headers/progress-markers stripped) before word/sentence
counting and lexicon matching. See text_normalize.py's module docstring.
"""
import re
from pathlib import Path

import yaml

from . import text_normalize as tn

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lexicons.yaml"


def load_lexicons(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compile_phrase_matcher(phrases):
    """One alternation regex with word boundaries per lexicon category. Phrases with
    non-word edge characters (e.g. "?") won't get a usable \\b match -- those categories
    are handled specially in compute_features() instead of through this path."""
    escaped = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "this", "that", "it", "as", "with", "at", "by", "from", "we", "you",
    "which", "each", "will", "should", "can", "not", "no", "if", "then", "so", "do",
    "does", "select", "from", "where", "join", "table", "column", "sql", "query",
}


def _prompt_vocabulary(prompt_text: str) -> set:
    """Significant terms from the prompt (schema identifiers, table/column names,
    question-specific words) used as the grounding-rate reference set."""
    if not prompt_text:
        return set()
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", prompt_text.lower())
    return {w for w in words if w not in _STOPWORDS}


class LexiconMatchers:
    def __init__(self, lexicons: dict):
        self.categories = {}
        for name, phrases in lexicons.items():
            if name in {"progress_markers", "question"}:
                continue  # progress_markers is normalization-only; question handled specially
            if phrases:
                self.categories[name] = _compile_phrase_matcher(phrases)

    def counts(self, text: str) -> dict:
        return {name: len(rx.findall(text)) for name, rx in self.categories.items()}


def compute_features(raw_text: str, prompt_text: str = None, stop_reason: str = None,
                      matchers: LexiconMatchers = None) -> dict:
    """One dict of features for one trace's generation-stage text. `raw_text` should be
    the UNMODIFIED trace text (pre-normalization) -- this function handles the
    flag-then-normalize ordering internally."""
    if matchers is None:
        matchers = LexiconMatchers(load_lexicons())

    flags = tn.detect_pre_normalization_flags(raw_text, stop_reason=stop_reason)

    normalized = tn.normalize_text(raw_text)
    words = normalized.split()
    n_words = len(words)
    sentences = tn.split_sentences(normalized)
    n_sentences = len(sentences)

    per100 = (lambda c: (c / n_words * 100.0) if n_words else None)

    lex_counts = matchers.counts(normalized) if n_words else {k: 0 for k in matchers.categories}

    n_questions = raw_text.count("?") if raw_text else 0

    grounding_rate = None
    prompt_vocab = _prompt_vocabulary(prompt_text) if prompt_text else set()
    if n_words and prompt_vocab:
        lowered_words = [w.strip(".,;:()[]{}\"'").lower() for w in words]
        n_grounded = sum(1 for w in lowered_words if w in prompt_vocab)
        grounding_rate = n_grounded / n_words * 100.0

    features = {
        "n_words": n_words,
        "n_sentences": n_sentences,
        "n_bold_headers": flags["n_bold_headers"],
        "n_atx_headers": flags["n_atx_headers"],
        "n_code_blocks": flags["n_code_blocks"],
        "n_bullets": flags["n_bullets"],
        "is_summary": flags["is_summary"],
        "truncated": flags["truncated"],
        "bullets_per_100w": per100(flags["n_bullets"]),
        "code_blocks_per_100w": per100(flags["n_code_blocks"]),
        "question_rate_per_100w": per100(n_questions),
        "grounding_rate_per_100w": grounding_rate,
    }
    for name, count in lex_counts.items():
        features[f"{name}_per_100w"] = per100(count)

    return features
