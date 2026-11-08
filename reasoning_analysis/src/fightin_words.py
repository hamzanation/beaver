"""
Fightin' Words (Monroe, Colaresi & Quinn 2008): log-odds-ratio with an informative
Dirichlet prior, for distinctive-vocabulary comparisons (brand vs. brand, size tier
vs. size tier).

No external background corpus is assumed -- the prior is drawn from the union of the
two corpora being compared (the standard default when a bespoke background isn't
available), scaled by `alpha0` (total prior pseudo-count mass).
"""
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def fightin_words(texts_a, texts_b, alpha0: float = 500.0, min_count: int = 3):
    """texts_a/texts_b: iterables of normalized trace text (one corpus per side).
    Returns a list of dicts sorted by |z| descending: word, count_a, count_b, z, log_odds.
    Positive z favors corpus A, negative favors corpus B."""
    counts_a = Counter()
    counts_b = Counter()
    for t in texts_a:
        counts_a.update(tokenize(t))
    for t in texts_b:
        counts_b.update(tokenize(t))

    vocab = set(counts_a) | set(counts_b)
    background = counts_a + counts_b
    total_background = sum(background.values()) or 1

    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())

    results = []
    for w in vocab:
        if background[w] < min_count:
            continue
        y_a = counts_a.get(w, 0)
        y_b = counts_b.get(w, 0)
        alpha_w = alpha0 * (background[w] / total_background)

        log_odds_a = math.log((y_a + alpha_w) / (n_a + alpha0 - y_a - alpha_w))
        log_odds_b = math.log((y_b + alpha_w) / (n_b + alpha0 - y_b - alpha_w))
        delta = log_odds_a - log_odds_b
        variance = 1.0 / (y_a + alpha_w) + 1.0 / (y_b + alpha_w)
        z = delta / math.sqrt(variance)

        results.append({
            "word": w, "count_a": y_a, "count_b": y_b,
            "log_odds": delta, "z": z,
        })

    results.sort(key=lambda r: abs(r["z"]), reverse=True)
    return results
