"""
Phase 2F: reasoning-move taxonomy, bootstrapped without any LLM prompting/labeling.

Pipeline:
  1. High-precision regex cues (config/move_cues.yaml) -- a sentence gets a regex seed
     label only if exactly one move's cues fire (ambiguous/no hits -> None).
  2. Zero-shot NLI (MoritzLaurer/deberta-v3-large-zeroshot-v2.0) as a small entailment
     classifier -- NOT a generative LLM asked to label things -- fills in a seed label
     for sentences the regex missed, only when confident (top score above a threshold,
     with a clear margin over the runner-up).
  3. Train a multinomial LogisticRegression on sentence EMBEDDINGS (bge-large, reusing
     the Phase 2A-E disk cache) using ONLY the seed-labeled sentences.
  4. Use the trained classifier to label every sentence (seeded or not).
  5. Export a stratified sample for hand-checking (outputs/validation/moves_to_check.csv).
  6. Move distributions, per-model Markov transition matrices, and Jensen-Shannon
     divergence between models' move distributions.
"""
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression

from . import embeddings as emb

NLI_CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "embeddings_cache" / "nli_seed_cache.json"

MOVES = ["restate", "interpret_ambiguity", "plan", "hypothesize", "verify",
         "self_correct", "meta_format", "execute"]

CUES_PATH = Path(__file__).resolve().parents[1] / "config" / "move_cues.yaml"

NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
NLI_CONFIDENCE_THRESHOLD = 0.55
NLI_MARGIN_THRESHOLD = 0.15


def load_cues(path=CUES_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compile_cue_matchers(cues_config):
    return {move: re.compile("|".join(f"(?:{p})" for p in spec["cues"]), re.IGNORECASE)
            for move, spec in cues_config.items()}


def regex_seed_labels(sentences, matchers):
    """One label (or None) per sentence -- None when zero or MORE THAN ONE move's
    cues fire (ambiguity resolved by NLI/the classifier, not guessed here)."""
    labels = []
    for s in sentences:
        hits = [move for move, rx in matchers.items() if rx.search(s)]
        labels.append(hits[0] if len(hits) == 1 else None)
    return labels


_NLI_PIPELINE = None


def _get_nli_pipeline(device="cuda"):
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline
        device_arg = 0 if device == "cuda" else -1
        _NLI_PIPELINE = pipeline("zero-shot-classification", model=NLI_MODEL, device=device_arg)
    return _NLI_PIPELINE


def _nli_cache_key(sentence, candidate_labels):
    h = hashlib.sha256((sentence + "||" + "|".join(candidate_labels)).encode("utf-8")).hexdigest()[:24]
    return h


def _load_nli_cache():
    if NLI_CACHE_PATH.exists():
        return json.loads(NLI_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_nli_cache(cache):
    NLI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NLI_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def nli_seed_labels(sentences, cues_config, device="cuda", batch_size=16, verbose=True):
    """Zero-shot NLI over the given sentences. Returns (labels, top_scores, second_scores),
    one entry per sentence; labels[i] is None when not confident enough. Results are
    cached to disk by (sentence, candidate label set) hash -- this step is the slow part
    of Phase 2F (minutes over thousands of sentences), and gets re-run whenever the
    classifier training step changes (e.g. class_weight), so caching it matters."""
    if not sentences:
        return [], [], []
    label_to_move = {spec["nli_label"]: move for move, spec in cues_config.items()}
    candidate_labels = list(label_to_move.keys())

    cache = _load_nli_cache()
    labels = [None] * len(sentences)
    top_scores = [None] * len(sentences)
    second_scores = [None] * len(sentences)

    to_compute_idx, to_compute_sentences = [], []
    for i, s in enumerate(sentences):
        key = _nli_cache_key(s, candidate_labels)
        cached = cache.get(key)
        if cached is not None:
            labels[i], top_scores[i], second_scores[i] = cached[0], cached[1], cached[2]
        else:
            to_compute_idx.append(i)
            to_compute_sentences.append(s)

    if to_compute_sentences:
        clf = _get_nli_pipeline(device=device)
        results = clf(to_compute_sentences, candidate_labels=candidate_labels, multi_label=True, batch_size=batch_size)
        if isinstance(results, dict):
            results = [results]
        for n_done, (i, s, res) in enumerate(zip(to_compute_idx, to_compute_sentences, results)):
            scores = res["scores"]
            top, second = scores[0], (scores[1] if len(scores) > 1 else 0.0)
            lbl = None
            if top >= NLI_CONFIDENCE_THRESHOLD and (top - second) >= NLI_MARGIN_THRESHOLD:
                lbl = label_to_move[res["labels"][0]]
            labels[i], top_scores[i], second_scores[i] = lbl, top, second
            cache[_nli_cache_key(s, candidate_labels)] = [lbl, top, second]
            if verbose and (n_done + 1) % 500 == 0:
                print(f"    NLI: {n_done + 1}/{len(to_compute_sentences)} (new; {len(sentences) - len(to_compute_sentences)} cached)")
        _save_nli_cache(cache)
    elif verbose:
        print(f"    NLI: all {len(sentences)} sentences served from cache")

    return labels, top_scores, second_scores


def build_seed_labels(sentences, cues_config, matchers, device="cuda"):
    """regex first (higher precision by construction); NLI fills in the rest when
    confident. Returns (seed_labels, source) with source in {"regex", "nli", None}."""
    regex_labels = regex_seed_labels(sentences, matchers)
    need_nli_idx = [i for i, l in enumerate(regex_labels) if l is None]
    need_nli_sentences = [sentences[i] for i in need_nli_idx]

    nli_labels, _top, _second = nli_seed_labels(need_nli_sentences, cues_config, device=device)

    seed_labels = list(regex_labels)
    source = ["regex" if l is not None else None for l in regex_labels]
    for idx, lbl in zip(need_nli_idx, nli_labels):
        if lbl is not None:
            seed_labels[idx] = lbl
            source[idx] = "nli"
    return seed_labels, source


def train_and_label_all(sentence_embeddings, seed_labels, random_state=emb.SEED, class_weight="balanced"):
    """Fits LogisticRegression on seed-labeled sentences, predicts on ALL sentences
    (including seeded ones -- the classifier may disagree with a seed label, which is
    expected and fine at this scale; the seed set exists to train it, not to be echoed back).

    class_weight="balanced" by default: the seed set is heavily imbalanced (e.g. restate
    and self_correct are each under 2% of seeds vs. execute's ~35%), and an unweighted
    multinomial LogisticRegression was found (via the hand-checked validation sample) to
    collapse those minority classes to zero predictions across the entire corpus. Balanced
    class weights is the standard, minimal fix -- pass None to reproduce the old behavior."""
    seeded_idx = [i for i, l in enumerate(seed_labels) if l is not None]
    if len(seeded_idx) < len(MOVES) * 2:
        raise ValueError(f"Only {len(seeded_idx)} seed-labeled sentences -- too few to train "
                          f"a reliable {len(MOVES)}-class classifier. Lower NLI thresholds or add more data.")
    X_train = sentence_embeddings[seeded_idx]
    y_train = np.array([seed_labels[i] for i in seeded_idx])

    clf = LogisticRegression(max_iter=3000, random_state=random_state, class_weight=class_weight)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(sentence_embeddings)
    pred_labels = clf.classes_[np.argmax(probs, axis=1)]
    pred_conf = np.max(probs, axis=1)
    return pred_labels, pred_conf, clf, len(seeded_idx)


def export_validation_sample(records, path, n=200, random_state=emb.SEED):
    """Stratified sample across predicted_move (roughly equal allocation, capped by
    availability), with an empty human_label column for hand-checking."""
    rng = random.Random(random_state)
    by_move = defaultdict(list)
    for r in records:
        by_move[r["predicted_move"]].append(r)

    per_move_target = max(1, n // max(1, len(by_move)))
    sample = []
    for _move, recs in by_move.items():
        shuffled = recs[:]
        rng.shuffle(shuffled)
        sample.extend(shuffled[:per_move_target])

    if len(sample) < n:
        sampled_ids = {(r["model"], r["instance_id"], r["sentence_idx"]) for r in sample}
        remaining = [r for r in records if (r["model"], r["instance_id"], r["sentence_idx"]) not in sampled_ids]
        rng.shuffle(remaining)
        sample.extend(remaining[: n - len(sample)])
    sample = sample[:n]
    rng.shuffle(sample)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "instance_id", "sentence_idx", "sentence", "predicted_move",
                  "predicted_confidence", "seed_source", "human_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            writer.writerow({k: r.get(k, "") for k in fieldnames[:-1]} | {"human_label": ""})
    return sample


def compute_agreement(validation_csv_path, current_records=None):
    """Cohen's kappa between human_label and the classifier's predicted move on the
    hand-checked sample. Rows with a blank human_label or 'unclear' are excluded
    (reported separately, not silently dropped without a count). Also flags any
    human_label value that isn't one of the 8 canonical MOVES strings, rather than let
    sklearn silently treat a typo as a distinct label.

    IMPORTANT: the validation CSV's own `predicted_move` column is frozen at export
    time. If the classifier has been retrained since (it was: class_weight="balanced"),
    scoring against that column measures the OLD classifier, not the one behind the
    figures. Pass `current_records` (the current run's per-sentence records) to score
    the human labels against the CURRENT predictions instead; rows whose sentence
    can't be found there (or whose text no longer matches) are excluded and counted.
    The export-time kappa is still reported alongside, for transparency."""
    import csv
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    with open(validation_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_unmatched = 0
    n_changed_since_export = 0
    kappa_export_time = None
    if current_records is not None:
        from . import text_normalize as _tn
        current = {(r["model"], r["instance_id"], int(r["sentence_idx"])): r for r in current_records}
        norm = lambda s: " ".join(_tn.normalize_text(s).split())

        labeled = [r for r in rows if (r.get("human_label") or "").strip().lower() not in ("", "unclear")
                   and (r.get("human_label") or "").strip() in MOVES]
        if labeled:
            kappa_export_time = cohen_kappa_score(
                [r["human_label"].strip() for r in labeled], [r["predicted_move"].strip() for r in labeled])

        rescored = []
        for r in rows:
            key = (r["model"], r["instance_id"], int(r["sentence_idx"]))
            cur = current.get(key)
            if cur is None or norm(cur["sentence"]) != norm(r["sentence"]):
                n_unmatched += 1
                r = {**r, "human_label": ""}  # can't score this row against the current run
                rescored.append(r)
                continue
            if cur["predicted_move"] != r["predicted_move"]:
                n_changed_since_export += 1
            rescored.append({**r, "predicted_move": cur["predicted_move"]})
        rows = rescored

    unrecognized = []
    valid_rows = []
    excluded_blank = 0
    for r in rows:
        h = (r.get("human_label") or "").strip()
        if not h or h.lower() == "unclear":
            excluded_blank += 1
            continue
        if h not in MOVES:
            unrecognized.append(h)
            continue
        valid_rows.append(r)

    y_true = [r["human_label"].strip() for r in valid_rows]
    y_pred = [r["predicted_move"].strip() for r in valid_rows]

    labels_present = sorted(set(y_true) | set(y_pred), key=MOVES.index)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels_present) if valid_rows else None
    raw_agreement = (sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)) if y_true else None
    cm = confusion_matrix(y_true, y_pred, labels=labels_present).tolist() if valid_rows else []

    return {
        "n_total_rows": len(rows),
        "n_excluded_blank_or_unclear": excluded_blank,
        "n_unrecognized_label_value": len(unrecognized),
        "unrecognized_values": sorted(set(unrecognized)),
        "n_used": len(valid_rows),
        "labels_present_in_sample": labels_present,
        "moves_never_appearing": [m for m in MOVES if m not in labels_present],
        "scored_against": "current classifier" if current_records is not None else "predictions stored in the CSV at export time",
        "n_rows_not_found_in_current_run": n_unmatched,
        "n_predictions_changed_since_export": n_changed_since_export,
        "cohen_kappa_at_export_time_predictions": kappa_export_time,
        "cohen_kappa": kappa,
        "raw_agreement_rate": raw_agreement,
        "confusion_matrix": cm,
    }


def move_distribution_by_model(records):
    """Returns a table: one row per model, one column per move (proportions)."""
    counts = defaultdict(Counter)
    for r in records:
        counts[r["model"]][r["predicted_move"]] += 1
    out = []
    for model, c in counts.items():
        total = sum(c.values())
        row = {"model": model, "n_sentences": total}
        for m in MOVES:
            row[m] = (c.get(m, 0) / total) if total else 0.0
        out.append(row)
    return out


def transition_matrix_by_model(records):
    """model -> row-normalized 8x8 transition matrix (P(next=B | current=A)), computed
    within each trace's sentence sequence only (no cross-trace transitions)."""
    traces = defaultdict(list)
    for r in records:
        traces[(r["model"], r["instance_id"])].append((r["sentence_idx"], r["predicted_move"]))

    move_idx = {m: i for i, m in enumerate(MOVES)}
    counts_by_model = defaultdict(lambda: np.zeros((len(MOVES), len(MOVES))))
    for (model, _iid), seq in traces.items():
        seq = sorted(seq, key=lambda x: x[0])
        for (_, a), (_, b) in zip(seq, seq[1:]):
            counts_by_model[model][move_idx[a], move_idx[b]] += 1

    out = {}
    for model, counts in counts_by_model.items():
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            trans = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
        out[model] = trans
    return out


def js_divergence_matrix(distribution_table):
    """distribution_table: output of move_distribution_by_model(). Returns (models,
    matrix) of pairwise Jensen-Shannon DIVERGENCE (squared JS distance, base 2)."""
    models = sorted(r["model"] for r in distribution_table)
    by_model = {r["model"]: r for r in distribution_table}
    vecs = {m: np.array([by_model[m].get(mv, 0.0) for mv in MOVES]) for m in models}
    n = len(models)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jensenshannon(vecs[models[i]], vecs[models[j]], base=2) ** 2
    return models, mat
