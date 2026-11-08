"""
Phase 4: 5-example human comparison. Descriptive only -- no significance tests (n=5),
per the spec. Human rationales are likely after-the-fact explanations of the gold
query, not a think-aloud toward it -- the "final answer" analog used throughout
(trajectory.py's final_answer_text) is gold SQL for humans, predicted_0.sql for models.

  - Step alignment: sentence-by-sentence cosine similarity matrix between a human
    trace and each model's trace on the SAME instance.
      - Coverage: share of human sentences with a match (similarity above threshold)
        somewhere in the model's trace.
      - Extra: share of model sentences with NO match anywhere in the human trace.
  - Ranking: models ranked by RAW (not residualized -- content similarity is exactly
    what's wanted here) trace-embedding similarity to the human trace, per example.

Uses its OWN tuned match threshold, separate from Phase 2E's revisit_threshold --
these look like the same question ("are these two sentences saying the same thing")
but aren't calibrated the same way: Phase 2E's threshold is tuned on WITHIN-trace
near-duplicate sentences (same author, same run, often near-repeats), while this is
CROSS-author paraphrase matching (human vs. model, different vocabulary describing
the same idea), which sits at a structurally lower cosine-similarity range. Reusing
the revisit_threshold here was tried first and found empirically wrong -- max
similarity for even the best-matching pairs topped out around 0.86, so a 0.884
threshold produced 0% coverage across every single example, which is a threshold
problem, not a real finding. Tuned separately below, on the actual pooled
human-vs-model sentence-pair similarity distribution.
"""
import numpy as np

from . import embeddings as emb
from . import text_normalize as tn


def _cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else None


def tune_alignment_threshold(human_rows, model_rows, percentile=90,
                              model_name=emb.DEFAULT_SENTENCE_MODEL, device="cuda"):
    """Pools every human-sentence-vs-model-sentence cosine similarity across all
    available (human example, model) pairs, and picks the threshold at `percentile`.
    Small-sample caveat applies here just as it does to revisit_threshold in Phase 2E
    (n=5 examples) -- report the chosen value, don't treat it as a validated cutoff."""
    model_rows_by_instance = group_model_rows_by_instance(model_rows)
    all_sims = []
    for hr in human_rows:
        if not hr["has_reasoning_trace"]:
            continue
        iid = hr["instance_id"]
        human_sentences = tn.split_sentences(tn.normalize_text(hr["raw_text"]))
        if not human_sentences:
            continue
        human_embs = emb.embed_texts(human_sentences, model_name=model_name, device=device)

        for model, mr in model_rows_by_instance.get(iid, {}).items():
            model_sentences = tn.split_sentences(tn.normalize_text(mr["raw_text"]))
            if not model_sentences:
                continue
            model_embs = emb.embed_texts(model_sentences, model_name=model_name, device=device)
            sim = human_embs @ model_embs.T
            all_sims.extend(sim.flatten().tolist())

    if not all_sims:
        return 0.75  # fallback if nothing to tune on
    return float(np.percentile(all_sims, percentile))


def step_alignment(human_sentences, human_embs, model_sentences, model_embs, threshold):
    """Returns {similarity_matrix, coverage, extra, n_human, n_model}.
    similarity_matrix[i, j] = cosine(human_sentence_i, model_sentence_j)."""
    if not human_sentences or not model_sentences:
        return {"similarity_matrix": np.zeros((len(human_sentences), len(model_sentences))),
                "coverage": None, "extra": None,
                "n_human": len(human_sentences), "n_model": len(model_sentences)}

    sim = human_embs @ model_embs.T  # (n_human, n_model), both already unit-normalized

    human_max = sim.max(axis=1)  # best match for each human sentence
    model_max = sim.max(axis=0)  # best match for each model sentence

    coverage = float((human_max > threshold).mean())
    extra = float((model_max <= threshold).mean())

    return {"similarity_matrix": sim, "coverage": coverage, "extra": extra,
            "n_human": len(human_sentences), "n_model": len(model_sentences)}


def group_model_rows_by_instance(model_rows):
    by_instance = {}
    for r in model_rows:
        if r["has_reasoning_trace"]:
            by_instance.setdefault(r["instance_id"], {})[r["model"]] = r
    return by_instance


def rank_models_by_trace_similarity(human_trace_emb, model_trace_embs: dict):
    """model_trace_embs: {model_name: embedding}. Returns a list of (model, similarity)
    sorted descending -- RAW trace embeddings (content similarity is the point here,
    not residualized style)."""
    ranked = [(m, _cosine(human_trace_emb, v)) for m, v in model_trace_embs.items()]
    ranked.sort(key=lambda x: (x[1] is not None, x[1]), reverse=True)
    return ranked


def build_comparison(human_rows, model_rows, threshold, model_name=emb.DEFAULT_SENTENCE_MODEL, device="cuda"):
    """Orchestrates step alignment + ranking for every (human example, available model)
    pair. Returns (alignment_summary_rows, alignment_matrices, ranking_rows).
    alignment_matrices: {(instance_id, model): similarity_matrix} for a heatmap figure later."""
    model_rows_by_instance = group_model_rows_by_instance(model_rows)

    alignment_summary_rows = []
    alignment_matrices = {}
    ranking_rows = []

    for hr in human_rows:
        if not hr["has_reasoning_trace"]:
            continue
        iid = hr["instance_id"]
        human_sentences = tn.split_sentences(tn.normalize_text(hr["raw_text"]))
        human_embs = emb.embed_texts(human_sentences, model_name=model_name, device=device) if human_sentences else np.zeros((0, 1))
        human_trace_emb = emb.embed_trace_mean_pooled(human_sentences, model_name=model_name, device=device)

        available_models = model_rows_by_instance.get(iid, {})
        if not available_models:
            print(f"  WARNING: no model has a reasoning trace for {iid} -- skipping (human example unmatched)")
            continue

        model_trace_embs = {}
        for model, mr in available_models.items():
            model_sentences = tn.split_sentences(tn.normalize_text(mr["raw_text"]))
            model_embs = emb.embed_texts(model_sentences, model_name=model_name, device=device) if model_sentences else np.zeros((0, 1))
            model_trace_embs[model] = emb.embed_trace_mean_pooled(model_sentences, model_name=model_name, device=device)

            result = step_alignment(human_sentences, human_embs, model_sentences, model_embs, threshold)
            alignment_matrices[(iid, model)] = result["similarity_matrix"]
            alignment_summary_rows.append({
                "instance_id": iid, "model": model, "n_human_sentences": result["n_human"],
                "n_model_sentences": result["n_model"], "coverage": result["coverage"], "extra": result["extra"],
            })

        ranked = rank_models_by_trace_similarity(human_trace_emb, model_trace_embs)
        for rank, (model, sim) in enumerate(ranked, start=1):
            ranking_rows.append({"instance_id": iid, "model": model, "rank": rank, "trace_similarity_to_human": sim})

    return alignment_summary_rows, alignment_matrices, ranking_rows
