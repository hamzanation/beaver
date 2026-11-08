"""
Phase 2E: per-trace trajectory metrics over sentence embeddings.

Unlike Phase 2 A-D (cross-model comparisons, restricted to the fully-overlapping
instance set), trajectory metrics are a WITHIN-trace property -- they run over every
attempted instance that has a reasoning trace, per the Phase 0 decision that
text-dependent, within-model analyses filter to has_reasoning_trace rather than the
cross-model overlap set.
"""
import numpy as np

from . import embeddings as emb
from . import text_normalize as tn

N_POSITION_BINS = 10


def _cosine(a, b):
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else None


def tune_revisit_threshold(rows, model_name=emb.DEFAULT_SENTENCE_MODEL, device="cuda",
                            percentile=90, max_traces=60):
    """Collects 'max similarity to any non-adjacent earlier sentence' across a sample
    of traces and picks the threshold at `percentile`. Report this value -- it's a
    tuned choice, not a fixed constant, per spec."""
    values = []
    count = 0
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        sentences = tn.split_sentences(tn.normalize_text(r["raw_text"]))
        if len(sentences) < 3:
            continue
        sent_embs = emb.embed_texts(sentences, model_name=model_name, device=device)
        for i in range(2, len(sentences)):
            earlier = sent_embs[: i - 1]  # exclude i-1 (the adjacent one) -- non-adjacent only
            sims = earlier @ sent_embs[i]
            values.append(float(np.max(sims)))
        count += 1
        if count >= max_traces:
            break
    if not values:
        return 0.9  # fallback default if nothing to tune on
    return float(np.percentile(values, percentile))


def compute_trajectory_metrics(rows, revisit_threshold, model_name=emb.DEFAULT_SENTENCE_MODEL, device="cuda",
                                verbose=True):
    """Returns (summary_rows, drift_rows).
    summary_rows: one row per trace with mean_step_size, revisit_rate, n_sentences,
      final_sentence_similarity_to_final_answer.
    drift_rows: one row per (trace, position_bin) with mean prompt-similarity."""
    summary_rows = []
    drift_rows = []
    n = sum(1 for r in rows if r["has_reasoning_trace"])
    done = 0

    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        done += 1
        if verbose and done % 100 == 0:
            print(f"    trajectory metrics: {done}/{n}")

        sentences = tn.split_sentences(tn.normalize_text(r["raw_text"]))
        base = {"model": r["model"], "instance_id": r["instance_id"], "brand": r.get("brand"),
                "size_tier": r.get("size_tier"), "correct": r.get("correct"), "n_sentences": len(sentences)}

        if len(sentences) < 2:
            summary_rows.append({**base, "mean_step_size": None, "revisit_rate": None,
                                  "final_sentence_similarity_to_final_answer": None})
            continue

        sent_embs = emb.embed_texts(sentences, model_name=model_name, device=device)

        # Mean step size: cosine DISTANCE (1 - similarity) between consecutive sentences.
        steps = [1.0 - float(np.dot(sent_embs[i], sent_embs[i + 1])) for i in range(len(sentences) - 1)]
        mean_step_size = float(np.mean(steps))

        # Revisit rate: share of sentences (from the 3rd on) whose max similarity to any
        # NON-adjacent earlier sentence exceeds the tuned threshold.
        revisits = 0
        eligible = 0
        for i in range(2, len(sentences)):
            earlier = sent_embs[: i - 1]
            max_sim = float(np.max(earlier @ sent_embs[i]))
            eligible += 1
            if max_sim > revisit_threshold:
                revisits += 1
        revisit_rate = (revisits / eligible) if eligible else None

        # Final-sentence similarity to the final answer. Callers attach `final_answer_text`
        # to each row -- predicted_0.sql for models, gold SQL for human rationales (Phase 4;
        # a human trace has no prediction of its own, and per the spec these are likely
        # after-the-fact explanations of the correct query, not a think-aloud toward one).
        final_sql = r.get("final_answer_text")
        final_answer_sim = None
        if final_sql:
            final_sql_emb = emb.embed_texts([final_sql], model_name=model_name, device=device)[0]
            final_answer_sim = _cosine(sent_embs[-1], final_sql_emb)

        summary_rows.append({
            **base, "mean_step_size": mean_step_size, "revisit_rate": revisit_rate,
            "final_sentence_similarity_to_final_answer": final_answer_sim,
        })

        # Prompt-similarity drift over normalized position, binned.
        prompt_text = r.get("prompt_text")
        if prompt_text:
            prompt_emb = emb.embed_texts([prompt_text], model_name=model_name, device=device)[0]
            bin_sims = {b: [] for b in range(N_POSITION_BINS)}
            n_sent = len(sentences)
            for i in range(n_sent):
                pos = i / (n_sent - 1) if n_sent > 1 else 0.0
                b = min(int(pos * N_POSITION_BINS), N_POSITION_BINS - 1)
                sim = _cosine(sent_embs[i], prompt_emb)
                if sim is not None:
                    bin_sims[b].append(sim)
            for b, sims in bin_sims.items():
                if sims:
                    drift_rows.append({
                        "model": r["model"], "instance_id": r["instance_id"], "brand": r.get("brand"),
                        "size_tier": r.get("size_tier"), "position_bin": b,
                        "mean_prompt_similarity": float(np.mean(sims)), "n_sentences_in_bin": len(sims),
                    })

    return summary_rows, drift_rows
