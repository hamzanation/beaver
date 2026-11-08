"""
Entrypoint for the reasoning-trace analysis pipeline.

Currently implements Phase 0 (data loading/joining + trace-coverage diagnostic) and
Phase 1 (lexical features + Fightin' Words). Later phases (embeddings, probes, UMAP,
move taxonomy, mixed-effects models, human comparison, figures) will plug into the
same --stage flag as they're built.

Usage:
    python run_all.py --stage lexical --sample 50
    python run_all.py --stage lexical            # full run, all instances
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_loading as dl
from src import lexical_features as lf
from src import fightin_words as fw
from src import text_normalize as tn
from src import embeddings as emb
from src import embedding_analysis as ea
from src import trajectory as traj
from src import move_taxonomy as mt
from src import stats_models as sm
from src import human_comparison as hc
from src import convergence as cv
from src import figures_overview as fig_ov
from src import figures_embeddings as fig_emb
from src import figures_lexical as fig_lex
from src import figures_moves as fig_mv
from src import figures_trajectory_stats as fig_tj
from src import figures_human as fig_hu
from src import figures_convergence as fig_cv

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TABLES = OUTPUTS / "tables"
EMBED_STORE = OUTPUTS / "embeddings"


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (skip) {path.name}: no rows")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(OUTPUTS.parent)} ({len(rows)} rows)")


def run_lexical(sample):
    print(f"Loading model instances (sample={sample or 'ALL'} per model)...")
    rows, models = dl.load_model_instances(sample=sample)
    print(f"  {len(rows)} attempted-instance rows across {len(models)} models: {models}")

    coverage = dl.trace_coverage_table(rows)
    write_csv(TABLES / "trace_coverage.csv", coverage)

    shared = dl.shared_instance_ids(rows, models=models)
    print(f"  fully-overlapping (all models have a trace) instance count: {len(shared)}")
    write_csv(TABLES / "shared_instance_ids.csv",
              [{"instance_id": iid} for iid in sorted(shared)],
              fieldnames=["instance_id"])

    print("Computing lexical features (this is the slow-ish step: text normalize + sentence split + lexicon match per trace)...")
    matchers = lf.LexiconMatchers(lf.load_lexicons())
    feature_rows = []
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue  # Phase 0 decision: text-dependent features filter to has_reasoning_trace, not the master join
        feats = lf.compute_features(
            r["raw_text"], prompt_text=r.get("prompt_text"), stop_reason=r.get("stop_reason"), matchers=matchers,
        )
        row = {
            "instance_id": r["instance_id"], "model": r["model"], "brand": r["brand"],
            "size_tier": r["size_tier"], "correct": r["correct"],
            "reasoning_tokens": r["reasoning_tokens"], "completion_tokens": r["completion_tokens"],
            **feats,
        }
        feature_rows.append(row)
    write_csv(TABLES / "lexical_features.csv", feature_rows)

    print("Running Fightin' Words: brand vs. brand, size tier vs. size tier (within-brand pairs)...")
    normed_by_model = {}
    for r in rows:
        if r["has_reasoning_trace"]:
            normed_by_model.setdefault(r["model"], []).append(tn.normalize_text(r["raw_text"]))

    normed_by_brand = {}
    for r in rows:
        if r["has_reasoning_trace"]:
            normed_by_brand.setdefault(r["brand"], []).append(tn.normalize_text(r["raw_text"]))

    brands = sorted(b for b in normed_by_brand if b != "human")
    if len(brands) == 2:
        result = fw.fightin_words(normed_by_brand[brands[0]], normed_by_brand[brands[1]])
        write_csv(TABLES / f"fightin_words_brand_{brands[0]}_vs_{brands[1]}.csv", result)
    else:
        print(f"  (skip) brand Fightin' Words: expected exactly 2 brands, found {brands}")

    tier_by_model = {r["model"]: r["size_tier"] for r in rows}
    brand_of_model = {r["model"]: r["brand"] for r in rows}
    for brand in brands:
        models_this_brand = sorted(m for m, b in brand_of_model.items() if b == brand)
        tiers_present = sorted({tier_by_model[m] for m in models_this_brand}, key=lambda t: {"small": 0, "mid": 1, "large": 2, "only": 0}.get(t, 9))
        for tier_a, tier_b in combinations(tiers_present, 2):
            models_a = [m for m in models_this_brand if tier_by_model[m] == tier_a]
            models_b = [m for m in models_this_brand if tier_by_model[m] == tier_b]
            texts_a = [t for m in models_a for t in normed_by_model.get(m, [])]
            texts_b = [t for m in models_b for t in normed_by_model.get(m, [])]
            if not texts_a or not texts_b:
                continue
            result = fw.fightin_words(texts_a, texts_b)
            write_csv(TABLES / f"fightin_words_size_{brand}_{tier_a}_vs_{tier_b}.csv", result)

    return rows, feature_rows


_LEXICAL_METADATA_COLS = {"instance_id", "model", "brand", "size_tier", "correct",
                           "reasoning_tokens", "completion_tokens"}


def _load_lexical_features_by_key():
    path = TABLES / "lexical_features.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `--stage lexical` first.")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        feature_cols = [c for c in fieldnames if c not in _LEXICAL_METADATA_COLS]
        by_key = {}
        for row in reader:
            key = (row["model"], row["instance_id"])
            vec = []
            for c in feature_cols:
                v = row[c]
                if v in ("", "None"):
                    vec.append(0.0)
                elif v == "True":
                    vec.append(1.0)
                elif v == "False":
                    vec.append(0.0)
                else:
                    vec.append(float(v))
            by_key[key] = vec
    return by_key, feature_cols


def run_embeddings(sample, device="cuda", embedder=None):
    print(f"Loading model instances (sample={sample or 'ALL'} per model)...")
    rows, models = dl.load_model_instances(sample=sample)
    shared = dl.shared_instance_ids(rows, models=models)
    print(f"  {len(rows)} rows, {len(models)} models, {len(shared)} fully-overlapping instances "
          f"(Phase 0 decision: restrict cross-model comparisons to this set)")

    emb.set_seed()
    EMBED_STORE.mkdir(parents=True, exist_ok=True)

    sentence_model = embedder or emb.DEFAULT_SENTENCE_MODEL
    print(f"Computing mean-pooled sentence embeddings with {sentence_model} (cached to disk by text hash)...")
    trace_embs = ea.compute_trace_embeddings(rows, model_name=sentence_model, device=device)

    print(f"Computing whole-trace long-context embeddings with {emb.LONG_CONTEXT_MODEL} (robustness check)...")
    long_ctx_embs = ea.compute_long_context_embeddings(rows, device=device)

    print("A. Residualizing (subtract per-instance mean across models)...")
    residuals, raw_subset = ea.residualize(trace_embs, shared, models)
    print(f"  {len(residuals)} (model, instance_id) residual vectors")

    np.savez(
        EMBED_STORE / "trace_embeddings_raw_and_residual.npz",
        **{f"raw__{m}__{iid}": v for (m, iid), v in raw_subset.items()},
        **{f"resid__{m}__{iid}": v for (m, iid), v in residuals.items()},
    )
    print(f"  wrote {EMBED_STORE / 'trace_embeddings_raw_and_residual.npz'}")

    print("D. UMAP of residual embeddings...")
    brand_of_model_umap = {r["model"]: r["brand"] for r in rows}
    tier_of_model_umap = {r["model"]: r["size_tier"] for r in rows}
    umap_pairs, umap_coords = ea.run_umap(residuals)
    umap_rows = [
        {"model": m, "instance_id": iid, "brand": brand_of_model_umap[m], "size_tier": tier_of_model_umap[m],
         "umap_x": float(x), "umap_y": float(y)}
        for (m, iid), (x, y) in zip(umap_pairs, umap_coords)
    ]
    write_csv(TABLES / "umap_residual_embeddings.csv", umap_rows)

    print("B. Within-instance model-by-model similarity + hierarchical clustering...")
    sim_result = ea.model_similarity_matrix(trace_embs, shared, models)
    sim_rows = [
        {"model_a": models[i], **{models[j]: sim_result["similarity_matrix"][i, j] for j in range(len(models))}}
        for i in range(len(models))
    ]
    write_csv(TABLES / "model_similarity_matrix.csv", sim_rows, fieldnames=["model_a"] + models)
    np.save(EMBED_STORE / "similarity_linkage.npy", sim_result["linkage"])
    print(f"  wrote {EMBED_STORE / 'similarity_linkage.npy'} (scipy linkage matrix, for a dendrogram figure later)")

    print("  Robustness check: same similarity computed from the long-context (bge-m3) embeddings...")
    sim_result_lc = ea.model_similarity_matrix(long_ctx_embs, shared, models)
    off_diag = ~np.eye(len(models), dtype=bool)
    a = sim_result["similarity_matrix"][off_diag]
    b = sim_result_lc["similarity_matrix"][off_diag]
    valid = ~(np.isnan(a) | np.isnan(b))
    corr = float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() >= 2 else None
    write_csv(TABLES / "similarity_robustness_check.csv", [{
        "mean_pooled_model": sentence_model, "long_context_model": emb.LONG_CONTEXT_MODEL,
        "pearson_corr_offdiag_similarities": corr, "n_model_pairs": int(valid.sum()),
    }])

    print("C. Probes: brand and size_tier, from residual embeddings vs. lexical features...")
    lexical_by_key, feature_cols = _load_lexical_features_by_key()
    brand_of_model = {r["model"]: r["brand"] for r in rows}
    tier_of_model = {r["model"]: r["size_tier"] for r in rows}

    pairs = sorted(residuals.keys())
    missing_lexical = [p for p in pairs if p not in lexical_by_key]
    if missing_lexical:
        print(f"  WARNING: {len(missing_lexical)} (model,instance_id) pairs have an embedding but no lexical "
              f"feature row -- excluding them from probes so both inputs use identical rows. e.g. {missing_lexical[:3]}")
        pairs = [p for p in pairs if p not in missing_lexical]

    emb_X = np.stack([residuals[p] for p in pairs])
    lex_X = np.array([lexical_by_key[p] for p in pairs])
    groups = np.array([iid for (_m, iid) in pairs])

    probe_summary = []
    for target_name, target_of in [("brand", brand_of_model), ("size_tier", tier_of_model)]:
        y = np.array([target_of[m] for (m, _iid) in pairs])
        for input_name, X in [("residual_embeddings", emb_X), ("lexical_features", lex_X)]:
            print(f"  probe: target={target_name} input={input_name} (n={len(y)})...")
            result = ea.run_probe(X, y, groups)
            if "error" in result:
                print(f"    skipped: {result['error']}")
                continue
            probe_summary.append({
                "target": target_name, "input": input_name,
                "cv_accuracy": result["cv_accuracy"], "majority_baseline": result["majority_baseline"],
                "permutation_pvalue": result["permutation_pvalue"], "n_permutations": result["n_permutations"],
                "n_samples": result["n_samples"], "n_groups": result["n_groups"],
            })
            if input_name == "lexical_features":
                top_feats = ea.top_lexical_features(result["coefficients"], result["classes"], feature_cols)
                write_csv(TABLES / f"probe_top_features_{target_name}.csv", top_feats)

    write_csv(TABLES / "probe_summary.csv", probe_summary)
    return rows, models


def run_trajectory(sample, device="cuda"):
    print(f"Loading model instances (sample={sample or 'ALL'} per model)...")
    rows, models = dl.load_model_instances(sample=sample)
    n_with_trace = sum(1 for r in rows if r["has_reasoning_trace"])
    print(f"  {len(rows)} rows, {n_with_trace} with a reasoning trace (E runs on all of these, "
          f"not just the cross-model-overlap set, per the Phase 0 within-model-analysis decision)")

    for r in rows:
        r["final_answer_text"] = dl.find_predicted_sql(r["run_name"], r["instance_id"])

    emb.set_seed()
    print("Tuning the revisit-similarity threshold on a sample of traces...")
    threshold = traj.tune_revisit_threshold(rows, device=device)
    print(f"  revisit_threshold = {threshold:.4f} (90th percentile of max-similarity-to-non-adjacent-earlier-sentence)")

    print("E. Computing per-trace trajectory metrics (mean step size, revisit rate, "
          "prompt-similarity drift, final-answer similarity)...")
    summary_rows, drift_rows = traj.compute_trajectory_metrics(rows, revisit_threshold=threshold, device=device)

    for row in summary_rows:
        row["revisit_threshold_used"] = threshold
    write_csv(TABLES / "trajectory_summary.csv", summary_rows)
    write_csv(TABLES / "trajectory_prompt_drift.csv", drift_rows)
    return summary_rows, drift_rows


def run_moves(sample, device="cuda", include_human=False):
    print(f"Loading model instances (sample={sample or 'ALL'} per model)...")
    rows, models = dl.load_model_instances(sample=sample)
    if include_human:
        human_rows = dl.load_human_instances()
        rows = rows + human_rows
        print(f"  + {len(human_rows)} human rows (Phase 4: 'Human' gets move labels the same way as any model)")
    emb.set_seed()

    print("Splitting traces into sentences...")
    all_sentences = []  # flat list of sentence strings
    meta = []  # parallel list of {model, instance_id, sentence_idx, brand, size_tier}
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        sentences = tn.split_sentences(tn.normalize_text(r["raw_text"]))
        for i, s in enumerate(sentences):
            all_sentences.append(s)
            meta.append({"model": r["model"], "instance_id": r["instance_id"], "sentence_idx": i,
                         "brand": r["brand"], "size_tier": r["size_tier"]})
    print(f"  {len(all_sentences)} sentences across {len({(m['model'], m['instance_id']) for m in meta})} traces")

    print("F1. Regex + zero-shot NLI seed labeling...")
    cues_config = mt.load_cues()
    matchers = mt.compile_cue_matchers(cues_config)
    seed_labels, seed_source = mt.build_seed_labels(all_sentences, cues_config, matchers, device=device)
    n_seeded = sum(1 for l in seed_labels if l is not None)
    seed_counts = {}
    for l in seed_labels:
        if l is not None:
            seed_counts[l] = seed_counts.get(l, 0) + 1
    print(f"  {n_seeded}/{len(all_sentences)} sentences got a seed label ({sum(1 for s in seed_source if s == 'regex')} regex, "
          f"{sum(1 for s in seed_source if s == 'nli')} nli)")
    print(f"  seed label distribution: {seed_counts}")

    print("F2. Embedding sentences (reuses the Phase 2A-E disk cache) and training the classifier on seed labels...")
    sentence_embs = emb.embed_texts(all_sentences, device=device)
    pred_labels, pred_conf, clf, n_seeded_used = mt.train_and_label_all(sentence_embs, seed_labels)
    print(f"  trained on {n_seeded_used} seed-labeled sentences, labeled all {len(all_sentences)}")

    records = []
    for m, lbl, conf, src, s in zip(meta, pred_labels, pred_conf, seed_source, all_sentences):
        records.append({**m, "sentence": s, "predicted_move": lbl,
                         "predicted_confidence": float(conf), "seed_source": src})

    write_csv(TABLES / "move_labels_per_sentence.csv", records)

    validation_path = OUTPUTS / "validation" / "moves_to_check.csv"
    already_labeled = False
    if validation_path.exists():
        with open(validation_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        already_labeled = any((row.get("human_label") or "").strip() for row in existing)

    if already_labeled:
        print("F3. moves_to_check.csv already has human labels in it -- NOT overwriting. "
              "Delete it first if you want a fresh stratified sample.")
        agreement = mt.compute_agreement(validation_path, current_records=records)
        write_csv(TABLES / "moves_agreement_summary.csv", [{k: v for k, v in agreement.items() if k != "confusion_matrix"}])
        print(f"  Cohen's kappa vs. CURRENT classifier: {agreement['cohen_kappa']} "
              f"(vs. the export-time predictions stored in the CSV: {agreement['cohen_kappa_at_export_time_predictions']}), "
              f"raw agreement: {agreement['raw_agreement_rate']}, "
              f"n_used={agreement['n_used']}/{agreement['n_total_rows']}, "
              f"rows not found in current run: {agreement['n_rows_not_found_in_current_run']}, "
              f"predictions that changed since export: {agreement['n_predictions_changed_since_export']}, "
              f"moves never appearing in the sample: {agreement['moves_never_appearing']}")
    else:
        print("F3. Exporting a stratified 200-sentence sample for hand-checking...")
        sample_records = mt.export_validation_sample(records, validation_path, n=200)
        print(f"  wrote outputs/validation/moves_to_check.csv ({len(sample_records)} rows) -- "
              f"fill in human_label, then re-run to get Cohen's kappa agreement")

    print("F4. Move distributions, transition matrices, Jensen-Shannon divergence...")
    dist_table = mt.move_distribution_by_model(records)
    write_csv(TABLES / "move_distribution_by_model.csv", dist_table)

    trans_by_model = mt.transition_matrix_by_model(records)
    trans_rows = []
    for model, mat in trans_by_model.items():
        for i, from_move in enumerate(mt.MOVES):
            trans_rows.append({"model": model, "from_move": from_move,
                                **{to_move: float(mat[i, j]) for j, to_move in enumerate(mt.MOVES)}})
    write_csv(TABLES / "move_transition_matrices.csv", trans_rows, fieldnames=["model", "from_move"] + mt.MOVES)

    js_models, js_mat = mt.js_divergence_matrix(dist_table)
    js_rows = [{"model_a": js_models[i], **{js_models[j]: float(js_mat[i, j]) for j in range(len(js_models))}}
               for i in range(len(js_models))]
    write_csv(TABLES / "move_js_divergence.csv", js_rows, fieldnames=["model_a"] + js_models)

    return records


def run_stats(sample, n_boot=200):
    print(f"Building master analysis table (sample={sample or 'ALL'} per model)...")
    df = sm.build_master_table(sample=sample)
    print(f"  {len(df)} rows across {df['model'].nunique()} models, {df['instance_id'].nunique()} distinct instances")
    df.to_csv(TABLES / "master_analysis_table.csv", index=False)

    print(f"Mixed-effects battery: feature ~ brand + size_tier + log(completion_tokens) + (1|instance_id), "
          f"bootstrap n_boot={n_boot}...")
    mixed_results = sm.run_mixed_effects_battery(df, n_boot=n_boot)
    mixed_results.to_csv(TABLES / "mixed_effects_results.csv", index=False)
    print(f"  wrote outputs/tables/mixed_effects_results.csv ({len(mixed_results)} rows)")

    print(f"Correctness battery (GEE, exchangeable, grouped by instance_id), all traces...")
    correctness_all = sm.run_correctness_battery(df, n_boot=n_boot, label="all")

    print(f"Correctness battery, robustness check: raw traces only (is_summary==False)...")
    print("  NOTE: is_summary is ~collinear with brand in this dataset (Phase 0/1 finding), "
          "so this is closer to an Anthropic-only check than a clean format-controlled one -- reported as such.")
    df_raw = df[df["is_summary"] == False]  # noqa: E712 (pandas boolean column comparison)
    correctness_raw = sm.run_correctness_battery(df_raw, n_boot=n_boot, label="raw_only")

    correctness_combined = pd.concat([correctness_all, correctness_raw], ignore_index=True)
    correctness_combined.to_csv(TABLES / "correctness_results.csv", index=False)
    print(f"  wrote outputs/tables/correctness_results.csv ({len(correctness_combined)} rows)")

    return df, mixed_results, correctness_combined


def run_human_comparison(device="cuda"):
    print("Loading human example rows and full model rows...")
    human_rows = dl.load_human_instances()
    model_rows, models = dl.load_model_instances(sample=None)
    n_human_traced = sum(r["has_reasoning_trace"] for r in human_rows)
    print(f"  {len(human_rows)} human examples, {n_human_traced} with a trace")

    traj_summary_path = TABLES / "trajectory_summary.csv"
    if not traj_summary_path.exists():
        raise SystemExit(f"{traj_summary_path} not found -- run --stage trajectory (full, no --sample) first.")
    with open(traj_summary_path, newline="", encoding="utf-8") as f:
        existing_traj = list(csv.DictReader(f))
    threshold = float(existing_traj[0]["revisit_threshold_used"])
    print(f"  reusing revisit_threshold={threshold:.4f} from Phase 2E as the step-alignment match threshold "
          f"(re-tuning from only 5 examples would be flimsy -- same underlying question either way)")

    emb.set_seed()

    print("Human lexical features (Phase 1, same feature set)...")
    matchers = lf.LexiconMatchers(lf.load_lexicons())
    human_lexical_rows = []
    for r in human_rows:
        if not r["has_reasoning_trace"]:
            continue
        feats = lf.compute_features(r["raw_text"], prompt_text=r.get("prompt_text"), stop_reason=None, matchers=matchers)
        human_lexical_rows.append({"instance_id": r["instance_id"], "model": "human", "brand": "human", **feats})
    write_csv(TABLES / "human_lexical_features.csv", human_lexical_rows)

    print("Human trajectory metrics (Phase 2E, final answer = gold SQL since a human trace "
          "has no prediction of its own)...")
    for r in human_rows:
        r["final_answer_text"] = r.get("gold_sql")
    human_traj_summary, human_traj_drift = traj.compute_trajectory_metrics(human_rows, revisit_threshold=threshold, device=device)
    for row in human_traj_summary:
        row["revisit_threshold_used"] = threshold
    write_csv(TABLES / "human_trajectory_summary.csv", human_traj_summary)
    write_csv(TABLES / "human_trajectory_prompt_drift.csv", human_traj_drift)

    print("Human move labels: rerunning --stage moves with humans folded in (Phase 2F, so 'human' "
          "gets move distributions/transitions the same way as any model)...")
    run_moves(sample=None, device=device, include_human=True)

    print("Tuning the step-alignment match threshold (a SEPARATE threshold from Phase 2E's "
          "revisit_threshold -- cross-author paraphrase similarity sits structurally lower "
          "than within-trace near-duplicate similarity; reusing revisit_threshold was tried "
          "first and produced 0% coverage everywhere, which was a threshold problem, not a finding)...")
    alignment_threshold = hc.tune_alignment_threshold(human_rows, model_rows, device=device)
    print(f"  alignment_threshold = {alignment_threshold:.4f} (90th percentile of pooled human-vs-model "
          f"sentence-pair similarity, n=5 examples -- report this value, don't treat it as validated)")

    print("Step alignment + model ranking by similarity to each human example...")
    alignment_rows, alignment_matrices, ranking_rows = hc.build_comparison(human_rows, model_rows, alignment_threshold, device=device)
    for row in alignment_rows:
        row["alignment_threshold_used"] = alignment_threshold
    write_csv(TABLES / "human_step_alignment_summary.csv", alignment_rows)
    write_csv(TABLES / "human_model_ranking.csv", ranking_rows)

    EMBED_STORE.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_STORE / "human_step_alignment_matrices.npz",
             **{f"{iid}__{model}": mat for (iid, model), mat in alignment_matrices.items()})
    print(f"  wrote {EMBED_STORE / 'human_step_alignment_matrices.npz'} (for a heatmap figure later)")

    return alignment_rows, ranking_rows


def run_convergence():
    """Answer convergence: re-execute predictions, cluster each instance's answers by
    result-set equivalence, and relate answer agreement to reasoning similarity."""
    moves_path = TABLES / "move_labels_per_sentence.csv"
    emb_path = EMBED_STORE / "trace_embeddings_raw_and_residual.npz"
    for p in (moves_path, emb_path):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run the embeddings/moves stages first.")
    # Every ATTEMPTED instance, not just traced ones: answer agreement doesn't need a
    # trace; the trace-similarity metrics are simply NaN where one is missing.
    rows, _models = dl.load_model_instances()
    master = pd.DataFrame(rows)[["model", "instance_id", "run_name", "correct", "has_reasoning_trace"]]

    store = np.load(emb_path)
    trace_emb = {}
    for k in store.files:
        kind, model, iid = k.split("__")
        if kind == "raw":
            trace_emb[(model, iid)] = store[k]
    move_labels = pd.read_csv(moves_path)
    shares = cv.move_shares(move_labels[move_labels.model != "human"])

    comparisons = {
        "same_vs_diff_answer (both wrong)": (["both_wrong_same_answer"], ["both_wrong_diff_answer"]),
        "both_correct vs both_wrong": (["both_correct"], ["both_wrong_same_answer", "both_wrong_diff_answer"]),
        "same_answer vs diff_answer (any)": (["both_correct", "both_wrong_same_answer"],
                                              ["one_correct", "both_wrong_diff_answer"]),
    }
    # "as_scored" reproduces the benchmark's inputs; "unescaped" html-unescapes the
    # GPT models' `&gt;`/`&lt;` first so their actual query logic gets compared.
    for variant, unescape in (("as_scored", False), ("unescaped", True)):
        print(f"Executing gold + predicted SQL ({variant})...")
        results, sql_of, outcome_rows = cv.execute_all(master, unescape=unescape)
        write_csv(TABLES / f"convergence_prediction_outcomes_{variant}.csv", outcome_rows)
        rescored = master.drop(columns="correct").merge(
            pd.DataFrame(outcome_rows)[["model", "instance_id", "correct"]], on=["model", "instance_id"])

        pairs, clusters = cv.pair_table(rescored, results, sql_of, trace_emb, shares)
        pairs.to_csv(TABLES / f"convergence_pairs_{variant}.csv", index=False)
        clusters.to_csv(TABLES / f"convergence_answer_clusters_{variant}.csv", index=False)
        print(f"  wrote convergence_pairs_{variant}.csv ({len(pairs)} rows), "
              f"convergence_answer_clusters_{variant}.csv ({len(clusters)} rows)")

        contrasts = []
        for metric in ("trace_cosine", "move_js", "table_jaccard"):
            for name, (ka, kb) in comparisons.items():
                contrasts.append({"metric": metric, "comparison": name,
                                  **cv.within_instance_contrast(pairs, metric, ka, kb)})
        write_csv(TABLES / f"convergence_contrasts_{variant}.csv", contrasts)


def run_figures(device="cuda"):
    print("Figure 1-2 (data overview, length distributions)...")
    fig_ov.fig1_data_overview()
    fig_ov.fig2_length_distributions()
    print("Figure 3-5 (similarity heatmap, UMAP, probes)...")
    fig_emb.fig3_similarity_heatmap()
    fig_emb.fig4_umap()
    fig_emb.fig5_probe_accuracy()
    print("Figure 6-7 (lexical profiles, Fightin' Words)...")
    fig_lex.fig6_lexical_profiles()
    fig_lex.fig7_fightin_words()
    print("Figure 8-9 (move composition, transitions)...")
    fig_mv.fig8_move_composition()
    fig_mv.fig9_move_transitions()
    print("Figure 10-12 (trajectory, revisit vs correctness, coefficient forest)...")
    fig_tj.fig10_trajectory(device=device)
    fig_tj.fig11_revisit_vs_correctness()
    fig_tj.fig12_coefficient_forest()
    print("Figure 13-14 (human alignment heatmaps, coverage vs extra)...")
    fig_hu.fig13_human_alignment_heatmaps()
    fig_hu.fig14_human_coverage_extra()
    if (TABLES / "convergence_pairs_unescaped.csv").exists():
        print("Figure 15-19 (answer convergence)...")
        fig_cv.all_figures()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="lexical",
                         choices=["lexical", "embeddings", "trajectory", "moves", "stats", "human", "convergence", "figures"],
                         help="Which pipeline stage to run (more stages land as they're built).")
    parser.add_argument("--sample", type=int, default=None,
                         help="Cap instances per model for a fast dev run.")
    parser.add_argument("--embedder", type=str, default=None,
                         help="Sentence embedding model override for --stage embeddings "
                              "(default: BAAI/bge-large-en-v1.5).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                         help="Device for stages that use GPU models (embeddings/trajectory/moves).")
    parser.add_argument("--n-boot", type=int, default=200,
                         help="Bootstrap resamples for --stage stats (resamples instance_id, not rows).")
    args = parser.parse_args()

    if args.stage == "lexical":
        run_lexical(args.sample)
    elif args.stage == "embeddings":
        run_embeddings(args.sample, device=args.device, embedder=args.embedder)
    elif args.stage == "trajectory":
        run_trajectory(args.sample, device=args.device)
    elif args.stage == "moves":
        run_moves(args.sample, device=args.device)
    elif args.stage == "stats":
        run_stats(args.sample, n_boot=args.n_boot)
    elif args.stage == "human":
        run_human_comparison(device=args.device)
    elif args.stage == "convergence":
        run_convergence()
    elif args.stage == "figures":
        run_figures(device=args.device)


if __name__ == "__main__":
    main()
