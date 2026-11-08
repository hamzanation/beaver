# Reasoning-trace analysis

Programmatic (no LLM prompting/labeling) analysis of reasoning traces across the 6
BEAVER models + 5 human reference rationales: brand, size, correctness, and a
human-comparison case study. See `outputs/report.md` for the findings write-up.

## Environment

Uses the repo's existing `beaver-eval` venv (one level up), which already has
`torch`+CUDA, `pandas`, `scikit-learn`, `sentence-transformers`, `transformers`,
`statsmodels`, `pysbd`, `umap-learn`, `matplotlib`, `seaborn`. No separate
environment needed.

```bash
cd reasoning_analysis
../beaver-eval/Scripts/python.exe run_all.py --stage <stage> [--sample N] [--device cuda|cpu]
```

## Pipeline stages, in dependency order

Each stage reads from `outputs/tables/` written by earlier stages, so run them in
this order the first time (a stage will tell you and exit if something it needs is
missing):

1. `lexical` — Phase 1. Joins traces + `summary_ex_acc.json` + `model_pricing.csv`,
   derives brand/size_tier, computes lexical features + Fightin' Words.
2. `embeddings` — Phase 2 A-D. Sentence/trace embeddings (`bge-large-en-v1.5`,
   `bge-m3` robustness check, both disk-cached by text hash), residualization,
   model-similarity + clustering, brand/size probes, UMAP.
3. `trajectory` — Phase 2E. Step size, revisit rate (threshold auto-tuned and
   reported), prompt-similarity drift, final-answer similarity.
4. `moves` — Phase 2F. Regex + zero-shot-NLI seeded reasoning-move taxonomy, trained
   classifier, move distributions/transitions/JS-divergence. NLI results are cached
   to disk (`outputs/embeddings_cache/nli_seed_cache.json`) since this is the slowest
   step (minutes on the full corpus, near-instant once cached).
5. `stats` — Phase 3. Mixed-effects models (style ~ brand + size + length,
   random intercept per instance), correctness models (GEE, grouped by instance),
   bootstrap CIs (resamples *instances*), Benjamini-Hochberg correction,
   raw-traces-only robustness rerun. `--n-boot` controls bootstrap resamples
   (default 200; expect this stage to be the slowest non-`moves` one).
6. `human` — Phase 4. Human lexical/trajectory features, folds humans into a full
   `moves` rerun (so "human" gets move labels the same way as a model), step
   alignment + model ranking against each of the 5 human examples. Requires
   `trajectory` (full, no `--sample`) to have run first, since it reuses that run's
   tuned revisit threshold for the human trajectory metrics.
7. `convergence` — re-executes every gold + predicted query against the local MySQL
   `dw` database (needs the repo-root `.env`), groups each problem's predictions by
   result-set equivalence, and compares reasoning similarity for same- vs.
   different-answer pairs. Runs twice: as scored, and with `&gt;`/`&lt;` html-unescaped.
   Write-up: `outputs/convergence_report.md`. ~20 min (query execution).
8. `figures` — renders all figures to `outputs/figures/` (`.png` + `.svg`); Figures
   15–19 only if `convergence` has run.

`--sample N` caps instances per model for a fast dev run; omit it for the full
dataset. Developed on `--sample 50` first, verified, then scaled to full.

## Rerunning everything from scratch

```bash
for stage in lexical embeddings trajectory moves stats human convergence figures; do
    ../beaver-eval/Scripts/python.exe run_all.py --stage $stage
done
```

`moves` and `stats` are the slow stages (NLI over ~9,200 sentences; ~20,000
bootstrap model fits respectively) — expect `moves` to take several minutes on a
cold cache and `stats` up to an hour depending on hardware.

## Layout

- `config/lexicons.yaml` — editable word/phrase lists for Phase 1 lexical features.
- `config/move_cues.yaml` — editable regex cues + NLI hypothesis labels per reasoning move.
- `src/` — one module per phase (`data_loading`, `text_normalize`, `lexical_features`,
  `fightin_words`, `embeddings`, `embedding_analysis`, `trajectory`, `move_taxonomy`,
  `stats_models`, `human_comparison`, `viz_common` + `figures_*`).
- `outputs/tables/` — every computed feature/stat/result as CSV.
- `outputs/figures/` — the 14 figures.
- `outputs/embeddings/`, `outputs/embeddings_cache/` — cached embeddings/NLI results
  (safe to delete to force recomputation; everything is content-hash-keyed).
- `outputs/validation/moves_to_check.csv` — a stratified 200-sentence sample for
  hand-checking the move classifier. Fill in `human_label` with one of the 8 move
  names (see the classifier's own categories in `move_taxonomy.MOVES`), then re-run
  `--stage moves` — it will NOT overwrite an already-labeled file. It scores your
  labels against the CURRENT run's predictions (not the `predicted_move` column
  stored in the file, which is frozen at export time and goes stale whenever the
  classifier is retrained) and writes `outputs/tables/moves_agreement_summary.csv`,
  which also records the export-time kappa for comparison. Best practice: hide the
  `predicted_move` column while labeling, to avoid anchoring.
- `outputs/report.md`, `outputs/report_with_figures.md` — the findings write-ups.
- `outputs/figure_guide.md` — the math, sample sizes, axes, assumptions and
  justification for every figure, plus a decoding of the lexical features and
  reasoning moves and a running list of known issues.
- `outputs/archive_20260925/` — the figures, tables and reports from before the
  2026-09-26 re-run (apostrophe-normalization fix), kept for before/after comparison.

## Known limitations (see `outputs/report.md` for the full list)

Brand and trace format (raw vs. summarized) are ~collinear in this dataset (100%
of OpenAI traces are summarized vs. 0-9% of Anthropic's) — most "brand" findings
are at least partly "format" findings. Size tier is a price-based proxy, not a real
parameter count. The mixed-effects brand/size p-values are anti-conservative
(only 3 models per brand); trust their direction/magnitude, not the raw p-value.
The move classifier agrees with hand labels at Cohen's kappa 0.69 (0.76 excluding the
two rare moves `restate`/`self_correct`, which it over-predicts); shares of the four
large moves are reasonably trustworthy, the rare ones are not validated.
