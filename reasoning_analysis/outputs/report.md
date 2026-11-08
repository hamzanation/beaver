# Reasoning-trace analysis: brand, size, correctness, and human comparison

*Updated 2026-09-26: pipeline re-run after fixing an apostrophe-normalization bug; Figures 1, 7, 11, 12 redrawn; move-classifier agreement corrected (κ=0.69, see §3). The previous version is archived in `outputs/archive_20260925/`.*

Full dataset: `dw`, setting 2, 6 models (`claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`) + 5 human reference rationales. 600 attempted instances, 515 with a captured reasoning trace, 47 instances with a trace from all 6 models. No LLM prompting or LLM-based labeling was used anywhere in this pipeline — all features are lexical, embedding-based, or a small trained classifier (logistic regression on sentence embeddings, seeded by regex cues + a zero-shot NLI entailment model).

---

## 1. Data shape and a structural confound you should read every other finding through

**Figure 1** — accuracy ranges from 15% (`gpt-5.6-sol`) to 37% (`claude-opus-5`); trace coverage is 100% for `claude-haiku-4-5`/`claude-opus-5` down to 66% for `gpt-5.6-terra`. Missingness is *not* random with respect to correctness for two OpenAI models (`gpt-5.6-luna`: 56% coverage on correct answers vs. 88% on incorrect; `gpt-5.6-terra`: 56% vs. 70%) — the OpenAI Responses API's `reasoning.summary` doesn't reliably return content even when `reasoning_tokens` shows real internal reasoning happened, and that omission correlates with correctness. Treat correctness-conditioned results for those two models as lower-confidence.

**Figure 2** — trace length varies by an order of magnitude (`claude-opus-5` median ~93 words vs. `claude-haiku-4-5` median ~488, after stripping markdown headers/code blocks — the quantity Figure 2 plots), and length is almost perfectly confounded with **format**: 100% of `gpt-5.6-*` traces are flagged `is_summary` (a bold-header block, the signature of OpenAI's Responses-API reasoning summary) vs. 0–9% of Anthropic traces. **This is the single most important caveat in this report**: Anthropic's extended-thinking traces are the model's actual raw thinking; OpenAI's are a model-generated *summary* of reasoning that happened, not the reasoning itself. Every "brand" finding below is at least partly a "raw vs. summarized" finding, and the two are not statistically separable in this dataset (see §4).

---

## 2. Brand and size are both real signals in the embeddings — but at very different strengths

**Figure 3 / Figure 4** — hierarchical clustering on model-to-model trace similarity (shared-instance cosine similarity, `bge-large-en-v1.5`) splits cleanly into two branches that are exactly the Anthropic/OpenAI split, before ever touching a label. The UMAP projection of *residual* embeddings (per-instance content removed) shows the same clean brand separation, and adds structure inside Anthropic: three clusters in total — `claude-haiku-4-5` on its own, `claude-opus-5` + `claude-sonnet-5` together, and all three OpenAI models fully intermixed in one blob (they are close to interchangeable in style). (An earlier draft said OpenAI's cluster was "tighter" than Anthropic's; that came from a dev-sample run and is not what the final figure shows. UMAP distances/cluster sizes shouldn't be compared anyway — see `figure_guide.md`.)

**Figure 5** — brand is trivially separable: a logistic-regression probe hits 100% (embeddings) / 99.6% (lexical features) cross-validated accuracy vs. a 50% majority baseline (permutation p=0.005, n=282 shared-instance rows, 47 groups). The single strongest lexical driver is `is_summary` itself — confirming this is substantially the format confound from §1, not a subtle stylistic tell. Size tier is a real but far weaker signal: 65% (embeddings) / 55% (lexical) vs. a 33% baseline, still p=0.005 — worth taking seriously, but nowhere near as separable as brand.

**Robustness check**: the mean-pooled-sentence similarity matrix correlates with an independent long-context whole-trace embedding (`bge-m3`) at r=0.57–0.59 off-diagonal — a moderate, not perfect, agreement. Don't over-read small differences in Figure 3's exact numbers as more precise than that.

---

## 3. Distinctive vocabulary and reasoning-move composition

**Figure 7** — Fightin' Words (log-odds, informative Dirichlet prior) on the brand comparison surfaces an interpretable pattern: `department`, `subject`, `name`, `code`, `offered` skew Anthropic (schema/domain-grounded vocabulary — consistent with raw thinking walking through the actual schema), while `might`, `seems`, `if`, `i` skew OpenAI (hedging + first-person narration — consistent with a narrated, retrospective summary rather than in-the-moment reasoning).

**Figure 8** — the reasoning-move taxonomy (8 categories: restate / interpret_ambiguity / plan / hypothesize / verify / self_correct / meta_format / execute, bootstrapped via regex cues + zero-shot NLI, then a trained sentence-embedding classifier) shows `claude-opus-5`/`claude-sonnet-5` are extremely `execute`-dominated (58–60% of sentences) with little else — **but this is very likely a length artifact**, not a stylistic one: those are also the two shortest-trace models (Figure 2), and a 5-sentence trace mostly *is* "writing the query." `claude-haiku-4-5` (much longer traces) shows a far more diverse mix. The `gpt-5.6-*` models spread evenly across plan/hypothesize/verify/meta_format/execute (~10–25% each) — a genuinely different compositional shape, not just "more of everything."

The most striking single number in this whole analysis: **humans restate the question in ~30% of their sentences; no model exceeds 7%** (Figure 8, `human` bar). Every model — regardless of brand or format — spends far less of its reasoning on restating/framing the problem than a human does when explaining their own reasoning.

**Figure 9** — per-model move-transition matrices are dominated by "→execute" for the terse Anthropic models; the JS-divergence matrix shows the three `gpt-5.6-*` models are nearly indistinguishable from each other (JS 0.001–0.005) while `human` is the single most distinct row in the whole matrix (JS 0.17–0.38 against every model, vs. 0.22–0.29 between an average Anthropic and average OpenAI model) — reinforcing that human explanations are compositionally unlike any of the 6 models, not just unlike one brand.

**Data-quality note (revised 2026-09-26)**: the move classifier initially had a real bug — an unweighted multinomial logistic regression collapsed the two rarest classes (`restate`, `self_correct`) to zero predictions across the corpus. Fixed with `class_weight="balanced"`. **But my earlier claim that agreement with your hand labels "held at κ=0.94" was wrong**: the check compared your labels against the predictions *stored in the validation file when it was first exported* (from the pre-fix classifier), not against the classifier behind the figures. Scored properly against the current classifier: **Cohen's κ = 0.69 (73.8% raw agreement; n=195 of 200, 5 rows could not be matched to the current run)**. Per move on your labels: `interpret_ambiguity`, `plan`, `hypothesize`, `meta_format` have F1 0.74–0.82; `execute` has precision 1.00 but recall 0.63; `verify` recall 0.59; and the classifier now *over-predicts* the rare moves (`self_correct` for 12 of your 200 sentences, which you labeled 1 time; `restate` for 5, which you labeled 0 times). Excluding both rare moves, κ = 0.76 (n=183). Two cautions: your 200 sentences were sampled using the *old* classifier's predictions and shown next to your label column, so agreement with the old predictions may be inflated by anchoring, and the sample is not a fair test of the two rare moves. Net: model-level shares for the four large moves are reasonably trustworthy; `restate` and `self_correct` shares (Fig 8 segments, the `self_correct` result in Fig 12) are unvalidated and probably inflated. Human sentences were never validated at all — my own spot check (not a substitute for yours) of the 17 human sentences labeled `restate`: about 13 read as genuine restatements of the question and about 4 look like `plan`, i.e. ~23% rather than the reported 30.4%, still ~3.5× the highest model (6.7%).

---

## 4. Trajectory: how reasoning moves through a trace

**Figure 10** — prompt-similarity drift (how much a sentence's meaning tracks the original prompt as the trace progresses) and step size (sentence-to-sentence semantic distance) do **not** split cleanly by brand. The two terse Anthropic models (`opus`, `sonnet`) have the highest prompt similarity (~0.60–0.68) and the *smallest* step sizes early in the trace (~0.28–0.30), with step size rising toward the end; `haiku` and all three OpenAI models sit lower on prompt similarity (~0.54–0.61) and take larger, steadier steps (~0.35–0.42). Caveats: the curves are noisy (no CIs are drawn, and `opus` traces have only ~4 sentences so its bins are 1–2 sentences each), and "similarity to the prompt" is measured against only the first 512 tokens of a ~13,000-token prompt (the embedding model's input limit) — i.e. mostly the generic instruction boilerplate, not the question or schema. Treat the left panel as weak evidence.

**Figure 11** — revisit rate (share of sentences that re-approach a non-adjacent earlier sentence above a tuned similarity threshold, 0.884 — 90th percentile of the within-trace distribution) is higher for Anthropic overall (mean 0.05 vs 0.01). Within Anthropic the *means* are similar (0.048 incorrect vs 0.041 correct), but the **share of traces with any revisit at all is 45% for incorrect (89/196) vs. 29% for correct (14/49)**. A naive two-proportion test on that gap gives p≈0.03, but it ignores that traces of the same problem are correlated and is not part of the corrected 20-feature battery (where revisit rate as a continuous feature shows nothing: p_BH=0.96), so treat it as a hint, not a finding. For OpenAI, all 36 correct traces have a revisit rate of exactly 0, versus 20 non-zero cases among 187 incorrect ones. Revisit rate is undefined for traces under 3 sentences (31 `opus` + 16 `sonnet` traces are dropped).

---

## 5. What predicts correctness (with real statistical caveats)

**Figure 12** — of 20 style features tested (`correct ~ feature + brand + size_tier`, fit via GEE with an exchangeable working correlation grouped by `instance_id`, 95% CI from 200 bootstrap resamples of *instances* — not rows — and Benjamini-Hochberg correction across the 20-feature family):

- **`hedging_per_100w` is the one feature that survives correction**: coefficient −0.39, 95% CI [−0.67, −0.17], p_BH=0.031, n=515. More hedging language in a trace predicts *lower* correctness. This only became significant at full data scale — it didn't clear correction on the 50-sample-per-model dev run, a useful reminder that some of these effects are real but need the full n to detect.
- `move_self_correct_rate` has a CI that excludes zero in both the full dataset (−7.76 to −0.37) and the raw-traces-only subset (−12.6 to −2.29), but doesn't survive BH correction (p_BH≈0.35 / 0.22). Combined with `self_correct` being an over-predicted, unvalidated move (see the data-quality note above), treat this as suggestive, not established.
- Nothing else in the battery reaches significance after correction.

**A caveat about the mixed-effects (brand/size) results specifically** (not shown as a figure, but in `mixed_effects_results.csv`): p-values there are astronomically small (e.g., p≈1e-127 for `first_person_per_100w` by brand) and **should not be taken at face value**. `brand` only varies at the model level — 3 Anthropic models vs. 3 OpenAI models — not at the row level, but the model's only random effect is on `instance_id`, which doesn't correct for the true replicate count being 3-vs-3, not 272-vs-243. The *direction and magnitude* of these effects are trustworthy and consistent with everything else in this report; the p-values are not a valid population-level significance claim.

**Robustness / stratification**: correctness models were rerun restricted to `is_summary==False` (raw traces only). Because of the brand/format confound in §1, this subset is ~100% Anthropic (143/143 rows) — so this check is closer to "does the finding hold within Anthropic" than a clean summary-vs-raw comparison. `hedging_per_100w`'s coefficient stays negative in that subset too (−0.46) but doesn't clear BH correction there (p_BH=0.22, smaller n). One feature (`question_rate_per_100w`) failed to converge in that subset (too little variation once restricted) and is reported as NaN rather than silently dropped.

---

## 6. Human comparison (5 examples, descriptive only — n=5, no significance tests)

**Figure 13 / Figure 14** — ranking models by raw trace-embedding similarity to each human explanation:

- **`claude-haiku-4-5` ranks #1 on all 5 of 5 examples**, by a consistent margin (0.92–0.94 similarity vs. 0.77–0.91 for every other model). It also has by far the highest step-alignment coverage (84% of human sentences matched, on average) — though this is likely partly a length effect (haiku's traces are also the longest, so there's simply more material to potentially match against).
- **`gpt-5.6-terra` is the least human-aligned**: lowest coverage (21%) and highest "extra" (73% of its own sentences match nothing the human said).
- This reinforces §2's brand-clustering finding through a completely independent method — real external human text, not internal model-to-model comparison — which makes it a more convincing piece of evidence than the internal clustering alone.

**Methodological note**: the step-alignment match threshold could *not* reuse the within-trace revisit threshold from §4 (0.884) — doing so first produced 0% coverage across all 30 (example, model) pairs, because cross-author paraphrase similarity (human vs. model, different vocabulary for the same idea) sits structurally lower than within-trace near-duplicate similarity. A separate threshold (0.703, 90th percentile of the pooled human-vs-model sentence-pair similarity distribution) was tuned instead. As with the revisit threshold, this is a calibrated choice from a small sample, not a validated ground-truth cutoff — reported, not asserted.

Human rationales in this dataset are after-the-fact explanations of the gold query (used here as the "final answer" analog for the human's own final-sentence-similarity metric, since a human trace has no `predicted_0.sql` of its own), not a think-aloud toward an answer the human is still constructing — keep that in mind when comparing "human reasoning style" to a model mid-generation.

---

## Limitations, in one place

1. **Brand ≈ format**, not cleanly separable in this dataset (§1). Every "brand" claim should be read with that in mind.
2. **Size tier is a price-based proxy**, ranked within brand from `model_pricing.csv` — not a real parameter count. Take "size" findings as "the cheaper vs. more expensive model in this family," not as a general size-of-model claim.
3. **Brand/size mixed-effects p-values are anti-conservative** (n=3 models per brand; see §5) — use effect direction/magnitude, not the raw p-value.
4. **Move-classifier agreement with your labels is moderate, not high**: κ=0.69 overall (0.76 excluding the two rare moves); the earlier 0.94 was measured against the pre-fix classifier's stored guesses. `restate` and `self_correct` are over-predicted and unvalidated, and human sentences were never validated.
5. **Length is a running confound** — it's plausible that some brand/format differences (move composition, revisit rate, coverage) are partly explained by trace length rather than "how" a model reasons. This wasn't formally partialled out beyond including `log(completion_tokens)` in the mixed-effects models.
6. **n=5 for the human comparison** — every human-comparison number is descriptive, not a significance-tested claim, per your original spec.
7. **Reasoning-tokens are OpenAI-only**: Anthropic's API doesn't report a separate reasoning-token count (thinking tokens are billed inside output tokens with no separate counter), so `reasoning_tokens` is empty for every Anthropic row — it was not usable as a cross-brand covariate.
8. **The 47 problems traced by all six models are a harder-than-average slice**: mean accuracy across models is 16.3% on them vs. 29.2% on the other 53, with more nested/multi-CTE problems. Figures 3–5 and the whole human comparison (Figures 13–14) use only this subset.
9. **Apostrophe normalization (fixed 2026-09-26)**: OpenAI's text uses curly apostrophes, Anthropic's straight ones. The normalizer now unifies them and every stage was re-run. Effect: Fig 7's OpenAI `i` z-score −39.4 → −32.7 (spurious `ll`/`m` tokens gone); OpenAI's `commitment_action` ~1.5 → ~1.85. No headline conclusion changed. The previous figures, tables and reports are in `outputs/archive_20260925/`.

See `figure_guide.md` for the math, sample sizes, assumptions, and per-figure caveats behind every figure.

---

## Figure index

| # | File | Question it answers |
|---|---|---|
| 1 | `fig01_data_overview` | How much data per model, and how accurate is each? |
| 2 | `fig02_length_distributions` | How long are traces, and how does that relate to raw vs. summarized format? |
| 3 | `fig03_similarity_heatmap` | Do models cluster by brand or by size? |
| 4 | `fig04_umap_residual` | Same question, in 2D, with instance-content removed |
| 5 | `fig05_probe_accuracy` | How separable is brand vs. size, from embeddings vs. lexical features? |
| 6 | `fig06_lexical_profiles` | Per-feature style profile per model, with uncertainty |
| 7 | `fig07_fightin_words` | Which specific words distinguish brand / size? |
| 8 | `fig08_move_composition` | What mix of reasoning moves does each model (and human) use? |
| 9 | `fig09_move_transitions` | What follows what, and how far apart are models' move usage? |
| 10 | `fig10_trajectory` | Does reasoning drift from the prompt, and how fast, over a trace? |
| 11 | `fig11_revisit_vs_correctness` | Does revisiting earlier reasoning associate with getting it right? |
| 12 | `fig12_coefficient_forest` | Which style features actually predict correctness? |
| 13 | `fig13_human_alignment_heatmaps` | Concretely, which reasoning steps match the human's, and which don't? |
| 14 | `fig14_human_coverage_extra` | Which models cover the human's points vs. introduce unrelated content? |

All figures are in `outputs/figures/` as both `.png` and `.svg`. All underlying numbers are in `outputs/tables/`.
