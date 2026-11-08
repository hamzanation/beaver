# Figure guide: the math, the samples, the axes, and the justification

> **Updated 2026-09-26.** Everything was re-run after fixing an apostrophe-normalization bug; Figures 1, 7, 11 and 12 were redrawn; and the move-classifier agreement was corrected (**κ = 0.69, not 0.94** — see Figure 8 and §6). All numbers below come from the new run. The previous version of the figures, tables and reports is in `outputs/archive_20260925/`.

Companion to `report_with_figures.md`. That file says *what we found*; this one says
*how each figure is built, what its axes mean, how many data points are behind it,
what it assumes, and why it's the right kind of figure for the question*. Written for
someone who hasn't used UMAP, log-odds coefficients, cross-validation, etc. before.

All numbers below were computed from the actual data files while writing this
guide (not recalled from memory). Where I found something wrong or weak in the
earlier figures/reports, it's flagged **[ISSUE]** and collected in the last section.

**Contents**
1. [The problems used, and how much they overlap across models](#1-the-problems-used-and-how-much-they-overlap)
2. [Ten ideas you need, in plain language](#2-ten-ideas-in-plain-language)
3. [Figure-by-figure (1–14)](#3-figure-by-figure)
4. [UMAP in depth (Figure 4)](#4-umap-in-depth)
5. [The lexical features and reasoning moves, decoded (Figures 6, 8, 9)](#5-lexical-features-and-reasoning-moves-decoded)
6. [Issues found while writing this guide](#6-issues-found-while-writing-this-guide)

---

## 1. The problems used, and how much they overlap

**Short answer: every model got exactly the same 100 problems with the same prompts. Claude does not have "more" problems. What differs is how many of those 100 have a *captured reasoning trace* — and that difference matters.**

### What the problems are
- **Benchmark**: BEAVER, dataset `dw` — text-to-SQL over a real enterprise data warehouse (MIT-style tables such as `COURSE_CATALOG_SUBJECT_OFFERED`, `SIS_DEPARTMENT`, `FCLT_ROOMS`). Each problem is a long natural-language question plus a gold SQL query. The full `dw` dev set has 5,787 questions; we use the fixed 100-question sample in `data/dw/dev_sampled.json`.
- **Composition of the 100**: 48 "domain-specific complex", 43 "complex", 9 "domain-specific" queries. By SQL shape: 26 `cte-nested`, 23 `cte`, 23 `nested-cte`, 19 `nested`, 9 `base`. 57 of 100 require domain knowledge (e.g. "Course 18" means the Mathematics department). Median 4 tables per problem (range 2–6), median 2 sub-queries (range 0–4), questions have a median of 90 words (29–210).
- **Prompt setting ("setting 2")**: the model is given the most help — the right tables, the column mapping, the join keys, the domain-knowledge notes, and the question decomposed into sub-questions — and must still write the final SQL. The prompt text is identical across models for 98 of 100 instances (the other 2 are two `claude-sonnet-5` rows where the prompt wasn't captured, not a real difference).
- **Accuracy** is *execution accuracy*: the predicted SQL is run and counts as correct only if its result table equals the gold query's result (column names ignored). Blank/failed SQL counts as wrong.

### Who has a trace for which problems

All six models attempted all 100 problems. Reasoning traces were captured for:

| model | attempted | has a trace | accuracy [95% Wilson CI] |
|---|---|---|---|
| claude-haiku-4-5 | 100 | 100 | 16% [10, 24] |
| claude-sonnet-5 | 100 | 92 | 28% [20, 37] |
| claude-opus-5 | 100 | 100 | 37% [28, 47] |
| gpt-5.6-luna | 100 | 83 | 16% [10, 24] |
| gpt-5.6-terra | 100 | 66 | 27% [19, 36] |
| gpt-5.6-sol | 100 | 74 | 15% [9, 23] |

The trace sets are nested: haiku and opus have all 100; sonnet's 92 sit inside those; then luna (83), sol (74), terra (66). Only **47 problems** have a trace from all six models. Pairwise overlap (number of problems both models have a trace for):

| | haiku | opus | sonnet | luna | sol | terra |
|---|---|---|---|---|---|---|
| haiku | 100 | 100 | 92 | 83 | 74 | 66 |
| opus | 100 | 100 | 92 | 83 | 74 | 66 |
| sonnet | 92 | 92 | 92 | 80 | 68 | 63 |
| luna | 83 | 83 | 80 | 83 | 66 | 59 |
| sol | 74 | 74 | 68 | 66 | 74 | 53 |
| terra | 66 | 66 | 63 | 59 | 53 | 66 |

Why traces are missing: Anthropic returns its extended-thinking text on every call (except 8 `sonnet-5` calls — cause not investigated; adaptive thinking may skip thinking on some prompts, which I have not verified). OpenAI only returns a *summary* of its reasoning when its summarizer decides to, even though `reasoning_tokens` shows real reasoning happened. Missing OpenAI summaries are correlated with correctness (luna: 56% of correct answers have a trace vs. 88% of incorrect ones).

### **[ISSUE]** The 47 shared problems are a harder-than-average slice
Figures 3, 4 and 5, and the whole human comparison (Figs 13–14), only use problems with traces from all models. That subset is **not a random sample of the 100**:

| | the 47 shared | the other 53 |
|---|---|---|
| mean accuracy across the 6 models | **16.3%** | **29.2%** |
| mean tables per problem | 4.2 | 3.7 |
| mean sub-queries per problem | 3.0 | 1.8 |
| `base` (simplest) problems | 0 of 9 | 9 of 9 |
| `cte-nested` problems | 20 of 26 | 6 of 26 |

Every model does worse on the shared 47 than on the rest (e.g. haiku 10.6% vs 20.8%, luna 8.5% vs 22.6%). So "how the models' reasoning differs" in Figs 3–5 describes their behavior on *harder* problems. The five human-annotated problems are all inside this set, and all are hard for the models (models correct out of 6: dw_104: 3, dw_1133: 2, dw_1795: 0, dw_3298: 1, dw_4771: 1).

### Which figure uses which data

| Fig | Data used | n |
|---|---|---|
| 1 | all attempted instances | 600 (6 × 100) |
| 2, 6, 7, 8, 9, 10, 11, 12 | every trace that exists (each model on *its own* traced problems) | 515 traces (9,170 model sentences) |
| 3, 4, 5 | only the 47 problems traced by all 6 models | 282 traces (47 × 6) |
| 8, 9 also | + human rationales | +5 traces, 56 sentences |
| 13, 14 | 5 human problems × 6 models | 30 (human, model) pairs |

Caveat for the 515-trace figures: because model traces cover different problem subsets, a model-vs-model bar is partly a problem-mix difference. I checked the most important case, Fig 6: restricted to the shared 47, the brand pattern is unchanged (hedging per 100 words: Anthropic 0.13–0.43 vs. OpenAI 2.53–2.89; grounding rate: 30.6–36.2 vs. 20.7–21.7).

---

## 2. Ten ideas in plain language

**1. Embedding.** A neural network (`bge-large-en-v1.5`) reads a sentence and outputs a list of 1,024 numbers — a point in a 1,024-dimensional space. Sentences with similar meaning land near each other. The numbers have no individual meaning; only distances between points do.

**2. Cosine similarity.** How close two embeddings are, measured as the cosine of the angle between them: `cos(a, b) = (a · b) / (‖a‖ ‖b‖)`. 1 = same direction (same meaning), 0 = unrelated, negative = opposite. Our embeddings are scaled to length 1, so it's just the dot product. **Caution:** two sentences about the same SQL problem will have high cosine (~0.6–0.9) even if they say different things; only *differences* in similarity carry information, not the raw value.

**3. Trace embedding (mean pooling).** A trace has many sentences. Its embedding is the average of its sentence embeddings, then rescaled to length 1. This treats the trace as a "bag" of sentences — it loses order and gives every sentence equal weight.

**4. Residualizing (removing the problem's content).** All six models talk about the same problem, so their traces share a lot of content. For each problem we compute the average trace embedding across the 6 models and subtract it from each model's embedding. What's left, `resid = v(model, problem) − mean over models of v(·, problem)`, is "how this model's reasoning differs from the average model's on this problem" — style, with content largely removed. Used in Figs 4 and 5.

**5. Cross-validated accuracy.** If you train a classifier and test it on the same data, it can memorize (we have 1,024 input numbers and only 282 examples). So we hold data out: split the 47 problems into 5 groups ("folds"), train on 4 groups, predict the 5th, rotate, and average. "**GroupKFold by problem**" means all 6 traces of a problem stay together in the same fold, so the classifier is never tested on a problem it partly saw. The number reported is the fraction of held-out traces classified correctly.

**6. Permutation test and the "p=0.005" floor.** To ask "could this accuracy arise by chance?", we shuffle the labels (within each problem) 200 times, rerun the whole cross-validation each time, and count how often the shuffled labels score as well as the real ones. `p = (1 + #shuffles ≥ real) / (200 + 1)`. In Fig 5 *no* shuffle ever matched, so every bar sits at the smallest p this test can output: `1/201 = 0.00498`. The four identical "p=0.005" labels are that floor, not four coincidentally equal results.

**7. Logistic regression, log-odds, and odds ratios.** Logistic regression models the probability a trace is correct as `P = 1/(1 + e^−(β₀ + β·x))`. Rearranged: `log(P/(1−P)) = β₀ + β·x`. The left side is the **log-odds**. So a coefficient β means: *each +1 in the feature adds β to the log-odds of being correct*. Exponentiating gives the **odds ratio** `OR = e^β`: OR = 0.68 means each +1 in the feature multiplies the odds of correct by 0.68 (32% lower); OR = 1 means no effect; β = 0 ↔ OR = 1. That's why Fig 12 has a vertical line at 0. Odds ≠ probability: odds of 1:3 = probability 25%.

**8. GEE (Generalized Estimating Equations).** Ordinary logistic regression assumes every row is independent, but traces of the *same problem* share difficulty. GEE fits the same logistic model while allowing rows within a problem to be correlated (we assume "exchangeable": any two traces of one problem are equally correlated) and computes standard errors that stay valid regardless.

**9. Bootstrap confidence interval.** Resample the 100 problems *with replacement* (each problem brings all its traces), refit the model, repeat 200 times, and take the 2.5th and 97.5th percentiles of the coefficient: a range that shows how much the estimate would jump around with a different draw of problems. (Fig 6's whiskers use the same idea on means with 500 resamples of traces.)

**10. Benjamini–Hochberg (BH) correction.** We test 20 features; at p < 0.05 we'd expect about 1 false alarm by luck alone. BH controls the *false discovery rate*: sort the m p-values `p(1) ≤ … ≤ p(m)`; adjusted `p(k) = min over j ≥ k of (m/j)·p(j)`. "Survives BH" = adjusted p < 0.05.

---

## 3. Figure by figure

Each section: **Question · Data · Axes/marks · Math · Assumptions · Why this figure · Read it as**.

### Figure 1 — Data volume and accuracy

- **Question**: how many instances does each model contribute, and how accurate is it?
- **Data**: 600 rows (6 models × 100). n = 100 per model.
- **Axes**: x = model, ordered Anthropic small→large then OpenAI small→large; color = brand hue, lightness = size tier (light = small). Left y = number of instances (colored part = has a captured trace, gray part = attempted but no trace). Right y = execution accuracy in %, whiskers = 95% Wilson interval (added 2026-09-26).
- **Math**: accuracy = (# instances with matching results) / 100. Uncertainty (drawn as whiskers): Wilson 95% interval for a proportion `p` from `n` trials: `[ (p + z²/2n) ± z·√(p(1−p)/n + z²/4n²) ] / (1 + z²/n)`, `z = 1.96` — the values are in the §1 table.
- **Assumptions**: problems independent; each model's accuracy is a binomial proportion.
- **Why a bar chart**: comparing one quantity (a count or a rate) across six categories; bars anchored at zero are the honest form for that. Fixed model order + brand-hue/tier-lightness color is reused in every figure so the eye learns it once.
- **Read it as**: with n = 100 each interval is roughly ±8–10 points. Clear differences: opus (37) vs. haiku/luna/sol (15–16). *Not* clear: opus vs. sonnet (28) vs. terra (27); haiku vs. luna vs. sol (15–16, indistinguishable). The 95% Wilson intervals are now drawn on the bars, so small gaps no longer look more meaningful than they are.

### Figure 2 — Trace length and raw-vs-summarized

- **Question**: how long are traces, and does length track with format?
- **Data**: 515 traces. Words per trace: haiku 100 traces (median 488), sonnet 92 (141), opus 100 (93), luna 83 (265), terra 66 (146), sol 74 (189).
- **Axes**: Left: x = model, y = **words per trace** — whitespace-separated words after removing markdown headers, code fences and "Writing the query…" lines and Unicode-normalizing (NFKC). These are plain words, *not* model tokens. Each "violin" is a mirrored density (a kernel-density estimate): wider = more traces at that length; the horizontal ticks are the minimum, median and maximum. Right: x = % of a model's traces flagged `is_summary`.
- **Math/definition**: `is_summary` = the trace contains at least one line that consists only of bold text (regex `^\s*\*\*[^*\n]{3,80}\*\*\s*$`) — the visual signature of OpenAI's reasoning-summary format. It's a proxy, not a proof: haiku's ~9% could be bold lines the model wrote itself.
- **Assumptions**: word count is a reasonable length measure; bold-header format ≈ "summarized".
- **Why violins**: length is heavily right-skewed (many short traces, a few very long); a bar of means would hide the shape. The right panel is needed because format is the biggest confounder in the whole study.
- **Read it as**: opus's traces are tiny (median 93 words, ~4 sentences); haiku's are ~5× longer. Format splits almost perfectly by brand.

### Figure 3 — Which models' reasoning resembles which?

- **Question**: do models cluster by brand or by size?
- **Data**: the 47 shared problems × 6 models = 282 traces. **[ISSUE: harder-than-average subset, §1]**
- **Axes**: rows/columns = models, reordered by the dendrogram; the top tree is the dendrogram; color bars beside the labels show brand hue + tier lightness; cell color/number = average cosine similarity.
- **Math**: trace vector `v = normalize(mean of the trace's sentence embeddings)` (raw embeddings, *not* residuals, so content is included). Cell `(i, j) = (1/47) Σ_problems cos(v_i, v_j)`. Diagonal = 1 by definition. Dendrogram: agglomerative clustering, **average linkage**, on distance `d = 1 − similarity`: repeatedly merge the two closest clusters; the height of a merge is the average distance between their members. Off-diagonal similarities: 0.858–0.931.
- **Assumptions**: mean-pooled sentence embeddings summarize a trace; equal weight per sentence (so a long trace is dominated by whatever it says most often); 47 problems represent "the" problem distribution (they don't quite).
- **Robustness**: replacing the trace vector with a whole-trace `bge-m3` embedding gives a similar-looking matrix (Pearson r = 0.59 across the 30 off-diagonal entries) — moderate agreement, not identical.
- **Why a similarity matrix + dendrogram**: the question is "who resembles whom", which a similarity matrix answers directly, and clustering with **no labels** is stronger evidence for a brand effect than a classifier that was told the labels (Fig 5).
- **Read it as**: the tree's two top branches are exactly {Anthropic} and {OpenAI}. Within-brand similarities are 0.90–0.93; cross-brand 0.86–0.91 — the ranges touch (haiku–luna = 0.909), so it's a tendency, not a wall. Remember the raw values are inflated because all traces discuss the same problem.

### Figure 4 — UMAP of residual embeddings
See [§4](#4-umap-in-depth) for the full explanation and the justification.

- **Data**: 282 points (47 problems × 6 models), 1,024 dimensions → 2.
- **Axes**: UMAP-1 and UMAP-2 have **no units and no meaning individually**; only which points are near each other matters. Color = brand, shape = tier (circle small, square mid, triangle large).

### Figure 5 — How separable are brand and size from the text alone?

- **Question**: how well can a simple classifier tell brand (or size tier) from a trace, on problems it hasn't seen?
- **Data**: 282 rows, 47 problem-groups. Inputs: (a) the 1,024-d *residual* embedding; (b) 17 lexical/structure features (`n_words`, `n_sentences`, bold-header count, code-block count, bullet count, `is_summary`, `truncated`, and the per-100-word rates of hedging, contrast, commitment, meta-format, first-person, question, grounding, bullets, code blocks). Targets: brand (2 classes, baseline 50%); size tier (3 classes, baseline 33%).
- **Axes**: x = which (target, input) pair; y = **cross-validated accuracy (%)**: the share of held-out traces classified correctly (mean over 5 folds). Black tick = the majority-class baseline (always guess the biggest class). Number above = permutation p (the 0.005 floor, see §2.6).
- **Math**: `LogisticRegression` (L2 penalty, default strength) on standardized inputs; `GroupKFold(5)` by problem; permutation test with 200 shuffles.
- **Assumptions**: problems are independent groups; a *linear* boundary is adequate; labels are balanced (they are: 3 models per brand, 47 traces each).
- **Why a probe instead of eyeballing a plot**: "can a classifier predict X from the text on unseen problems?" is a number with a chance baseline and a significance test; a scatterplot isn't.
- **Read it as**: brand ≈ 100% — but the biggest input is `is_summary`, so this measures the format confound, not subtle style. "Size tier" 65% (embeddings) / 55% (lexical) vs. 33% chance.
- **[ISSUE] the pooled size number hides two opposite stories.** I ran the same probe *within* each brand: Anthropic's three models → **93%** (chance 33%), opus vs. sonnet alone → **91%** (chance 50%); OpenAI's three models → **52%** (chance 33%). Claude's models are very distinguishable from each other; OpenAI's are close to interchangeable. Also, "size tier" here is really *which specific model* it is: there's exactly one model per brand × tier, so tier and model identity can't be separated.
- **[ISSUE, minor]** the inputs were standardized using all 282 rows before cross-validation (the scaler saw held-out rows). It's unsupervised so the leak is small, but strictly it should be fit inside each fold. No CI is drawn for accuracy (only the permutation p).

### Figure 6 — Style profile per model

- **Question**: how do specific, interpretable style features differ by model, with uncertainty?
- **Data**: 515 traces (100, 92, 100, 83, 66, 74 for haiku, sonnet, opus, luna, terra, sol).
- **Axes**: one panel per feature; x = model (c- = claude, g- = gpt); **y = mean rate per 100 words** (panels have different scales on purpose). Whiskers = 95% bootstrap CI.
- **Math**: for each trace, `rate = 100 × (# words/phrases matching the feature's word list) / (words in the trace)`. Bar = mean of the model's per-trace rates. Whisker = 2.5th–97.5th percentile of 500 resampled means (resample a model's traces with replacement). Feature definitions and word lists: [§5](#5-lexical-features-and-reasoning-moves-decoded).
- **Assumptions**: word lists capture the concept; a trace is one independent observation; each trace counts equally regardless of length (so short traces make noisy rates).
- **Why small multiples**: eight features on different scales can't share an axis; identical layout per panel lets you compare models within each feature.
- **Read it as**: OpenAI's summaries hedge ~5–12× more (2.4–2.9 vs. 0.2–0.5 per 100 words), use first-person words ~1.7–3.3× more (5.6–5.9 vs. 1.8–3.4), and mention format/instructions more (0.7–0.9 vs. 0.2–0.4); Anthropic text is far more grounded in the prompt's actual vocabulary (30–35% vs. ~21%). Robust to problem mix (checked on the 47 shared problems).
- **Two features are basically uninformative here**: `question_rate` is ~0 everywhere (opus/sonnet exactly 0), and `bullets` is non-zero *only* for haiku — a formatting habit, not a reasoning trait. (The OpenAI `commitment_action` bars were understated by a curly-apostrophe bug until 2026-09-26; now fixed they are 1.77–1.86 — roughly equal to Anthropic's 1.37–1.88.)

### Figure 7 — Which words distinguish the groups? ("Fightin' Words")

- **Question**: which specific words does one group use unusually often relative to the other?
- **Data**: brand panel: Anthropic corpus 92,427 words (haiku alone is 59% of it) vs. OpenAI 60,568 words; 1,991 distinct words with ≥3 total uses. Size panel: haiku 54,701 words vs. opus 12,574; 1,253 distinct words.
- **Axes**: each dot is one word. **x = log₁₀(total uses in both groups + 1)** (2 = ~100 uses, 3 = ~1,000 — more common to the right). **y = z-score**: positive = over-used by group A (blue), negative = over-used by group B (orange); 0 = used equally often. Colored/labelled dots are the 12 most extreme on each side.
- **Math** (Monroe, Colaresi & Quinn 2008). For word `w`: `yA, yB` = counts in each group, `nA, nB` = total words in each group. Add a "prior" that pulls rare words toward the pooled rate: `α_w = 500 × (yA + yB) / (nA + nB)`, `α₀ = 500`. Then
  `logodds_A = log[ (yA + α_w) / (nA + α₀ − yA − α_w) ]` (same for B), `δ = logodds_A − logodds_B`, `variance ≈ 1/(yA + α_w) + 1/(yB + α_w)`, **`z = δ / √variance`**.
- **Why not just compare frequencies?** Raw differences are dominated by common words ("the"). The plain log-odds ratio explodes for rare words (a word used 3 times vs. 0 looks "infinitely" distinctive). The prior shrinks rare words toward "no difference", and dividing by the standard error turns δ into "how much evidence is there for a difference". That's the **funnel shape**: a word used ~10 times physically can't reach a big |z|; only frequent words can.
- **Assumptions**: words are independent draws from each group's vocabulary (false but standard); words are lowercase runs of `[a-z']`; no stop-word removal (which is why "the/to/it/a" appear).
- **Why this figure**: a plain word-cloud can't show direction or evidence; this ranks words by both. Treat z as a *ranking*, not a p-value (thousands of words are tested).
- **Read it as**: Anthropic → schema vocabulary (`department, subject, name, code, offered, course`); OpenAI → hedge/narration words (`might, seems, if, could`) and first-person `i`. Haiku vs. opus: haiku names schema objects (`code, subject, sis`), opus writes prose about the query (`the, a, query, i'm`).
- **[ISSUES]** (1) Verbose traces dominate the pooled counts (haiku = 59% of the Anthropic side). (2) *Fixed 2026-09-26:* a curly-apostrophe bug had inflated OpenAI's `i` (z = −39.4 → −32.7 after unifying apostrophes; the spurious `ll`/`m` tokens are gone; and because OpenAI's contractions are no longer split, `i'm` (z = −13.8) and `i'll` (−5.9) now also read as OpenAI-distinctive). The top-5 word lists on both sides did not change. (3) *Fixed 2026-09-26:* the right panel now uses haiku's and opus's own tier colors (light/dark blue) instead of orange. (4) Labels are now nudged apart with leader lines; a few in crowded areas still touch.

### Figure 8 — Mix of reasoning moves

- **Question**: what share of each model's sentences does each of 8 kinds of reasoning step make up?
- **Data**: 9,226 sentences: haiku 4,071, sonnet 910, opus 460, luna 1,781, terra 786, sol 1,162, human 56 (from 5 traces).
- **Axes**: x = model (human last); y = share of that model's sentences (0–100%); each of the 8 colored segments = one move; segments sum to 100%. (Colors here encode *moves*, not brand — a different palette from the other figures.)
- **Math (labeling pipeline, no LLM prompting)**: (1) split traces into sentences (`pysbd`); (2) *regex seeds*: a sentence gets a label only if cue phrases of exactly one move match (787 sentences); (3) *zero-shot NLI seeds*: a small entailment model (`deberta-v3-large-zeroshot-v2.0`) scores how well each of 8 descriptions ("planning an approach…") fits the sentence; accept the top move if its score ≥ 0.55 and it beats the runner-up by ≥ 0.15 (3,974 sentences); → 4,761 of 9,226 sentences (52%) have a seed label; (4) train logistic regression (`class_weight="balanced"`) on the seed sentences' embeddings and predict all 9,226. Agreement with **your 200 hand labels, scored against the current classifier: Cohen's κ = 0.69** (`κ = (p_observed − p_chance)/(1 − p_chance)`; raw agreement 73.8%, n = 195 — 5 rows couldn't be matched to the current run). *Earlier versions of this guide said 0.94. That number compared your labels with the predictions stored in the CSV at export time (the pre-fix classifier), not with the classifier behind the figures.* Per move (F1): interpret_ambiguity 0.82, meta_format 0.78, plan 0.77, execute 0.77 (precision 1.00 but recall 0.63), hypothesize 0.74, verify 0.69 (recall 0.59). The classifier predicts `self_correct` for 12 of your 200 sentences (you labeled 1) and `restate` for 5 (you labeled 0). κ excluding both rare moves: 0.76 (n = 183).
- **Assumptions**: each sentence is exactly one move; moves are mutually exclusive; bars are *sentence*-weighted, so verbose traces count more.
- **Why 100% stacked bars**: the question is about *composition* (parts of a whole) across a handful of models; stacked bars with a shared baseline make the shares comparable, and the last bar puts the human on the same scale.
- **Read it as**: opus/sonnet ≈ 58–60% `execute` — likely a length effect (their traces are ~4–5 sentences, and short traces are mostly "writing the query"). Humans: 30.4% `restate` (17 of 56 sentences; 95% Wilson interval **20%–43%**) vs. ≤ 6.7% for any model. Even the low end of the human range is ~3× the highest model. *My own spot check of those 17 sentences (not a substitute for your labels):* ~13 read as genuine restatements of the question and ~4 look like `plan` ("From the dept table I can get…"), i.e. ~23% — still ~3.5× the highest model.
- **[ISSUES]** (1) The 200-sentence validation sample was drawn from *model* sentences before humans were added, so the classifier's accuracy on **human** sentences has never been checked. (2) `restate` never appeared in that sample and `self_correct` appeared once, so those two moves are the least validated. (3) The human data is 5 explanations, presumably by one writer — "humans restate more" is a claim about this writer on these 5 problems; note also that writing an *explanation* invites restating the question in a way a model mid-generation wouldn't.
- **[ISSUE, added 2026-09-26] the classifier is less accurate than the earlier κ suggested, and it over-predicts the rare moves.** Shares of the four large moves are reasonably trustworthy at the model level; `restate` and `self_correct` shares (Fig 8 segments; the `self_correct` result in Fig 12) are unvalidated and probably inflated. Two further cautions on the 200-sentence sample: it was drawn using the *old* classifier's predictions (so it is not a fair test of the two rare moves), and the old predictions were visible next to the label column while you labeled (possible anchoring: on the 48 sentences where the old and current classifiers disagree, your label matched the old prediction 44 times and the current one once). A fresh sample drawn from the *current* classifier, with the prediction column hidden, would settle this.

### Figure 9 — Move transitions and JS divergence

- **Question**: does one move tend to follow another in a model-specific way, and how different are models' overall move mixes?
- **Data**: consecutive-sentence pairs within a trace: haiku 3,971, luna 1,698, sol 1,088, sonnet 818, terra 720, opus 360 (human 51, not plotted as a matrix — too sparse for 64 cells).
- **Axes**: small panels: row = the current sentence's move, column = the next sentence's move (labels shortened to 4 letters); cell = **P(next = column | current = row)**, each row sums to 1; darker = more likely (shared 0–1 scale). Right panel: a model × model matrix of Jensen–Shannon divergence.
- **Math**: `P(B|A) = count(A→B) / count(A→anything)`. `JSD(P, Q) = ½·KL(P‖M) + ½·KL(Q‖M)`, `M = (P+Q)/2`, `KL(P‖M) = Σ P(x)·log₂(P(x)/M(x))`. Base 2 → JSD runs from 0 (identical move mixes) to 1 (no overlap). (The code uses the squared `scipy` Jensen–Shannon *distance*, which equals this divergence.)
- **Assumptions**: only the immediately preceding sentence matters (first-order Markov); cells with few counts are noisy (each row of opus's matrix rests on a handful of transitions).
- **Why**: Fig 8 shows *how much* of each move; transitions show *order* (e.g. "plan → execute"). JSD compresses the 8 shares into one comparable number per pair, and is symmetric and bounded, unlike raw KL.
- **Read it as**: the three OpenAI models are nearly identical (JSD 0.001–0.005); Anthropic–OpenAI pairs 0.22–0.29; **human vs. every model 0.17–0.38** — the most distinct row.

### Figure 10 — Trajectory over the trace

- **Question**: does reasoning drift away from the prompt, and how big are the semantic jumps, as a trace progresses?
- **Data**: all traces with ≥ 2 sentences (3,797 trace-by-position-bin means).
- **Axes**: x = normalized position (0 = first sentence, 1 = last; sentence `i` of `n` sits at `i/(n−1)`, grouped into 10 equal bins). Left y = mean cosine similarity between a sentence and the whole prompt. Right y = **step size** = `1 − cos(sentence i, sentence i+1)` (bigger = a bigger semantic jump).
- **Assumptions**: relative position within a trace is comparable across traces of very different length; whole-prompt similarity measures "staying on topic".
- **Read it as**: it does **not** split by brand. opus/sonnet have the highest prompt similarity (~0.60–0.68) and the smallest early steps (~0.28–0.30, rising late); haiku and the OpenAI models are lower on similarity (~0.54–0.61) with larger, steadier steps (~0.35–0.42).
- **[ISSUES]** (1) **The embedder reads only the first 512 tokens; the prompt is ~13,000 tokens** (I measured 13,159 on one example → 3.9% seen). "Similarity to the prompt" therefore mostly means "similarity to the generic instruction boilerplate at the start", not to the question or the schema. Treat the left panel as weak evidence. (2) No CIs; opus has a median of 4 sentences spread over 10 bins (about half a sentence per bin per trace), so its lines are noise-driven. (3) Earlier versions of the report described this figure as splitting by brand; that was wrong and is corrected.

### Figure 11 — Revisit rate vs. correctness

- **Question**: do traces that loop back to earlier points tend to be more or less correct?
- **Data**: revisit rate needs ≥ 3 sentences, so 31 opus and 16 sonnet traces drop out. Anthropic: 196 incorrect (89 non-zero) + 49 correct (14 non-zero); OpenAI: 187 incorrect (20 non-zero) + 36 correct (0 non-zero).
- **Axes**: x = correct/incorrect within each brand panel (tick labels give n and the share of traces with any revisit at all); y = **revisit rate** per trace = (# sentences whose most similar *non-adjacent earlier* sentence has cosine > 0.884) / (# sentences from the 3rd on). Each dot is one trace (jittered sideways so they don't overlap); the black diamond is the mean. Threshold 0.884 = the 90th percentile of that similarity across the first 60 traces (a tuned choice; by construction ~10% of sentences in that sample exceeded it).
- **Read it as**: Anthropic: the *means* are similar — 0.048 (incorrect) vs. 0.041 (correct) — but the **share of traces with any revisit is 45% (89/196) for incorrect vs. 29% (14/49) for correct**. A naive two-proportion test on that gap gives p ≈ 0.03, but it ignores that traces of the same problem are correlated and is not part of the corrected 20-feature battery; the continuous revisit-rate feature shows nothing there (p_BH = 0.96). Treat it as a hint. OpenAI: correct = all zeros (0 of 36); 20 of 187 incorrect traces are non-zero.
- **[Redrawn 2026-09-26]** violins were replaced by one dot per trace (a violin's smoothed density misrepresents data that is mostly exactly zero), and the red/green pair (not colorblind-safe) by brand colors. Remaining limits: only 20 non-zero OpenAI traces exist to compare, and traces under 3 sentences are excluded. Earlier text claiming "some separation by correctness" was unsupported and is corrected.

### Figure 12 — Which features predict correctness?

- **Question**: after accounting for brand and size, does any style feature predict a correct answer?
- **Data**: 515 traces (484 for `mean_step_size`, 468 for `revisit_rate`, 464 for final-sentence similarity). 20 features, 20 separate models.
- **Axes**: y = feature, with its standard deviation (what "+1 SD" means for that row); **x = odds ratio per +1 SD of the feature, log scale**; 1 (vertical line) = no relationship, left of 1 = lower odds of a correct answer; whiskers = 95% bootstrap CI (200 resamples of problems). Red = BH-adjusted p < 0.05, gray = not. *Redrawn 2026-09-26:* the earlier version plotted the raw coefficient β (per +1 unit), which made rows incomparable (see below). `question_rate_per_100w` is omitted because its bootstrap CI failed to compute.
- **Math**: for feature `f`: `logit P(correct) = β₀ + β_f·f + β_brand·1[OpenAI] + β_tier terms`, fit by GEE (exchangeable correlation within a problem; §2.8). p-values from the Wald test; BH-adjusted across the 20 features (§2.10). Plotted value = `exp(β × SD_f)`; the CI endpoints transform the same way (the mapping is monotone, so percentiles carry over).
- **Assumptions**: log-odds is linear in the feature; one feature at a time (not jointly); traces of the same problem are equally correlated; the traces we have are representative (they aren't: OpenAI's missing traces correlate with correctness).
- **Why a forest plot**: the reader wants magnitude *and* uncertainty for many effects at once; a dot with a whisker per row does that, and the zero line makes "is there an effect?" a visual test.
- **Read it as**: only `hedging_per_100w` is significant: β = −0.386 → **OR = 0.68 per +1 hedging word per 100 words** (95% CI 0.51–0.84). In plainer units: hedging has mean 1.32 and SD 1.43, so a trace one SD hedgier has **0.57× the odds** of being correct (CI 0.38–0.78).
- **Why per-SD**: β means "per +1 unit of the feature", and features have very different natural ranges (hedging SD = 1.4 words/100; move rates and revisit rate SD ≈ 0.06–0.33; grounding SD = 7.8), so raw coefficients aren't comparable across rows — a move rate's "+1" is a 100-percentage-point change, which made those rows look enormous with huge whiskers in the earlier version. The plotted per-SD odds ratios, same models:

  | feature | SD | odds ratio per +1 SD [95% CI] | p_BH |
  |---|---|---|---|
  | hedging_per_100w | 1.44 | **0.57** [0.38, 0.78] | 0.031 |
  | grounding_rate_per_100w | 7.77 | 0.76 [0.56, 1.00] | 0.34 |
  | move_self_correct_rate | 0.065 | 0.76 [0.60, 0.98] | 0.35 |
  | contrast_self_correction_per_100w | 0.79 | 0.82 [0.63, 0.99] | 0.35 |
  | move_verify_rate | 0.116 | 0.84 [0.67, 1.04] | 0.45 |
  | (the other 15 features) | | 0.89 – 1.20; every computable CI includes 1 (one CI, `question_rate`, failed to compute) | ≥ 0.58 |

  In these units the hedging effect is visibly the largest and the only one to survive correction. Several others (grounding, self-correct) lean the same direction but don't survive.
- **[ISSUE] problem difficulty is not controlled.** Hard problems may cause both more hedging *and* more errors, in which case hedging predicts failure without causing it. The model clusters by problem but doesn't include a per-problem effect. This is an association, not a mechanism.

### Figure 13 — Which reasoning steps line up with the human's?

- **Question**: concretely, which parts of a model's reasoning correspond to the human's steps?
- **Data**: 5 human explanations (8, 8, 16, 13, 11 sentences) against the model with the highest coverage (haiku: 22–37 sentences) and the lowest (terra: 5–6 sentences).
- **Axes**: y = human sentence index, x = model sentence index; **cell color = cosine similarity** between that human sentence and that model sentence (0 = white, 1 = dark blue).
- **Assumptions/choices**: haiku and terra were picked *after* seeing which had the best/worst coverage (post hoc). Panels have different widths because the models write different numbers of sentences.
- **Read it as**: dark cell = a close semantic match; a row that is dark somewhere = that human step is "covered". Terra's panels are more uniform and lighter.

### Figure 14 — Coverage vs. extra

- **Question**: which models cover the human's points, and which add content the human never mentioned?
- **Data**: 30 points (5 problems × 6 models); large marker = mean of a model's 5.
- **Axes**: x = **coverage** = share of the human's sentences that have at least one model sentence with similarity > τ. y = **extra** = share of the model's sentences with *no* human sentence above τ.
- **Math**: from each similarity matrix `S` (human × model): `coverage = mean_i [max_j S_ij > τ]`, `extra = mean_j [max_i S_ij ≤ τ]`. **τ = 0.703** = the 90th percentile of all human×model sentence-pair similarities pooled over the 30 matrices (so ~10% of all pairs count as matches by construction). It is *not* the Fig 11 threshold: reusing 0.884 gave 0% coverage everywhere, because cross-author paraphrases are less similar than within-trace near-repeats.
- **Also reported in text**: ranking models by raw mean-pooled trace-embedding similarity to the human trace: haiku is #1 on all 5 problems (0.92–0.94 vs. 0.77–0.91 for others).
- **Sensitivity (I checked)**: coverage by threshold —

  | model | τ=0.60 | 0.65 | **0.703** | 0.75 | 0.80 |
  |---|---|---|---|---|---|
  | haiku | 0.98 | 0.93 | **0.84** | 0.61 | 0.26 |
  | sonnet | 0.97 | 0.84 | **0.56** | 0.29 | 0.08 |
  | opus | 0.87 | 0.83 | **0.58** | 0.31 | 0.08 |
  | luna | 0.94 | 0.82 | **0.49** | 0.23 | 0.04 |
  | terra | 0.83 | 0.58 | **0.21** | 0.13 | 0.05 |
  | sol | 0.80 | 0.62 | **0.49** | 0.17 | 0.05 |

  Haiku is #1 at every threshold — that ranking is robust. Terra is last at τ = 0.65–0.75 (at 0.60 sol is last); at 0.80 nearly everything is close to zero (luna 0.04, terra/sol 0.05) and the ordering means little. **The absolute numbers are not**: haiku's "84% coverage" is 98% at τ = 0.60 and 26% at τ = 0.80. Use the ordering, not the percentages.
- **[ISSUES]** coverage depends on length (Spearman ρ = 0.46 between a model's sentence count and its coverage across the 30 pairs); n = 5 problems and one annotator; descriptive only, no significance test.

---

## 4. UMAP in depth

### The problem it solves
Each trace is a point in 1,024 dimensions. You can't look at 1,024 dimensions. You need a 2-D picture in which points that are close in the real space are close on the page.

### What UMAP does (intuition, then a little math)
1. **Build a neighborhood graph.** For every point, find its 15 nearest neighbors in the original space (using cosine distance) and connect them, with edge weights that fade with distance (weights are calibrated per point, so dense and sparse regions are treated fairly).
2. **Find a 2-D layout of the same graph.** Start with random 2-D positions, then repeatedly nudge points: pull connected points together, push unconnected points apart, until the 2-D neighborhood graph resembles the original one. Formally it minimizes a cross-entropy between the high-dimensional and low-dimensional neighbor-probabilities.

The layout is *stochastic* (random start), so we fix the seed (`random_state=42`) for reproducibility. Settings: `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`.

### What the axes mean
Nothing, individually. UMAP-1 and UMAP-2 are just coordinates the optimizer ended up with; the whole picture could be rotated or flipped and mean the same thing. (A live demonstration: the figure was regenerated in the 2026-09-26 re-run and its coordinates are completely different from the archived version — e.g. haiku's cluster moved from roughly (10, −4) to (21, 0) — while the three-cluster structure is identical.) Only *who is near whom* is meaningful.

### What you can trust
- **Membership**: points that sit together are near neighbors in the original space; tight, well-separated groups are real groups.
- **Not**: how far apart two clusters are; how big or spread out a cluster looks; the shape of a cluster. UMAP stretches and squeezes space non-uniformly, so a cluster looking "tighter" than another proves nothing. (An earlier version of my report claimed OpenAI's cluster was tighter — that was exactly this mistake.)
- **Not**: that overlap in 2-D means the groups are the same. Squashing 1,024 dimensions into 2 hides differences: opus and sonnet look mixed in Fig 4, but a classifier separates them 91% of the time in the full space.

### Why UMAP rather than something simpler?
The plain alternative is PCA (project onto the two directions of greatest variance). PCA is linear and preserves large-scale variance, but for these data the top two directions explain only **31.5%** of the variance (PC1 21.0%, PC2 10.5%) — a lot is thrown away. UMAP instead tries to preserve *local neighborhoods*, which is what "which traces are like which" is about, and it usually shows cluster structure more clearly.

### Why we can defend the picture: we checked it against the real space
UMAP is easy to over-read, so it is *illustration, not evidence* here. The evidence is non-visual (Fig 3's unsupervised clustering, Fig 5's classifier). I also tested whether the 2-D picture keeps the real neighbor structure, by asking: *for each point, what fraction of its 10 nearest neighbors have the same label as it?* (a "neighbor purity"; chance shown for reference)

| label | chance | in the full 1,024-D space | in the UMAP 2-D picture | in a PCA 2-D picture |
|---|---|---|---|---|
| brand | 0.50 | 1.00 | 1.00 | 1.00 |
| which model (6-way) | 0.17 | 0.58 | 0.57 | 0.52 |

- Brand separation is 100% in the original space, in UMAP, *and* in plain PCA — it is a property of the data, not something UMAP invented.
- For finer structure, UMAP keeps 0.57 of the 0.58 neighbor purity, clearly better than PCA (0.52). (Model and size-tier purity are identical because, once brand is pure, each brand × tier is exactly one model.)
- Where the clusters come from: clustering the 2-D points into 3 groups recovers exactly {all three OpenAI models}, {opus + sonnet}, {haiku} — 47 points per model, no exceptions.

### Why residual embeddings for this figure
On raw embeddings, points would first group by *problem* (all traces about problem A are near each other regardless of model), drowning out model differences. Subtracting the per-problem average across models removes that shared content (§2.4), so what's left is how models differ *from each other* on the same problem. Because we subtract the mean of all six, each brand's points are pushed to opposite sides of the origin, which is part of why brand separation is so stark here.

### How to say it in a talk
"Each dot is one model's reasoning on one problem, converted to 1,024 numbers. UMAP squashes that to two so we can see who resembles whom. Only closeness matters, not the axes or the sizes. We use it to illustrate what the classifier and the clustering already established: the two brands are separable, and inside Anthropic haiku differs from opus/sonnet."

---

## 5. Lexical features and reasoning moves, decoded

There are **two separate systems** and it's easy to conflate them:

- **Lexical features (Fig 6, Fig 12)**: counts of words/phrases from short editable lists (`config/lexicons.yaml`), reported per 100 words. Simple, transparent, and blunt.
- **Reasoning moves (Figs 8–9, 12)**: a per-*sentence* classification into 8 categories (`config/move_cues.yaml` + an NLI model + a trained classifier). Slower and more interpretive, but understands sentences rather than isolated words.

### 5a. Lexical features (all "per 100 words")

Matching is whole-word/phrase, case-insensitive, on normalized text.

| feature | what it counts | word list | what it's *meant* to capture | caveats |
|---|---|---|---|---|
| **hedging** | tentative wording | might, may, could, seems, seem, perhaps, likely, possibly, probably, not sure, unsure, i think, i believe, it appears, appears to, presumably, uncertain | uncertainty in the reasoner's voice | "may" can be the month; "could/might" also appear in neutral phrasing. The **only** feature that predicts correctness (Fig 12). |
| **contrast_self_correction** | turns and reversals | however, but, yet, wait, actually, alternatively, hmm, on second thought, on the other hand, although, though, instead, rather, let me reconsider, let me re-check, i was wrong, correction, re-reading | "oscillation" — changing direction | **Broad**: "but/instead/though" are ordinary connectives, not corrections. Do not read it as self-correction; the *move* `self_correct` is the stricter measure. |
| **commitment_action** | committing to and doing | i'll, i will, let me, i need to, i have to, writing, so the answer, final, therefore, thus, so i'll, the final query, the final answer, final sql, going with | resolve/action language | Understated for OpenAI before the 2026-09-26 apostrophe fix (now 1.77–1.86). |
| **meta_format** | talking about the task's form | output, format, tags, instructions, the user, the prompt, the question asks, the task, `<ans>`, `</ans>`, wrap, wrapped, json object, markdown | attention to instructions/format rather than the problem | "output" also appears in SQL discussion. |
| **first_person** | self-reference | i, i'm, i've, i'll, i'd, me, my, myself | narration of the reasoner's own actions | Checked: the SQL alias `i.` is almost never present (≤2 cases), so it isn't inflating this. |
| **question_rate** | question marks | `?` (counted directly) | self-questioning | ~0 for every model; uninformative. |
| **grounding_rate** | words that also appear in the prompt | (not a word list) the share of a trace's words found in the prompt's vocabulary: lowercase words of ≥3 characters from the prompt, minus a small stop list | how much the reasoning uses the schema's/question's own terms | Proxy: common words present in a long prompt inflate it (20–35% baseline). |
| **bullets** | bullet-list lines | (structure) lines starting with `-`, `*` or `•` | list-style reasoning | Non-zero only for haiku — a formatting habit. |

### 5b. The eight reasoning moves

Each sentence gets exactly one label. Examples are the highest-confidence sentences the classifier produced (real data).

| move | what it is | typical cues | real examples |
|---|---|---|---|
| **restate** | putting the question into your own words before working on it | "the question is asking", "I need to find/determine…", "the user wants", "in other words" | (haiku) "First, I need to understand what the question is asking for:" · (human) "The question is asking for subject information regarding subjects in the math department during regular terms." |
| **interpret_ambiguity** | resolving something unclear in the question | "ambiguous", "not entirely clear", "could mean", "I'll assume", "assuming that" | (haiku) "The phrasing 'number of subjects offered by the department' is ambiguous but likely just means a row count per department…" |
| **plan** | laying out an approach before doing it | "I'll start by", "my approach is", "let me break this down", "the plan is", "steps:" | (luna) "I'm considering various ways to structure it efficiently." · (sol) "I'm considering how to handle the combination of needs for each subject code." |
| **hypothesize** | a tentative guess not yet committed to | "maybe", "perhaps", "could be", "might be", "what if", "suppose" | (luna) "It seems like this would help narrow down the results." · (sonnet) "That suggests a simple cross join between subq1 and subq2 might actually be what's expected…" |
| **verify** | checking something against the requirements | "let me check/verify/confirm", "double-check", "make sure", "sanity check" | (luna) "I want to check if that's valid in MySQL." · (terra) "I need to ensure I'm providing valid MySQL code." |
| **self_correct** | catching and fixing an earlier mistake | "wait,", "actually,", "on second thought", "I made a mistake", "that's wrong", "let me redo" | (haiku) "Wait, looking at the mapping more carefully:" · (sonnet) "Maybe I misidentified the column entirely, and 'Harvard Cross-Enrollment Program' should actually map to OFFER_SCHOOL_NAME instead." |
| **meta_format** | talking about the answer's form/instructions, not the problem | "the output should", "wrap … tags", `<ans>`, "the instructions say", "formatted as" | (luna) "The developer's instructions are clear: I need to wrap the output with just tags." |
| **execute** | actually writing or finalizing the SQL | "so the final query is", "here's the query", "final answer:", "```sql" | (sonnet) "Finishing the query by joining the two subqueries on department name to bring together the enrollment standard deviation and room area statistics." |

**How the two systems relate**: `hedging` ≈ words that often occur in `hypothesize` sentences; `meta_format` (lexical) ≈ the `meta_format` move; `commitment_action` ≈ `plan`/`execute`; `contrast_self_correction` is a loose cousin of `self_correct`. They're computed independently, so they can disagree.

**What each tells you about a trace**: a trace heavy in `plan`/`hypothesize`/`verify` is *narrating a process*; heavy in `execute` is *mostly writing the answer*; heavy in `meta_format` is *thinking about the instructions*; `restate` is *framing the problem first*. Whether a trace is "good" is not what these say — only Fig 12 asks that, and only hedging shows up.

**Known weak spots of the moves**: a hedge word can push a sentence toward `hypothesize` even when it's really commentary (e.g. "It seems like this would help narrow down the results" above); `restate` and `self_correct` are rare (each ~2–7% and ~2–10% of a model's sentences) and the least validated — and over-predicted by the current classifier (κ = 0.69 overall; see Fig 8).

---

## 6. Issues found, and their status (updated 2026-09-26)

**Statements I had made that the data didn't support — corrected in the reports ✔**
1. ✔ Fig 4: "OpenAI's cluster is tighter than Anthropic's." The figure shows three clusters (haiku alone; opus+sonnet; all OpenAI intermixed), and cluster tightness isn't meaningful in UMAP anyway.
2. ✔ Fig 10: "shows brand-consistent patterns." It doesn't; it separates the terse Anthropic models (opus/sonnet) from everything else. I hadn't looked at the figure before describing it.
3. ✔ Fig 11: "shows some separation by correctness." Not in the means (Anthropic 0.048 vs 0.041). The redrawn figure does show a difference in the *share of traces with any revisit* (45% vs 29%), which is a hint only.
4. ✔ Fig 3: within/cross-brand similarity ranges were misquoted (correct: 0.90–0.93 vs. 0.86–0.91; they touch).
5. ✔ Fig 2: median lengths (~94 / ~567) came from raw text; the figure plots normalized text (~93 / ~488).
6. ✔ **Move classifier: "agreement held at κ=0.94 after the class-weight fix."** This was never tested. The agreement check compared your labels with the predictions stored in the validation file at export time (the pre-fix classifier); that is why κ printed as exactly 0.9400569… on every rerun. Scored against the current classifier it is **κ = 0.69** (details under Fig 8). The check now scores against the current run's predictions and also prints the export-time κ for transparency.

**Fixed in the 2026-09-26 re-run ✔** (old outputs in `outputs/archive_20260925/`)
7. ✔ **Curly vs. straight apostrophes.** OpenAI's text uses typographic apostrophes (`I’ll`) for ~40% of its apostrophes; Anthropic's never does. The normalizer now unifies them. Measured effect: OpenAI's `i` z-score in Fig 7 −39.4 → −32.7 (spurious `ll`/`m` tokens gone; `i'm`/`i'll` now register as OpenAI-distinctive); OpenAI's `commitment_action` ~1.5 → ~1.85. Nothing else moved materially (correctness results, similarity matrix, probes, move shares, JS divergences and human-comparison numbers changed by ≤ 0.5 percentage point or by rounding; the top-5 distinctive-word lists are identical; hedging and first-person are unchanged).
8. ✔ Fig 1: accuracy CIs are now drawn. Fig 7 (right): colors no longer clash with the brand colors. Fig 11: dots instead of violins, no red/green. Fig 12: odds ratio per +1 SD instead of raw coefficients.

**Open — the largest one**
9. **The move labels are less reliable than the earlier κ implied** (κ = 0.69; 0.76 without the two rare moves). Shares of the four large moves are reasonably trustworthy at the model level; `restate` and `self_correct` are over-predicted and unvalidated; the human sentences were never validated (my own spot check puts the human `restate` share nearer 23% than 30%, still ~3.5× the highest model). Your labeling sample was drawn from the old classifier's predictions and shown beside the label column (possible anchoring). **Recommended:** a fresh ~150-sentence sample drawn from the *current* classifier — including ~30 each predicted `restate` and `self_correct` and ~30 human sentences — with `predicted_move` hidden from the labeler, then rescore.

**Open — methodological weaknesses (not changed)**
10. **The 47-problem shared subset is harder** (16% vs 29% mean accuracy; more nested/multi-CTE problems). Figs 3–5 and the human comparison describe behavior on that subset (§1).
11. **Fig 10 (left)**: the prompt is truncated to 512 of ~13,000 tokens by the embedder; no CIs on either panel.
12. **Fig 12**: no control for problem difficulty (hedging may track hard problems rather than cause errors).
13. **Fig 5**: pooled "size" accuracy hides Anthropic 93% vs. OpenAI 52%; the scaler was fit before cross-validation; tier ≡ specific model.
14. **Fig 14**: absolute coverage is threshold-dependent (the ranking is robust; the percentages aren't) and correlates with model trace length (ρ = 0.46).
15. **Fig 6**: `question_rate` and `bullets` panels carry almost no information.
