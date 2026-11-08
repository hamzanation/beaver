# BEAVER dw pre-test

A 15-question subset of the 100 `dw` dev_sampled questions. Run it on a model **before**
spending a full run, to find out:

1. **Which hint setting the model is ready for**: the fewest hints (setting 2 → 1 → 0)
   at which it still scores clearly above the floor.
2. **Roughly what it will score** on the full 100 at that setting (±2–6 points).
3. **Whether the harness will under-measure it**: truncated answers, SQL that won't
   execute, API errors. These are the failures that made Sol, Luna and Haiku look
   weaker than they are.

It never runs automatically. You run it by hand, per model, when you want to.

## How to run it

```bash
cd eval/pretest
# same env vars you'd use for the full run (BEAVER_REASONING_EFFORT, BEAVER_THINKING_EFFORT, ...)
PYTHON=../../beaver-eval/Scripts/python.exe ./run_pretest.sh --model <model> --setting 2
```

This makes 15 LLM calls (about 1/7 the cost of a full run), scores them against local MySQL,
and prints a verdict. The report is also saved to `pretest_report.txt` inside the run directory
under `eval/unified-output/fewshot/pretest/`. Pre-test runs go in that `pretest/` subfolder so
the comparison and reasoning-analysis scripts never mistake them for full runs.

**The ladder:** start at setting 2 (all hints). If it says READY, run again at setting 1. If
that says READY, run at setting 0. The model's setting is the last READY one. A model taken
all the way down costs at most 45 calls.

## What the verdict means

| Verdict | Rule | What to do |
|---|---|---|
| **READY** | ≥ 3 of 15 correct (predicted full-100 accuracy ≥ 10%), health checks pass | Full run at this setting is worth it; try the next setting down |
| **NOT READY** | ≤ 2 of 15 correct | A full run would land at 2–6%, where models can't be told apart. Use the next setting up (more hints) |
| **FIX HARNESS FIRST** | Any answer truncated at the token cap, > 20% of SQL fails to execute, or any API error | The model isn't being measured fairly yet. Fix that first, then re-run the pre-test |

**Predicted accuracy = 0.505 × (correct / 15).** Each pre-test question is worth about 3.4
points of full-100 accuracy. 0.505 is the least-squares scale factor fitted on the six
calibration models. The 80% interval comes from the binomial uncertainty of 15 questions.

### If a health check fails

- **Truncated at token cap.** The model ran out of `max_tokens` while thinking, before it
  wrote SQL. Raise `BEAVER_MAX_TOKENS` (Anthropic: it must exceed `BEAVER_THINKING_BUDGET_TOKENS`
  with room left for the answer; 16384 fixed Haiku). This was Haiku 4.5's problem: 28/100
  answers were cut off at 5,120 tokens.
- **SQL fails to execute.** Look at `pred_error` in the run's `summary_ex_acc.json`. A cluster
  of MySQL 1064 (syntax) errors usually means an output-format problem, not a reasoning one.
  Look at the raw response in `eval/fewshot/output/<run>/<id>/generation.log`.
- **API errors.** Credits, rate limits, or the model name.
- *Info lines (don't block):* "HTML-escaped SQL" is now unescaped automatically
  (`eval/sql_sanitize.py`). This was Sol's and Luna's problem. "Answered None" means the model
  chose to abstain, as the prompt allows. It counts as wrong.

## What's in it, and why

**Bands.** Each of the 100 questions is put in a band by how many of the 6 current-generation
models (Haiku 4.5, Sonnet 5, Opus 5, GPT-5.6 Luna/Terra/Sol) solved it at setting 2:
easy = 4–6, medium = 2–3, hard = 1, unsolved = 0. Out of 100: 22 easy, 12 medium, 16 hard,
50 unsolved.

**The 15 questions (6 easy / 5 medium / 3 hard / 1 unsolved):**

| id | band | solved by (6 current / all 19 runs) | gold SQL: joins, subqueries | why it's useful |
|---|---|---|---|---|
| dw_461 | easy | 6/6 · 15/19 | 1, 2 | Solved even at setting 0 by half the old models; baseline competence |
| dw_19 | easy | 5/6 · 14/19 | 3, 2 | Separates setting 0 from 1 (solved 25% → 80%) |
| dw_2669 | easy | 6/6 · 14/19 | 3, 1 | Same: 0% at setting 0, 80% at setting 1 |
| dw_5166 | easy | 5/6 · 13/19 | 1, 0 | Short query; tests following a function hint (STDDEV) |
| dw_663 | easy | 6/6 · 9/19 | 3, 2 | Easy for current models, missed by all older ones at settings 0/1 |
| dw_3585 | easy | 5/6 · 7/19 | 4, 2 | Easy now, rare for older models (0% at setting 0, 20% at setting 1) |
| dw_4961 | medium | 3/6 · 4/19 | 8, 3 | Most joins in the set |
| dw_140 | medium | 2/6 · 2/19 | 3, 1 | Separates the top models at setting 2 |
| dw_287 | medium | 2/6 · 2/19 | 3, 2 | ″ |
| dw_4351 | medium | 2/6 · 2/19 | 5, 5 | ″; long query (truncation canary) |
| dw_777 | medium | 2/6 · 2/19 | 2, 4 | ″ |
| dw_2097 | hard | 1/6 · 6/19 | 2, 2 | Old models solve it at setting 1 (60%), yet only 1 of 6 current models does at setting 2 |
| dw_5668 | hard | 1/6 · 4/19 | 2, 0 | Short query; solved at setting 0 by one old model |
| dw_4610 | hard | 1/6 · 1/19 | 6, 3 | Top-end discrimination |
| dw_4374 | unsolved | 0/6 · 0/19 | 4, 6 | Ceiling detector: a model that solves this is beyond the calibration set |

(Full per-question data for all 100 questions is in `item_bank.csv`.)

**Why this mix:**

- **Weighted toward easy and medium.** At settings 0 and 1, nearly the only questions any
  model solves are ones that are easy at setting 2. The easy questions carry the signal for
  "ready for fewer hints". The medium and hard ones separate strong models at setting 2.
- **Only one unsolved question.** Half the benchmark (50 questions) is unsolved by every model
  at every setting, so those questions say nothing about current models. One is kept so a
  future, stronger model shows up as such.
- **Format and truncation canaries.** 12 of the 15 have `<`/`>` in the gold SQL, which is what
  exposed the HTML-escaping bug. Two are in the longest quartile, which is what exposed
  truncation. The selection required at least 3 and at least 2 respectively.
- **How the exact 15 were chosen.** 3,000 random draws within the band quotas that met the
  canary constraints. Kept the one whose prediction best matched the 6 calibration models'
  real scores. This picks a somewhat better draw than random but doesn't guarantee one. The
  honest test is the holdout below.
- **Why a straight line and not something fancier.** I compared it against a Rasch (IRT)
  ability model and a band-weighted estimator. The line `0.505 × fraction correct` had the
  lowest error on held-out runs and is the easiest to reason about.

## How well it works

Validated on **13 runs that played no part in building it**: gpt-5.4-mini, gpt-5.5,
claude-haiku-4-5 (July run) and claude-sonnet-4-5 at settings 0, 1 and 2, plus
qwen2.5-coder-14b at setting 1. For each, I took only its answers on the 15 pre-test
questions, predicted its full-100 score, and compared that with the real score.

- Mean absolute error **2.2 points**, worst **5.8** (gpt-5.5 at setting 2: predicted 20%, actual 26%).
- Random 15-question subsets with the same estimator: median 4.7 points.
- The READY / NOT READY verdict matched the full-run result on **13/13** runs. All four
  old models were NOT READY at setting 0 (actual 2–6%). At setting 1, Haiku 4.5 and Qwen were
  NOT READY (actual 8–9%) and the rest were READY.

Per-run numbers are in `validation.csv`. Headline numbers are in `calibration.json`.

## Limitations

- **Small calibration set.** 6 models define the bands and the scale factor, and 13 runs
  validate them. Treat the ±2 points as approximate, not a guarantee.
- **Settings 0 and 1 are validated only on older models.** No current-generation model has
  had a full run at settings 0 or 1. The first time one does, check it against its pre-test.
- **Ceiling.** 15/15 maps to about 50%. A model clearly stronger than Opus 5 (the scorer warns
  at ≥ 13/15) will be under-predicted. Fold its full run in and rebuild.
- **Run-to-run noise.** One question changing is about 3.4 points. A model at exactly 3/15
  is borderline either way.
- **The 10% "ready" floor is a judgment call.** It sits above the 2–6% that every old model
  scored at setting 0. Change it in `calibration.json` (`readiness.min_predicted_accuracy`).
  The "≥ N of 15" rule moves with it.
- **dw and the fewshot baseline only.** Use the same model settings (effort, token caps) as
  the full run would. The pre-test only predicts runs made under the same conditions.

## Rebuilding

Once a new full run should count (for example a new model, or a current model at settings
0 or 1), score it with `--sanitize --output_name summary_ex_acc_sanitized.json`, add it to
`CALIBRATION_MODELS` in `build_pretest.py` if it is current-generation, and run:

```bash
cd eval/pretest && ../../beaver-eval/Scripts/python.exe build_pretest.py
```

Rebuilding can change the 15 questions. Pre-test results from an older question set aren't
comparable with ones from the new set, so keep `pretest_ids.json` under version control.

## Files

| file | what |
|---|---|
| `run_pretest.sh` | run + score + verdict for one model at one setting |
| `score_pretest.py` | the verdict logic (can be run alone on a scored run directory) |
| `build_pretest.py` | chooses the questions, fits the scale factor, validates |
| `pretest_ids.json` | the 15 question ids |
| `item_bank.csv` | band, solve counts per setting, gold-SQL features for all 100 questions |
| `calibration.json` | scale factor, readiness thresholds, validation summary |
| `validation.csv` | predicted vs actual for every run used |
