#!/bin/bash
set -e

# ============================================================================
# BEAVER dw pre-test: run the 15 questions in pretest_ids.json for one model at one
# setting, score them, and print a readiness verdict. Manual only -- nothing calls this.
# Usage: ./run_pretest.sh --model claude-opus-5 --setting 2
# Env vars (BEAVER_THINKING_EFFORT, BEAVER_REASONING_EFFORT, BEAVER_MAX_TOKENS, ...) are
# passed straight through to fewshot/run.sh -- use the same ones as the full run would.
# ============================================================================

MODEL=""
SETTING=2
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2";   shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run_pretest.sh --model <model> [--setting <0|1|2>]   (default setting 2)"
            echo "Start at setting 2; if READY, repeat at 1, then 0. See README.md."
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -z "$MODEL" ] && { echo "--model is required"; exit 1; }

PRETEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${PRETEST_DIR}/.." && pwd)"

# 1. Generate (15 calls) + unify, via the normal fewshot pipeline
bash "${EVAL_DIR}/fewshot/run.sh" --model "$MODEL" --dataset dw --setting "$SETTING" \
    --ids_file "${PRETEST_DIR}/pretest_ids.json" --tag pretest
RUN_NAME=$(ls -td "${EVAL_DIR}"/unified-output/fewshot/"${MODEL}"-beaver-dw-setting"${SETTING}"-pretest-log-* | head -1 | xargs basename)

# 2. Keep pre-test runs out of the top-level results the analysis scripts scan
mkdir -p "${EVAL_DIR}/unified-output/fewshot/pretest"
mv "${EVAL_DIR}/unified-output/fewshot/${RUN_NAME}" "${EVAL_DIR}/unified-output/fewshot/pretest/"
RUN_DIR="${EVAL_DIR}/unified-output/fewshot/pretest/${RUN_NAME}"

# 3. Score (sanitized, so format problems show up in the health check instead of as wrong answers)
cd "$EVAL_DIR"
"$PYTHON" evaluate_ex_acc.py --dataset dw --input_dir "$RUN_DIR" --sanitize > /dev/null

# 4. Verdict
cd "$PRETEST_DIR"
"$PYTHON" score_pretest.py "$RUN_DIR" | tee "${RUN_DIR}/pretest_report.txt"
