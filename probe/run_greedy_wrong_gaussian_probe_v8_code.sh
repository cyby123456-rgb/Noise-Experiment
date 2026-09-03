#!/usr/bin/env bash

# V8 single-token Gaussian probe for prepared code datasets.
#
# Required:
#   INPUT_PARQUET=/path/to/prepared_code.parquet
#
# The parquet must use a supported code data_source (apps, taco, codecontests,
# codeforces, livecodebench/*) and contain executable test cases in
# reward_model.ground_truth. A code answer is correct only when it passes all
# tests. Run this script only inside an isolated container: generated code is
# executed by the evaluator.

set -euo pipefail
set -x

# ================= Environment =================

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${EXPERIMENT_ROOT}/.git" ]]; then
    DEFAULT_REPO_ROOT="${EXPERIMENT_ROOT}"
else
    DEFAULT_REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/.." && pwd)"
fi

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
VERL_ROOT="${VERL_ROOT:-${EXPERIMENT_ROOT}/verl}"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-test123}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# ================= Experiment =================

if [[ -z "${INPUT_PARQUET:-}" ]]; then
    echo "Set INPUT_PARQUET to a prepared code parquet." >&2
    exit 1
fi
if [[ ! -f "${INPUT_PARQUET}" ]]; then
    echo "Input parquet not found: ${INPUT_PARQUET}" >&2
    exit 1
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-code-v8-pilot}"
DATASET_TAG="${DATASET_TAG:-$(basename "${INPUT_PARQUET}" .parquet)}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/model_merged/grpo-8k}"

LAYER="${LAYER:-24}"
NOISE_STD="${NOISE_STD:-0.1}"
NOISE_SCALE_MODE="${NOISE_SCALE_MODE:-relative_rms}"

# Every (question, response position, trial) receives a distinct Gaussian
# vector by default. BASE_NOISE_SEED still makes the full experiment exactly
# reproducible; it does not cause directions to be reused.
NOISE_SEED_MODE="${NOISE_SEED_MODE:-independent}"
NUM_NOISE_SEEDS="${NUM_NOISE_SEEDS:-32}"
BASE_NOISE_SEED="${BASE_NOISE_SEED:-20260827}"
NOISE_NAMESPACE="${NOISE_NAMESPACE:-${EXPERIMENT_NAME}:${DATASET_TAG}:v8-code}"
NOISE_BATCH_SIZE="${NOISE_BATCH_SIZE:-1}"

# Defaults are deliberately small enough for a first run. Set
# MAX_QUESTIONS=20 (or 30) for the positional pilot after a smoke test.
MAX_QUESTIONS="${MAX_QUESTIONS:-1}"
ROW_INDICES="${ROW_INDICES:-}"
REQUIRE_ALL_INPUT_ROLLOUTS_WRONG="${REQUIRE_ALL_INPUT_ROLLOUTS_WRONG:-false}"

RESPONSE_POSITION="${RESPONSE_POSITION:-}"
RESPONSE_POSITION_FRACTION="${RESPONSE_POSITION_FRACTION:-}"
NUM_RESPONSE_POSITIONS="${NUM_RESPONSE_POSITIONS:-10}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-16}"

FILTER_ARGS=()
if [[ "${REQUIRE_ALL_INPUT_ROLLOUTS_WRONG}" == "true" ]]; then
    FILTER_ARGS+=(--require-all-input-rollouts-wrong)
fi

ROW_ARGS=()
if [[ -n "${ROW_INDICES}" ]]; then
    ROW_ARGS+=(--row-indices "${ROW_INDICES}")
fi

POSITION_ARGS=()
if [[ -n "${RESPONSE_POSITION}" ]]; then
    POSITION_ARGS+=(--response-position "${RESPONSE_POSITION}")
    POSITION_TAG="pos${RESPONSE_POSITION}"
elif [[ -n "${RESPONSE_POSITION_FRACTION}" ]]; then
    POSITION_ARGS+=(--response-position-fraction "${RESPONSE_POSITION_FRACTION}")
    POSITION_TAG="frac${RESPONSE_POSITION_FRACTION}"
elif [[ -n "${NUM_RESPONSE_POSITIONS}" ]]; then
    POSITION_ARGS+=(--num-response-positions "${NUM_RESPONSE_POSITIONS}")
    POSITION_TAG="k${NUM_RESPONSE_POSITIONS}"
else
    echo "Configure one response-position selection mode." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/greedy_gaussian_w2r/${DATASET_TAG}-layer${LAYER}-${POSITION_TAG}-${NOISE_SEED_MODE}-${NOISE_SCALE_MODE}-std${NOISE_STD}-n${NUM_NOISE_SEEDS}-seed${BASE_NOISE_SEED}}"

python3 "${EXPERIMENT_ROOT}/probe/run_greedy_wrong_gaussian_probe_v8.py" \
    --input-parquet "${INPUT_PARQUET}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --layer "${LAYER}" \
    --noise-std "${NOISE_STD}" \
    --noise-scale-mode "${NOISE_SCALE_MODE}" \
    --noise-seed-mode "${NOISE_SEED_MODE}" \
    --num-noise-seeds "${NUM_NOISE_SEEDS}" \
    --base-noise-seed "${BASE_NOISE_SEED}" \
    --noise-namespace "${NOISE_NAMESPACE}" \
    --noise-batch-size "${NOISE_BATCH_SIZE}" \
    --max-questions "${MAX_QUESTIONS}" \
    "${ROW_ARGS[@]}" \
    "${POSITION_ARGS[@]}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --dtype "${DTYPE}" \
    --log-every "${LOG_EVERY}" \
    "${FILTER_ARGS[@]}" \
    "$@"

echo "Code V8 Gaussian W2R collection completed: ${OUTPUT_DIR}"
