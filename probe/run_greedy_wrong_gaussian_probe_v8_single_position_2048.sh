#!/usr/bin/env bash
# Strict full-path probe: one question, one response position, 2048 noises.
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
CONDA_BASE="/root/miniconda3"
CONDA_ENV="test123"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

# ================= Fixed-condition experiment =================
# These defaults reproduce the previously tested AIME24 row and its earliest
# selected response position. Override them through environment variables when
# deliberately choosing another question or position.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-greedy-r1-1p5b-32-8k-grpo-clean-seed1003}"
DATASET_TAG="${DATASET_TAG:-aime24}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/model_merged/grpo-8k}"
INPUT_PARQUET="${INPUT_PARQUET:-${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/${DATASET_TAG}.parquet}"

LAYER="${LAYER:-24}"
ROW_INDEX="${ROW_INDEX:-2}"
# One-based clean response-token position. The perturbation first affects
# response token RESPONSE_POSITION + 1.
RESPONSE_POSITION="${RESPONSE_POSITION:-745}"

NOISE_STD="${NOISE_STD:-0.1}"
NOISE_SCALE_MODE="${NOISE_SCALE_MODE:-relative_rms}"
NOISE_SEED_MODE="independent"
NUM_NOISE_SEEDS="${NUM_NOISE_SEEDS:-2048}"
BASE_NOISE_SEED="${BASE_NOISE_SEED:-2026090201}"
# This new namespace prevents the first 32 vectors from duplicating an older
# v8 run even when the question, layer, and response position are unchanged.
NOISE_NAMESPACE="${NOISE_NAMESPACE:-${EXPERIMENT_NAME}:${DATASET_TAG}:single-pos-2048-bank1}"

# Batch size 1 is intentional: every trial follows exactly the same full
# batch-1 greedy path up to the intervention point.
NOISE_BATCH_SIZE=1
MAX_QUESTIONS=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-16}"

if [[ ! -f "${INPUT_PARQUET}" ]]; then
    echo "Input parquet not found: ${INPUT_PARQUET}" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/greedy_gaussian_w2r/${DATASET_TAG}-row${ROW_INDEX}-layer${LAYER}-pos${RESPONSE_POSITION}-${NOISE_SCALE_MODE}-${NOISE_SEED_MODE}-std${NOISE_STD}-n${NUM_NOISE_SEEDS}-seed${BASE_NOISE_SEED}}"

python3 "${EXPERIMENT_ROOT}/probe/run_greedy_wrong_gaussian_probe_v8.py" \
    --input-parquet "${INPUT_PARQUET}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --layer "${LAYER}" \
    --row-indices "${ROW_INDEX}" \
    --response-position "${RESPONSE_POSITION}" \
    --noise-std "${NOISE_STD}" \
    --noise-scale-mode "${NOISE_SCALE_MODE}" \
    --noise-seed-mode "${NOISE_SEED_MODE}" \
    --noise-namespace "${NOISE_NAMESPACE}" \
    --num-noise-seeds "${NUM_NOISE_SEEDS}" \
    --base-noise-seed "${BASE_NOISE_SEED}" \
    --noise-batch-size "${NOISE_BATCH_SIZE}" \
    --max-questions "${MAX_QUESTIONS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --dtype "${DTYPE}" \
    --log-every "${LOG_EVERY}" \
    "$@"

echo "Single-position 2048-noise collection completed: ${OUTPUT_DIR}"
