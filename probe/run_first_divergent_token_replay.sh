#!/usr/bin/env bash

# Strict batch-1 first-divergent-token replay pilot for existing V8 math data.
#
# Required:
#   SOURCE_DIRS="/path/to/v8/result_dir [/path/to/another/v8/result_dir]"
#   MODEL_PATH=/path/to/the/exact/source/checkpoint

set -euo pipefail
set -x

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

if [[ -z "${SOURCE_DIRS:-}" ]]; then
    echo "Set SOURCE_DIRS to one or more completed strict batch-1 V8 result directories." >&2
    exit 1
fi
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "Set MODEL_PATH to the exact checkpoint used by the source V8 experiment." >&2
    exit 1
fi

read -r -a SOURCE_DIR_ARRAY <<< "${SOURCE_DIRS}"
SOURCE_ARGS=()
for SOURCE_DIR in "${SOURCE_DIR_ARRAY[@]}"; do
    if [[ ! -f "${SOURCE_DIR}/manifest.jsonl" ]]; then
        echo "Source manifest not found: ${SOURCE_DIR}/manifest.jsonl" >&2
        exit 1
    fi
    SOURCE_ARGS+=(--source-dir "${SOURCE_DIR}")
done

EXPERIMENT_NAME="${EXPERIMENT_NAME:-first-divergent-token-replay-pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}}"

# Start with 10 pairs as a smoke test, then increase to 50.
MAX_W2R_TRIALS="${MAX_W2R_TRIALS:-10}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-1,4,16,64}"
MIN_UNFORCED_SOURCE_TOKENS="${MIN_UNFORCED_SOURCE_TOKENS:-128}"
SELECTION_SEED="${SELECTION_SEED:-20260904}"

# This must equal the source V8 generation budget whenever a source response
# reached its length cap; otherwise the exact-reproduction gate will stop.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-10}"

python3 "${EXPERIMENT_ROOT}/probe/run_first_divergent_token_replay.py" \
    "${SOURCE_ARGS[@]}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --prefix-lengths "${PREFIX_LENGTHS}" \
    --min-unforced-source-tokens "${MIN_UNFORCED_SOURCE_TOKENS}" \
    --max-w2r-trials "${MAX_W2R_TRIALS}" \
    --selection-seed "${SELECTION_SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}" \
    --log-every "${LOG_EVERY}" \
    "$@"

echo "First-divergent-token replay completed: ${OUTPUT_DIR}"
