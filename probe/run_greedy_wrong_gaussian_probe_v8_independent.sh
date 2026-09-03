#!/usr/bin/env bash
# Strict full-path single-response-token Gaussian W2R probe.
# This launcher uses an independent per-(row, position, trial) seed mode.
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

# ================= Experiment (edit here) =================
EXPERIMENT_NAME="greedy-r1-1p5b-32-8k-grpo-clean-seed1003"
DATASET_TAG="aime24"
MODEL_PATH="${REPO_ROOT}/model_merged/grpo-8k"

# Use an evaluated clean parquet when REQUIRE_ALL_INPUT_ROLLOUTS_WRONG=true.
# It must contain prompt, data_source, reward_model, responses, and preferably
# response_scores. The newly generated greedy answer is always scored again.
INPUT_PARQUET="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/${DATASET_TAG}.parquet"

LAYER=24
NOISE_STD=0.1
NOISE_SCALE_MODE="relative_rms"
# "independent" gives every row x position x trial a distinct Gaussian vector.
# A master seed is retained only so the experiment can be reproduced.
NOISE_SEED_MODE="independent"
# Included in seed derivation. Keep it unchanged for matched std sweeps; change
# it when you intentionally want a completely new independent noise bank.
NOISE_NAMESPACE="${EXPERIMENT_NAME}:${DATASET_TAG}:v8-independent"
NUM_NOISE_SEEDS=32
BASE_NOISE_SEED=20260827
NOISE_BATCH_SIZE=1
MAX_QUESTIONS=1
# Match the completed mid-hard paired run exactly. Row indices are zero-based.
ROW_INDICES="2"
REQUIRE_ALL_INPUT_ROLLOUTS_WRONG=false

# Probe approximately uniform clean-response positions. At every selected
# position, each rollout still receives noise at that one token only.
# To probe one position instead, set either RESPONSE_POSITION or
# RESPONSE_POSITION_FRACTION and leave NUM_RESPONSE_POSITIONS empty.
RESPONSE_POSITION=""
RESPONSE_POSITION_FRACTION=""
NUM_RESPONSE_POSITIONS=10

# Match the project's 8k rollout setting. MAX_INPUT_TOKENS must also fit the
# replayed prompt plus the selected clean response prefix.
MAX_NEW_TOKENS=8192
MAX_INPUT_TOKENS=16384
DTYPE="bfloat16"
LOG_EVERY=16

if [[ ! -f "${INPUT_PARQUET}" ]]; then
    echo "Input parquet not found: ${INPUT_PARQUET}" >&2
    exit 1
fi

FILTER_ARGS=()
if [[ "${REQUIRE_ALL_INPUT_ROLLOUTS_WRONG}" == "true" ]]; then
    FILTER_ARGS+=(--require-all-input-rollouts-wrong)
fi

ROW_ARGS=()
if [[ -n "${ROW_INDICES}" ]]; then
    ROW_ARGS+=(--row-indices "${ROW_INDICES}")
    ROW_TAG="rows${ROW_INDICES//,/-}"
else
    ROW_TAG="auto"
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
    echo "Configure one response-position selection mode" >&2
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/greedy_gaussian_w2r/${DATASET_TAG}-${ROW_TAG}-layer${LAYER}-${POSITION_TAG}-${NOISE_SCALE_MODE}-${NOISE_SEED_MODE}-std${NOISE_STD}-seed${BASE_NOISE_SEED}"

python3 "${EXPERIMENT_ROOT}/probe/run_greedy_wrong_gaussian_probe_v8.py" \
    --input-parquet "${INPUT_PARQUET}" \
    --model "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --layer "${LAYER}" \
    --noise-std "${NOISE_STD}" \
    --noise-scale-mode "${NOISE_SCALE_MODE}" \
    --noise-seed-mode "${NOISE_SEED_MODE}" \
    --noise-namespace "${NOISE_NAMESPACE}" \
    --num-noise-seeds "${NUM_NOISE_SEEDS}" \
    --base-noise-seed "${BASE_NOISE_SEED}" \
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

echo "Greedy Gaussian W2R collection completed: ${OUTPUT_DIR}"
