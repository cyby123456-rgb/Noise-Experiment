#!/usr/bin/env bash
set -euo pipefail

# Required environment variables: MODEL_PATH, INPUT_PARQUET, OUTPUT_PARQUET.
# NOISE_STD=0 creates the clean counterpart.  Use a distinct OUTPUT_PARQUET
# and NOISE_SEED for each noisy seed.

: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint}"
: "${INPUT_PARQUET:?Set INPUT_PARQUET to prompts parquet}"
: "${OUTPUT_PARQUET:?Set OUTPUT_PARQUET to generation output parquet}"

NOISE_STD="${NOISE_STD:-0.0}"
NOISE_LAYER="${NOISE_LAYER:-null}"
NOISE_SEED="${NOISE_SEED:-null}"
N_SAMPLES="${N_SAMPLES:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$VERL_ROOT"

python -m verl.trainer.main_generation \
  model.path="$MODEL_PATH" \
  data.path="$INPUT_PARQUET" \
  data.output_path="$OUTPUT_PARQUET" \
  data.n_samples="$N_SAMPLES" \
  rollout.temperature="$TEMPERATURE" \
  rollout.response_length="$MAX_RESPONSE_LENGTH" \
  trainer.validation_noise.std="$NOISE_STD" \
  trainer.validation_noise.layer_idx="$NOISE_LAYER" \
  trainer.validation_noise.layer_seed="$NOISE_SEED" \
  "$@"
