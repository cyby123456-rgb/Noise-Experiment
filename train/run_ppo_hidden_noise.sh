#!/usr/bin/env bash
set -euo pipefail

# Required environment variables: MODEL_PATH, TRAIN_DATA, VAL_DATA.
# Example:
# MODEL_PATH=/models/grpo TRAIN_DATA=data/train.parquet VAL_DATA=data/val.parquet \
# TRAIN_NOISE_STD=0.02 TRAIN_NOISE_LAYER=20 TRAIN_NOISE_SEED=1235 \
# bash noise_experiments/train/run_ppo_hidden_noise.sh \
#   trainer.n_gpus_per_node=8 trainer.total_epochs=1

: "${MODEL_PATH:?Set MODEL_PATH to the actor checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a training parquet}"
: "${VAL_DATA:?Set VAL_DATA to a validation parquet}"

TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.0}"
TRAIN_NOISE_LAYER="${TRAIN_NOISE_LAYER:-null}"
TRAIN_NOISE_SEED="${TRAIN_NOISE_SEED:-0}"
TRAIN_NOISE_MODE="${TRAIN_NOISE_MODE:-rollout_only}" # rollout_only|update_only|all
VAL_NOISE_STD="${VAL_NOISE_STD:-0.0}"
VAL_NOISE_LAYER="${VAL_NOISE_LAYER:-null}"
VAL_NOISE_SEED="${VAL_NOISE_SEED:-null}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$VERL_ROOT"

python -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="${CRITIC_MODEL_PATH:-$MODEL_PATH}" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  trainer.train_hidden_noise.std="$TRAIN_NOISE_STD" \
  trainer.train_hidden_noise.layer_idx="$TRAIN_NOISE_LAYER" \
  trainer.train_hidden_noise.base_seed="$TRAIN_NOISE_SEED" \
  trainer.train_hidden_noise.apply_mode="$TRAIN_NOISE_MODE" \
  trainer.validation_noise.std="$VAL_NOISE_STD" \
  trainer.validation_noise.layer_idx="$VAL_NOISE_LAYER" \
  trainer.validation_noise.layer_seed="$VAL_NOISE_SEED" \
  "$@"
