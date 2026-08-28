#!/usr/bin/env bash
set -euo pipefail

# ================= Paths =================
# Works both from the standalone Noise-Experiment repository and from the
# noise_experiments/ directory in the larger Exploration workspace.
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$EXPERIMENT_ROOT/.git" ]]; then
  DEFAULT_REPO_ROOT="$EXPERIMENT_ROOT"
else
  DEFAULT_REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
fi
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
VERL_ROOT="${VERL_ROOT:-$EXPERIMENT_ROOT/verl}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/model_merged/grpo-8k}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-r1-1p5b-32-8k-grpo-clean-seed1002}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/evaluation/results/$EXPERIMENT_NAME}"

# ================= CUDA / generation =================
export CUDA_VISIBLE_DEVICES=0,1
# The hidden-state collector is implemented against vLLM 0.8.4's V0
# ModelRunner. This must be set before Python imports vLLM.
export VLLM_USE_V1=0

NNODES=1
N_GPUS=2
BATCH_SIZE=1024
MAX_BATCH_TOKENS=40960
GPU_MEMORY_UTILIZATION=0.8
TENSOR_MODEL_PARALLEL_SIZE=1

N_SAMPLES=32
TEMPERATURE=1.0
TOP_K=-1
TOP_P=1.0
PROMPT_LENGTH=766
MAX_RESPONSE_LENGTH=8192
# One fixed base seed makes the run reproducible. main_generation and the
# vLLM rollout derive a different request seed for every (row, sample_index),
# so the 32 responses are independent samples rather than 32 clones.
ROLLOUT_SEED=1002

# ================= Hidden-state capture =================
# The all-wrong filter only needs responses and response_scores. Keep capture
# off by default to avoid unnecessary storage; enable it for separate analyses.
CAPTURE_HIDDEN_STATES=false
HIDDEN_STATE_LAYERS='[24]'
HIDDEN_STATE_RESPONSE_POSITIONS='[0,2048,4096,6144]'
RUN_MAIN_EVAL=true

# ================= Noise =================
# This is the clean run because NOISE_STD=0.0.
NOISE_STD=0.0
NOISE_LAYER=24
NOISE_SEED=1002
NOISE_ALL_LAYERS=false

export PYTHONPATH="$VERL_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Make the patched snapshot win over any unmodified top-level verl checkout.
cd "$VERL_ROOT"

# Override this list with a whitespace-separated DATASET_PATHS variable when
# needed.  The default exactly matches the six datasets in this project.
if [[ -n "${DATASET_PATHS:-}" ]]; then
  read -r -a DATASET_FILES <<< "$DATASET_PATHS"
else
  DATASET_FILES=(
    "$DATA_ROOT/aime24.parquet"
    "$DATA_ROOT/aime25.parquet"
    "$DATA_ROOT/amc23.parquet"
    "$DATA_ROOT/minerva.parquet"
    "$DATA_ROOT/olympiad_bench.parquet"
    "$DATA_ROOT/deepscaler/math.parquet"
  )
fi

mkdir -p "$OUTPUT_DIR"

for INPUT_PARQUET in "${DATASET_FILES[@]}"; do
  if [[ ! -f "$INPUT_PARQUET" ]]; then
    echo "Dataset not found: $INPUT_PARQUET" >&2
    exit 1
  fi

  DATASET_TAG="$(basename "$INPUT_PARQUET" .parquet)"
  OUTPUT_PARQUET="$OUTPUT_DIR/${DATASET_TAG}.parquet"

  echo "======================================"
  echo "Dataset: $DATASET_TAG"
  echo "Input:   $INPUT_PARQUET"
  echo "Output:  $OUTPUT_PARQUET"

  python -m verl.trainer.main_generation \
    model.path="$MODEL_PATH" \
    data.path="$INPUT_PARQUET" \
    data.output_path="$OUTPUT_PARQUET" \
    data.n_samples="$N_SAMPLES" \
    data.batch_size="$BATCH_SIZE" \
    rollout.temperature="$TEMPERATURE" \
    rollout.do_sample=true \
    rollout.top_k="$TOP_K" \
    rollout.top_p="$TOP_P" \
    rollout.prompt_length="$PROMPT_LENGTH" \
    rollout.response_length="$MAX_RESPONSE_LENGTH" \
    rollout.enforce_eager=true \
    rollout.enable_chunked_prefill=true \
    rollout.max_num_batched_tokens="$MAX_BATCH_TOKENS" \
    rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
    ++rollout.seed="$ROLLOUT_SEED" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.validation_noise.std="$NOISE_STD" \
    trainer.validation_noise.layer_idx="$NOISE_LAYER" \
    trainer.validation_noise.all_layers="$NOISE_ALL_LAYERS" \
    trainer.validation_noise.layer_seed="$NOISE_SEED" \
    analysis.hidden_states.enable="$CAPTURE_HIDDEN_STATES" \
    analysis.hidden_states.layers="$HIDDEN_STATE_LAYERS" \
    analysis.hidden_states.response_positions="$HIDDEN_STATE_RESPONSE_POSITIONS" \
    "$@"

  if [[ "$RUN_MAIN_EVAL" == "true" ]]; then
    python -m verl.trainer.main_eval \
      data.path="$OUTPUT_PARQUET" \
      data.output_path="$OUTPUT_PARQUET" \
      data.prompt_key=prompt \
      data.response_key=responses \
      data.response_scores_key=response_scores \
      data.data_source_key=data_source \
      data.reward_model_key=reward_model
  fi
done
