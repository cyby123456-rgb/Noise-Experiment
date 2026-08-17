#!/usr/bin/env bash
set -euo pipefail

# This invokes the causal one-shot probe.  It never modifies checkpoints.
# Required: CLEAN_PARQUET, MODEL_PATH, OUTPUT_JSONL, LAYER.

: "${CLEAN_PARQUET:?Set CLEAN_PARQUET to clean generation parquet}"
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint}"
: "${OUTPUT_JSONL:?Set OUTPUT_JSONL to a new JSONL output path}"
: "${LAYER:?Set LAYER to the intervention layer}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$VERL_ROOT"

python "$EXPERIMENT_ROOT/probe/run_fixed_state_direction_probe.py" \
  --input-parquet "$CLEAN_PARQUET" \
  --model "$MODEL_PATH" \
  --output-jsonl "$OUTPUT_JSONL" \
  --layer "$LAYER" \
  --alphas "${ALPHAS:-0.025,0.05}" \
  --directions "${DIRECTIONS:-64}" \
  --direction-batch-size "${DIRECTION_BATCH_SIZE:-8}" \
  --max-states "${MAX_STATES:-48}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  "$@"
