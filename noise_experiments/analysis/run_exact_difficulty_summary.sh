#!/usr/bin/env bash
set -euo pipefail

# Rebuild exact 0/32, 1-7/32, 8-23/32, 24-31/32, 32/32 buckets from
# question_seed_flips.csv created by analyze_layer20_multiseed_consistency.py.

: "${QUESTION_FLIPS_CSV:?Set QUESTION_FLIPS_CSV}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$EXPERIMENT_ROOT/analysis/summarize_exact_difficulty_buckets.py" \
  --input "$QUESTION_FLIPS_CSV" \
  --output-dir "$OUTPUT_DIR"
