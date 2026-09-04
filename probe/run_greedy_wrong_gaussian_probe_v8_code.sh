#!/usr/bin/env bash

# Backward-compatible alias. Code collection moved to the independent v9
# single-GPU batched implementation; no v8 serial code path remains here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_greedy_wrong_gaussian_probe_v9_code_parallel.sh" "$@"
