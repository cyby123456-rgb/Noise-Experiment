#!/usr/bin/env bash

# Launch two different shell scripts concurrently, one on each GPU.
#
# Usage:
#   bash probe/run_two_scripts_2gpu.sh path/to/script_a.sh path/to/script_b.sh
#
# Each child script keeps its own experiment parameters and output directory.
# It must respect the inherited CUDA_VISIBLE_DEVICES value instead of replacing
# it with a hard-coded physical GPU index.

set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${EXPERIMENT_ROOT}/.git" ]]; then
    DEFAULT_REPO_ROOT="${EXPERIMENT_ROOT}"
else
    DEFAULT_REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/.." && pwd)"
fi
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
cd "${REPO_ROOT}"

SCRIPT0_INPUT="${1:-${GPU0_SCRIPT:-}}"
SCRIPT1_INPUT="${2:-${GPU1_SCRIPT:-}}"
if [[ -z "${SCRIPT0_INPUT}" || -z "${SCRIPT1_INPUT}" ]]; then
    echo "Usage: bash probe/run_two_scripts_2gpu.sh SCRIPT_FOR_GPU0 SCRIPT_FOR_GPU1" >&2
    exit 2
fi
if (( $# > 2 )); then
    echo "Unexpected arguments after the two script paths: ${*:3}" >&2
    exit 2
fi

resolve_script_path() {
    local candidate="$1"
    if [[ "${candidate}" = /* ]]; then
        printf '%s\n' "${candidate}"
    else
        printf '%s\n' "${REPO_ROOT}/${candidate#./}"
    fi
}

SCRIPT0="$(resolve_script_path "${SCRIPT0_INPUT}")"
SCRIPT1="$(resolve_script_path "${SCRIPT1_INPUT}")"
if [[ ! -f "${SCRIPT0}" ]]; then
    echo "GPU 0 script not found: ${SCRIPT0}" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT1}" ]]; then
    echo "GPU 1 script not found: ${SCRIPT1}" >&2
    exit 1
fi
if [[ "${SCRIPT0}" == "${SCRIPT1}" ]]; then
    echo "Provide two different scripts; both paths resolved to ${SCRIPT0}." >&2
    exit 1
fi

GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]; then
    echo "GPU0_ID and GPU1_ID must be different." >&2
    exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/two_gpu/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"
LOG0="${LOG_DIR}/gpu0_$(basename "${SCRIPT0}" .sh).log"
LOG1="${LOG_DIR}/gpu1_$(basename "${SCRIPT1}" .sh).log"

run_script() {
    local label="$1"
    local physical_gpu="$2"
    local script_path="$3"
    local log_path="$4"

    (
        export CUDA_VISIBLE_DEVICES="${physical_gpu}"
        echo "physical GPU: ${physical_gpu}"
        echo "script: ${script_path}"
        bash "${script_path}"
    ) 2>&1 | sed -u "s/^/[${label}] /" | tee "${log_path}"
}

echo "Launching two different scripts"
echo "  GPU ${GPU0_ID}: ${SCRIPT0}"
echo "  GPU ${GPU1_ID}: ${SCRIPT1}"
echo "  logs: ${LOG_DIR}"

run_script gpu0 "${GPU0_ID}" "${SCRIPT0}" "${LOG0}" &
PID0=$!
run_script gpu1 "${GPU1_ID}" "${SCRIPT1}" "${LOG1}" &
PID1=$!

terminate_children() {
    kill "${PID0}" "${PID1}" 2>/dev/null || true
}
trap terminate_children INT TERM

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM

echo "GPU 0 exit status: ${STATUS0}; log: ${LOG0}"
echo "GPU 1 exit status: ${STATUS1}; log: ${LOG1}"
if (( STATUS0 != 0 || STATUS1 != 0 )); then
    exit 1
fi

echo "Both scripts completed successfully."
