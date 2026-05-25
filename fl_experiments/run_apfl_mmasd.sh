#!/usr/bin/env bash
#
# Run APFL (Adaptive Personalized Federated Learning) on MMASD+ / FreqMixFormer.
#
# Usage (from FreqMixFormer root):
#   bash fl_experiments/run_apfl_mmasd.sh [GPU_ID]
#

set -e

cd "$(dirname "$0")/.."   # go to FreqMixFormer root

GPU="${1:-0}"

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_ROOT}/fl_apfl_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running APFL on GPU ${GPU}, log -> ${LOG_FILE}"
{
  echo "APFL run started."
  CUDA_VISIBLE_DEVICES="${GPU}" bash fl_experiments/run_fl_mmasd_simulator.sh --method apfl 2>&1
  echo "APFL run finished."
} | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee "${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log written to ${LOG_FILE}"

