#!/usr/bin/env bash
#
# Run the "local (FL framework)" method with the 5-epochs × 6-rounds setting.
# This uses the nvflare_job_mmasd_local_5x6 job (num_rounds=6, local_epochs=5).
#
# Usage (from FreqMixFormer root):
#   bash fl_experiments/run_local_5x6_mmasd_fl.sh [GPU_ID]
#

set -e

cd "$(dirname "$0")/.."   # go to FreqMixFormer root

GPU="${1:-0}"

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_ROOT}/fl_local_5x6_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running local (FL framework, personalization=local) with 5 epochs × 6 rounds on GPU ${GPU}, log -> ${LOG_FILE}"
{
  echo "Local 5x6 FL run started."
  CUDA_VISIBLE_DEVICES="${GPU}" bash fl_experiments/run_fl_mmasd_simulator.sh --method local_5x6 2>&1
  echo "Local 5x6 FL run finished."
} | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee "${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log written to ${LOG_FILE}"

