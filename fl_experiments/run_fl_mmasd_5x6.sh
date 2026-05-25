#!/usr/bin/env bash
#
# Quick 5-epochs × 6-rounds runs for all FL methods (fedavg, fedprox, fedbn, fedper, local).
# Each method uses its own *_5x6 NVFLARE job with num_rounds=6 and --local_epochs=5.
#
# Usage (from FreqMixFormer root):
#   bash fl_experiments/run_fl_mmasd_5x6.sh          # default GPUs 0 1 2 3
#   bash fl_experiments/run_fl_mmasd_5x6.sh 0 1 2 3  # explicitly set GPUs
#

set -e

cd "$(dirname "$0")/.."   # go to FreqMixFormer root

GPU0="${1:-0}"
GPU1="${2:-1}"
GPU2="${3:-2}"
GPU3="${4:-3}"

METHODS=("fedavg_5x6" "fedprox_5x6")
GPUS=("$GPU1" "$GPU2")

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running FL methods with 5 epochs × 6 rounds (run ${RUN_TS}):"
for i in "${!METHODS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')]   ${METHODS[$i]}  -> GPU ${GPUS[$i]}"
done
echo

pids=()

run_method_5x6() {
  local method="$1"
  local gpu="$2"
  local log_file="${LOG_ROOT}/fl_${method}_${RUN_TS}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${method} on GPU ${gpu}, log -> ${log_file}"
  {
    echo "${method} run started."
    CUDA_VISIBLE_DEVICES="${gpu}" bash fl_experiments/run_fl_mmasd_simulator.sh --method "${method}" 2>&1
    echo "${method} run finished."
  } | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done >> "${log_file}" &
  pids+=($!)
}

for i in "${!METHODS[@]}"; do
  run_method_5x6 "${METHODS[$i]}" "${GPUS[$i]}"
done

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for 5x6 FL runs (fedavg_5x6/fedprox_5x6/fedbn_5x6/fedper_5x6) to finish..."

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 5x6 FL methods (fedavg/fedprox/fedbn/fedper) completed."

LOG_LOCAL="${LOG_ROOT}/fl_local_5x6_${RUN_TS}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting local_5x6 on GPU ${GPU0}, log -> ${LOG_LOCAL}"
{
  echo "local_5x6 run started."
  CUDA_VISIBLE_DEVICES="${GPU0}" bash fl_experiments/run_fl_mmasd_simulator.sh --method local_5x6 2>&1
  echo "local_5x6 run finished."
} | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee "${LOG_LOCAL}" || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5x6 FL methods (including local_5x6) completed. Logs in ${LOG_ROOT} with timestamp ${RUN_TS}."

