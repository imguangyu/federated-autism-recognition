#!/usr/bin/env bash
#
# Run only personalized FL methods (fedbn, fedper) for MMASD+ / FreqMixFormer.
# 默认用后两块 GPU（例如 2、3），避免和前两个方法冲突。
#
# Usage（在 FreqMixFormer 根目录）:
#   bash fl_experiments/run_personalized_fl_mmasd_methods.sh        # 默认 GPU 2 3
#   bash fl_experiments/run_personalized_fl_mmasd_methods.sh 4 5    # 显式指定两个 GPU
#

set -e

cd "$(dirname "$0")/.."   # 切到 FreqMixFormer 根目录

GPU_BN="${1:-2}"
GPU_PER="${2:-3}"

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running personalized FL methods (run ${RUN_TS}):"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   fedbn  -> GPU ${GPU_BN}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   fedper -> GPU ${GPU_PER}"
echo

pids=()

run_method() {
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

run_method "fedbn" "${GPU_BN}"
run_method "fedper" "${GPU_PER}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Personalized FL runs started. Waiting for them to finish..."

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Personalized FL methods (fedbn, fedper) completed. Logs in ${LOG_ROOT} with timestamp ${RUN_TS}."

