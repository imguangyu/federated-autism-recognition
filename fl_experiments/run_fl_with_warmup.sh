#!/usr/bin/env bash
#
# Run all FL methods (fedavg, fedprox, fedbn, fedper, local) with warmup enabled.
# Warmup is controlled by the --warmup_rounds argument in nvflare_mmasd_client.py
# (currently默认 5 轮，从 0 -> base_lr 线性升高)。
#
# Usage（在 FreqMixFormer 根目录）:
#   bash fl_experiments/run_fl_with_warmup.sh          # 默认 GPU 0 1 2 3 依次跑前四种方法
#   bash fl_experiments/run_fl_with_warmup.sh 0 1 2 3  # 显式指定 4 块 GPU
#

set -e

cd "$(dirname "$0")/.."   # 切到 FreqMixFormer 根目录

GPU0="${1:-0}"
GPU1="${2:-1}"
GPU2="${3:-2}"
GPU3="${4:-3}"

METHODS=("fedavg" "fedprox" "fedbn" "fedper")
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

echo "Running FL methods with warmup (default --warmup_rounds=5):"
for i in "${!METHODS[@]}"; do
  echo "  ${METHODS[$i]}  -> GPU ${GPUS[$i]}"
done
echo

pids=()

run_method() {
  local method="$1"
  local gpu="$2"
  local log_file="${LOG_ROOT}/fl_${method}_warmup.log"

  echo "[$(date +%H:%M:%S)] Starting ${method} (with warmup) on GPU ${gpu}, log -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash fl_experiments/run_fl_mmasd_simulator.sh --method "${method}" \
    > "${log_file}" 2>&1 &
  pids+=($!)
}

for i in "${!METHODS[@]}"; do
  run_method "${METHODS[$i]}" "${GPUS[$i]}"
done

echo
echo "All warmup FL runs (fedavg/fedprox/fedbn/fedper) started. Waiting for them to finish..."

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "[$(date +%H:%M:%S)] FedAvg/FedProx/FedBN/FedPer with warmup completed."

# Also run local-only baseline under FL framework with warmup, reusing GPU0.
LOG_LOCAL="${LOG_ROOT}/fl_local_warmup.log"
echo "[$(date +%H:%M:%S)] Starting local (FL framework, personalization=local) with warmup on GPU ${GPU0}, log -> ${LOG_LOCAL}"
CUDA_VISIBLE_DEVICES="${GPU0}" bash fl_experiments/run_fl_mmasd_simulator.sh --method local \
  > "${LOG_LOCAL}" 2>&1 || true

echo "[$(date +%H:%M:%S)] All warmup FL methods (including local) completed."

