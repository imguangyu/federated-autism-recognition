#!/usr/bin/env bash
#
# Grid search for FedProx mu on MMASD+ / FreqMixFormer.
# Each GPU runs one FedProx experiment with a different mu value.
#
# Usage（在 FreqMixFormer 根目录）:
#   bash fl_experiments/run_fedprox_grid.sh               # 默认 GPU 0 1 2 3 和预设 mus
#   bash fl_experiments/run_fedprox_grid.sh 0 1 2 3       # 显式指定 4 块 GPU
#

set -e

cd "$(dirname "$0")/.."   # 切到 FreqMixFormer 根目录

# GPUs to use (can override via positional args)
GPU0="${1:-0}"
GPU1="${2:-1}"
GPU2="${3:-2}"
GPU3="${4:-3}"
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")

# FedProx mu candidates（可自行改成你想要的取值）
MUS=("0.0" "0.0005" "0.001" "0.005")

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

echo "FedProx grid search (mu per GPU):"
for i in "${!MUS[@]}"; do
  if [ "$i" -ge "${#GPUS[@]}" ]; then
    break
  fi
  echo "  mu=${MUS[$i]}  -> GPU ${GPUS[$i]}"
done
echo

pids=()

run_one() {
  local mu="$1"
  local gpu="$2"
  local tag="${mu//./p}"           # e.g., 0.0005 -> 0p0005
  local log_file="${LOG_ROOT}/fl_fedprox_mu${tag}.log"

  echo "[$(date +%H:%M:%S)] Starting FedProx (mu=${mu}) on GPU ${gpu}, log -> ${log_file}"
  FEDPROX_MU="${mu}" CUDA_VISIBLE_DEVICES="${gpu}" \
    bash fl_experiments/run_fl_mmasd_simulator.sh --method fedprox \
    > "${log_file}" 2>&1 &
  pids+=($!)
}

for i in "${!MUS[@]}"; do
  if [ "$i" -ge "${#GPUS[@]}" ]; then
    break
  fi
  run_one "${MUS[$i]}" "${GPUS[$i]}"
done

echo
echo "FedProx grid search jobs started. Waiting for them to finish..."

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "[$(date +%H:%M:%S)] FedProx grid search completed."

