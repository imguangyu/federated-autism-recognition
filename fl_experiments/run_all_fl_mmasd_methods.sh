#!/usr/bin/env bash
#
# Run all FL methods (fedavg, fedprox, fedbn, fedper, apfl, local) for MMASD+ / FreqMixFormer.
# Uses only 4 GPUs: when a GPU becomes free, the next method is started (wait if no available GPU).
#
# Usage (from FreqMixFormer root):
#   bash fl_experiments/run_all_fl_mmasd_methods.sh           # default GPUs 0 1 2 3
#   bash fl_experiments/run_all_fl_mmasd_methods.sh 0 1 2 3  # explicit GPU ids
#

set -e

cd "$(dirname "$0")/.."   # FreqMixFormer root

GPU0="${1:-0}"
GPU1="${2:-1}"
GPU2="${3:-2}"
GPU3="${4:-3}"

# Only 4 GPUs; 6 methods will run in waves (4 concurrent max)
METHODS=(fedavg fedprox fedbn fedper apfl local)
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")
NUM_GPUS=4

LOG_ROOT="fl_experiments/logs"
mkdir -p "$LOG_ROOT"

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${#METHODS[@]} FL methods on ${NUM_GPUS} GPUs (run ${RUN_TS}); next method starts when a GPU is free."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs: ${GPUS[*]}"
echo

run_one() {
  local method="$1"
  local gpu="$2"
  local log_file="${LOG_ROOT}/fl_${method}_${RUN_TS}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${method} on GPU ${gpu}, log -> ${log_file}"
  {
    echo "${method} run started."
    CUDA_VISIBLE_DEVICES="${gpu}" bash fl_experiments/run_fl_mmasd_simulator.sh --method "${method}" 2>&1
    echo "${method} run finished."
  } | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done >> "${log_file}"
}

# Queue of method indices still to run
queue=(0 1 2 3 4 5)
# For each GPU slot (0..3), pid of current job (empty string if free). We use indices 0..3 for slots.
declare -a slot_pid
declare -a slot_gpu
for i in $(seq 0 $((NUM_GPUS - 1))); do
  slot_pid[i]=""
  slot_gpu[i]=${GPUS[$i]}
done

start_next() {
  if [ ${#queue[@]} -eq 0 ]; then return 1; fi
  local idx
  for idx in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -z "${slot_pid[$idx]}" ]; then
      local method_idx="${queue[0]}"
      queue=("${queue[@]:1}")
      local method="${METHODS[$method_idx]}"
      local gpu="${slot_gpu[$idx]}"
      run_one "$method" "$gpu" &
      slot_pid[$idx]=$!
      return 0
    fi
  done
  return 1
}

# Start first min(4, 6) jobs
for _ in $(seq 1 $NUM_GPUS); do
  start_next || break
done

# When a job finishes, free its slot and start next from queue (requires Bash 4.3+ for wait -n)
running() {
  local i
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    [ -n "${slot_pid[$i]}" ] && kill -0 "${slot_pid[$i]}" 2>/dev/null && return 0
  done
  return 1
}

while running || [ ${#queue[@]} -gt 0 ]; do
  if [ ${#queue[@]} -gt 0 ]; then
    # Wait for any one job to finish so we can start the next on that GPU
    wait -n 2>/dev/null || true
  else
    # No queue left, wait for all running jobs then exit
    for idx in $(seq 0 $((NUM_GPUS - 1))); do
      [ -n "${slot_pid[$idx]}" ] && wait "${slot_pid[$idx]}" 2>/dev/null || true
    done
    break
  fi
  # Find which slot (GPU) freed up and start next method on it
  for idx in $(seq 0 $((NUM_GPUS - 1))); do
    pid="${slot_pid[$idx]}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      slot_pid[$idx]=""
      start_next || true
      break
    fi
  done
done

# Reap any remaining background jobs
for idx in $(seq 0 $((NUM_GPUS - 1))); do
  [ -n "${slot_pid[$idx]}" ] && wait "${slot_pid[$idx]}" 2>/dev/null || true
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All FL methods completed. Logs in ${LOG_ROOT} with timestamp ${RUN_TS}."
