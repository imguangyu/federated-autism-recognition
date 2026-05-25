#!/usr/bin/env bash
#
# Local (non-FL) baseline: train one model per MMASD+ theme (theme1-3).
# Creates a timestamped output folder so results from different runs don't overwrite.
#
# Usage (from FreqMixFormer directory):
#   bash fl_experiments/run_local_mmasd_baseline.sh              # theme1 on GPU 0, theme2 on GPU 1, theme3 on GPU 2 (parallel)
#   bash fl_experiments/run_local_mmasd_baseline.sh 2 3 4        # parallel on GPUs 2/3/4
#   bash fl_experiments/run_local_mmasd_baseline.sh 0            # sequential on GPU 0
#

set -e
cd "$(dirname "$0")/.."

if [ -n "$3" ]; then
  GPU1="${1:-0}"
  GPU2="${2:-1}"
  GPU3="${3:-2}"
  PARALLEL=1
elif [ -n "$1" ]; then
  GPU1="$1"
  GPU2="$1"
  GPU3="$1"
  PARALLEL=0
else
  GPU1=0
  GPU2=1
  GPU3=2
  PARALLEL=1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="work_dir/baselines/local_${TIMESTAMP}"
mkdir -p "$RUN_ROOT"

run_theme() {
  local theme="$1"
  local dev="$2"
  local work_dir="${RUN_ROOT}/${theme}"
  echo "[$(date +%H:%M:%S)] Starting LOCAL baseline $theme on GPU $dev (CUDA_VISIBLE_DEVICES=$dev)"
  mkdir -p "$work_dir"
  CUDA_VISIBLE_DEVICES="$dev" python main.py \
    --config "config/mmasd/joint_${theme}.yaml" \
    --work-dir "$work_dir" \
    --num-epoch 30 \
    --base-lr 0.1 \
    --weight-decay 0.0005 \
    --batch-size 32 \
    --device 0 \
    --phase train
  echo "[$(date +%H:%M:%S)] Finished $theme"
}

echo "=============================================="
echo " LOCAL BASELINE (no FL)"
echo " Output root: $RUN_ROOT"
echo "=============================================="

mkdir -p "$RUN_ROOT/logs"
if [ "$PARALLEL" -eq 1 ]; then
  run_theme theme1 "$GPU1" > "${RUN_ROOT}/logs/theme1.log" 2>&1 &
  PID1=$!
  run_theme theme2 "$GPU2" > "${RUN_ROOT}/logs/theme2.log" 2>&1 &
  PID2=$!
  run_theme theme3 "$GPU3" > "${RUN_ROOT}/logs/theme3.log" 2>&1 &
  PID3=$!
  wait $PID1 || true
  wait $PID2 || true
  wait $PID3 || true
else
  run_theme theme1 "$GPU1" | tee "${RUN_ROOT}/logs/theme1.log"
  run_theme theme2 "$GPU2" | tee "${RUN_ROOT}/logs/theme2.log"
  run_theme theme3 "$GPU3" | tee "${RUN_ROOT}/logs/theme3.log"
fi

echo ""
echo "=============================================="
echo " Summary:"
echo "=============================================="
for theme in theme1 theme2 theme3; do
  LOG="${RUN_ROOT}/${theme}/log.txt"
  if [ -f "$LOG" ]; then
    echo "--- $theme ---"
    grep -E "Best accuracy|Epoch number" "$LOG" | tail -2 || true
  else
    echo "--- $theme ---"
    echo "No log.txt found at $LOG"
  fi
done

echo ""
echo "Artifacts are under: $RUN_ROOT"

