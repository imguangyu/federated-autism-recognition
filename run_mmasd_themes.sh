#!/usr/bin/env bash
# Train and test FreqMixFormer on each MMASD+ theme subset (theme1, theme2, theme3).
# Uses CUDA_VISIBLE_DEVICES to pin each theme to one GPU (avoids multi-process GPU conflicts).
# Requires theme npz files: MMASD+_theme1.npz, MMASD+_theme2.npz, MMASD+_theme3.npz
#
# Usage: from FreqMixFormer directory
#   bash run_mmasd_themes.sh              # theme1 on GPU 0, theme2 on GPU 1, theme3 on GPU 2 (parallel)
#   bash run_mmasd_themes.sh 2 3 4        # theme1 on GPU 2, theme2 on GPU 3, theme3 on GPU 4 (parallel)
#   bash run_mmasd_themes.sh 0            # all three on GPU 0, one after another (sequential)

set -e
cd "$(dirname "$0")"

if [ -n "$3" ]; then
  # Three GPUs: run in parallel, one theme per GPU
  GPU1="${1:-0}"
  GPU2="${2:-1}"
  GPU3="${3:-2}"
  PARALLEL=1
elif [ -n "$1" ]; then
  # Single GPU: run sequentially
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

run_theme() {
  local theme="$1"
  local dev="$2"
  echo "[$(date +%H:%M:%S)] Starting $theme on GPU $dev (CUDA_VISIBLE_DEVICES=$dev)"
  WORK_DIR="work_dir/mmasd/${theme}"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
  CUDA_VISIBLE_DEVICES="$dev" python main.py \
    --config "config/mmasd/joint_${theme}.yaml" \
    --work-dir "$WORK_DIR" \
    --device 0 \
    --phase train
  echo "[$(date +%H:%M:%S)] Finished $theme"
}

mkdir -p work_dir/mmasd
if [ "$PARALLEL" -eq 1 ]; then
  echo "=============================================="
  echo " Running theme1 (GPU $GPU1), theme2 (GPU $GPU2), theme3 (GPU $GPU3) in parallel"
  echo " Logs: work_dir/mmasd/theme1.log, theme2.log, theme3.log"
  echo "=============================================="
  run_theme theme1 "$GPU1" > work_dir/mmasd/theme1.log 2>&1 &
  PID1=$!
  run_theme theme2 "$GPU2" > work_dir/mmasd/theme2.log 2>&1 &
  PID2=$!
  run_theme theme3 "$GPU3" > work_dir/mmasd/theme3.log 2>&1 &
  PID3=$!
  wait $PID1 || true
  wait $PID2 || true
  wait $PID3 || true
else
  echo "=============================================="
  echo " Running theme1, theme2, theme3 sequentially on GPU $GPU1"
  echo "=============================================="
  run_theme theme1 "$GPU1"
  run_theme theme2 "$GPU2"
  run_theme theme3 "$GPU3"
fi

echo ""
echo "=============================================="
echo " All three themes done. Summary:"
echo "=============================================="
for theme in theme1 theme2 theme3; do
  LOG="work_dir/mmasd/${theme}/log.txt"
  if [ -f "$LOG" ]; then
    echo "--- $theme ---"
    grep -E "Best accuracy|Epoch number" "$LOG" | tail -2
  fi
done
echo ""
echo "To run test only for a theme (e.g. theme1):"
echo "  CUDA_VISIBLE_DEVICES=0 python main.py --config work_dir/mmasd/theme1/config.yaml --work-dir work_dir/mmasd/theme1 --phase test --weights work_dir/mmasd/theme1/runs-<epoch>-<step>.pt --device 0"
