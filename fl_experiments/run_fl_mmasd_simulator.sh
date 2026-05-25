#!/usr/bin/env bash
#
# Run NVFLARE simulator for MMASD+ / FreqMixFormer with 3 clients (theme1–3).
# Usage:
#   bash run_fl_mmasd_simulator.sh                        # create workspace with timestamp
#   bash run_fl_mmasd_simulator.sh /path/to/workspace     # use given workspace
#   bash run_fl_mmasd_simulator.sh --method fedavg        # FedAvg (global shared model)
#   bash run_fl_mmasd_simulator.sh --method fedprox       # FedProx (prox loss on clients)
#   bash run_fl_mmasd_simulator.sh --method fedbn         # FedBN (global weights, local BN)
#   bash run_fl_mmasd_simulator.sh --method fedper        # FedPer (global backbone, local head)
#   bash run_fl_mmasd_simulator.sh --method local         # Local-only (no global params loaded, but under FL framework)
#   bash run_fl_mmasd_simulator.sh --method apfl          # APFL (adaptive personalized FL, shared+personal models)
#   bash run_fl_mmasd_simulator.sh --method fedavg_5x6    # FedAvg quick run:   5 epochs × 6 rounds
#   bash run_fl_mmasd_simulator.sh --method fedprox_5x6   # FedProx quick run:  5 epochs × 6 rounds
#   bash run_fl_mmasd_simulator.sh --method fedbn_5x6     # FedBN quick run:    5 epochs × 6 rounds
#   bash run_fl_mmasd_simulator.sh --method fedper_5x6    # FedPer quick run:   5 epochs × 6 rounds
#   bash run_fl_mmasd_simulator.sh --method local_5x6     # Local quick run:    5 epochs × 6 rounds
#

set -e

# Resolve job directory to an absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_METHOD="fedavg"
METHOD="$DEFAULT_METHOD"
WORKSPACE=""

# Parse args:
#   positional: [workspace]
#   flags: --method {fedavg|fedprox|fedbn|fedper|local|apfl|fedavg_5x6|fedprox_5x6|fedbn_5x6|fedper_5x6|local_5x6}
while [ $# -gt 0 ]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    -m|--method=*)
      METHOD="${1#*=}"
      shift 1
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      if [ -z "$WORKSPACE" ]; then
        WORKSPACE="$1"
        shift 1
      else
        echo "Unexpected extra arg: $1"
        exit 1
      fi
      ;;
  esac
done

case "$METHOD" in
  fedavg|fedprox|fedbn|fedper|local|apfl|fedavg_5x6|fedprox_5x6|fedbn_5x6|fedper_5x6|local_5x6) ;;
  *)
    echo "Unsupported --method: $METHOD (supported: fedavg, fedprox, fedbn, fedper, local, apfl, fedavg_5x6, fedprox_5x6, fedbn_5x6, fedper_5x6, local_5x6)"
    exit 1
    ;;
esac

JOB_DIR="${SCRIPT_DIR}/nvflare_job_mmasd_${METHOD}"
if [ ! -d "$JOB_DIR" ]; then
  # Backward-compatible fallback to the original job folder.
  JOB_DIR="${SCRIPT_DIR}/nvflare_job_mmasd"
fi

if [ -z "$WORKSPACE" ]; then
  TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
  WORKSPACE="${SCRIPT_DIR}/workspaces/nvflare_workspace_${METHOD}_${TIMESTAMP}"
  mkdir -p "$WORKSPACE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Created workspace: $WORKSPACE"
fi

if [ ! -d "$JOB_DIR" ]; then
  echo "Job directory not found: $JOB_DIR"
  exit 1
fi

if ! command -v nvflare &> /dev/null; then
  echo "Error: 'nvflare' command not found. Please activate the NVFLARE environment first."
  exit 1
fi

# FreqMixFormer root (for model/ and feeders/ imports when client runs)
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"
export FLARE_PROJECT_ROOT="$PROJ_ROOT"

# CUDA libs: if you get "libcufft.so.10: cannot open shared object file", your PyTorch
# was built for CUDA 10.x but the env has CUDA 11/12. Fix one of:
#   conda install -c nvidia libcufft   # try matching version
#   pip install torch --upgrade        # reinstall PyTorch for your CUDA
#   Or add your CUDA/cudatoolkit lib path:
if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using method:            $METHOD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using NVFLARE workspace: $WORKSPACE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] IMPORTANT: Clear old job cache if you changed the job: rm -rf \$WORKSPACE/*/simulate_job"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using job directory:     $JOB_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using project root:      $PROJ_ROOT (for imports)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting simulator with 3 clients (theme1, theme2, theme3)..."
echo

# For your NVFLARE version, the simulator CLI is:
#   nvflare simulator [options] job_folder
# and client names are given via -c/--clients as a comma-separated list.
nvflare simulator \
  -w "$WORKSPACE" \
  -n 3 \
  -t 3 \
  -c theme1,theme2,theme3 \
  "$JOB_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Simulator finished."

