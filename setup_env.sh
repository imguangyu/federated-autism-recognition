#!/usr/bin/env bash
# Create conda env 'freqmixformer' with Python 3.9 and required packages.
# Usage: bash setup_env.sh [cpu]
#   No arg: install PyTorch with CUDA 11.8 (pip). For other CUDA versions, create env then: pip install torch --index-url https://download.pytorch.org/whl/cu118
#   cpu:    install PyTorch CPU-only.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="${ENV_NAME:-freqmixformer}"
USE_CPU="${1:-}"

echo "Creating conda env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing PyTorch..."
if [ "$USE_CPU" = "cpu" ]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo "Installing other requirements..."
pip install -r requirements.txt

# Use the project's local torchlight (has DictAction); avoid conflict with PyPI torchlight
pip uninstall -y torchlight 2>/dev/null || true
pip install -e ./torchlight

echo "Done. Activate with: conda activate $ENV_NAME"
echo "Verify: python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
