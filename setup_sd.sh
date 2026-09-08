#!/usr/bin/env bash
# Linux / vast.ai. Keep SDXL dependencies separate from reconstruction models.
set -eo pipefail
SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_ENV_NAME="${SD_ENV_NAME:-sdxl-spike}"
command -v conda >/dev/null || { echo 'conda is required'; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$SD_ENV_NAME"; then
  conda create -y -n "$SD_ENV_NAME" python=3.10
fi
conda activate "$SD_ENV_NAME"
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$SPIKE_DIR/requirements-sd.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name())'
echo "Ready: conda activate $SD_ENV_NAME"
echo "python sd_spike.py --dry-run"
