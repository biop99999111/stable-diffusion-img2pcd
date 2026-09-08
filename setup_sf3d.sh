#!/usr/bin/env bash
# Linux + CUDA devel image; SF3D texture baker/UV unwrapper build extensions.
set -eo pipefail
SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SF3D_DIR="${SF3D_DIR:-$(dirname "$SPIKE_DIR")/stable-fast-3d}"
SF3D_ENV_NAME="${SF3D_ENV_NAME:-sf3d-spike}"
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) is required'; exit 1; }
command -v conda >/dev/null || { echo 'conda is required'; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$SF3D_ENV_NAME"; then
  conda create -y -n "$SF3D_ENV_NAME" python=3.10
fi
conda activate "$SF3D_ENV_NAME"
python -m pip install 'huggingface-hub>=0.26,<1'
# Fail before CUDA builds if access to the gated weights has not been granted.
python -c 'from huggingface_hub import hf_hub_download; hf_hub_download("stabilityai/stable-fast-3d", "config.yaml")'
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install setuptools==69.5.1 wheel
if [ ! -d "$SF3D_DIR" ]; then
  git clone https://github.com/Stability-AI/stable-fast-3d.git "$SF3D_DIR"
fi
test -f "$SF3D_DIR/sf3d/system.py" || { echo "Invalid SF3D_DIR: $SF3D_DIR"; exit 1; }
cd "$SF3D_DIR"
python -m pip install --no-build-isolation -c "$SPIKE_DIR/constraints-sf3d.txt" -r requirements.txt 'open3d>=0.18' 'matplotlib>=3.8' 'pyyaml>=6'
python -m pip check
git rev-parse HEAD
echo "Ready: conda activate $SF3D_ENV_NAME"
echo "python $SPIKE_DIR/img2pcd.py --backend sf3d --repo-dir $SF3D_DIR --out out_sf3d"
