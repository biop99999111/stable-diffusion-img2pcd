#!/usr/bin/env bash
# vast.ai(Linux + CUDA) 에서 Hunyuan3D-2.1 + 이 스파이크를 세팅한다.
#
#   bash setup_hunyuan3d.sh
#
# TRELLIS.2 와 달리 gated 모델 의존성이 없어 승인 없이 바로 돈다.
# VRAM: shape 10GB / texture 21GB (24GB 에서는 shape 를 내리고 texture 를 올린다)
#
# set -u 는 쓰지 않는다 — conda 활성화 스크립트가 미설정 변수를 참조해서 죽는다.
set -eo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_DIR="${HUNYUAN3D_DIR:-$(dirname "$SPIKE_DIR")/Hunyuan3D-2.1}"
ENV_NAME="${ENV_NAME:-hunyuan3d}"

echo "=== 0. 환경 확인 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi 실패 — GPU 인스턴스가 맞는지 확인하세요"; exit 1
}
echo "설치 위치: $HY_DIR"
df -h "$(dirname "$HY_DIR")" | tail -1

if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
echo "CUDA_HOME=${CUDA_HOME:-(unset)}"

echo
echo "=== 1. conda 훅 로드 ==="
# 비대화형 서브셸에서는 conda activate 가 그냥은 안 된다.
command -v conda >/dev/null 2>&1 || { echo "!! conda 없음"; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo
echo "=== 2. conda env '$ENV_NAME' (python 3.10) ==="
# TRELLIS.2 env(torch 2.6)와 섞지 않는다 — Hunyuan3D 는 torch 2.5.1 을 쓴다.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "이미 있음: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
echo "python: $(command -v python)"

echo
echo "=== 3. Hunyuan3D-2.1 클론 ==="
if [ -d "$HY_DIR/.git" ]; then
  echo "이미 있음: $HY_DIR"
else
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY_DIR"
fi
cd "$HY_DIR"

echo
echo "=== 4. torch 2.5.1 (cu124) ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo
echo "=== 5. requirements ==="
pip install -r requirements.txt

echo
echo "=== 6. texture 확장 빌드 (실패해도 shape 는 돈다) ==="
# custom_rasterizer / DifferentiableRenderer 는 --texture 에서만 쓰인다.
# 여기서 깨져도 shape-only 실행은 가능하므로 전체를 중단시키지 않는다.
TEXTURE_OK=1
(
  cd hy3dpaint/custom_rasterizer && pip install -e .
) || { echo "!! custom_rasterizer 빌드 실패 — --texture 는 못 씁니다"; TEXTURE_OK=0; }
(
  cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
) || { echo "!! DifferentiableRenderer 빌드 실패 — --texture 는 못 씁니다"; TEXTURE_OK=0; }

echo
echo "=== 7. RealESRGAN 체크포인트 (texture 용) ==="
mkdir -p hy3dpaint/ckpt
if [ -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]; then
  echo "이미 있음"
else
  wget -q --show-progress \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P hy3dpaint/ckpt || echo "!! 다운로드 실패 — --texture 사용 시 필요합니다"
fi

echo
echo "=== 8. 스파이크 의존성 + Jupyter 커널 ==="
pip install -r "$SPIKE_DIR/requirements.txt"
pip install ipykernel
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo
echo "=== 9. 임포트 검증 ==="
cd "$HY_DIR"
python - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "hy3dshape"))
sys.path.insert(0, os.path.join(os.getcwd(), "hy3dpaint"))

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, f"{p.total_memory / 2**30:.1f} GiB")

missing = []
try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    print("  OK    hy3dshape (shape 생성)")
except Exception as e:
    print(f"  FAIL  hy3dshape: {e}")
    missing.append("hy3dshape")

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline  # noqa: F401
    print("  OK    hy3dpaint (texture, --texture 옵션)")
except Exception as e:
    print(f"  WARN  hy3dpaint: {e}")
    print("        -> shape-only 는 정상 동작합니다(--texture 만 불가)")

for m in ("trimesh", "open3d", "yaml", "matplotlib"):
    try:
        __import__(m)
        print(f"  OK    {m}")
    except Exception as e:
        print(f"  FAIL  {m}: {e}")
        missing.append(m)

print("\n=> shape 실행 가능" if not missing else f"\n=> 실패: {missing}")
PY

cat <<EOF

=== 완료 ===
  cd $SPIKE_DIR
  conda activate $ENV_NAME

  # 1) shape 만 (빠름 · VRAM ~10GB · PCD 색은 회색)
  python img2pcd.py --backend hunyuan3d --only bumper_cover --out out

  # 2) 텍스처까지 (느림 · VRAM ~21GB · PCD 에 색이 들어감)
  python img2pcd.py --backend hunyuan3d --texture --only bumper_cover --out out

레포 자동 탐색이 안 되면:  --repo-dir $HY_DIR
산출물: out/<부품>/{mesh.glb, mesh_mm.ply, <부품>.pcd, preview.png, manifest.json}
EOF
