#!/usr/bin/env bash
# vast.ai(Linux + CUDA) 에서 TRELLIS.2 + 이 스파이크를 세팅한다.
#
#   bash setup_vast.sh
#
# 인스턴스 요구: NVIDIA 24GB+ (4090/A10G/L40S...), 디스크 100GB+, CUDA 12.x
# 소요: 모델 ~15GB + 의존성 ~30GB 다운로드라 20~40분 걸린다.
set -euo pipefail

TRELLIS_DIR="${TRELLIS_DIR:-$HOME/TRELLIS.2}"
SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 0. 환경 확인 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi 실패 — GPU 인스턴스가 맞는지 확인하세요"; exit 1
}
df -h "$HOME" | tail -1
echo "CUDA_HOME=${CUDA_HOME:-(unset)}"

# CUDA 가 여러 개 깔린 이미지에서 CuMesh 빌드가 깨지는 걸 막는다(이슈 #106).
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
  echo "CUDA_HOME 을 $CUDA_HOME 로 설정"
fi

echo
echo "=== 1. TRELLIS.2 클론 ==="
if [ -d "$TRELLIS_DIR/.git" ]; then
  echo "이미 있음: $TRELLIS_DIR"
else
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
fi

echo
echo "=== 2. TRELLIS.2 설치 (공식 setup.sh · conda env 'trellis2' 생성) ==="
cd "$TRELLIS_DIR"
# 공식 README 의 명령 그대로. cumesh/o-voxel/flexgemm 이 빠지면 to_glb 가 안 돈다.
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

echo
echo "=== 3. 스파이크 의존성 ==="
# setup.sh 가 conda env 'trellis2' 를 활성화한 상태여야 한다.
python -c "import sys; print('python:', sys.executable)"
pip install -r "$SPIKE_DIR/requirements.txt"

echo
echo "=== 4. Jupyter 커널 등록 (노트북에서 trellis2 env 선택용) ==="
pip install ipykernel
python -m ipykernel install --user --name trellis2 --display-name "Python (trellis2)"

echo
echo "=== 5. 임포트 검증 ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
for m in ("trellis2", "o_voxel", "trimesh", "open3d"):
    try:
        __import__(m)
        print(f"  {m:10s} OK")
    except Exception as e:
        print(f"  {m:10s} FAIL: {e}")
PY

cat <<EOF

=== 완료 ===
다음 순서로 진행하세요.

  1) 이미지 2장을 $SPIKE_DIR/inputs/ 에 올린다
       inputs/bumper.jpg   (범퍼 커버)
       inputs/hood.jpg     (본네트)
     배경이 단순하고 부품 하나만 크게 찍힌 이미지가 잘 나옵니다.

  2) parts.yaml 의 target_mm 을 실측값으로 고친다 (스케일 보정 기준)

  3) 실행
       cd $SPIKE_DIR
       conda activate trellis2      # 새 셸이면 필요
       python img2pcd.py --config parts.yaml --out out

  4) 노트북으로 하려면 run_spike.ipynb 를 열고 커널을 "Python (trellis2)" 로 바꾼다

산출물: out/<부품>/{mesh.glb, mesh_mm.ply, <부품>.pcd, preview.png, manifest.json}
EOF
