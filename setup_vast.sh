#!/usr/bin/env bash
# vast.ai(Linux + CUDA) 에서 TRELLIS.2 + 이 스파이크를 세팅한다.
#
#   bash setup_vast.sh
#
# 인스턴스 요구: NVIDIA 24GB+ (4090/A10G/L40S...), 디스크 100GB+, CUDA 12.x
# 먼저 `python check_env.py` 로 FAIL 0 을 확인할 것.
# 소요: 모델 ~15GB + 의존성 ~30GB 다운로드라 20~40분 걸린다.
#
# set -u 는 쓰지 않는다 — conda 활성화 스크립트가 미설정 변수를 참조해서 죽는다.
set -eo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 기본 위치는 이 레포의 형제 폴더. vast.ai 는 $HOME 이 아니라 /workspace 가
# 큰 디스크인 경우가 많아서 $HOME 을 기본으로 쓰지 않는다.
TRELLIS_DIR="${TRELLIS_DIR:-$(dirname "$SPIKE_DIR")/TRELLIS.2}"

echo "=== 0. 환경 확인 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi 실패 — GPU 인스턴스가 맞는지 확인하세요"; exit 1
}
echo "설치 위치: $TRELLIS_DIR"
df -h "$(dirname "$TRELLIS_DIR")" | tail -1

# gated 레포 접근을 여기서 먼저 본다. 확장 빌드 30분을 태운 뒤 401 을 만나면
# 빌린 GPU 시간이 그대로 날아간다. DINOv3 와 briaai/RMBG-2.0 둘 다 필요하다.
if ! python3 "$SPIKE_DIR/check_env.py" --hf; then
  echo
  echo "!! HF 접근 확인 실패 — 위 안내대로 약관 동의/토큰을 처리한 뒤 다시 실행하세요."
  echo "   (RMBG-2.0 승인 없이 가려면 나중에 --rembg ZhengPeng7/BiRefNet 로 실행)"
  exit 1
fi

# CUDA 가 여러 개 깔린 이미지에서 CuMesh 빌드가 깨지는 걸 막는다(이슈 #106).
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
  echo "CUDA_HOME 을 $CUDA_HOME 로 설정"
else
  echo "CUDA_HOME=${CUDA_HOME:-(unset)}"
fi

echo
echo "=== 1. conda 훅 로드 ==="
# `bash setup_vast.sh` 는 비대화형 서브셸이라 conda activate 가 그냥은 안 된다
# ("Your shell has not been properly configured to use 'conda activate'").
# 공식 setup.sh 가 --new-env 에서 activate 를 하므로 훅을 먼저 읽어둔다.
if ! command -v conda >/dev/null 2>&1; then
  echo "!! conda 가 없습니다 — check_env.py 를 먼저 돌려보세요"; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "conda base: $CONDA_BASE"

echo
echo "=== 2. TRELLIS.2 클론 ==="
if [ -d "$TRELLIS_DIR/.git" ]; then
  echo "이미 있음: $TRELLIS_DIR"
else
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
fi

echo
echo "=== 3. TRELLIS.2 설치 (공식 setup.sh · conda env 'trellis2' 생성) ==="
cd "$TRELLIS_DIR"
# 공식 README 의 명령 그대로. cumesh/o-voxel/flexgemm 이 빠지면 to_glb 가 안 돈다.
# setup.sh 안에서 비영(非零) 종료가 섞여도 전체가 죽지 않도록 -e 를 잠시 끈다.
set +e
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
SETUP_RC=$?
set -e
[ "$SETUP_RC" -ne 0 ] && echo "!! setup.sh 종료코드 $SETUP_RC — 아래 검증 결과로 실제 상태를 확인하세요"

# setup.sh 가 activate 를 못 했을 경우를 대비해 한 번 더 시도한다.
if [ "${CONDA_DEFAULT_ENV:-}" != "trellis2" ]; then
  echo "trellis2 env 가 활성화되지 않음 — 직접 activate 시도"
  conda activate trellis2
fi
echo "현재 env: ${CONDA_DEFAULT_ENV:-(none)} · python: $(command -v python)"

echo
echo "=== 4. 스파이크 의존성 ==="
pip install -r "$SPIKE_DIR/requirements.txt"

# TRELLIS.2 setup.sh 가 transformers 를 핀 없이 깔아 5.x 가 들어오면 DINOv3
# 특징 추출이 죽는다. requirements.txt 의 핀이 실제로 먹었는지 확인한다.
python -c "import transformers as t; v=t.__version__; print('transformers', v); print('!! 5.x 입니다. 4.57.x 를 권장: pip install transformers==4.57.6') if int(v.split(chr(46))[0])>=5 else None" || true

echo
echo "=== 4.5. 가중치 미리 받기 (16.4 GiB) ==="
python "$SPIKE_DIR/prefetch_models.py" || {
  echo "!! 가중치 다운로드 실패 — 위 로그의 레포를 확인하세요"; exit 1
}

echo
echo "=== 5. Jupyter 커널 등록 (노트북에서 trellis2 env 선택용) ==="
pip install ipykernel
python -m ipykernel install --user --name trellis2 --display-name "Python (trellis2)"

echo
echo "=== 6. 임포트 검증 ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
missing = []
for m in ("trellis2", "o_voxel", "trimesh", "open3d"):
    try:
        __import__(m)
        print(f"  OK    {m}")
    except Exception as e:
        print(f"  FAIL  {m}: {e}")
        missing.append(m)
print("\n=> 설치 완료" if not missing else f"\n=> 실패: {missing}")
PY

cat <<EOF

=== 완료 ===
  1) parts.yaml 의 target_mm 을 실측값으로 고친다 (스케일 보정 기준)
     입력 이미지는 이미 $SPIKE_DIR/inputs/ 에 들어 있다.

  2) 실행
       cd $SPIKE_DIR
       conda activate trellis2      # 새 셸이면 필요
       python img2pcd.py --backend trellis2 --out out_trellis

     24GB 에서 메시 후처리 OOM 이 나면 자동으로 512 로 한 번 재시도한다.
     처음부터 가볍게 가려면 --pipeline-type 512
     RMBG-2.0 승인이 없으면 --rembg ZhengPeng7/BiRefNet

  3) 노트북으로 하려면 run_spike.ipynb 를 열고 커널을 "Python (trellis2)" 로 바꾼다

산출물: out/<부품>/{mesh.glb, mesh_mm.ply, <부품>.pcd, preview.png, manifest.json}
EOF
