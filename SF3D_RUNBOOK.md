# SF3D 원본 사진 실험 실행 안내

## 서버 실행 (Linux Jupyter / RTX 4090)

새 서버는 아래 명령으로 코드를 받는다. 원본 범퍼·본네트 사진도 저장소에 포함되어 있다. Python 3.10 conda 환경을 생성하므로 conda가 필요하며 CUDA 확장을 빌드하기 위한 nvcc와 g++도 필요하다. 현재 설치 스크립트는 torch cu124를 사용하므로 CUDA toolkit 12.4 devel 환경을 기준으로 준비한다. nvidia-smi의 CUDA 표시는 toolkit 설치 확인을 대신하지 않는다.

```bash
cd /workspace
git clone https://github.com/biop99999111/stable-diffusion-img2pcd.git
cd stable-diffusion-img2pcd
git log -1 --oneline
nvidia-smi
nvcc --version
conda --version
g++ --version
```

이미 같은 폴더를 클론했다면 저장소에서 `git status --short`를 확인한 뒤 `git pull --ff-only`로 갱신한다. 모델 계정의 접근 조건 동의와 읽기 권한이 필요하다. [모델 페이지](https://huggingface.co/stabilityai/stable-fast-3d)에서 동의한 계정의 토큰을 기존 로그인 캐시 또는 `HF_TOKEN` 환경에서 제공한다. 토큰을 화면에 표시하지 않고 입력하려면 Bash에서 다음을 실행한다.

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
export HF_TOKEN
echo
```

```bash
cd /workspace/stable-diffusion-img2pcd
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p logs
set -o pipefail
bash setup_sf3d.sh 2>&1 | tee logs/sf3d_install.log
```

설치 성공 후:

```bash
conda activate sf3d-spike
python sf3d_workflow.py --dry-run --out out_sf3d_check_01
python sf3d_workflow.py --repo-dir /workspace/stable-fast-3d --out out_sf3d_bumper_01
```

`--dry-run`은 입력 디코딩·원본 해시·설정만 검사하며 GPU나 모델 다운로드를 사용하지 않는다. 출력 폴더는 실행마다 새 이름을 사용한다. 최초 추론은 텍스처 1024, 범퍼, 원본 사진, 30만 점, seed 42, 스무딩 없음, remesh none이다. SF3D가 텍스처를 자체 생성한다.

처음 결과의 `input_nobg.png`와 `validation.json`을 검토한 뒤 2048을 실행한다.

```bash
python sf3d_workflow.py --texture-resolution 2048 \
  --repo-dir /workspace/stable-fast-3d --out out_sf3d_bumper_02
```

`--only hood`는 본네트 외판을 선택한다. 모델 재현은 이전 `workflow.json`의 `model_revision` SHA를 `--revision`에 전달한다. 설치 소스 재현은 `SF3D_COMMIT`에 이전 `source_commit`을 설정하고 설치한다. 지정 commit은 로컬 clone에 존재해야 하며 수정된 소스는 자동으로 덮어쓰지 않는다.

설치 조합은 torch 2.5.1/cu124이며 서버의 nvcc 12.8과 확장 빌드 호환성은 실제 설치 로그로 확인한다. `setup_sf3d.sh`는 pip check와 확장 import까지 검사하고, 워크플로는 GPU·nvcc·소스 상태·환경을 기록한다. 가중치 snapshot은 HF revision을 SHA로 고정하여 다운로드하고 해당 로컬 snapshot을 추론에 사용한다.

## 산출물·검증

실행 루트에 `workflow.json`, `parts_sf3d.yaml`, `freeze.txt`, `inference.log`, `run.json`을 저장한다. `bumper_cover/`에는 GLB, mm PLY, 색상 PCD, 배경 제거 입력, 미리보기, manifest와 `validation.json`이 저장된다. 실패 시 workflow 상태와 오류를 보존한다. 기존 결과를 덮어쓰지 않고 새 출력에서 재시도한다.

```bash
python validate_sf3d.py out_sf3d_bumper_01/bumper_cover
python compare_sf3d.py out_sf3d_bumper_01/bumper_cover/manifest.json \
  out_sf3d_bumper_02/bumper_cover/manifest.json --out sf3d_comparison.md
```

비교기는 Hunyuan `manifest.json`도 받는다. 기록되지 않은 성능·입력 해시는 미기록으로 표시한다. 입력이 다른 SDXL→Hunyuan 결과는 참고군이다. 생성 시간은 모델별 포함 범위가 다르므로 그대로 순위를 매기지 않는다.

검증기는 GLB 각 인스턴스의 실제 UV/텍스처, mm 메시 bbox, PCD 점 수·유한 좌표·색·범위, 이미지 재읽기를 확인한다. 색이 거의 일정하면 수동 검토 경고를 남긴다. bbox 형상 판정과 기술적 산출 성공은 구분하며 계측은 계속 미검증으로 표시한다. 정면·측면·후면 품질은 사람이 확인한다.

## GLB 직접 회전 영상

로컬 또는 OpenGL 지원 환경에서 실행한다. Windows 작업 공간에 기존 `.video_tools`가 있으면 재사용한다. 다른 환경은 다음 패키지가 필요하다.

```bash
python -m pip install -r requirements-sf3d-video.txt
python render_sf3d_bumper_video.py out_sf3d_bumper_01/bumper_cover/mesh.glb \
  --preview --out out_sf3d_angles_01
python render_sf3d_bumper_video.py out_sf3d_bumper_01/bumper_cover/mesh.glb \
  --out out_sf3d_video_01
```

Linux headless에서는 지원 드라이버 환경에 맞게 `--gl-backend egl`을 추가한다. `--front-angle 180`처럼 정면 기준을 조절할 수 있다. 출력은 `bumper_turntable.mp4`, `bumper_preview.jpg`, `angle_check.jpg`, `video_midpoint.jpg`, `render_report.json`이다. 1280×720, 30fps, 12초, 360프레임을 전체 디코딩하여 검증한다.

scene transform과 여러 geometry/material을 유지하고, 내장 base-color 이미지·색 factor와 metallic/roughness 이미지 또는 scalar factor를 처리한다. OPAQUE/MASK 지원. 투명 BLEND는 오류로 안내하며 normal/occlusion/emissive 맵은 이 간단한 렌더러 범위에 포함하지 않는다. 원본 GLB는 수정하지 않는다. 기존 영상의 조명·배경·33도 시야각·회전 속도 곡선을 재사용하며 화면 잘림 방지를 위해 모델별 bounding sphere로 카메라 거리를 조절한다. 엄밀한 비교 시 보고서의 표시 조건과 방향을 함께 확인한다.

## 로컬 검증 결과와 남은 확인

2026-09-08: CPU 테스트 24개 통과. 실제 원본 범퍼 dry-run과 합성 GLB(2개 인스턴스, 내장 텍스처, scalar PBR 재질)의 12초 영상 생성·360프레임 디코딩을 확인했다. AMD Radeon 860M에서 4방향 미리보기의 텍스처와 인스턴스 배치도 확인했다. 합성 fixture는 SF3D 추론 결과가 아니다.

RTX 4090 SF3D 설치·가중치 로딩·실제 추론은 아직 검증되지 않았다. 서버 실행 후 `inference.log`, `workflow.json`, 부품 폴더를 수령하여 실제 결과 품질과 영상을 확인해야 한다.
