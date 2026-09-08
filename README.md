# stable-diffusion-img2pcd

[trellis2-img2pcd](https://github.com/biop99999111/trellis2-img2pcd)의 이력을 유지한 Stable Diffusion 스파이크용 파생 저장소입니다. 동일 계정 소유 저장소에서 분리했으므로 GitHub의 fork 표시가 붙는 포크는 아닙니다.

**사진을 SDXL로 변형한 뒤 기존 3D 모델에 넣어 PCD를 만드는 실험**과 **Stable Fast 3D 직접 비교 실험**을 제공합니다.

| 경로 | 실행 | 확인할 것 |
|---|---|---|
| 원본 사진 → Hunyuan3D / TRELLIS → PCD | `img2pcd.py` | 기존 대조군 |
| 원본 사진 → SDXL img2img → 같은 3D 모델 → PCD | `sd_spike.py` 후 `img2pcd.py` | SD 전처리에 따른 형상·색 변화 |
| 원본 사진 → Stable Fast 3D → PCD | `img2pcd.py --backend sf3d` | 3D 복원 모델 자체 비교 |

SDXL은 2D 이미지 모델이며 PCD를 직접 출력하지 않습니다. Stable Fast 3D는 Stability AI의 별도 feedforward 3D 복원 모델이며 Stable Diffusion과 같은 모델이 아닙니다. [SDXL 공식 문서](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/sdxl), [SF3D 공식 저장소](https://github.com/Stability-AI/stable-fast-3d).

## 1. SDXL 스파이크 (Linux / vast.ai)

CUDA 12.4용 PyTorch를 설치하는 스크립트입니다. 기존 실험과 같은 RTX 4090급 환경을 대상으로 작성했으며 GPU 추론·설치는 아직 실측 검증하지 않았습니다. SDXL과 3D 모델은 서로 다른 conda 환경과 프로세스에서 실행합니다.

```bash
git clone https://github.com/biop99999111/stable-diffusion-img2pcd.git
cd stable-diffusion-img2pcd
bash setup_sd.sh
conda activate sdxl-spike

# 입력 파일, 설정, 이미지 디코딩 확인. 모델 다운로드/GPU 호출 없음
python sd_spike.py --dry-run

# 동일 사진, 낮은 변형 강도, seed 3개로 비교
python sd_spike.py --only bumper_cover --seeds 42 43 44 --strength 0.25 --out out_sd_images
```

`parts.yaml`의 이미지와 치수를 사용합니다. `--prompt`, `--negative-prompt`, `--steps`, `--guidance`, `--size`로 실험을 바꿀 수 있습니다. 기본 모델은 `stabilityai/stable-diffusion-xl-base-1.0`, 1024px, 30 steps입니다. 원본 비율을 유지해 흰색 패딩을 추가하고 투명 이미지도 흰 배경에 합성합니다. seed는 SD 이미지 생성에만 적용되고 후속 3D/샘플링 seed는 설정값을 유지합니다.

```text
out_sd_images/
  bumper_cover_input.png       SDXL 입력(패딩 적용)
  bumper_cover_sd_s42.png      변형 이미지(seed별)
  parts_baseline.yaml          원본 사진 대조군 설정
  parts_sd.yaml                변형 이미지 설정
  sd_manifest.json            프롬프트·seed·입력 SHA256·시간·VRAM·완료 여부
```

결과 혼합을 막기 위해 출력 폴더는 비어 있어야 합니다. 오류 발생 후에는 새 `--out` 경로를 사용하세요. 생성된 YAML의 이미지 경로는 절대 경로이므로 같은 머신에서 후속 처리를 실행합니다. 다른 머신으로 옮기면 YAML 경로도 수정해야 합니다.

### 같은 3D 모델로 대조군과 비교

```bash
bash setup_hunyuan3d.sh       # 미설치 시 한 번
conda activate hunyuan3d
python img2pcd.py --config out_sd_images/parts_baseline.yaml --out out_baseline
python img2pcd.py --config out_sd_images/parts_sd.yaml --out out_sd_pcd
```

양쪽 `run.json`의 `shape_check`, `bbox_mm`, `verify`와 부품별 `preview.png`를 비교합니다. `target_mm`로 최장축을 강제 보정하므로 그 축이 맞는 것만으로 형상이 정확하다고 판정하지 않습니다. `expect_mm`의 나머지 축과 최소축 배율도 확인해야 합니다. 기본 치수는 대략치이므로 실제 측정값으로 교체하세요.

SDXL은 외형을 바꿀 수 있습니다. 원본 사진 대조군과 비교하는 이 실험은 SDXL 전처리의 효과를 평가하며 3D 모델 교체 실험과 구분됩니다. 낮은 strength와 형태 유지 프롬프트도 치수 보존을 보장하지 않습니다. 세밀한 0.1~1mm 결함 계측 정확도는 별도 실측이 필요합니다.

## 2. Stable Fast 3D 직접 비교

[모델 페이지](https://huggingface.co/stabilityai/stable-fast-3d)에서 접근 권한을 받은 Hugging Face 계정의 토큰을 `HF_TOKEN`으로 설정하세요. 설치 스크립트는 CUDA 확장 빌드 전에 모델 설정 파일 접근부터 확인합니다. 모델 사용 조건은 해당 모델 페이지를 따릅니다.

```bash
export HF_TOKEN=hf_...       # 실제 토큰은 커밋하지 않음
bash setup_sf3d.sh          # conda + nvcc 있는 Linux CUDA devel 환경
conda activate sf3d-spike
python img2pcd.py --backend sf3d --out out_sf3d
```

기본 소스 위치는 형제 폴더 `stable-fast-3d`입니다. `--repo-dir` 또는 `SF3D_DIR`로 지정할 수 있습니다. SF3D는 알파가 없으면 배경을 제거하고, foreground ratio 0.85로 패딩 후 텍스처 메시를 생성합니다. `--texture` 없이도 색을 생성합니다. 텍스처 해상도는 `parts.yaml`의 `texture_size`를 사용합니다. 재메싱 옵션 예:

```bash
python img2pcd.py --backend sf3d --run-kwargs '{"remesh":"triangle","vertex_count":10000}' --out out_sf3d_remesh
```

SDXL→SF3D를 실험하려면 이미지 생성 시 `--backend sf3d`를 주고, 출력 YAML을 `sf3d-spike` 환경의 `img2pcd.py`에 전달하면 됩니다.

SF3D API는 [공식 run.py](https://github.com/Stability-AI/stable-fast-3d/blob/main/run.py)를 기준으로 작성했습니다. `setup_sf3d.sh`는 설치된 SF3D commit을 출력합니다. upstream 소스 설치는 고정 커밋이 아니므로 재현 실험 시 해당 commit도 보관하세요.

## 검증 상태

### Hunyuan 텍스처 단계 복구 (Python 3.10)

`ModuleNotFoundError: No module named 'bpy'`는 구형 설치 스크립트에서 bpy를 제외한 경우 발생합니다.
[Blender 공식 안내](https://pypi.org/project/bpy/)에 따라 보관소에서 설치합니다.

```bash
conda activate hunyuan3d
python -m pip install 'numpy<2' 'bpy==4.0.0' --extra-index-url https://download.blender.org/pypi/
python -c 'import bpy; print(bpy.app.version_string)'
git pull --ff-only
python img2pcd.py --config out_sd_bumper_01/parts_sd.yaml --texture-only --out out_sd_textured_01
```

`--texture-only`는 해당 출력 폴더의 `mesh_shape.glb`와 `input_nobg.png`를 재사용하며 shape 모델을 로드하지 않습니다.
원본 형상과 대응하는 이미지 설정으로만 재개하세요. OBJ·MTL·텍스처 중간 파일은 부품 폴더의 `paint_*/`에 보존됩니다.
텍스처 경로는 OBJ를 생성하고 Blender로 GLB 변환 후 바이너리 헤더 검증을 통과한 파일만 `mesh.glb`로 채택합니다.
모델 설정·RealESRGAN 가중치 경로는 Hunyuan 소스 기준 절대 경로로 지정합니다.

### CPU 검사

BasicSR 1.4.2는 삭제된 `torchvision.transforms.functional_tensor` 경로를 참조합니다.
Hunyuan 텍스처 실행 시 해당 모듈이 없는 경우에만 공개 API의 `rgb_to_grayscale`에 연결합니다.
설치 패키지나 torch/torchvision 버전은 변경하지 않습니다.
`python -m unittest test_basicsr_compat -v`로 import 호환 동작 3개를 검증합니다.

```bash
python -m unittest test_sd_spike -v
python smoke_test.py
```

- CPU 테스트 11개 통과: 입력 비율·투명도, 잘못된 설정, seed·설정·manifest, SF3D 호출 계약, 실제 GLB→PCD 통합, Hunyuan 텍스처 재개·GLB 검증.
- 기존 CPU 스모크 테스트 12/12 통과, 실제 범퍼 사진의 SDXL `--dry-run` 통과.
- 모델 호출은 모의 객체로 검증했습니다. **SDXL/SF3D 실제 GPU 추론과 설치 스크립트의 GPU 서버 실행은 미검증**입니다. 성능·형상 개선 수치는 아직 없습니다.
- `sd_manifest.json`의 시간은 SDXL 단계, 부품 `manifest.json`의 시간은 3D 단계입니다. VRAM은 PyTorch allocated peak이며 드라이버 전체 사용량과 다릅니다.

원본 Hunyuan3D 실측 결과와 TRELLIS 설치 기록은 [README_UPSTREAM.md](README_UPSTREAM.md), [HANDOFF.md](HANDOFF.md)에 보존했습니다. 해당 실측은 이번 SDXL/SF3D 결과가 아닙니다. 기존 `track_a/` 스캔 기반 결함 합성도 유지합니다.
