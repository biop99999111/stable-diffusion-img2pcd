# 인계 문서 — 2026-09-08

## 목적

자동차 부품(범퍼 커버·본네트) 사진 1장 → 3D → **mm 스케일 점군(`.pcd`)** 이 되는지 입증.
상위 프로젝트는 별도 private 레포 `biop99999111/bumper-synth`(합성 결함 데이터 PoC)이고,
여기서 만든 PCD 가 쓸 만하면 그쪽 FR-1(`bumper asset add`) 입력으로 넘긴다.

## 현재 상태

**파이프라인 관통 성공 (Hunyuan3D-2.1)** — 그러나 **에셋 품질은 사용 불가**.

| 항목 | 결과 |
|---|---|
| 이미지 → 3D → mm PCD | ✅ 전 구간 동작 |
| 소요 | 33.5s/부품 (Diffusion 50스텝 14s + Volume decoding 18s) |
| VRAM 피크 | **7.63 GiB** (RTX 4090 24GB 기준 여유) |
| 산출 | 300,000 points · 4.58 MB · 되읽기 검증 통과 |
| **형상 품질** | ❌ 부풀림(pillow inflation) |

### 실측 bbox — 이게 핵심 문제

| 부품 | 실제 대략치 (mm) | 생성 결과 (mm) | 판정 |
|---|---|---|---|
| 범퍼 커버 | 1750 × 450 × 350 | 1750 × **963** × **516** | 높이 2배 |
| 본네트 | 1500 × 1300 × 200 | 1490 × 1240 × **1500** | 거의 정육면체 |

- 본네트는 `fit_axis=z` 로 잡혔다 = **최장축이 깊이**. 판때기라면 불가능한 결과.
- 본네트 표면적 16.2 m² (실제 양면 합쳐 ~4 m²) → 4배.
- 본네트가 해상도 4배(1299×1023 vs 600×342)인데 결과는 더 나쁨
  → **입력 해상도 문제가 아니라 방법의 한계**.

원인: 단일 뷰 image-to-3D 는 보이지 않는 뒷면을 지어내야 하는데, 학습 데이터가
대체로 "부피 있는 물체"라 **얇은 판을 얇게 유지하지 못한다**. 정면 직교 사진은
깊이 단서가 가장 적은 각도라 더 불리하다.

**결론**: 결함 깊이 0.1~1mm 를 다루는 파이프라인에 깊이가 7배 틀린 메시는 못 넣는다.
`bumper-synth` spec 이 FR-1/FR-2 에서 **3D 카메라 점군**을 원본으로 잡은 것은 여전히 옳다.
생성 모델은 대체재가 아니라 잘해야 형상 다양성 보조.

## 다음에 할 일 (우선순위)

1. **TRELLIS.2 로 같은 사진 재현** — DINOv3 gated 승인 완료(아래 참조). 미검증 백엔드.
   `bbox_mm` 의 깊이가 300mm 대로 나오면 모델 차이가 실재, 비슷하게 부풀면 방법의 한계 확정.
2. **3/4 측면 뷰 사진으로 재시도** — 모델 교체보다 이쪽이 효과가 클 가능성이 높다.
   정면 직교가 깊이 단서 최소 각도이므로.
3. (선택) **SPAR3D 백엔드 추가** — 중간 산출이 점군이라 FR-1 과 궁합이 좋고,
   생성 점군을 사람이 직접 편집 가능. 7GB/0.7초로 비용 거의 없음. 어댑터 30분.

## 환경 (중요)

**vast.ai 인스턴스는 삭제됨.** 재개 시 처음부터 설치 필요.

| | 크기 | 시간 |
|---|---|---|
| Hunyuan3D env + 의존성 | ~10GB | 10~15분 |
| Hunyuan3D 모델 | 8GB | ~1분 (169MB/s 실측) |
| TRELLIS.2 env + 확장 빌드 | ~30GB | 20~40분 |
| TRELLIS.2 가중치 | ~15GB | ~2분 |

### GPU 선택

- ✅ **RTX 4090 24GB** — 검증됨. Hunyuan3D 피크 7.63GiB 로 여유.
- ✅ L40S / A6000 / RTX 6000 Ada (48GB), A100 — 여유가 필요하면 이쪽.
- ❌ **RTX 5090 금지** — Blackwell(sm_120)인데 두 프로젝트 모두 **torch cu124**(sm_90 까지)를
  핀으로 박는다. `no kernel image is available` 로 죽는다.
- ⚠️ **`*-devel` 이미지 또는 PyTorch 템플릿** 필수. `*-runtime` 은 nvcc 가 없어 소스 빌드 실패.
- 디스크 **100GB+**.

### HuggingFace

- DINOv3 는 **Gating Group Collection** 으로 묶여 있어 컬렉션 승인이 개별 모델 전체에 적용된다.
  2026-09-08 **ACCEPTED 확인** (`facebook/dinov3-vitl16-pretrain-lvd1689m` 파일 목록 접근 가능).
- `export HF_TOKEN=...` 필요. **Classic → Read** 토큰이 확실하다
  (fine-grained 는 'Read access to contents of all public gated repos' 체크가 빠지면 403).
- 이전 세션에서 노출된 토큰 3개는 **revoke 할 것**.

## 재개 절차

```bash
git clone https://github.com/biop99999111/trellis2-img2pcd.git
cd trellis2-img2pcd
export HF_TOKEN=<새 토큰>

python check_env.py            # FAIL 0 확인 (gated 접근 포함)

# TRELLIS.2 (미검증 백엔드 — 1순위 과제)
bash setup_vast.sh 2>&1 | tee setup.log
conda activate trellis2
python img2pcd.py --backend trellis2 --only bumper_cover --out out_trellis 2>&1 | tee t1.log

# Hunyuan3D (검증됨 — 비교 기준)
bash setup_hunyuan3d.sh 2>&1 | tee setup_hy.log
conda activate hunyuan3d
python img2pcd.py --only bumper_cover --out out_hy
```

`--out` 을 나눠 `out_trellis/*/preview.png` 와 `out_hy/*/preview.png` 를 나란히 비교한다.
**판정 기준은 `bbox_mm`** — 위 표의 실제 치수와 대조.

## 해결된 함정 (스크립트에 반영 완료 — 다시 안 만난다)

| # | 증상 | 원인 | 조치 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: trellis2` | pip 패키지가 아니라 레포 소스 트리 | 자동 탐색 후 `sys.path` 추가 |
| 2 | DINOv3 401 → 403 | `gated: manual` (Meta 수동 승인) | 승인 완료 |
| 3 | pip 무한 정지 | basicsr 빌드 격리가 torch 재다운로드(~2.5GB) | `--no-build-isolation` + cython 선설치 |
| 4 | 1.1 MB/s 스로틀 (회선은 169 MB/s) | `requirements.txt` 안에 중국 미러 `--extra-index-url` | grep 으로 해당 줄 제외 |
| 5 | requirements 전체 실패 | `bpy==4.0` 이 PyPI 에서 삭제됨(4.2.0+ 는 py>=3.11) | grep 으로 제외. shape 경로엔 불필요함 확인 |
| 6 | `conda activate` 실패 | 비대화형 서브셸 | `profile.d/conda.sh` 훅 선로드, `set -u` 제거 |

⚠️ 4번 관련: **설치 도중에 pip 인덱스를 바꾸지 말 것.** 캐시 키가 URL 기준이라
무효화되어 받은 걸 다시 받는다. 처음부터 정리된 requirements 로 시작할 것.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `img2pcd.py` | 오케스트레이션 + CPU 3~6단계 (스케일·스무딩·샘플링·PCD·검증·미리보기) |
| `backends.py` | GPU 1~2단계. `Trellis2Backend` / `Hunyuan3DBackend` |
| `check_env.py` | 설치 전 진단 (stdlib 만). nvcc·VRAM·디스크·conda·gated 접근 |
| `smoke_test.py` | GPU 없이 CPU 단계 검증. **8/8 passed** |
| `setup_vast.sh` / `setup_hunyuan3d.sh` | 설치 자동화 |
| `run_spike.ipynb` | Jupyter 9섹션 |
| `parts.yaml` | 부품 정의 + `target_mm`(실측 치수) + defaults |

**백엔드 경계**: GPU 를 쓰는 1~2단계만 `backends.py` 에 격리. 3~6단계는 GLB 하나만
받으므로 모델을 갈아끼워도 그대로 돌고, 두 모델 결과를 같은 뒷단으로 비교할 수 있다.

**버전 방어**: `inspect.signature()` 로 호출 대상이 실제로 받는 인자만 넘긴다.
`run(image, seed=...)` 처럼 문서에 없는 인자로 죽는 것을 막는다.

**GPU/CPU 분리**: `--skip-generate` 로 기존 GLB 에서 3~6단계만 재실행(몇 초).
스케일·점 개수·스무딩을 GPU 없이 반복할 수 있다.

## 알려진 제약

- **mm 스케일은 자동이 안 된다** — 생성 메시는 단위 박스로 정규화되어 나온다.
  `parts.yaml` 의 `target_mm` 은 **사람이 넣는 실측값**이고, 틀리면 뒤가 전부 틀린다.
  현재 값(범퍼 1750 / 본네트 1500)은 대략치이므로 실측으로 교체 필요.
- `fit_axis: auto` 는 최장축을 고르는데, **형상이 틀리면 엉뚱한 축을 고른다**
  (본네트가 실제로 그랬다). 형상 검증 없이 스케일 숫자만 믿으면 안 된다.
- `--texture` 는 아직 미검증. Hunyuan3D 는 shape 10GB / texture 21GB 라
  shape 를 완전히 해제한 뒤 texture 를 올리도록 짜두었다(paint 가 메시를 경로로 받는 점 이용).
  `custom_rasterizer` · `DifferentiableRenderer` 빌드 + RealESRGAN 체크포인트가 필요하다.
- texture 없이 돌리면 **PCD 의 rgb 가 회색**이 된다(형상 입증에는 무관).
- `run.json` 은 그 실행에서 처리한 부품만 담고 덮어쓴다. 부품별 `manifest.json` 은 보존된다.

## 참고

- TRELLIS.2 이슈 [#188](https://github.com/microsoft/TRELLIS.2/issues/188) — 23GB 급에서
  1024³/1536³ 메시 후처리 OOM. 중간 텐서 21GB 잔류 → `del` + `empty_cache` 로 11GB.
  코드에 `expandable_segments:True` + 부품 사이 `empty_cache()` 반영됨.
- TRELLIS.2 이슈 [#106](https://github.com/microsoft/TRELLIS.2/issues/106) — CuMesh 빌드에 `CUDA_HOME` 필요.
- 상위 프로젝트 spec: `bumper-synth` private 레포 `docs/bumper-synth-spec.md` (FR-1~18).
  ⚠️ 그 레포는 병렬 구현 트랙이 같은 워크트리에서 동시에 스테이징하므로
  **반드시 `git commit -- <경로>`** 로 경로를 지정해 커밋할 것.
