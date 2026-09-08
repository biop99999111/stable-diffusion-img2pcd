# trellis2-img2pcd

이미지 **1장** → [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) → **mm 스케일 점군(`.pcd`)**.

자동차 부품(범퍼 커버·본네트) 사진을 3D 로 만들어 점군으로 뽑는 게 실제로 되는지 확인하려고
만든 스파이크다. 부품에 종속된 건 `parts.yaml` 뿐이라 다른 물체에도 그대로 쓸 수 있다.

## 파이프라인

| 단계 | 하는 일 | GPU |
|---|---|---|
| 1 | TRELLIS.2 `pipeline.run(image)` → 메시 (단위 박스 `[-0.5, 0.5]`) | ✅ |
| 2 | `o_voxel.postprocess.to_glb` → GLB(PBR) | ✅ |
| 3 | 최장축을 `target_mm` 에 맞춰 스케일, 원점 = bbox 중심 | — |
| 4 | (선택) Taubin 스무딩 | — |
| 5 | 표면 균등 샘플링 → xyz(mm) + rgb | — |
| 6 | `.pcd` 저장 + 3면도 PNG + `manifest.json` + 되읽기 검증 | — |

3~6단계는 CPU 전용이라 `--skip-generate` 로 기존 GLB 에서 몇 초 만에 다시 돌릴 수 있다.
점 개수·스무딩·단위 조정은 GPU 재실행 없이 여기서 반복한다.

## 환경 요구사항

| 항목 | 필요 | 미리 준비? |
|---|---|---|
| OS | **Linux** | ✅ 인스턴스 선택 |
| GPU | NVIDIA **24GB+** (4090 / A10G / L40S) | ✅ 인스턴스 선택 |
| **CUDA toolkit (`nvcc`)** | **필수** | ✅ **`*-devel` 이미지 또는 PyTorch 템플릿** |
| conda | 필수 (`setup.sh --new-env` 가 씀) | ✅ |
| gcc | 필수 (CUDA 확장 빌드) | ✅ 보통 이미지에 있음 |
| 디스크 | 100GB+ (모델 15GB + 의존성 30GB) | ✅ 인스턴스 선택 |
| **PyTorch** | 필수 | ❌ **`setup.sh` 가 `torch==2.6.0`(cu124) 를 직접 설치** |

**PyTorch 는 미리 깔 필요가 없다.** `setup.sh --new-env` 가 `conda create -n trellis2 python=3.10`
후 torch 를 그 env 에 설치한다. 이미지에 이미 있는 torch 는 쓰이지 않는다.

**대신 CUDA toolkit 은 반드시 있어야 한다.** CuMesh · o-voxel · flexgemm · nvdiffrast 가
**소스 빌드**라서 `nvcc` 가 필요하다. vast.ai 에서 `*-runtime` 이미지를 고르면 여기서 실패한다.

### HuggingFace gated 모델 (필수)

TRELLIS.2 는 이미지 인코더로 **[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)**
을 쓰는데 이게 **gated repo** 다. 승인·토큰 없이 돌리면 파이프라인 로드가 이렇게 죽는다:

```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo for url .../dinov3-vitl16-pretrain-lvd1689m/resolve/main/config.json
```

1. https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m 에서 **약관 동의** (보통 즉시 승인)
2. https://huggingface.co/settings/tokens 에서 **read 토큰** 발급
   - ⚠️ fine-grained 토큰이면 **"Read access to contents of all public gated repos you can access"**
     를 반드시 체크한다. 안 그러면 승인을 받아도 계속 401 이 난다. Classic → Read 가 확실하다.
3. `export HF_TOKEN=hf_...`

토큰이 있으면 TRELLIS.2-4B 가중치(~15GB) 다운로드의 rate limit 도 같이 풀린다.
`check_env.py` 가 이걸 **실제 HTTP 요청으로** 확인하므로, 15GB 를 받기 전에 걸러진다.

## 실행

```bash
git clone https://github.com/biop99999111/trellis2-img2pcd.git
cd trellis2-img2pcd

python check_env.py         # 1. 진단 — FAIL 0 확인 (표준 라이브러리만 씀)
python smoke_test.py        # 2. (선택) GPU 없이 후반부 검증
                            #    pip install trimesh open3d matplotlib pyyaml 필요
bash setup_vast.sh          # 3. TRELLIS.2 + 의존성 + Jupyter 커널 (20~40분)

conda activate trellis2
python img2pcd.py --config parts.yaml --out out
```

노트북으로 하려면 **`run_spike.ipynb`** 를 열고 커널을 **Python (trellis2)** 로 바꾼다.
(진단 → 스모크 → 설치 → 입력 → 로드 → 생성 → 3면도 → 검증표)

> Jupyter 셀 안에서는 `!python check_env.py`, 터미널에서는 `python check_env.py`.
> 터미널에 `!` 를 붙이면 bash 히스토리 확장으로 `event not found` 가 난다.

## 입력

`inputs/` 에 샘플 3장이 들어 있다. 배경이 단순하고 부품 하나만 크게 찍힌 이미지가 잘 나온다.

| 파일 | 크기 | 내용 |
|---|---|---|
| `inputs/bumper.jpg` | 600×342 | 전면 범퍼 커버 |
| `inputs/hood.png` | 1299×1023 | 본네트 외판(앞면) |
| `inputs/hood_inner.png` | 1366×1042 | 본네트 내판(뒷면) — 참고용, `parts.yaml` 에 주석 처리 |

`parts.yaml` 에서 이미지 경로와 **`target_mm`(실측 치수)** 를 지정한다.

## 산출물

```
out/<부품>/
  mesh.glb          TRELLIS.2 원본 (단위 박스, PBR 텍스처)
  mesh_mm.ply       mm 스케일 메시
  <부품>.pcd        ★ xyz(mm) + rgb
  preview.png       XY/XZ/YZ 3면도 (헤드리스에서도 생성)
  manifest.json     이미지·seed·스케일 배율·bbox·소요시간·VRAM 피크
out/run.json        전체 요약
```

## 옵션

| 플래그 | 뜻 |
|---|---|
| `--only hood` | 부품 하나만 |
| `--skip-generate` | GPU 없이 기존 GLB 에서 3~6단계만 |
| `--points 500000` | 샘플 점 개수 |
| `--spacing-mm 0.8` | 점 간격 기준으로 개수 자동 산출(면적 기반) |
| `--smooth 8` | Taubin 스무딩 반복 |
| `--single-view` | hidden point removal — 3D 카메라 1시점처럼 앞면만 |
| `--seed 42` | 샘플링 재현성 |
| `--run-kwargs '{"resolution": 512}'` | `run()` 추가 인자. 안 받는 이름은 자동으로 걸러짐 |

## 알려진 함정

- **VRAM** — 24GB 급에서 고해상도(1024³/1536³)는 메시 후처리(`decode_shape_slat`/`fill_holes`)에서
  OOM 이 보고돼 있다([이슈 #188](https://github.com/microsoft/TRELLIS.2/issues/188)).
  기본 해상도로 먼저 돌리고, 터지면 부품을 하나씩(`--only`) 돌린다.
  코드는 부품 사이에서 `torch.cuda.empty_cache()` 를 부르고
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 켜둔다.
- **CuMesh 빌드 실패** — CUDA 가 여러 개면 `CUDA_HOME` 을 명시해야 한다([이슈 #106](https://github.com/microsoft/TRELLIS.2/issues/106)).
  `setup_vast.sh` 가 `/usr/local/cuda` 로 기본 설정한다.
- **스케일은 자동이 안 된다** — 생성 메시는 단위 박스로 정규화되어 나온다.
  `target_mm` 은 **사람이 넣는 실측값**이고, 이게 틀리면 뒤 단계가 전부 틀린다.
- **표면 울렁임** — 생성 메시는 sub-mm 에서 매끄럽지 않다. 결함 검출용 데이터로 쓸 거라면
  결함 깊이(0.1~1mm)와 배경 표면 노이즈가 겹치지 않는지 봐야 한다. `--smooth` 로 눌러야 할 수 있다.
- **`--single-view` 는 닫힌 메시에서만 의미** — 열린 판때기면 다 보이므로 점이 안 줄어든다.
- **앞뒤 사진은 합쳐지지 않는다** — 같은 부품의 외판·내판을 각각 넣으면 하나로 합쳐진 3D 가 아니라
  별개 오브젝트 두 개가 나온다.
- **env 격리** — TRELLIS.2 는 torch 2.6~2.8 대를 쓴다. 다른 프로젝트 env 와 섞지 말고
  conda env `trellis2` 안에서만 돌린다. 오가는 건 파일(GLB/PCD)뿐이다.

## 검증 상태

- ✅ **CPU 단계(3~6)** — `smoke_test.py` **8/8 passed**. 합성 메시로 스케일 정확도(±1%)·
  seed 재현성·PCD 되읽기·미리보기 생성까지 확인.
- ⚠️ **GPU 단계(1~2)** — **미검증**. `pipeline.run(image)[0]` 과 `o_voxel.postprocess.to_glb(...)` 는
  TRELLIS.2 공식 `example.py` 기준이다. `run()` 이 `seed`·해상도 인자를 받는지는 문서에 없어서,
  코드가 시그니처를 보고 안 받는 인자를 자동으로 걸러낸다. 실제 인자 목록은 노트북 5번 셀이 출력한다.

## 라이선스

이 레포의 코드는 자유롭게 쓰되, TRELLIS.2 본체와 그 의존성의 라이선스는 각 프로젝트를 따른다.
