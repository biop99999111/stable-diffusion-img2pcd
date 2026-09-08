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

### HuggingFace 모델 (필수)

파이프라인이 받는 레포는 **4곳**이고 그중 **2곳이 gated** 다. 하나라도 막히면
`from_pretrained` 가 401 로 죽는다. `pipeline.json` 에 박혀 있어 코드로 우회할 수 없다.

| 레포 | 크기 | gated | 용도 |
|---|---|---|---|
| `microsoft/TRELLIS.2-4B` | 14.3 GiB | — | 본체 가중치 8종 |
| `microsoft/TRELLIS-image-large` | 0.14 GiB | — | `pipeline.json` 이 참조하는 **v1** sparse structure decoder |
| [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 1.13 GiB | **manual** | 이미지 인코더. Meta **수동 승인** — 대기 시간이 있다 |
| [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) | 0.82 GiB | **auto** | 배경 제거. 약관 동의 즉시 통과. **비상업 라이선스** |

1. 위 두 gated 페이지에서 각각 **약관 동의**
2. https://huggingface.co/settings/tokens 에서 **read 토큰** 발급
   - ⚠️ fine-grained 토큰이면 **"Read access to contents of all public gated repos you can access"**
     를 반드시 체크한다. 안 그러면 승인을 받아도 계속 401 이 난다. Classic → Read 가 확실하다.
3. `export HF_TOKEN=hf_...`

```bash
python check_env.py --hf        # 4곳 전부 실제 HTTP 요청으로 확인 (GPU·설치 불필요)
```

`RMBG-2.0` 은 배경 제거에만 쓰이므로 승인이 없거나 상업 사용이 걸리면 구조가 같은
MIT 모델로 바꿔도 된다 — `--rembg ZhengPeng7/BiRefNet`.
다만 배경 제거 결과가 달라지면 생성 형상도 조금 달라진다는 점은 감안해야 한다.

## 실행

```bash
git clone https://github.com/biop99999111/trellis2-img2pcd.git
cd trellis2-img2pcd

export HF_TOKEN=hf_...      # gated 2곳에 필요

python check_env.py         # 1. 진단 — FAIL 0 확인 (표준 라이브러리만 씀)
python smoke_test.py        # 2. (선택) GPU 없이 후반부 검증
python test_compat.py       # 2b.(선택) transformers 4.x/5.x 호환 패치 검증 (torch 만 필요)
bash setup_vast.sh          # 3. TRELLIS.2 + 의존성 + 가중치 16.4GiB (20~40분)

conda activate trellis2
python img2pcd.py --backend trellis2 --out out_trellis
python img2pcd.py --backend hunyuan3d --out out_hy      # 비교 기준
```

`setup_vast.sh` 는 시작하자마자 `check_env.py --hf` 를 돌린다. 확장 빌드 30분을 태운
뒤에 gated 401 을 만나는 사고를 막기 위해서다. 가중치만 따로 받으려면
`python prefetch_models.py` (`--dry-run` 이면 접근 확인만).

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
| `--backend trellis2` | 생성 백엔드 (`trellis2` \| `hunyuan3d`) |
| `--pipeline-type 512` | trellis2 생성 해상도 (`512` \| `1024` \| `1024_cascade` \| `1536_cascade`). 기본 `1024_cascade` |
| `--max-num-tokens 32768` | trellis2 토큰 상한. 낮추면 해상도가 자동으로 내려간다 |
| `--rembg ZhengPeng7/BiRefNet` | 배경 제거 모델 교체 (gated 인 RMBG-2.0 회피) |
| `--run-kwargs '{"preprocess_image": false}'` | `run()` 추가 인자. 안 받는 이름은 자동으로 걸러짐 |

## 알려진 함정

- **transformers 5.x** — TRELLIS.2 `setup.sh` 는 transformers 를 **핀 없이** 깔아서 오늘 설치하면
  5.x 가 들어온다. 5.x 는 `DINOv3ViTModel` 의 블록을 encoder 하위로 옮겨서 원본 코드가
  `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'` 로 죽는다
  ([이슈 #147](https://github.com/microsoft/TRELLIS.2/issues/147) · PR [#148](https://github.com/microsoft/TRELLIS.2/pull/148)/[#156](https://github.com/microsoft/TRELLIS.2/pull/156), 셋 다 미머지).
  `requirements.txt` 가 `transformers>=4.56,<5` 로 되돌리고, `backends.py` 가 양쪽 레이아웃을
  모두 타는 패치를 건다(`test_compat.py` 가 두 경로의 결과가 같음을 확인).
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

- ✅ **CPU 단계(3~6)** — `smoke_test.py` **12/12 passed**. 합성 메시로 스케일 정확도(±1%)·
  seed 재현성·PCD 되읽기·미리보기 생성·형상 판정(`expect_mm`)까지 확인.
- ✅ **호환 패치** — `test_compat.py` **10/10 passed**. transformers 4.x/5.x 두 레이아웃의
  DINOv3 특징이 **완전히 동일**함(max diff 0)과, 모르는 레이아웃에서는 조용히 틀리는 대신
  죽는 것을 확인. GPU·가중치 없이 돈다.
- ✅ **Hunyuan3D 백엔드(1~2)** — 실행 검증됨. 33.5s/부품 · VRAM 피크 7.63 GiB.
  다만 형상이 부풀어(깊이 최대 7배) **계측용으로는 못 쓴다** — `HANDOFF.md` 참조.
- ⚠️ **TRELLIS.2 백엔드(1~2)** — **미검증**. `run()` 시그니처는 확인했다:
  `(image, num_samples, seed, sparse_structure_sampler_params, shape_slat_sampler_params,`
  `tex_slat_sampler_params, preprocess_image, return_latent, pipeline_type, max_num_tokens)`.
  텍스처는 별도 단계가 아니라 `run()` 안에 포함돼 있어서 PCD 색이 회색이 되지 않는다.

## 라이선스

이 레포의 코드는 자유롭게 쓰되, TRELLIS.2 본체와 그 의존성의 라이선스는 각 프로젝트를 따른다.
