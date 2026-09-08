"""파이프라인이 쓰는 가중치를 미리 내려받는다(선택).

용도는 두 가지다.
  1) 실패를 앞당긴다 — 확장 빌드 30분을 다 태운 뒤 gated 401 을 만나는 대신
     여기서 먼저 죽는다. (접근 권한만 볼 거면 `python check_env.py --hf` 가 더 빠르다)
  2) 파일 목록을 고정한다 — from_pretrained 가 게으르게 받는 것과 달리
     무엇을 몇 GB 받는지 로그에 남는다.

    python prefetch_models.py                 # trellis2 (기본)
    python prefetch_models.py --rembg ZhengPeng7/BiRefNet
    python prefetch_models.py --dry-run       # 접근 확인만

HF_TOKEN 이 필요하다(gated 2곳). HF_HOME 을 큰 디스크로 지정하고 싶으면
    export HF_HOME=/workspace/hf
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# pipeline.json 을 그대로 옮긴 것. 파일 단위로 적는 이유는 레포 통째로 받으면
# 쓰지도 않는 encoder 2종(1.3GB)까지 딸려오기 때문이다.
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS2_CKPTS = [
    "ss_flow_img_dit_1_3B_64_bf16",
    "shape_dec_next_dc_f16c32_fp16",
    "slat_flow_img2shape_dit_1_3B_512_bf16",
    "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_dec_next_dc_f16c32_fp16",
    "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
]
# pipeline.json 이 v1 레포를 가리키는 항목. 이것만 다른 레포에서 온다.
V1_REPO = "microsoft/TRELLIS-image-large"
V1_CKPTS = ["ss_dec_conv3d_16l8_fp16"]

DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_REMBG = "briaai/RMBG-2.0"


def plan(rembg: str) -> list[tuple[str, str]]:
    """(repo, 파일) 목록. 순서는 작은 것부터 — 권한 문제를 빨리 만나게."""
    items: list[tuple[str, str]] = [(TRELLIS2_REPO, "pipeline.json")]
    for name in V1_CKPTS:
        items += [(V1_REPO, f"ckpts/{name}.json"), (V1_REPO, f"ckpts/{name}.safetensors")]
    for name in TRELLIS2_CKPTS:
        items += [
            (TRELLIS2_REPO, f"ckpts/{name}.json"),
            (TRELLIS2_REPO, f"ckpts/{name}.safetensors"),
        ]
    for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
        items.append((DINOV3, f))
    # rembg 는 remote code 라 .py 도 같이 받아야 한다(trust_remote_code=True).
    for f in ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"):
        items.append((rembg, f))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TRELLIS.2 가중치 미리 받기")
    ap.add_argument("--rembg", default=DEFAULT_REMBG, help=f"배경 제거 모델(기본 {DEFAULT_REMBG})")
    ap.add_argument("--dry-run", action="store_true", help="받지 않고 접근 가능 여부만 확인")
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        print("huggingface_hub 가 없습니다: pip install huggingface_hub")
        return 1

    if not any(os.environ.get(v) for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
        print("[warn] HF_TOKEN 미설정 — gated 레포 2곳(DINOv3 · rembg)에서 401 이 납니다")

    items = plan(args.rembg)
    print(f"대상 {len(items)}개 파일 · HF_HOME={os.environ.get('HF_HOME', '(기본 ~/.cache/huggingface)')}")

    total_mb, failed = 0.0, []
    for repo, filename in items:
        t0 = time.time()
        try:
            if args.dry_run:
                from huggingface_hub import get_hf_file_metadata, hf_hub_url

                meta = get_hf_file_metadata(hf_hub_url(repo, filename))
                size_mb = (meta.size or 0) / 2**20
                print(f"  OK    {repo}/{filename}  {size_mb:.1f} MB")
            else:
                path = hf_hub_download(repo, filename)
                size_mb = os.path.getsize(path) / 2**20
                print(f"  받음  {repo}/{filename}  {size_mb:.1f} MB  {time.time() - t0:.1f}s")
            total_mb += size_mb
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"  FAIL  {repo}/{filename}  접근 불가: {type(e).__name__}")
            failed.append(repo)
        except Exception as e:  # 네트워크·디스크 등
            print(f"  FAIL  {repo}/{filename}  {type(e).__name__}: {str(e)[:120]}")
            failed.append(repo)

    print(f"\n합계 {total_mb / 1024:.2f} GiB")
    if failed:
        uniq = sorted(set(failed))
        print(f"실패한 레포: {uniq}")
        print("  -> python check_env.py --hf  로 원인(401/403)을 확인하세요")
        if args.rembg == DEFAULT_REMBG and DEFAULT_REMBG in uniq:
            print(f"  -> 승인 없이 가려면: python prefetch_models.py --rembg ZhengPeng7/BiRefNet")
            print(f"     실행도 같은 값으로: python img2pcd.py --backend trellis2 --rembg ZhengPeng7/BiRefNet")
        return 1
    print("전부 준비됨 — img2pcd.py 를 돌려도 됩니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
