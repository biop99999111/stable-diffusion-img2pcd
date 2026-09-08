"""스캔 1개 -> 미리보기 PNG 묶음 + 배경 평면/부품 마스크 초안.

    python preview.py ../zdf/2-2.zdf --out out/2-2

만드는 것
  rgb.png  depth.png  normal.png  valid.png  snr.png  part_mask.png
  stats.json — 유효율·점 간격·배경 평면(RANSAC)·부품 마스크 비율

부품 마스크는 "결함을 어디에 놓을 수 있는가"를 정한다. 지그 판(가장 큰 평면)과
너무 먼 배경을 빼고 남은 가장 큰 덩어리를 부품으로 본다. 초안이므로 눈으로 확인한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import zdf_io


def colorize(v: np.ndarray, mask: np.ndarray, lo=None, hi=None, cmap="turbo") -> np.ndarray:
    import matplotlib

    m = matplotlib.colormaps[cmap]
    lo = np.nanpercentile(v[mask], 1) if lo is None else lo
    hi = np.nanpercentile(v[mask], 99) if hi is None else hi
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (m(t)[..., :3] * 255).astype(np.uint8)
    img[~mask] = 0
    return img


def save_png(path: Path, img: np.ndarray, scale: int = 1) -> None:
    from PIL import Image

    im = Image.fromarray(img)
    if scale > 1:
        im = im.resize((im.width // scale, im.height // scale), Image.Resampling.BOX)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def ransac_plane(pts: np.ndarray, iters: int = 300, thresh: float = 1.5, seed: int = 0):
    """가장 많은 점을 품는 평면 (n, d): n·p + d = 0. thresh 는 mm."""
    rng = np.random.default_rng(seed)
    best_n, best_d, best_cnt = None, 0.0, 0
    sub = pts[rng.choice(len(pts), min(len(pts), 200_000), replace=False)]
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -n @ a
        cnt = int((np.abs(sub @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    # 최소제곱으로 다듬기
    inl = sub[np.abs(sub @ best_n + best_d) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1]
    if n[2] > 0:  # 카메라를 향하도록
        n = -n
    return n, float(-n @ c), best_cnt / len(sub)


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zdf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=int, default=4, help="미리보기 축소 배율")
    ap.add_argument("--plane-thresh", type=float, default=1.5, help="배경 평면 거리(mm)")
    ap.add_argument("--part-margin", type=float, default=4.0, help="평면에서 이만큼(mm) 떨어져야 부품")
    args = ap.parse_args(argv)

    d = zdf_io.load(args.zdf)
    xyz, rgba, snr, nrm = d["xyz"], d["rgba"], d["snr"], d["normals"]
    valid = zdf_io.valid_mask(xyz)
    H, W = valid.shape
    out = args.out

    save_png(out / "rgb.png", rgba[..., :3], args.scale)
    save_png(out / "valid.png", (valid * 255).astype(np.uint8), args.scale)
    save_png(out / "depth.png", colorize(xyz[..., 2], valid), args.scale)
    save_png(out / "snr.png", colorize(snr, valid, 0, 100, "RdYlGn"), args.scale)
    nimg = np.zeros((H, W, 3), np.uint8)
    nimg[valid] = ((nrm[valid] * 0.5 + 0.5) * 255).astype(np.uint8)
    save_png(out / "normal.png", nimg, args.scale)

    # 배경 평면 → 부품 마스크
    pts = xyz[valid]
    n, dd, inlier_ratio = ransac_plane(pts, thresh=args.plane_thresh)
    dist = np.full((H, W), np.nan, np.float32)
    dist[valid] = pts @ n + dd  # 카메라 쪽이 양수
    on_plane = valid & (np.abs(dist) < args.plane_thresh)
    above = valid & (dist > args.part_margin)
    part = largest_component(above)
    save_png(out / "plane.png", (on_plane * 255).astype(np.uint8), args.scale)
    save_png(out / "part_mask.png", (part * 255).astype(np.uint8), args.scale)
    save_png(out / "height_above_plane.png", colorize(dist, valid, -5, 150), args.scale)

    # 점 간격
    sp = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)
    sp = sp[np.isfinite(sp)]
    stats = {
        "file": str(args.zdf),
        "grid": [H, W],
        "valid_ratio": round(float(valid.mean()), 4),
        "z_median_mm": round(float(np.median(xyz[..., 2][valid])), 1),
        "spacing_mm_median": round(float(np.median(sp)), 4),
        "spacing_mm_p90": round(float(np.percentile(sp, 90)), 4),
        "plane_normal": [round(float(v), 4) for v in n],
        "plane_inlier_ratio": round(float(inlier_ratio), 3),
        "on_plane_ratio": round(float(on_plane.sum() / valid.sum()), 3),
        "part_mask_ratio": round(float(part.sum() / valid.sum()), 3),
        "part_pixels": int(part.sum()),
        "snr_median": round(float(np.median(snr[valid])), 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
