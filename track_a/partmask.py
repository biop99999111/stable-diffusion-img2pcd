"""결함을 놓을 수 있는 영역(부품 마스크)을 스캔에서 뽑는다.

두 가지 모드가 필요하다는 것을 실데이터에서 확인했다.
  above  지그 판이 가장 큰 평면이고 부품이 그 위에 떠 있다 (범퍼 커버 2-2.zdf)
  near   부품 앞면 자체가 가장 큰 평면이다 (용접 브래킷 0023.zdf)
"""

from __future__ import annotations

import numpy as np


def ransac_plane(pts: np.ndarray, iters: int = 300, thresh: float = 1.5, seed: int = 0):
    """가장 많은 점을 품는 평면 (n, d): n·p + d = 0. n 은 카메라(-z)를 향한다."""
    rng = np.random.default_rng(seed)
    if len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("Plane fitting requires at least three finite points")
    sub = pts[rng.choice(len(pts), min(len(pts), 200_000), replace=False)]
    best_n, best_d, best_cnt = None, 0.0, 0
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -n @ a
        cnt = int((np.abs(sub @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    if best_n is None:
        raise ValueError("No non-degenerate plane found")
    inl = sub[np.abs(sub @ best_n + best_d) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1]
    if n[2] > 0:
        n = -n
    return n, float(-n @ c), best_cnt / len(sub)


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def part_mask(xyz: np.ndarray, valid: np.ndarray, mode: str = "above",
              plane_thresh: float = 1.5, margin: float = 4.0, window: float = 30.0):
    """(mask, plane_n, plane_d, dist) — dist 는 평면에서 카메라 쪽으로 잰 mm."""
    pts = xyz[valid]
    n, d, _ = ransac_plane(pts, thresh=plane_thresh)
    dist = np.full(valid.shape, np.nan, np.float32)
    dist[valid] = pts @ n + d
    if mode == "above":
        cand = valid & (dist > margin)
    elif mode == "near":
        cand = valid & (np.abs(dist) < window)
    else:
        raise ValueError(f"모르는 part mode: {mode}")
    return largest_component(cand), n, d, dist


def erode_mm(mask: np.ndarray, radius_mm: float, spacing_mm: float) -> np.ndarray:
    """mm 단위 반경으로 침식. 결함 전체가 부품 안에 들어오게 하고 얇은 탭을 걷어낸다."""
    from scipy import ndimage

    # 원판 커널 침식은 반경 200px 에서 메모리가 터진다. 거리 변환이면 격자 1장 메모리로 같은 결과.
    if spacing_mm <= 0 or not np.isfinite(spacing_mm) or radius_mm < 0:
        raise ValueError("Invalid erosion radius or spacing")
    r = radius_mm / spacing_mm
    return ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] > r
