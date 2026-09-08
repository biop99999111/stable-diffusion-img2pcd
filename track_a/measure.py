"""실스캔 계측 — 합성 파라미터를 실측에 묶는다.

    python measure.py data/2-2.zdf --dent 600 1120 --out out/measure/2-2
    python measure.py data/0023.zdf --out out/measure/0023

1) 노이즈 프로파일 (spec U-1 · scanner_noise_profile)
   - 평면 잔차 RMS: 가장 큰 평면(지그 판 또는 부품 앞면)의 인라이어에서
   - 국소 잔차 RMS: 부품 위 8mm 창마다 2차 곡면을 맞추고 남는 것 = 곡면 위 깊이 노이즈
   - 법선 노이즈: SDK 법선 vs 평활 법선의 각도 std. 우리 추정기(step 1~4)도 같이 재서
     SDK 와 가장 가까운 step 을 고른다 → parts.yaml normal_step
   - 결측률 vs 입사각: 법선·시선 각도 10° 구간별 결측 비율

2) 실물 찍힘 계측 (--dent row col)
   중심 주변 고리(r1<r<r2)에 2차 곡면을 맞춰 "찍힘이 없었다면"의 표면을 만들고,
   실제와의 차이를 깊이로 본다. 최대 깊이·50% 깊이 반경·단면 프로파일을 남긴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import inject
import partmask
import zdf_io


def fit_quadric(pts: np.ndarray, center: np.ndarray, n: np.ndarray):
    """접평면 좌표 (u,v,w) 에서 w ≈ a + bu + cv + du² + euv + fv² 를 맞춘다."""
    t1, t2 = inject.tangent_basis(n)
    rel = pts - center
    u, v, w = rel @ t1, rel @ t2, rel @ n
    A = np.c_[np.ones_like(u), u, v, u * u, u * v, v * v]
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    return coef, (t1, t2)


def quadric_eval(coef, pts, center, n, basis):
    t1, t2 = basis
    rel = pts - center
    u, v = rel @ t1, rel @ t2
    A = np.c_[np.ones_like(u), u, v, u * u, u * v, v * v]
    return A @ coef, rel @ n


def local_residual_rms(xyz, valid, part, spacing, win_mm=8.0, n_windows=300, seed=0):
    """부품 위 무작위 창에서 2차 곡면 잔차 RMS 의 중앙값 — 곡면 위 깊이 노이즈."""
    rng = np.random.default_rng(seed)
    r = int(win_mm / 2 / spacing)
    ys, xs = np.nonzero(partmask.erode_mm(part, win_mm, spacing))
    out = []
    for i in rng.choice(len(ys), min(n_windows, len(ys)), replace=False):
        y, x = ys[i], xs[i]
        w = xyz[y - r : y + r + 1, x - r : x + r + 1]
        m = valid[y - r : y + r + 1, x - r : x + r + 1]
        p = w[m]
        if len(p) < 50:
            continue
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[-1]
        n = n if n[2] < 0 else -n
        coef, basis = fit_quadric(p, c, n)
        pred, w_actual = quadric_eval(coef, p, c, n, basis)
        out.append(float(np.sqrt(np.mean((w_actual - pred) ** 2))))
    if not out:
        raise ValueError("No usable surface windows for residual measurement")
    return float(np.median(out)), float(np.percentile(out, 90))


def normal_noise(nrm_sdk, xyz, valid, part, spacing):
    """법선 노이즈 std(°): SDK 와 우리 추정기(step 별). 평활 법선을 기준으로 잰다."""
    ref = inject.smooth_normals(nrm_sdk, valid, sigma_px=6.0)
    core = partmask.erode_mm(part, 6.0, spacing)

    def ang_std(n):
        ok = core & inject.finite_normals(n) & inject.finite_normals(ref)
        cos = np.clip((n[ok] * ref[ok]).sum(-1), -1, 1)
        return float(np.degrees(np.arccos(cos)).std())

    res = {"sdk": ang_std(nrm_sdk)}
    for step in (1, 2, 3, 4):
        res[f"step{step}"] = ang_std(inject.estimate_normals(xyz, step))
    return res


def dropout_vs_angle(nrm, xyz, valid, part):
    """입사각(법선 vs 시선) 10° 구간별 결측률. 결측 픽셀은 법선이 없어 이웃 평활 법선으로 대신한다."""
    ref = inject.smooth_normals(nrm, valid, sigma_px=8.0)
    # 시선 = 픽셀의 3D 점 방향. 결측 픽셀은 이웃 xyz 평균으로 근사
    from scipy import ndimage

    w = ndimage.uniform_filter(valid.astype(np.float32), 9)
    xyz_f = np.stack([ndimage.uniform_filter(np.where(valid, xyz[..., i], 0), 9) / np.maximum(w, 1e-6) for i in range(3)], -1)
    view = xyz_f / np.maximum(np.linalg.norm(xyz_f, axis=-1, keepdims=True), 1e-6)
    cos = np.clip(-(ref * view).sum(-1), -1, 1)
    ang = np.degrees(np.arccos(cos))
    from scipy import ndimage as ndi

    region = ndi.binary_dilation(part, iterations=6) & (w > 0.3)
    bins = np.arange(0, 91, 10)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = region & (ang >= lo) & (ang < hi)
        if m.sum() < 500:
            continue
        rows.append({"angle": f"{lo}-{hi}", "pixels": int(m.sum()), "dropout": round(float(1 - valid[m].mean()), 4)})
    return rows


def _quadric_residual(xyz, valid, row, col, r1_mm, r2_mm, spacing):
    """(창 슬라이스, 잔차, 고리 마스크, 안쪽 마스크, 3D 거리). 잔차는 카메라 쪽 양수 → 찍힘은 음수."""
    c0 = xyz[row, col]
    if not valid[row, col] or not (0 < r1_mm < r2_mm):
        raise ValueError("Measurement requires a valid center and 0 < r1 < r2")
    r_px = int(r2_mm / spacing) + 5
    ys, xs = inject.window(row, col, r_px, *valid.shape)
    w = xyz[ys, xs]
    m = valid[ys, xs]
    dist = np.linalg.norm(w - c0, axis=-1)
    ring = m & (dist > r1_mm) & (dist < r2_mm)
    p = w[ring]
    if len(p) < 12:
        raise ValueError("Not enough valid ring points for quadric fitting")
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    n = n if n[2] < 0 else -n
    coef, basis = fit_quadric(p, c, n)
    pred, actual = quadric_eval(coef, w[m], c, n, basis)
    resid = np.full(m.shape, np.nan, np.float32)
    resid[m] = actual - pred
    resid[dist > r2_mm] = np.nan
    return (ys, xs), resid, ring, m & (dist < r1_mm), dist


def measure_dent(xyz, valid, row, col, r1_mm, r2_mm, out: Path, spacing: float = 0.15):
    """실물 찍힘: 고리 영역 2차 곡면 → 잔차 = 깊이.

    2패스: 대략 중심으로 한 번 맞춰 최저점을 찾고, 거기로 재중심해 다시 맞춘다.
    50% 깊이 반경은 최저점과 **연결된** 영역에서만 잰다 — 고리 밖 곡면 적합 오차가
    섞이면 반경이 수십 mm 로 튄다.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    inject.configure_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 1패스: 최저점 찾기
    (ys, xs), resid, ring, inner, _ = _quadric_residual(xyz, valid, row, col, r1_mm, r2_mm, spacing)
    iy, ix = np.unravel_index(np.nanargmin(np.where(inner, resid, np.nan)), resid.shape)
    row2, col2 = ys.start + int(iy), xs.start + int(ix)
    # 2패스: 재중심
    (ys, xs), resid, ring, inner, dist = _quadric_residual(xyz, valid, row2, col2, r1_mm, r2_mm, spacing)
    iy, ix = np.unravel_index(np.nanargmin(np.where(inner, resid, np.nan)), resid.shape)
    depth = float(-resid[iy, ix])
    half = np.nan_to_num(resid, nan=0.0) < -0.5 * depth
    lab, _ = ndimage.label(half)
    comp = lab == lab[iy, ix]
    w = xyz[ys, xs]
    d_from_min = np.linalg.norm(w - w[iy, ix], axis=-1)
    radius_half = float(np.nanmax(np.where(comp, d_from_min, np.nan)))
    ring_rms = float(np.sqrt(np.nanmean(np.where(ring, resid, np.nan) ** 2)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    im = ax[0].imshow(resid, cmap="RdBu", vmin=-depth, vmax=depth)
    ax[0].contour(comp, levels=[0.5], colors="k", linewidths=0.6)
    ax[0].set_title(f"잔차 (mm) · 검은 선 = 50% 깊이 영역")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot((np.arange(w.shape[1]) - ix) * spacing, resid[iy, :], label="가로")
    ax[1].plot((np.arange(w.shape[0]) - iy) * spacing, resid[:, ix], label="세로")
    ax[1].axhline(-depth / 2, ls="--", c="gray", lw=0.8)
    ax[1].set_xlim(-r2_mm, r2_mm); ax[1].set_ylim(-depth * 1.2, depth * 0.4)
    ax[1].set_xlabel("mm"); ax[1].set_ylabel("깊이 (mm)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"단면 · 최대 깊이 {depth:.2f}mm · 50% 반경 {radius_half:.1f}mm")
    ax[2].imshow(np.where(inner, 1, 0) + np.where(ring, 2, 0), cmap="viridis")
    ax[2].set_title(f"안쪽(<{r1_mm}mm) / 고리(<{r2_mm}mm) · 고리 잔차 RMS {ring_rms:.3f}mm")
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"center_px": [row2, col2], "max_depth_mm": round(depth, 3), "radius_at_half_depth_mm": round(radius_half, 2),
            "ring_residual_rms_mm": round(ring_rms, 4), "r1_mm": r1_mm, "r2_mm": r2_mm}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zdf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part-mode", default="above")
    ap.add_argument("--dent", nargs=2, type=int, metavar=("ROW", "COL"), help="실물 찍힘 중심(전체 해상도 픽셀)")
    ap.add_argument("--dent-r1", type=float, default=18.0)
    ap.add_argument("--dent-r2", type=float, default=30.0)
    args = ap.parse_args(argv)

    d = zdf_io.load(args.zdf)
    xyz, nrm, valid = d["xyz"], d["normals"], zdf_io.valid_mask(d["xyz"])
    sp = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)
    spacing = float(np.median(sp[np.isfinite(sp)]))
    part, n, dd, dist = partmask.part_mask(xyz, valid, args.part_mode)
    args.out.mkdir(parents=True, exist_ok=True)

    plane_res = dist[valid & (np.abs(dist) < 1.5)]
    prof = {
        "file": str(args.zdf),
        "spacing_mm": round(spacing, 4),
        "valid_ratio": round(float(valid.mean()), 4),
        "plane_residual_rms_mm": round(float(np.sqrt(np.mean(plane_res**2))), 4),
        "local_quadric_residual_rms_mm": dict(zip(("median", "p90"), [round(v, 4) for v in local_residual_rms(xyz, valid, part, spacing)])),
        "normal_noise_std_deg": {k: round(v, 3) for k, v in normal_noise(nrm, xyz, valid, part, spacing).items()},
        "dropout_vs_incidence": dropout_vs_angle(nrm, xyz, valid, part),
    }
    sdk = prof["normal_noise_std_deg"]["sdk"]
    steps = {k: v for k, v in prof["normal_noise_std_deg"].items() if k.startswith("step")}
    prof["recommended_normal_step"] = int(min(steps, key=lambda k: abs(steps[k] - sdk))[4:])

    if args.dent:
        prof["real_dent"] = measure_dent(xyz, valid, args.dent[0], args.dent[1], args.dent_r1, args.dent_r2, args.out / "real_dent.png", spacing)

    (args.out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(prof, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
