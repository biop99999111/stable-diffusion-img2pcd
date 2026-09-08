"""실스캔(organized Zivid 격자)에 결함을 2.5D 로 주입해 합성 샘플을 만든다.

    python inject.py --scan bumper_cover --samples 5 --out out/synth --seed 0
    python inject.py --scan bumper_cover --samples 1 --pcd     # PCD 까지

원리
  점 xyz(H,W,3) 를 그대로 두고, 결함 창 안의 점만 **스캔 법선 방향으로 변위**한다.
  찍힘 = 코사인 벨(타원), 긁힘 = 선분 홈. 변위 뒤 법선을 재계산하고 RGB 는 법선이
  카메라를 향하는 정도가 바뀐 만큼 음영을 바꾼다(광원≈카메라). 결측·노이즈·해상도는
  실데이터 것이 그대로 남는다 — 그래서 센서 모델 없이 도메인 갭이 작다.

산출 (샘플마다, spec FR-7/8 형식)
  rgb.png  depth_mm.png(uint16, 0.1mm)  normal.png  seg.png(255=없음)  inst.png
  labels.txt(YOLO seg 폴리곤)  defects.json  overlay.png  crops/<inst>.png
  --pcd 이면 points.pcd(xyz mm+rgb, 유효 픽셀만) + point_labels.npy (길이 == 유효 픽셀 수)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import partmask
import zdf_io

NO_CLASS = 255


def configure_fonts():
    import matplotlib
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans"):
        if name in names:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_config(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in cfg["scans"]:
        entry["zdf"] = str((path.parent / entry["zdf"]).resolve())
    return cfg


def finite_normals(n):
    return np.isfinite(n).all(axis=-1) & (np.linalg.norm(n, axis=-1) > 1e-6)


def shade_ratio(light, before, after, xyz, ambient=0.2, specular=0.0, shininess=32.0):
    """Shared Lambert + optional Blinn-Phong ratio; specular is uncalibrated by default."""
    if ambient <= 0 or specular < 0 or shininess <= 0:
        raise ValueError("Invalid shading parameters")
    view = -xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-6)
    half = view + light
    half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-6)
    def intensity(n):
        return ambient + np.maximum(n @ light, 0) + specular * np.maximum((n * half).sum(-1), 0) ** shininess
    ratio = intensity(after) / intensity(before)
    ok = finite_normals(before) & finite_normals(after) & np.isfinite(ratio)
    return np.where(ok, np.clip(ratio, 0.25, 1.75), 1.0)


# ---------------------------------------------------------------- 기하


def estimate_normals(xyz: np.ndarray, step: int = 2) -> np.ndarray:
    """organized 격자에서 중앙차분으로 법선. 카메라(-z)를 향하게 맞춘다. 결측은 NaN.

    step 은 차분 폭(px). SDK 법선의 노이즈 수준에 맞추려고 measure.py 로 정한다 —
    결함 창만 재계산할 때 그 창의 '결'이 주변과 달라 보이면 안 되기 때문이다.
    """
    k = int(step)
    if k < 1 or min(xyz.shape[:2]) <= 2 * k:
        raise ValueError("normal_step must be positive and smaller than half the grid")
    du = np.full_like(xyz, np.nan)
    dv = np.full_like(xyz, np.nan)
    du[:, k:-k] = xyz[:, 2 * k :] - xyz[:, : -2 * k]
    dv[k:-k, :] = xyz[2 * k :, :] - xyz[: -2 * k, :]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    flip = n[..., 2] > 0
    n[flip] *= -1
    n[~zdf_io.valid_mask(xyz)] = np.nan
    return n.astype(np.float32)


def smooth_normals(n: np.ndarray, valid: np.ndarray, sigma_px: float = 6.0) -> np.ndarray:
    """변위 방향으로 쓸 법선. 픽셀 노이즈를 따라가면 변위가 울퉁불퉁해지므로 평활한다."""
    from scipy import ndimage

    out = np.zeros_like(n)
    valid = valid & finite_normals(n)
    w = ndimage.gaussian_filter(valid.astype(np.float32), sigma_px)
    for c in range(3):
        ch = np.where(valid, n[..., c], 0.0).astype(np.float32)
        out[..., c] = ndimage.gaussian_filter(ch, sigma_px) / np.maximum(w, 1e-6)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norm, 1e-6)).astype(np.float32)


def estimate_light(rgb: np.ndarray, normals: np.ndarray, mask: np.ndarray, rng=None) -> np.ndarray:
    """스캔 자체에서 조명 방향을 뽑는다: 밝기 ≈ a + b·(n·L) 램버트 회귀.

    결함의 RGB 음영을 실물처럼 비대칭(한쪽 어둡고 반대 림 밝음)으로 만들려면
    광원이 카메라와 정확히 같은 곳에 있지 않다는 것을 반영해야 한다.
    광택 하이라이트가 회귀를 흔들므로 상위 밝기 5% 는 뺀다.
    """
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(mask & np.isfinite(normals[..., 0]))
    if len(ys) < 10:
        return np.array([0.0, 0.0, -1.0])
    sel = rng.choice(len(ys), min(len(ys), 200_000), replace=False)
    n = normals[ys[sel], xs[sel]].astype(np.float64)
    i = rgb[ys[sel], xs[sel]].astype(np.float64).mean(-1)
    keep = i < np.percentile(i, 95)
    if keep.sum() < 10:
        return np.array([0.0, 0.0, -1.0])
    A = np.c_[np.ones(keep.sum()), n[keep]]
    coef, *_ = np.linalg.lstsq(A, i[keep], rcond=None)
    L = coef[1:]
    norm = np.linalg.norm(L)
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    L = L / norm
    return L if L[2] < 0 else -L


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


# ---------------------------------------------------------------- 결함 정의


@dataclass
class Defect:
    inst_id: int
    cls: str
    class_id: int
    center_px: list[int]        # (row, col)
    center_mm: list[float]
    params: dict
    bbox_px: list[int] | None = None   # x0 y0 x1 y1
    mask_area_px: int = 0
    point_count: int = 0


def dent_field(xyz_win, valid_win, center, n, radius, depth, aspect, angle, rim=0.0):
    """타원 코사인 벨 변위(mm). 창 안 점마다 접평면 좌표로 반경을 잰다."""
    if radius <= 0 or depth <= 0 or aspect < 1 or rim < 0:
        raise ValueError("dent requires radius/depth > 0, aspect >= 1, rim >= 0 mm")
    t1, t2 = tangent_basis(n)
    c, s = math.cos(angle), math.sin(angle)
    u_axis = c * t1 + s * t2
    v_axis = -s * t1 + c * t2
    rel = xyz_win - center
    u = rel @ u_axis
    v = rel @ v_axis
    r = np.sqrt((u / radius) ** 2 + (v / (radius / aspect)) ** 2)
    d = np.where(r < 1.0, depth * 0.5 * (1.0 + np.cos(np.pi * np.clip(r, 0, 1))), 0.0)
    # Raised annulus outside the zero-crossing radius; rim is peak height in mm.
    phase = np.clip((r - 1.0) / 0.3, 0, 1)
    d -= np.where((r > 1) & (r < 1.3), rim * np.sin(np.pi * phase) ** 2, 0)
    d[~valid_win] = 0.0
    return d.astype(np.float32)


def scratch_field(xyz_win, valid_win, p0, p1, width, depth):
    """선분 홈 변위(mm). 단면은 가우시안, 양 끝은 점점 얕아진다."""
    seg = p1 - p0
    L2 = float(seg @ seg)
    rel = xyz_win - p0
    t = np.clip((rel @ seg) / max(L2, 1e-9), 0.0, 1.0)
    closest = p0 + t[..., None] * seg
    dist = np.linalg.norm(xyz_win - closest, axis=-1)
    sigma = width / 2.355  # FWHM = width
    taper = np.sin(np.pi * t) ** 0.5  # 끝에서 0
    d = depth * np.exp(-0.5 * (dist / sigma) ** 2) * taper
    d[dist > 3 * sigma] = 0.0
    d[~valid_win] = 0.0
    return d.astype(np.float32)


# ---------------------------------------------------------------- 배치


def pick_location(rng, allowed: np.ndarray, valid: np.ndarray, r_px: int, min_valid: float = 0.97):
    """허용 마스크에서 무작위 픽셀. 창 안 유효율이 낮으면(결측 가장자리) 다시 뽑는다."""
    ys, xs = np.nonzero(allowed)
    if len(ys) == 0:
        raise RuntimeError("결함을 놓을 수 있는 영역이 없습니다 — placement_erode_mm 을 줄이세요")
    for _ in range(200):
        i = rng.integers(len(ys))
        y, x = int(ys[i]), int(xs[i])
        win = valid[max(0, y - r_px) : y + r_px + 1, max(0, x - r_px) : x + r_px + 1]
        if win.mean() >= min_valid:
            return y, x
    raise RuntimeError("No valid placement after 200 attempts")


def window(y, x, r_px, H, W):
    y0, y1 = max(0, y - r_px), min(H, y + r_px + 1)
    x0, x1 = max(0, x - r_px), min(W, x + r_px + 1)
    return slice(y0, y1), slice(x0, x1)


def place_field(scan, allowed, occupied, rng, r_px, field, at=None):
    """Reject clipped, off-part, excluded, overlapping or degenerate footprints."""
    H, W = allowed.shape
    candidates = np.argwhere(allowed)
    if at is not None:
        y, x = at
        if not (0 <= y < H and 0 <= x < W) or not allowed[y, x]:
            raise ValueError("Requested center is not an allowed surface point")
        candidates = np.array([[y, x]])
    if not len(candidates):
        raise RuntimeError("No placement centers available")
    for _ in range(1 if at is not None else 200):
        y, x = map(int, candidates[rng.integers(len(candidates))])
        if y - r_px < 0 or x - r_px < 0 or y + r_px >= H or x + r_px >= W:
            continue
        ys, xs = window(y, x, r_px, H, W)
        valid = scan["valid"][ys, xs]
        if valid.mean() < 0.97:
            continue
        d = field(scan["xyz"][ys, xs], valid, scan["xyz"][y, x], scan["n_smooth"][y, x])
        support = np.abs(d) > 0
        if not np.isfinite(d).all() or not support.any():
            continue
        if support[0].any() or support[-1].any() or support[:, 0].any() or support[:, -1].any():
            continue
        safe = scan["part"][ys, xs] & ~scan["exclude"][ys, xs] & ~occupied[ys, xs]
        safe &= finite_normals(scan["n_smooth"][ys, xs])
        if np.any(support & ~safe):
            continue
        return y, x, ys, xs, d
    raise RuntimeError("Cannot fit complete non-overlapping defect; reduce size/count or change location")


def apply_displacement(xyz, normals, d, ys, xs):
    support = np.abs(d) > 0
    patch = xyz[ys, xs]
    patch[support] -= normals[ys, xs][support] * d[support, None]


def render_changed(scan, xyz, rgb, displacement):
    from scipy import ndimage
    touched = ndimage.binary_dilation(np.abs(displacement) > 0, iterations=scan["normal_step"])
    n_est = estimate_normals(xyz, scan["normal_step"])
    ok = touched & scan["valid"] & finite_normals(n_est) & finite_normals(scan["n_est0"])
    n_new = scan["normals"].copy()
    n_new[ok] = n_est[ok]
    shade = np.ones(touched.shape, np.float32)
    shade[ok] = shade_ratio(scan["light_dir"], scan["n_est0"][ok], n_est[ok], xyz[ok], **scan.get("shading", {}))
    rgb = np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8)
    if not np.array_equal(zdf_io.valid_mask(xyz), scan["valid"]):
        raise RuntimeError("Injection changed the scan validity mask")
    return rgb, n_new, touched, shade


# ---------------------------------------------------------------- 샘플 1개


def make_sample(scan: dict, cfg: dict, classes: dict, rng, spacing_mm: float, pcd: bool) -> tuple[dict, list[Defect]]:
    xyz0, rgba0, nrm0 = scan["xyz"], scan["rgba"], scan["normals"]
    valid = scan["valid"]
    part = scan["part"]
    H, W = valid.shape

    xyz = xyz0.copy()
    rgb = rgba0[..., :3].astype(np.float32)
    seg = np.full((H, W), NO_CLASS, np.uint8)
    inst = np.zeros((H, W), np.uint16)
    disp_total = np.zeros((H, W), np.float32)
    n_smooth = scan["n_smooth"]

    # 결함끼리 겹치면 마스크가 서로를 잘라 라벨이 깨진다. 놓은 창은 이후 배치에서 뺀다.
    allowed = scan["allowed"].copy()
    occupied = np.zeros_like(allowed)

    defects: list[Defect] = []
    inst_id = 0
    for cls, spec in cfg["defects"].items():
        lo, hi = spec["count"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            inst_id += 1
            if cls == "dent":
                radius = float(rng.uniform(*spec["radius_mm"]))
                depth = float(rng.uniform(*spec["depth_mm"]))
                aspect = float(rng.uniform(*spec["aspect"]))
                angle = float(rng.uniform(0, math.pi))
                rim = float(rng.uniform(*spec.get("rim_height_mm", [0, 0])))
                r_px = int(radius / spacing_mm * 1.7) + 4
                field = lambda w, v, c, n: dent_field(w, v, c, n, radius, depth, aspect, angle, rim)
                y, x, ys, xs, d = place_field(scan, allowed, occupied, rng, r_px, field)
                c = xyz0[y, x]
                params = dict(radius_mm=radius, depth_mm=depth, aspect=aspect, angle_rad=angle, rim_height_mm=rim)
            elif cls == "scratch":
                length = float(rng.uniform(*spec["length_mm"]))
                width = float(rng.uniform(*spec["width_mm"]))
                depth = float(rng.uniform(*spec["depth_mm"]))
                gain = float(rng.uniform(*spec["rgb_gain"]))
                angle = float(rng.uniform(0, math.pi))
                r_px = int(length / 2 / spacing_mm * 1.2) + 8
                def field(w, v, c, n):
                    t1, t2 = tangent_basis(n)
                    dirv = math.cos(angle) * t1 + math.sin(angle) * t2
                    return scratch_field(w, v, c - dirv * length / 2, c + dirv * length / 2, width, depth)
                y, x, ys, xs, d = place_field(scan, allowed, occupied, rng, r_px, field)
                c = xyz0[y, x]
                params = dict(length_mm=length, width_mm=width, depth_mm=depth, angle_rad=angle, rgb_gain=gain)
            else:
                raise ValueError(f"미구현 결함: {cls}")

            # 변위: 법선(카메라 쪽) 반대 방향으로 밀어 넣는다 = 표면 안쪽
            apply_displacement(xyz, n_smooth, d, ys, xs)
            disp_total[ys, xs] += d
            pad = int(3.0 / spacing_mm)  # 3mm 여유
            occupied[max(0, ys.start - pad) : ys.stop + pad, max(0, xs.start - pad) : xs.stop + pad] = True
            allowed &= ~occupied

            # 인스턴스 마스크 = 변위가 최대의 10% 를 넘는 곳
            m = np.abs(d) > 0.1 * float(np.abs(d).max())
            seg_win = seg[ys, xs]
            inst_win = inst[ys, xs]
            seg_win[m] = classes[cls]
            inst_win[m] = inst_id

            if cls == "scratch":
                soft = np.clip(d / max(float(d.max()), 1e-6), 0, 1)
                rgb[ys, xs] *= (1.0 + (gain - 1.0) * soft)[..., None]

            yy, xx = np.nonzero(m)
            bbox = [int(xs.start + xx.min()), int(ys.start + yy.min()),
                    int(xs.start + xx.max()), int(ys.start + yy.max())] if len(yy) else None
            defects.append(Defect(
                inst_id=inst_id, cls=cls, class_id=classes[cls],
                center_px=[y, x], center_mm=[round(float(v), 2) for v in c],
                params={k: round(v, 4) for k, v in params.items()},
                bbox_px=bbox, mask_area_px=int(m.sum()), point_count=int((m & valid[ys, xs]).sum()),
            ))

    # 법선 재계산: 바뀐 창만. 같은 추정기로 원본 법선과의 차이를 기록해 검증에 쓴다.
    rgb, n_new, touched, _ = render_changed(scan, xyz, rgb, disp_total)

    return dict(xyz=xyz, rgb=rgb, normals=n_new, seg=seg, inst=inst, disp=disp_total, touched=touched), defects


# ---------------------------------------------------------------- 저장


def yolo_polygons(inst: np.ndarray, seg: np.ndarray, defects: list[Defect]) -> list[str]:
    import cv2

    H, W = inst.shape
    lines = []
    for d in defects:
        m = (inst == d.inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1 or hierarchy is None or hierarchy[0, 0, 2] != -1:
            raise ValueError("YOLO cannot represent disconnected masks/holes exactly; use --mask-only")
        cnt = contours[0]
        if len(cnt) < 3:
            raise ValueError("YOLO contour has fewer than three vertices; use --mask-only")
        pts = cnt.reshape(-1, 2).astype(np.float64)
        coords = " ".join(f"{x / W:.10f} {y / H:.10f}" for x, y in pts)
        lines.append(f"{d.class_id} {coords}")
    return lines


def save_png(path: Path, img: np.ndarray, scale: int = 1) -> None:
    from PIL import Image

    im = Image.fromarray(img)
    if scale > 1:
        im = im.resize((im.width // scale, im.height // scale), Image.Resampling.BOX)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


CLASS_COLORS = {0: (255, 40, 40), 1: (40, 90, 255), 2: (255, 220, 0), 3: (200, 60, 255)}


def overlay(rgb: np.ndarray, seg: np.ndarray, scale: int = 4) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for cid, col in CLASS_COLORS.items():
        m = seg == cid
        out[m] = out[m] * 0.45 + np.array(col, np.float32) * 0.55
    return out.astype(np.uint8)


def crop_panel(scan: dict, s: dict, d: Defect, spacing_mm: float, out: Path) -> None:
    """결함 1개 확대: 원본 RGB | 합성 RGB | 변위(mm) | 합성 normal | 원본 normal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False

    H, W = s["seg"].shape
    y, x = d.center_px
    half = int(max(d.params.get("radius_mm", 0) * 1.6, d.params.get("length_mm", 0) * 0.65, 8) / spacing_mm) + 10
    ys, xs = window(y, x, half, H, W)
    rgb0 = scan["rgba"][ys, xs, :3]
    rgb1 = s["rgb"][ys, xs]
    disp = s["disp"][ys, xs]
    n1 = (np.nan_to_num(s["normals"][ys, xs]) * 127.5 + 127.5).astype(np.uint8)
    n0 = (np.nan_to_num(scan["normals"][ys, xs]) * 127.5 + 127.5).astype(np.uint8)
    m = s["inst"][ys, xs] == d.inst_id

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    ax[0].imshow(rgb0); ax[0].set_title("원본 RGB")
    ax[1].imshow(rgb1); ax[1].contour(m, levels=[0.5], colors="cyan", linewidths=0.6); ax[1].set_title("합성 RGB + 마스크")
    im = ax[2].imshow(disp, cmap="magma"); ax[2].set_title("변위 (mm)"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].imshow(n1); ax[3].set_title("합성 normal")
    ax[4].imshow(n0); ax[4].set_title("원본 normal")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    p = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.params.items() if k != "angle_rad")
    fig.suptitle(f"#{d.inst_id} {d.cls}  {p}  · 창 {2 * half * spacing_mm:.0f}mm", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def write_pcd(path: Path, xyz: np.ndarray, rgb: np.ndarray, valid: np.ndarray) -> int:
    import open3d as o3d

    pts = xyz[valid].astype(np.float64)
    col = rgb[valid].astype(np.float64) / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(col)
    if not o3d.io.write_point_cloud(str(path), pc, write_ascii=False):
        raise IOError(f"PCD write failed: {path}")
    restored = o3d.io.read_point_cloud(str(path))
    actual = np.asarray(restored.points)
    colors = np.asarray(restored.colors)
    if actual.shape != pts.shape or not np.allclose(actual, pts, atol=1e-4, rtol=0):
        raise IOError("PCD readback changed point count, coordinates or order")
    if colors.shape != col.shape or not np.allclose(colors, col, atol=1 / 255, rtol=0):
        raise IOError("PCD readback changed colors or order")
    return len(pts)


def save_sample(out: Path, scan: dict, s: dict, defects: list[Defect], spacing_mm: float, pcd: bool, preview_scale: int, mask_only: bool = False, mesh: bool = False, mesh_format: str = 'ply', mesh_max_edge_mm: float | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite sample: {out}")
    polygons = None if mask_only else yolo_polygons(s["inst"], s["seg"], defects)
    out.mkdir(parents=True, exist_ok=True)
    valid = scan["valid"] & ~scan["exclude"]
    rgb = s["rgb"].copy()
    rgb[scan["exclude"]] = 0
    save_png(out / "rgb.png", rgb)
    save_png(out / "valid.png", valid.astype(np.uint8) * 255)
    save_png(out / "ignore.png", scan["exclude"].astype(np.uint8) * 255)
    z = np.where(valid, s["xyz"][..., 2], 0.0)
    save_png(out / "depth_mm.png", np.clip(np.round(z * 10), 0, 65535).astype(np.uint16))
    nimg = np.zeros((*valid.shape, 3), np.uint8)
    nimg[valid] = (np.nan_to_num(s["normals"][valid]) * 127.5 + 127.5).astype(np.uint8)
    save_png(out / "normal.png", nimg)
    save_png(out / "seg.png", s["seg"])
    save_png(out / "inst.png", s["inst"])
    save_png(out / "overlay.png", overlay(rgb, s["seg"]), preview_scale)
    if polygons is not None:
        (out / "labels.txt").write_text("\n".join(polygons) + "\n", encoding="utf-8")

    for d in defects:
        crop_panel(scan, s, d, spacing_mm, out / "crops" / f"{d.inst_id:02d}_{d.cls}.png")

    info = {"defects": [asdict(d) for d in defects], "valid_pixels": int(valid.sum()),
            "source_valid_pixels": int(scan["valid"].sum()), "excluded_pixels": int(scan["exclude"].sum()),
            "yolo_exported": polygons is not None, "shading": scan.get("shading", {})}
    if pcd:
        n = write_pcd(out / "points.pcd", s["xyz"], s["rgb"], valid)
        labels = s["seg"][valid]
        assert n == len(labels), "NFR-2: 점 수 != 라벨 수"
        np.save(out / "point_labels.npy", labels)
        np.save(out / "point_pixel_indices.npy", np.flatnonzero(valid))
        if not np.array_equal(np.load(out / "point_labels.npy"), labels):
            raise IOError("Point label readback mismatch")
        info["point_count"] = n
    if mesh:
        from mesh_export import export_mesh
        info['mesh'] = export_mesh(out / 'mesh', s['xyz'], s['rgb'], valid & scan['part'],
                                   spacing_mm * 3 if mesh_max_edge_mm is None else mesh_max_edge_mm,
                                   mesh_format, labels=s['seg'])
    (out / "defects.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------- 스캔 준비


def prepare_scan(entry: dict) -> tuple[dict, float]:
    d = zdf_io.load(Path(entry["zdf"]))
    xyz, valid = d["xyz"], zdf_io.valid_mask(d["xyz"])
    sp = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)
    spacing = float(np.median(sp[np.isfinite(sp)]))
    part, n, dd, dist = partmask.part_mask(
        xyz, valid, entry.get("part_mode", "above"),
        margin=entry.get("part_margin_mm", 4.0), window=entry.get("part_window_mm", 30.0),
    )
    exclude = np.zeros_like(valid)
    yy, xx = np.ogrid[:valid.shape[0], :valid.shape[1]]
    for region in entry.get("exclude_circles_px", []):
        y, x, radius = region
        if radius <= 0:
            raise ValueError("Exclusion radius must be positive")
        exclude |= (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    allowed = partmask.erode_mm(part & ~exclude, entry.get("placement_erode_mm", 20.0), spacing)
    step = int(entry.get("normal_step", 2))
    n_est0 = estimate_normals(xyz, step)
    light = estimate_light(d["rgba"][..., :3], d["normals"], part)
    scan = dict(
        xyz=xyz, rgba=d["rgba"], normals=d["normals"], valid=valid, part=part, allowed=allowed,
        n_est0=n_est0, n_smooth=smooth_normals(d["normals"], valid), normal_step=step, light_dir=light,
        exclude=exclude, shading=entry.get("shading", {}),
    )
    print(f"[scan] 조명 방향 추정 L={np.round(light, 3)} (카메라 축 = [0,0,-1])")
    # 우리 추정기 vs SDK 법선: 각도 차 중앙값. 크면 재계산한 법선이 원본과 이질적이라는 뜻.
    both = valid & np.isfinite(n_est0[..., 0]) & np.isfinite(d["normals"][..., 0])
    cosang = np.clip((n_est0[both] * d["normals"][both]).sum(-1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    print(f"[scan] {entry['name']}: 격자 {valid.shape} 유효 {valid.mean():.1%} 간격 {spacing:.3f}mm "
          f"부품 {part.sum():,}px 배치가능 {allowed.sum():,}px · 법선 추정기 vs SDK 중앙값 {np.median(ang):.1f}°")
    return scan, spacing


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="parts.yaml", type=Path)
    ap.add_argument("--scan", required=True, help="parts.yaml 의 scans[].name")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/synth"))
    ap.add_argument("--pcd", action="store_true", help="points.pcd + point_labels.npy 도 저장(느림·큼)")
    ap.add_argument("--mask-only", action="store_true", help="Skip lossy YOLO polygons; retain exact masks and point labels")
    ap.add_argument('--mesh', action='store_true', help='Export triangle surface mesh of the part')
    ap.add_argument('--mesh-format', choices=['ply', 'obj', 'both'], default='ply')
    ap.add_argument('--mesh-max-edge-mm', type=float, default=None, help='Reject longer triangle edges; default 3x scan spacing')
    ap.add_argument("--preview-scale", type=int, default=4)
    args = ap.parse_args(argv)

    if args.samples < 1 or args.preview_scale < 1:
        ap.error("samples and preview-scale must be positive")
    cfg = load_config(args.config)
    entry = next((s for s in cfg["scans"] if s["name"] == args.scan), None)
    if entry is None:
        raise SystemExit(f"--scan {args.scan} 이 {args.config} 에 없습니다")
    classes = cfg["classes"]

    scan, spacing = prepare_scan(entry)
    root = args.out / args.scan
    if root.exists():
        raise FileExistsError(f"Use a new output directory: {root}")
    summary = []
    for i in range(args.samples):
        rng = np.random.default_rng([args.seed, i])
        s, defects = make_sample(scan, entry, classes, rng, spacing, args.pcd)
        info = save_sample(root / f"{i:04d}", scan, s, defects, spacing, args.pcd, args.preview_scale,
                           args.mask_only, args.mesh, args.mesh_format, args.mesh_max_edge_mm)
        summary.append({"idx": i, "seed": [args.seed, i], "n_defects": len(defects),
                        "classes": [d.cls for d in defects]})
        print(f"[sample {i:04d}] {len(defects)} 결함: " + ", ".join(
            f"{d.cls}(d={d.params['depth_mm']:.2f}mm, {d.mask_area_px}px)" for d in defects))
    (root / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {args.samples} 샘플 -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
