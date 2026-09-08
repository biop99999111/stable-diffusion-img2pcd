"""이미지 1장 -> TRELLIS.2 -> mm 스케일 메시 -> PCD (스파이크).

목적은 "3D 생성/점군화가 되는가" 입증 하나다. spec(docs/bumper-synth-spec.md)의
FR/NFR 을 따르지 않는다 — DB 도, run 도, 라벨도 없다. 산출 PCD 가 쓸 만하면
나중에 FR-1(`bumper asset add`) 입력으로 넘긴다.

단계
  1) TRELLIS.2 로 이미지 -> 메시 (단위 박스 [-0.5, 0.5])
  2) o_voxel.postprocess.to_glb -> GLB(PBR)
  3) 최장축을 실측 치수에 맞춰 mm 스케일 보정
  4) (선택) Taubin 스무딩
  5) 표면 샘플링 -> xyz(mm) + rgb
  6) .pcd 저장 + 3면도 미리보기 PNG + manifest.json

TRELLIS.2 는 무겁고 CUDA 가 필요하므로 1)~2) 만 GPU 를 쓴다.
3)~6) 은 CPU 전용이라 --skip-generate 로 기존 GLB 에서 다시 돌릴 수 있다.
"""

from __future__ import annotations

import os

# example.py 와 동일. expandable_segments 는 24GB 급에서 단편화 OOM 을 줄인다.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# 로컬(Windows cp949) 에서 스모크 테스트할 때 한글/em dash 로 죽지 않게.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------- 설정


@dataclass
class PartSpec:
    name: str
    image: str
    target_mm: float
    fit_axis: str = "auto"


@dataclass
class Settings:
    points: int = 300_000
    spacing_mm: float | None = None
    smooth_iterations: int = 0
    texture_size: int = 2048
    decimation_target: int = 500_000
    simplify_target: int = 16_777_216
    single_view: bool = False
    model_id: str = "microsoft/TRELLIS.2-4B"
    seed: int | None = 42
    run_kwargs: dict = field(default_factory=dict)


def load_config(path: Path) -> tuple[list[PartSpec], Settings]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    parts = [PartSpec(**p) for p in raw["parts"]]
    defaults = {k: v for k, v in (raw.get("defaults") or {}).items() if v is not None}
    known = set(Settings.__dataclass_fields__)
    settings = Settings(**{k: v for k, v in defaults.items() if k in known})
    unknown = set(defaults) - known
    if unknown:
        print(f"[warn] parts.yaml defaults 에서 모르는 키 무시: {sorted(unknown)}")
    return parts, settings


# ---------------------------------------------------------------- 1~2단계 (GPU)


def load_pipeline(model_id: str):
    """TRELLIS.2 파이프라인 로드. trellis2 가 설치된 env 에서만 된다."""
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"[gen] loading {model_id} ...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    print(f"[gen] loaded in {time.time() - t0:.1f}s")
    print(f"[gen] pipeline.run signature: {inspect.signature(pipeline.run)}")
    print(f"[gen] cuda mem allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return pipeline


def _accepted_kwargs(fn, wanted: dict) -> dict:
    """run() 이 실제로 받는 인자만 남긴다.

    TRELLIS.2 공식 example.py 는 run(image) 만 쓴다. seed/resolution 류 인자 이름이
    버전마다 다를 수 있어 시그니처를 보고 걸러낸다(틀린 이름으로 죽는 것 방지).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    ok = {k: v for k, v in wanted.items() if k in params}
    dropped = set(wanted) - set(ok)
    if dropped:
        print(f"[gen] run() 이 안 받는 인자 제외: {sorted(dropped)}")
    return ok


def generate_glb(pipeline, image_path: Path, out_glb: Path, st: Settings) -> dict:
    """이미지 -> 메시 -> GLB. 소요 시간과 VRAM 피크를 반환."""
    import o_voxel
    import torch
    from PIL import Image

    image = Image.open(image_path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    print(f"[gen] {image_path.name} {image.size} {image.mode}")

    wanted: dict = {}
    if st.seed is not None:
        wanted["seed"] = st.seed
    wanted.update(st.run_kwargs)
    kwargs = _accepted_kwargs(pipeline.run, wanted)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = pipeline.run(image, **kwargs)[0]
    t_gen = time.time() - t0
    print(f"[gen] mesh in {t_gen:.1f}s")

    if st.simplify_target:
        mesh.simplify(st.simplify_target)

    t1 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=st.decimation_target,
        texture_size=st.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    t_glb = time.time() - t1
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_glb), extension_webp=True)
    print(f"[gen] glb in {t_glb:.1f}s -> {out_glb}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    # 이슈 #188: 샘플링 중간 텐서가 남아 메시 후처리에서 24GB 급이 터진다.
    # 부품을 여러 개 돌리므로 부품 사이에서 확실히 비운다.
    del mesh, glb
    torch.cuda.empty_cache()
    print(f"[gen] peak cuda mem: {peak:.2f} GiB (empty_cache 완료)")
    return {
        "t_generate_s": round(t_gen, 2),
        "t_glb_s": round(t_glb, 2),
        "peak_vram_gib": round(peak, 2),
    }


# ---------------------------------------------------------------- 3~5단계 (CPU)


def load_trimesh(glb_path: Path):
    """GLB -> 단일 trimesh.Trimesh (색을 정점색으로 베이크)."""
    import trimesh

    loaded = trimesh.load(str(glb_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    # 텍스처 -> 정점색. 샘플링 때 색을 같이 뽑기 위해 필요.
    if hasattr(loaded.visual, "to_color"):
        try:
            loaded.visual = loaded.visual.to_color()
        except Exception as exc:  # 텍스처 없는 GLB 등
            print(f"[mesh] to_color 실패({exc}) — 색 없이 진행")
    print(f"[mesh] vertices={len(loaded.vertices)} faces={len(loaded.faces)}")
    return loaded


def scale_to_mm(tri, target_mm: float, fit_axis: str = "auto") -> tuple[float, np.ndarray]:
    """최장축(또는 지정축)이 target_mm 이 되도록 스케일. 원점은 bbox 중심."""
    extents = tri.bounds[1] - tri.bounds[0]
    axis = int(np.argmax(extents)) if fit_axis == "auto" else "xyz".index(fit_axis)
    if extents[axis] <= 0:
        raise ValueError(f"축 {axis} 크기가 0 — 메시가 비었거나 평면입니다")
    scale = float(target_mm) / float(extents[axis])
    tri.apply_translation(-tri.bounds.mean(axis=0))
    tri.apply_scale(scale)
    bbox_mm = tri.bounds[1] - tri.bounds[0]
    print(
        f"[mesh] fit_axis={'xyz'[axis]} scale={scale:.2f} "
        f"bbox_mm=({bbox_mm[0]:.1f}, {bbox_mm[1]:.1f}, {bbox_mm[2]:.1f})"
    )
    return scale, bbox_mm


def smooth_mesh(tri, iterations: int):
    if iterations <= 0:
        return tri
    import trimesh

    # Taubin: Laplacian 과 달리 부피가 줄지 않는다(결함 깊이를 재야 하므로 중요).
    trimesh.smoothing.filter_taubin(tri, iterations=iterations)
    print(f"[mesh] taubin smoothing x{iterations}")
    return tri


def sample_points(tri, n_points: int, spacing_mm: float | None, seed: int | None):
    """표면 균등 샘플링 -> (xyz mm, rgb 0~1)."""
    import trimesh

    area_mm2 = float(tri.area)
    if spacing_mm:
        n_points = max(1000, int(area_mm2 / (spacing_mm**2)))
        print(f"[pcd] area={area_mm2 / 100:.0f}cm2 spacing={spacing_mm}mm -> {n_points} points")

    # trimesh 5.x 는 자체 RNG 를 쓰므로 np.random.seed 로는 고정되지 않는다.
    # seed 인자를 직접 넘겨야 같은 점군이 재현된다(구버전 대비 np.random.seed 도 유지).
    if seed is not None:
        np.random.seed(seed)
    params = inspect.signature(trimesh.sample.sample_surface).parameters
    kw: dict = {}
    if "sample_color" in params:
        kw["sample_color"] = True
    if "seed" in params and seed is not None:
        kw["seed"] = seed

    result = trimesh.sample.sample_surface(tri, n_points, **kw)
    colors = None
    if len(result) == 3:
        pts, face_idx, colors = result
    else:
        pts, face_idx = result

    if colors is None:
        vc = getattr(tri.visual, "vertex_colors", None)
        if vc is not None and len(vc):
            colors = np.asarray(vc)[tri.faces[face_idx]].mean(axis=1)
        else:
            colors = np.full((len(pts), 4), 200, dtype=np.uint8)

    rgb = np.asarray(colors, dtype=np.float64)[:, :3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    print(f"[pcd] sampled {len(pts)} points (area {area_mm2 / 100:.0f}cm2)")
    return np.asarray(pts, dtype=np.float64), np.clip(rgb, 0.0, 1.0)


def to_open3d(xyz: np.ndarray, rgb: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def keep_single_view(pcd, distance_factor: float = 100.0):
    """hidden point removal — 3D 카메라 1시점 촬영처럼 앞면만 남긴다.

    전체 껍데기(closed shell)는 실제 스캐너 산출과 다르므로, 실데이터에 가까운
    부분 점군이 필요할 때 쓴다.
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    diameter = float(np.linalg.norm(bbox.get_extent()))
    camera = bbox.get_center() + np.array([0.0, 0.0, diameter])
    _, idx = pcd.hidden_point_removal(camera, diameter * distance_factor)
    out = pcd.select_by_index(idx)
    print(f"[pcd] single view: {len(pcd.points)} -> {len(out.points)} points")
    return out


def write_pcd(pcd, path: Path) -> int:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    if not ok:
        raise RuntimeError(f"PCD 저장 실패: {path}")
    return len(pcd.points)


def verify_pcd(path: Path) -> dict:
    """저장한 파일을 다시 읽어 실제로 유효한 PCD 인지 확인(입증 단계)."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    if len(xyz) == 0:
        raise RuntimeError(f"읽었더니 점이 0개: {path}")
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    info = {
        "path": str(path),
        "point_count": int(len(xyz)),
        "has_colors": bool(pcd.has_colors()),
        "bbox_mm": [round(float(v), 2) for v in extent],
        "size_mb": round(path.stat().st_size / 2**20, 2),
    }
    print(f"[verify] {json.dumps(info, ensure_ascii=False)}")
    return info


def preview_png(pcd, path: Path, title: str, max_points: int = 60_000):
    """헤드리스에서도 되는 3면도 산점도. 렌더 창이 없는 vast.ai 용."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = len(np.asarray(pcd.points))
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.full((total, 3), 0.6)
    if total > max_points:
        sel = np.random.default_rng(0).choice(total, max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j), label in zip(axes, [(0, 1), (0, 2), (1, 2)], ["XY", "XZ", "YZ"]):
        ax.scatter(xyz[:, i], xyz[:, j], c=rgb, s=0.4, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{label} (mm)")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{title} — {total} points")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[preview] {path}")


# ---------------------------------------------------------------- 부품 1개 처리


def process_part(part: PartSpec, st: Settings, out_root: Path, pipeline=None, base_dir: Path | None = None) -> dict:
    print(f"\n{'=' * 60}\n[part] {part.name}\n{'=' * 60}")
    base_dir = base_dir or Path.cwd()
    out_dir = out_root / part.name
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "mesh.glb"
    manifest: dict = {"part": part.name, "settings": asdict(st), "spec": asdict(part)}

    if pipeline is not None:
        image_path = Path(part.image)
        if not image_path.is_absolute():
            image_path = (base_dir / part.image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"입력 이미지 없음: {image_path}")
        manifest["image"] = str(image_path)
        manifest.update(generate_glb(pipeline, image_path, glb_path, st))
    else:
        if not glb_path.exists():
            raise FileNotFoundError(f"--skip-generate 인데 GLB 가 없음: {glb_path}")
        print(f"[gen] skip — 기존 {glb_path} 사용")

    tri = load_trimesh(glb_path)
    scale, bbox_mm = scale_to_mm(tri, part.target_mm, part.fit_axis)
    tri = smooth_mesh(tri, st.smooth_iterations)
    manifest["scale_factor"] = round(scale, 4)
    manifest["bbox_mm"] = [round(float(v), 2) for v in bbox_mm]

    # 나중에 FR-2(Open3D Poisson 재구성)와 비교할 수 있게 mm 메시도 남긴다.
    tri.export(str(out_dir / "mesh_mm.ply"))

    xyz, rgb = sample_points(tri, st.points, st.spacing_mm, st.seed)
    pcd = to_open3d(xyz, rgb)
    if st.single_view:
        pcd = keep_single_view(pcd)

    pcd_path = out_dir / f"{part.name}.pcd"
    write_pcd(pcd, pcd_path)
    manifest["verify"] = verify_pcd(pcd_path)
    preview_png(pcd, out_dir / "preview.png", part.name)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="이미지 1장 -> TRELLIS.2 -> PCD (스파이크)")
    ap.add_argument("--config", default="parts.yaml", type=Path)
    ap.add_argument("--out", default="out", type=Path)
    ap.add_argument("--only", help="부품 이름 하나만 처리")
    ap.add_argument(
        "--skip-generate", action="store_true", help="GPU 없이 기존 GLB 에서 PCD 만 다시 생성"
    )
    ap.add_argument("--points", type=int)
    ap.add_argument("--spacing-mm", type=float)
    ap.add_argument("--single-view", action="store_true")
    ap.add_argument("--smooth", type=int, help="Taubin 반복 횟수")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model", help="기본 microsoft/TRELLIS.2-4B")
    ap.add_argument("--run-kwargs", help="run() 추가 인자 JSON")
    args = ap.parse_args(argv)

    config_path = args.config.resolve()
    parts, st = load_config(config_path)
    if args.points:
        st.points = args.points
    if args.spacing_mm:
        st.spacing_mm = args.spacing_mm
    if args.single_view:
        st.single_view = True
    if args.smooth is not None:
        st.smooth_iterations = args.smooth
    if args.seed is not None:
        st.seed = args.seed
    if args.model:
        st.model_id = args.model
    if args.run_kwargs:
        st.run_kwargs = json.loads(args.run_kwargs)
    if args.only:
        parts = [p for p in parts if p.name == args.only]
        if not parts:
            raise SystemExit(f"--only {args.only} 에 맞는 부품이 {config_path.name} 에 없습니다")

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline = None if args.skip_generate else load_pipeline(st.model_id)

    results = []
    for part in parts:
        results.append(process_part(part, st, out_root, pipeline, base_dir=config_path.parent))

    summary = out_root / "run.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\n[done] {len(results)} 부품 -> {out_root}")
    for r in results:
        v = r["verify"]
        print(
            f"  {r['part']:14s} {v['point_count']:>9,} pts  "
            f"bbox_mm={v['bbox_mm']}  {v['size_mb']}MB"
        )
    print(f"[done] 요약: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
