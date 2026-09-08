"""GPU 없이 CPU 단계(3~6)만 검증하는 스모크 테스트.

TRELLIS.2 없이 합성 메시로 GLB 를 만들어 스케일·샘플링·PCD·미리보기를 통째로 돌린다.
vast.ai 에 올리기 전에 로컬에서 이 파일만 돌려보면 파이프라인 후반부 버그가 걸린다.

    python smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import img2pcd


def make_fake_glb(path: Path) -> None:
    """TRELLIS.2 출력을 흉내낸 GLB: 단위 박스 [-0.5, 0.5] 안의 곡면 판 + 정점색."""
    import trimesh

    # 범퍼처럼 완만하게 휜 판때기
    nu, nv = 80, 40
    u = np.linspace(-0.5, 0.5, nu)
    v = np.linspace(-0.15, 0.15, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ww = 0.12 * np.cos(uu * np.pi) - 0.5 * vv**2

    verts = np.stack([uu.ravel(), vv.ravel(), ww.ravel()], axis=1)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b = i * nv + j, i * nv + j + 1
            c, d = (i + 1) * nv + j, (i + 1) * nv + j + 1
            faces.append([a, b, d])
            faces.append([a, d, c])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    colors[:, 0] = np.clip(120 + 200 * (uu.ravel() + 0.5), 0, 255)
    colors[:, 1] = 90
    colors[:, 2] = np.clip(200 - 150 * (vv.ravel() + 0.15), 0, 255)
    colors[:, 3] = 255
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    extent = mesh.bounds[1] - mesh.bounds[0]
    print(f"[fake] GLB 생성: {path.name} bbox={np.round(extent, 3)}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="img2pcd_smoke_"))
    out_root = tmp / "out"
    part = img2pcd.PartSpec(
        name="bumper_cover",
        image="unused.jpg",
        target_mm=1750.0,
        # 가짜 GLB 는 두께가 거의 없는 판이라 최소축이 실측보다 작게 나온다.
        # 형상 판정이 실제로 FAIL 을 낼 수 있는지(=늘 PASS 가 아닌지) 확인하는 값.
        expect_mm=[1750.0, 450.0, 350.0],
    )
    st = img2pcd.Settings(points=50_000, smooth_iterations=2, seed=42)

    make_fake_glb(out_root / part.name / "mesh.glb")

    # backend=None -> 생성 단계 건너뛰고 기존 GLB 사용
    manifest = img2pcd.process_part(part, st, out_root, backend=None)

    checks: list[tuple[str, bool, str]] = []
    v = manifest["verify"]
    checks.append(("PCD 점 개수 > 0", v["point_count"] > 0, str(v["point_count"])))
    checks.append(("PCD 색상 있음", v["has_colors"], str(v["has_colors"])))

    longest = max(v["bbox_mm"])
    checks.append(
        ("최장축 == target_mm(±1%)", abs(longest - 1750.0) < 17.5, f"{longest:.1f}mm"),
    )
    checks.append(("스케일 배율 기록됨", manifest["scale_factor"] > 0, str(manifest["scale_factor"])))

    # 형상 판정: 정렬 비교라 축 순서와 무관해야 하고, 최장축은 항상 1.00 이어야 한다.
    sc = manifest["shape_check"]
    checks.append(("형상 판정 기록됨", sc["verdict"] in ("PASS", "FAIL"), sc["verdict"]))
    checks.append(("최장축 배율 == 1.00", abs(sc["ratio"][0] - 1.0) < 0.01, str(sc["ratio"][0])))
    flipped = img2pcd.shape_check([350.0, 1750.0, 450.0], [450.0, 350.0, 1750.0], 0.30)
    checks.append(("축 순서 무관 (완전 일치 = PASS)", flipped["verdict"] == "PASS", str(flipped["ratio"])))
    inflated = img2pcd.shape_check([1750.0, 963.0, 516.0], [1750.0, 450.0, 350.0], 0.30)
    checks.append(
        ("부풀림 감지 (Hunyuan3D 실측값)", inflated["verdict"] == "FAIL", f"최소축 x{inflated['thinnest_axis_ratio']}")
    )

    for name in ("preview.png", "mesh_mm.ply", "manifest.json"):
        f = out_root / part.name / name
        checks.append((f"{name} 생성", f.exists() and f.stat().st_size > 0, f"{f.stat().st_size if f.exists() else 0}B"))

    # 재현성: 같은 seed 로 다시 샘플링하면 같은 점이 나온다
    tri = img2pcd.load_trimesh(out_root / part.name / "mesh.glb")
    img2pcd.scale_to_mm(tri, part.target_mm, part.fit_axis)
    a, _ = img2pcd.sample_points(tri, 5000, None, seed=7)
    b, _ = img2pcd.sample_points(tri, 5000, None, seed=7)
    checks.append(("같은 seed = 같은 점군", np.allclose(a, b), "allclose"))

    print("\n" + "=" * 60)
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
        failed += not ok
    print("=" * 60)
    print(f"{len(checks) - failed}/{len(checks)} passed — 산출물: {out_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
