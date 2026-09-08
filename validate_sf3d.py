"""Read back SF3D artifacts; geometric/visual accuracy remains a separate review."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image


def inspect_glb(path):
    scene = trimesh.load(path, force="scene", process=False)
    if not scene.graph.nodes_geometry:
        raise ValueError("GLB has no geometry instances")
    items = []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        mesh = scene.geometry[key]
        if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Empty or nonfinite mesh")
        if not np.isfinite(transform).all():
            raise ValueError("Nonfinite scene transform")
        uv = getattr(mesh.visual, "uv", None)
        mat = getattr(mesh.visual, "material", None)
        texture = getattr(mat, "baseColorTexture", None)
        valid_uv = uv is not None and np.shape(uv) == (len(mesh.vertices), 2) and np.isfinite(uv).all()
        items.append({"node": str(node), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                      "has_uv": bool(valid_uv), "has_texture": texture is not None,
                      "texture_size": list(texture.size) if texture is not None else None})
    return {"instances": items, "textured": all(i["has_uv"] and i["has_texture"] for i in items)}


def validate(directory, expected_points=300000):
    import open3d as o3d
    directory = Path(directory)
    errors, warnings = [], []
    result = {"passed": False, "errors": errors, "warnings": warnings,
              "visual_quality": "pending_review", "metrology": "unverified"}
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        part = manifest["part"]
        if Path(part).name != part or part in {".", ".."} or "\\" in part:
            raise ValueError("Invalid part name")
        result["glb"] = inspect_glb(directory / "mesh.glb")
        if not result["glb"]["textured"]:
            errors.append("GLB needs real base-color texture and finite UVs on every instance")
        mm = trimesh.load(directory / "mesh_mm.ply", force="mesh", process=False)
        if not len(mm.faces) or not np.isfinite(mm.vertices).all():
            errors.append("Invalid mm mesh")
        elif not np.allclose(mm.extents, manifest["bbox_mm"], atol=.1, rtol=1e-4):
            errors.append("mm mesh bbox differs from manifest")
        pcd = o3d.io.read_point_cloud(str(directory / f"{part}.pcd"))
        xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)
        if len(xyz) != expected_points or not np.isfinite(xyz).all():
            errors.append("PCD point count or coordinates invalid")
        if not pcd.has_colors() or not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            errors.append("PCD colors missing or invalid")
        elif np.max(np.ptp(rgb, axis=0)) < 1 / 255:
            warnings.append("Nearly uniform PCD color: inspect fallback gray versus actual object")
        if len(xyz):
            if (xyz < mm.bounds[0] - .1).any() or (xyz > mm.bounds[1] + .1).any():
                errors.append("PCD lies outside mm mesh bounds")
        result["pcd"] = {"points": len(xyz), "has_colors": pcd.has_colors(),
                         "bbox_mm": np.ptp(xyz, axis=0).tolist() if len(xyz) else None}
        for filename in ("preview.png", "input_nobg.png"):
            with Image.open(directory / filename) as image:
                image.load()
                if filename == "input_nobg.png" and not np.asarray(image.convert("RGBA"))[:, :, 3].any():
                    errors.append("Empty foreground")
        result["shape_check"] = manifest.get("shape_check")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["passed"] = not errors
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--points", type=int, default=300000)
    args = ap.parse_args()
    report = validate(args.directory, args.points)
    (args.directory / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
