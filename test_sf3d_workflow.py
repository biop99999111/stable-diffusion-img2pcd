"""CPU fixtures test actual textured GLB/PCD readback; no model downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from img2pcd import PartSpec, Settings, process_part
from sf3d_workflow import build_config
from validate_sf3d import inspect_glb, validate


def make_fixture(root):
    root = Path(root)
    directory = root / "panel"
    directory.mkdir(parents=True, exist_ok=True)
    texture = Image.new("RGB", (16, 16), (210, 40, 20))
    texture.paste((20, 90, 220), (8, 0, 16, 16))
    mesh = trimesh.creation.box(extents=[1, .35, .12])
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.column_stack((mesh.vertices[:, 0]+.5, mesh.vertices[:, 1]/.35+.5)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,
                                                   metallicFactor=.1, roughnessFactor=.7))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="first")
    transform = np.eye(4); transform[0, 3] = 1.3
    scene.add_geometry(mesh, node_name="second", transform=transform)
    scene.export(directory / "mesh.glb")
    Image.new("RGBA", (32, 32), (200, 20, 30, 255)).save(directory / "input_nobg.png")
    process_part(PartSpec("panel", "unused.png", 1750), Settings(points=500), root, None, root)
    return directory


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_config_uses_original_absolute_image(self):
        import yaml
        Image.new("RGB", (10, 10)).save(self.root / "photo.png")
        config = self.root / "parts.yaml"
        config.write_text(yaml.safe_dump({"parts": [{"name": "bumper", "image": "photo.png", "target_mm": 1750}]}))
        result = build_config(config, "bumper", 1024)
        self.assertEqual(result["parts"][0]["image"], str(self.root / "photo.png"))
        self.assertEqual(result["defaults"]["backend"], "sf3d")
        self.assertEqual(result["defaults"]["texture_size"], 1024)
        with self.assertRaises(ValueError):
            build_config(config, "missing", 1024)

    def test_textured_scene_and_pcd_pass(self):
        directory = make_fixture(self.root)
        result = validate(directory, 500)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(result["glb"]["instances"]), 2)
        self.assertEqual(result["metrology"], "unverified")

    def test_metadata_cannot_hide_missing_texture(self):
        directory = make_fixture(self.root)
        trimesh.creation.box().export(directory / "mesh.glb")
        result = validate(directory, 500)
        self.assertFalse(result["passed"])
        self.assertFalse(result["glb"]["textured"])

    def test_wrong_point_count_and_empty_foreground_fail(self):
        directory = make_fixture(self.root)
        Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(directory / "input_nobg.png")
        result = validate(directory, 501)
        self.assertFalse(result["passed"])
        self.assertIn("Empty foreground", result["errors"])
        self.assertIn("PCD point count or coordinates invalid", result["errors"])

    def test_renderer_preserves_scene_instances_and_scalar_material(self):
        from render_sf3d_bumper_video import load_instances
        directory = make_fixture(self.root)
        meshes = load_instances(directory / "mesh.glb")
        self.assertEqual(len(meshes), 2)
        self.assertGreater(abs(meshes[1].centroid[0]-meshes[0].centroid[0]), 1)
        self.assertAlmostEqual(meshes[0].visual.material.roughnessFactor, .7)
        self.assertIsNone(meshes[0].visual.material.metallicRoughnessTexture)

    def test_failed_preflight_records_failure_and_protects_existing_output(self):
        import sf3d_workflow
        output = self.root / "attempt"
        argv = ["sf3d_workflow.py", "--out", str(output)]
        with patch("sys.argv", argv), patch("sf3d_workflow.preflight", side_effect=RuntimeError("No GPU")):
            with self.assertRaisesRegex(RuntimeError, "No GPU"):
                sf3d_workflow.main()
        report = json.loads((output / "workflow.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "No GPU")
        original = (output / "workflow.json").read_bytes()
        with patch("sys.argv", argv), patch("sf3d_workflow.preflight") as preflight:
            with self.assertRaises(SystemExit):
                sf3d_workflow.main()
            preflight.assert_not_called()
        self.assertEqual(original, (output / "workflow.json").read_bytes())

    def test_comparison_leaves_missing_metrics_unrecorded(self):
        from compare_sf3d import comparison
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"backend": "hunyuan3d", "t_texture_s": 88.7}))
        result = comparison([path])
        self.assertIn("88.7", result)
        self.assertIn("unrecorded", result)
        self.assertIn("not a controlled same-input comparison", result)


if __name__ == "__main__":
    unittest.main()
