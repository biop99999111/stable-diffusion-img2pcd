"""CPU contract/integration tests. No downloads or GPU inference."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

import sd_spike
import img2pcd
from sf3d_backend import StableFast3DBackend


def fake_torch():
    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                           reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 1024**3,
                           is_bf16_supported=lambda: True)
    return SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16",
                           manual_seed=lambda s: None, no_grad=nullcontext,
                           autocast=lambda **kw: nullcontext(),
                           Generator=lambda **kw: SimpleNamespace(manual_seed=lambda s: s))


class SpikeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(self.root / "part.png")
        self.raw = {"parts": [{"name": "panel", "image": "part.png", "target_mm": 100,
                               "expect_mm": [100, 50, 10]}], "defaults": {"seed": 7}}
        config = self.root / "parts.yaml"
        config.write_text(yaml.safe_dump(self.raw), encoding="utf-8")
        self.args = argparse.Namespace(config=config, only=None, strength=.25, steps=30,
                                       size=64, guidance=5., seeds=[42, 43], out=self.root / "out",
                                       model=sd_spike.MODEL, prompt="panel", negative_prompt="",
                                       backend="sf3d")

    def test_padding_preserves_proportions(self):
        image = sd_spike.prepare_image(self.root / "part.png", 64)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.getpixel((32, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((32, 32)), (255, 0, 0))

    def test_transparency_composites_white(self):
        Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(self.root / "alpha.png")
        self.assertEqual(sd_spike.prepare_image(self.root / "alpha.png", 64).getpixel((32, 32)),
                         (255, 255, 255))

    def test_invalid_settings_fail_before_model_load(self):
        for attr, value in (("strength", 0), ("strength", float("nan")), ("steps", 1),
                            ("size", 63), ("guidance", float("inf")), ("seeds", [42, 42]),
                            ("only", "missing")):
            with self.subTest(attr=attr, value=value):
                args = argparse.Namespace(**vars(self.args))
                setattr(args, attr, value)
                with self.assertRaises(ValueError):
                    sd_spike.build_plan(args)

    def test_source_paths_relative_to_config(self):
        _, jobs = sd_spike.build_plan(self.args)
        self.assertEqual(Path(jobs[0]["source"]), self.root / "part.png")
        self.assertEqual([job["seed"] for job in jobs], [42, 43])

    def test_generation_produces_reusable_configs_and_provenance(self):
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[kwargs["image"]])
        raw, jobs = sd_spike.build_plan(self.args)
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            config = sd_spike.generate(self.args, raw, jobs, pipeline=pipeline)
        parts, settings = img2pcd.load_config(config)
        self.assertEqual([p.name for p in parts], ["panel_sd_s42", "panel_sd_s43"])
        self.assertEqual(settings.backend, "sf3d")
        self.assertEqual(parts[0].expect_mm, [100, 50, 10])
        self.assertTrue(all(Path(p.image).is_file() for p in parts))
        baseline, _ = img2pcd.load_config(self.args.out / "parts_baseline.yaml")
        self.assertEqual(baseline[0].image, str(self.root / "part.png"))
        manifest = json.loads((self.args.out / "sd_manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(manifest["runs"][0]["source_sha256"]), 64)
        self.assertEqual([c["generator"] for c in calls], [42, 43])
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            with self.assertRaises(ValueError):
                sd_spike.generate(self.args, raw, jobs, pipeline=pipeline)

    def test_sf3d_export_flows_through_actual_pcd_pipeline(self):
        import trimesh
        calls = []
        mesh = trimesh.creation.box(extents=[1, .5, .1])
        def run_image(image, **kwargs):
            calls.append((image.mode, kwargs))
            return mesh, {}
        st = img2pcd.Settings(backend="sf3d", points=500, texture_size=1024)
        backend = StableFast3DBackend(st)
        backend.model = SimpleNamespace(run_image=run_image)
        backend.model_id = backend.default_model
        backend.load_seconds = 0
        backend.session = None
        rembg = SimpleNamespace(new_session=lambda **kwargs: object())
        utils = SimpleNamespace(remove_background=lambda image, session: image,
                                resize_foreground=lambda image, ratio: image)
        part = img2pcd.PartSpec(**self.raw["parts"][0])
        with patch.dict(sys.modules, {"torch": fake_torch(), "rembg": rembg, "sf3d.utils": utils}):
            result = img2pcd.process_part(part, st, self.root / "pcd", backend, self.root)
        self.assertEqual(calls[0], ("RGBA", {"bake_resolution": 1024}))
        self.assertEqual(result["verify"]["point_count"], 500)
        self.assertEqual(result["shape_check"]["verdict"], "PASS")
        self.assertTrue((self.root / "pcd/panel/preview.png").is_file())

    def test_sf3d_empty_alpha_rejected(self):
        Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(self.root / "empty.png")
        st = img2pcd.Settings(backend="sf3d")
        backend = StableFast3DBackend(st)
        utils = SimpleNamespace(remove_background=None, resize_foreground=None)
        with patch.dict(sys.modules, {"torch": fake_torch(), "rembg": SimpleNamespace(), "sf3d.utils": utils}):
            with self.assertRaisesRegex(ValueError, "transparent"):
                backend.generate(self.root / "empty.png", self.root / "mesh.glb", st)


if __name__ == "__main__":
    unittest.main()
