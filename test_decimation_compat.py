"""Exercise the installed-upstream function rewrite without GPU dependencies."""
import sys
import types
import unittest
from unittest.mock import patch

from backends import _patch_hunyuan_decimation


def mesh_simplify_trimesh(courent, outputpath, target_count=40000):
    if len(courent.faces) > target_count:
        courent = courent.simplify_quadric_decimation(target_count)
    courent.export(outputpath)


class Mesh:
    def __init__(self, faces):
        self.faces = range(faces)
        self.calls = []
        self.output = None

    def simplify_quadric_decimation(self, percent=None, face_count=None):
        if percent is not None and not 0 <= percent <= 1:
            raise ValueError("target_reduction must be between 0 and 1")
        self.calls.append((percent, face_count))
        return self

    def export(self, path):
        self.output = path


class DecimationTests(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("utils.simplify_mesh_utils")
        self.module.mesh_simplify_trimesh = mesh_simplify_trimesh
        self.modules = patch.dict(sys.modules, {
            "utils.simplify_mesh_utils": self.module,
            "trimesh": types.SimpleNamespace(Trimesh=Mesh),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_original_error_and_fixed_face_count(self):
        mesh = Mesh(80000)
        with self.assertRaises(ValueError):
            mesh_simplify_trimesh(mesh, "mesh.obj")
        _patch_hunyuan_decimation()
        self.module.mesh_simplify_trimesh(mesh, "mesh.obj")
        self.assertEqual(mesh.calls, [(None, 40000)])
        self.assertEqual(mesh.output, "mesh.obj")

    def test_small_mesh_still_skips_reduction(self):
        mesh = Mesh(12)
        _patch_hunyuan_decimation()
        self.module.mesh_simplify_trimesh(mesh, "small.obj")
        self.assertEqual(mesh.calls, [])
        self.assertEqual(mesh.output, "small.obj")

    def test_custom_target_and_repeated_patch(self):
        _patch_hunyuan_decimation()
        first = self.module.mesh_simplify_trimesh
        _patch_hunyuan_decimation()
        self.assertIs(first, self.module.mesh_simplify_trimesh)
        mesh = Mesh(1000)
        first(mesh, "custom.obj", target_count=200)
        self.assertEqual(mesh.calls, [(None, 200)])


if __name__ == "__main__":
    unittest.main()
