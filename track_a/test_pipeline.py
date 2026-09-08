"""CPU regression tests: python -m unittest discover -s . -p test_pipeline.py -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import inject
import compare
import zdf_io
import mesh_export


def plane_scan():
    yy, xx = np.mgrid[:160, :160]
    xyz = np.stack([xx - 80, yy - 80, np.full_like(xx, 400)], -1).astype(np.float32)
    normals = np.zeros_like(xyz); normals[..., 2] = -1
    valid = np.ones((160, 160), bool)
    allowed = valid.copy(); allowed[:20] = False; allowed[-20:] = False
    allowed[:, :20] = False; allowed[:, -20:] = False
    return dict(xyz=xyz, rgba=np.full((160, 160, 4), 150, np.uint8), normals=normals,
                n_smooth=normals.copy(), n_est0=inject.estimate_normals(xyz), normal_step=2,
                valid=valid, part=valid.copy(), allowed=allowed, exclude=np.zeros_like(valid),
                light_dir=np.array([0., 0., -1.]), shading={})


class PipelineTests(unittest.TestCase):
    def test_mesh_faces_holes_and_discontinuities(self):
        yy, xx = np.mgrid[:4, :4]
        xyz = np.stack([xx, yy, np.full_like(xx, 100)], -1).astype(np.float32)
        rgb = np.full_like(xyz, 100, dtype=np.uint8)
        valid = np.ones((4, 4), bool)
        v, c, f, pixels = mesh_export.triangulate(xyz, rgb, valid, 2)
        self.assertEqual(len(f), 18)
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        self.assertTrue(np.all(n[:, 2] < 0))
        valid[1, 1] = False
        v, c, f, pixels = mesh_export.triangulate(xyz, rgb, valid, 2)
        self.assertNotIn(5, pixels)
        self.assertLess(len(f), 18)
        xyz[:, 2:, 2] += 100
        v, c, f, pixels = mesh_export.triangulate(xyz, rgb, valid, 2)
        self.assertTrue(np.all(np.ptp(v[f, 2], axis=1) == 0))

    def test_mesh_ply_obj_and_labels_roundtrip(self):
        s = plane_scan()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'mesh'
            labels = np.ones((160, 160), np.uint8)
            info = mesh_export.export_mesh(out, s['xyz'], s['rgba'][..., :3], s['valid'], 2, 'both', labels)
            self.assertTrue(info['readback_verified'])
            self.assertEqual(info['triangles'], 2 * 159 * 159)
            np.testing.assert_array_equal(np.load(out / 'vertex_labels.npy'), np.ones(160 * 160, np.uint8))

    def test_empty_mesh_fails(self):
        s = plane_scan()
        with self.assertRaises(ValueError):
            mesh_export.triangulate(s['xyz'], s['rgba'][..., :3], np.zeros_like(s['valid']), 2)

    def test_valid_requires_all_coordinates(self):
        p = np.array([[[np.nan, 0, 400], [0, 0, 400]]])
        self.assertEqual(zdf_io.valid_mask(p).tolist(), [[False, True]])

    def test_normal_nan_does_not_spread(self):
        s = plane_scan(); s['normals'][80, 80] = np.nan
        smooth = inject.smooth_normals(s['normals'], s['valid'])
        self.assertTrue(np.isfinite(smooth).all())
        np.testing.assert_allclose(smooth[..., 2], -1)

    def test_dent_and_rim_sign(self):
        p = np.array([[[0., 0, 400], [0, 11.5, 400], [0, 14, 400]]])
        d = inject.dent_field(p, np.ones((1, 3), bool), p[0, 0], np.array([0., 0, -1]), 10, 2, 1, 0, .25)
        np.testing.assert_allclose(d, [[2, -.25, 0]], atol=1e-6)

    def test_identity_shading(self):
        s = plane_scan()
        r = inject.shade_ratio(s['light_dir'], s['normals'], s['normals'], s['xyz'], specular=1.0)
        np.testing.assert_array_equal(r, np.ones_like(r))

    def test_off_part_overlap_and_exclusion_rejected(self):
        for reason in ('part', 'exclude', 'occupied'):
            s = plane_scan(); occupied = np.zeros_like(s['valid'])
            if reason == 'part': s['part'][80, 82] = False
            if reason == 'exclude': s['exclude'][80, 82] = True
            if reason == 'occupied': occupied[80, 82] = True
            field = lambda w, v, c, n: inject.dent_field(w, v, c, n, 8, 1, 1, 0)
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                inject.place_field(s, s['allowed'], occupied, np.random.default_rng(0), 20, field, [80, 80])

    def test_no_fallback_on_failed_placement(self):
        with self.assertRaises(RuntimeError):
            inject.pick_location(np.random.default_rng(0), np.ones((10, 10), bool), np.zeros((10, 10), bool), 2)

    def test_generation_preserves_geometry_and_labels(self):
        s = plane_scan()
        s['xyz'][10, 10] = np.nan; s['valid'][10, 10] = False
        cfg = {'defects': {'dent': {'count': [2, 2], 'radius_mm': [6, 8], 'depth_mm': [1, 2], 'aspect': [1, 1.3], 'rim_height_mm': [.1, .2]}}}
        a, defects = inject.make_sample(s, cfg, {'dent': 1}, np.random.default_rng(10), 1, False)
        b, _ = inject.make_sample(s, cfg, {'dent': 1}, np.random.default_rng(10), 1, False)
        np.testing.assert_array_equal(a['xyz'], b['xyz'])
        np.testing.assert_array_equal(zdf_io.valid_mask(a['xyz']), s['valid'])
        unchanged = a['disp'] == 0
        np.testing.assert_array_equal(a['xyz'][unchanged], s['xyz'][unchanged])
        for d in defects:
            self.assertEqual(int((a['inst'] == d.inst_id).sum()), d.mask_area_px)
        self.assertFalse(np.any((a['seg'] != 255) & ~s['part']))

    def test_yolo_rejects_holes_and_disconnected_masks(self):
        for hole in (False, True):
            inst = np.zeros((20, 20), np.uint16); inst[2:10, 2:10] = 1
            if hole: inst[4:6, 4:6] = 0
            else: inst[12:16, 12:16] = 1
            d = inject.Defect(1, 'dent', 1, [5, 5], [0, 0, 0], {})
            with self.assertRaises(ValueError):
                inject.yolo_polygons(inst, inst, [d])

    def test_yolo_simple_polygon_roundtrip(self):
        import cv2
        inst = np.zeros((20, 20), np.uint16); inst[2:10, 3:12] = 1
        d = inject.Defect(1, 'dent', 1, [5, 5], [0, 0, 0], {})
        line = inject.yolo_polygons(inst, inst, [d])[0]
        pts = np.array(list(map(float, line.split()[1:]))).reshape(-1, 2)
        out = np.zeros_like(inst); cv2.fillPoly(out, [np.rint(pts * 20).astype(np.int32)], 1)
        np.testing.assert_array_equal(out, inst)

    def test_comparison_ignores_invalid_normals(self):
        n = np.array([[[0., 0, -1], [np.nan, np.nan, np.nan]]])
        stats = compare.angular_stats(n, n, np.ones((1, 2), bool))
        self.assertEqual(stats['normal_pixels'], 1)
        self.assertEqual(stats['angular_std_deg'], 0)

    def test_identical_profiles(self):
        x = np.arange(-20, 21, dtype=float); y = -np.exp(-(x / 5) ** 2)
        stats = compare.profile_metrics((x, y), (x, y), 1)
        self.assertAlmostEqual(stats['profile_correlation'], 1)
        self.assertEqual(stats['profile_rmse_mm'], 0)

    def test_pcd_roundtrip(self):
        s = plane_scan()
        with tempfile.TemporaryDirectory() as td:
            count = inject.write_pcd(Path(td) / 'test.pcd', s['xyz'], s['rgba'][..., :3], s['valid'])
        self.assertEqual(count, 160 * 160)

    def test_saved_exclusion_and_point_mapping(self):
        from PIL import Image
        s = plane_scan(); s['exclude'][70:90, 70:90] = True
        sample = {'xyz': s['xyz'].copy(), 'rgb': s['rgba'][..., :3].copy(), 'normals': s['normals'].copy(),
                  'seg': np.full((160, 160), 255, np.uint8), 'inst': np.zeros((160, 160), np.uint16)}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'sample'
            info = inject.save_sample(out, s, sample, [], 1, True, 1, mask_only=True)
            self.assertEqual(info['point_count'], 160 * 160 - 400)
            self.assertFalse((out / 'labels.txt').exists())
            for name in ('rgb.png', 'overlay.png', 'normal.png', 'depth_mm.png', 'valid.png'):
                pixels = np.array(Image.open(out / name))
                self.assertTrue(np.all(pixels[s['exclude']] == 0), name)
            np.testing.assert_array_equal(np.load(out / 'point_pixel_indices.npy'), np.flatnonzero(~s['exclude']))
            with self.assertRaises(FileExistsError):
                inject.save_sample(out, s, sample, [], 1, False, 1, mask_only=True)


if __name__ == '__main__':
    unittest.main()
