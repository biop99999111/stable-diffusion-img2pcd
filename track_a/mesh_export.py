"""Organized scan -> colored triangle PLY / geometry OBJ, without filling holes."""
from pathlib import Path
import json
import numpy as np


def triangulate(xyz, rgb, valid, max_edge_mm):
    if xyz.ndim != 3 or xyz.shape[2] != 3 or valid.shape != xyz.shape[:2] or rgb.shape != xyz.shape:
        raise ValueError('Expected organized XYZ/RGB (H,W,3) and validity (H,W)')
    if not np.isfinite(max_edge_mm) or max_edge_mm <= 0:
        raise ValueError('max_edge_mm must be finite and positive')
    h, w = valid.shape
    flat = xyz.reshape(-1, 3)
    valid = (valid & np.isfinite(xyz).all(-1)).reshape(-1)
    blocks = []
    # Bound temporary coordinate/edge arrays even on full-resolution scans.
    for start in range(0, h - 1, 64):
        rows = np.arange(start, min(start + 64, h - 1))[:, None]
        cols = np.arange(w - 1)[None, :]
        a = (rows * w + cols).reshape(-1)
        faces = np.concatenate([np.stack([a, a + w, a + 1], 1),
                                np.stack([a + 1, a + w, a + w + 1], 1)])
        faces = faces[valid[faces].all(1)]
        p = flat[faces]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        cross = np.cross(e1, e2)
        keep = (np.linalg.norm(e1, axis=1) <= max_edge_mm)
        keep &= np.linalg.norm(e2, axis=1) <= max_edge_mm
        keep &= np.linalg.norm(p[:, 2] - p[:, 1], axis=1) <= max_edge_mm
        keep &= np.linalg.norm(cross, axis=1) > 1e-10
        # Winding faces the camera at the origin.
        flip = (cross * p.mean(1)).sum(1) > 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        blocks.append(faces[keep].astype(np.int32))
    faces = np.concatenate(blocks) if blocks else np.empty((0, 3), np.int32)
    if not len(faces):
        raise ValueError('No mesh faces: check part mask and max-edge-mm')
    used = np.zeros(h * w, bool)
    used[faces.reshape(-1)] = True
    pixels = np.flatnonzero(used)
    mapping = np.full(h * w, -1, np.int32)
    mapping[pixels] = np.arange(len(pixels), dtype=np.int32)
    return flat[pixels], rgb.reshape(-1, 3)[pixels], mapping[faces], pixels


def write_ply(path, vertices, colors, faces):
    header = ('ply\nformat binary_little_endian 1.0\ncomment coordinates_in_millimeters\n'
              f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        for start in range(0, len(vertices), 100_000):
            v = vertices[start:start + 100_000]
            block = np.empty(len(v), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
            block['xyz'], block['rgb'] = v, colors[start:start + len(v)]
            stream.write(block.tobytes())
        for start in range(0, len(faces), 100_000):
            f = faces[start:start + 100_000]
            block = np.empty(len(f), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
            block['count'], block['indices'] = 3, f
            stream.write(block.tobytes())


def write_obj(path, vertices, faces):
    # OBJ geometry is standard; use PLY for vertex RGB (no nonstandard OBJ color extension).
    with Path(path).open('w', encoding='ascii', newline='\n') as stream:
        stream.write('# coordinates in millimeters\no scanned_part\n')
        for start in range(0, len(vertices), 100_000):
            np.savetxt(stream, vertices[start:start + 100_000], fmt='v %.9g %.9g %.9g')
        for start in range(0, len(faces), 100_000):
            np.savetxt(stream, faces[start:start + 100_000] + 1, fmt='f %d %d %d')


def verify_mesh(path, vertices, colors, faces, check_colors):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if Path(path).suffix == '.obj':
        # OBJ importers may reorder vertices. Verify file indices directly, then
        # confirm that the independent importer can reconstruct all triangles.
        from itertools import islice
        if len(f) != len(faces):
            raise IOError('OBJ importer triangle count mismatch')
        with Path(path).open(encoding='ascii') as stream:
            next(stream); next(stream)
            for source, prefix in ((vertices, 'v '), (faces + 1, 'f ')):
                for start in range(0, len(source), 100_000):
                    expected = source[start:start + 100_000]
                    lines = list(islice(stream, len(expected)))
                    if len(lines) != len(expected) or any(not line.startswith(prefix) for line in lines):
                        raise IOError('OBJ records missing or malformed')
                    actual = np.fromstring(' '.join(line[2:] for line in lines), sep=' ').reshape(-1, 3)
                    if not np.allclose(actual, expected, atol=1e-4 if prefix == 'v ' else 0, rtol=0):
                        raise IOError('OBJ coordinates or topology changed')
            if stream.read().strip():
                raise IOError('Unexpected extra OBJ records')
        return
    if v.shape != vertices.shape or not np.allclose(v, vertices, atol=1e-4, rtol=0):
        raise IOError(f'Mesh vertex readback mismatch: {path}')
    if not np.array_equal(f, faces):
        raise IOError(f'Mesh topology readback mismatch: {path}')
    if check_colors:
        c = np.asarray(mesh.vertex_colors)
        if c.shape != colors.shape or not np.allclose(c * 255, colors, atol=1e-5, rtol=0):
            raise IOError(f'Mesh color readback mismatch: {path}')


def export_mesh(out, xyz, rgb, valid, max_edge_mm, formats='ply', labels=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Use a new mesh output directory: {out}')
    if formats not in ('ply', 'obj', 'both'):
        raise ValueError('formats must be ply, obj or both')
    vertices, colors, faces, pixels = triangulate(xyz, rgb, valid, max_edge_mm)
    out.mkdir(parents=True)
    files = []
    for ext in (('ply', 'obj') if formats == 'both' else (formats,)):
        path = out / ('surface.' + ext)
        if ext == 'ply':
            write_ply(path, vertices, colors, faces)
        else:
            write_obj(path, vertices, faces)
        verify_mesh(path, vertices, colors, faces, ext == 'ply')
        files.append(path.name)
    np.save(out / 'vertex_pixel_indices.npy', pixels)
    if labels is not None:
        if labels.shape != valid.shape:
            raise ValueError('Vertex labels must match the organized grid')
        np.save(out / 'vertex_labels.npy', labels.reshape(-1)[pixels])
    info = {'vertices': len(vertices), 'triangles': len(faces), 'units': 'mm',
            'max_edge_mm': float(max_edge_mm), 'files': files, 'readback_verified': True,
            'surface_only': True, 'holes_filled': False, 'grid_shape': list(valid.shape)}
    (out / 'mesh.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


def main(argv=None):
    import argparse
    import inject
    ap = argparse.ArgumentParser(description='Export the original scanned part as a polygon object')
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('parts.yaml'))
    ap.add_argument('--scan', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--format', choices=['ply', 'obj', 'both'], default='both')
    ap.add_argument('--max-edge-mm', type=float, default=None)
    ap.add_argument('--exclude-known-defects', action='store_true')
    args = ap.parse_args(argv)
    cfg = inject.load_config(args.config)
    entry = next(s for s in cfg['scans'] if s['name'] == args.scan)
    scan, spacing = inject.prepare_scan(entry)
    valid = scan['valid'] & scan['part']
    if args.exclude_known_defects:
        valid &= ~scan['exclude']
    info = export_mesh(args.out, scan['xyz'], scan['rgba'][..., :3], valid,
                       args.max_edge_mm if args.max_edge_mm is not None else spacing * 3, args.format)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
