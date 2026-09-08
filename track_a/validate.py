"""Real-scan CPU validation, optionally including PCD readback and dent comparison."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import yaml
import inject
import compare
import zdf_io


def validate_sample(scan, sample, defects):
    np.testing.assert_array_equal(zdf_io.valid_mask(sample['xyz']), scan['valid'])
    changed = np.abs(sample['disp']) > 0
    if np.any(changed & (~scan['part'] | scan['exclude'])):
        raise AssertionError('Off-part or excluded surface changed')
    np.testing.assert_array_equal(sample['xyz'][~changed], scan['xyz'][~changed])
    for defect in defects:
        m = sample['inst'] == defect.inst_id
        if int(m.sum()) != defect.mask_area_px or not np.all(sample['seg'][m] == defect.class_id):
            raise AssertionError('Instance/class label mismatch, possibly overlapping defects')
    return {'source_valid_preserved': True, 'outside_geometry_unchanged': True,
            'off_part_changes': 0, 'excluded_changes': 0, 'instance_counts_match': True}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('parts.yaml'))
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--out', type=Path, default=Path('out/validation'))
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--pcd', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--mesh', action='store_true')
    ap.add_argument('--mesh-format', choices=['ply', 'obj', 'both'], default='ply')
    args = ap.parse_args(argv)
    if args.samples < 1 or args.out.exists():
        ap.error('samples must be positive and out must be a new directory')
    cfg = inject.load_config(args.config)
    if args.data_dir:
        for entry in cfg['scans']:
            entry['zdf'] = str(args.data_dir.resolve() / Path(entry['zdf']).name)
    args.out.mkdir(parents=True)
    effective = args.out / 'effective_config.yaml'
    effective.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    results = []
    for entry in cfg['scans']:
        start = time.perf_counter()
        scan, spacing = inject.prepare_scan(entry)
        for i in range(args.samples):
            sample, defects = inject.make_sample(scan, entry, cfg['classes'], np.random.default_rng([0, i]), spacing, args.pcd)
            checks = validate_sample(scan, sample, defects)
            out = args.out / entry['name'] / f'{i:04d}'
            info = inject.save_sample(out, scan, sample, defects, spacing, args.pcd, 4, mask_only=True,
                                      mesh=args.mesh, mesh_format=args.mesh_format)
            if args.mesh:
                checks['mesh_readback'] = info['mesh']['readback_verified']
                indices = np.load(out / 'mesh' / 'vertex_pixel_indices.npy')
                np.testing.assert_array_equal(np.load(out / 'mesh' / 'vertex_labels.npy'), sample['seg'].reshape(-1)[indices])
                checks['mesh_vertex_labels'] = True
            if args.pcd:
                idx = np.load(out / 'point_pixel_indices.npy')
                labels = np.load(out / 'point_labels.npy')
                np.testing.assert_array_equal(labels, sample['seg'].reshape(-1)[idx])
                np.testing.assert_array_equal(idx, np.flatnonzero(scan['valid'] & ~scan['exclude']))
                checks['pcd_readback_and_pixel_labels'] = True
            results.append({'scan': entry['name'], 'sample': i, 'defects': len(defects),
                            'checks': checks, 'valid_pixels': info['valid_pixels'],
                            'elapsed_seconds_including_prepare': time.perf_counter() - start})
            del sample
        del scan
        gc.collect()
    (args.out / 'validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    if args.compare:
        compare.main(['--config', str(effective), '--real', '559', '1114', '--at', '760', '1560', '--out', str(args.out / 'compare')])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
