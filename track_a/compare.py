"""Real/synthetic dent comparison with shared geometry and shading."""
import argparse
import json
from pathlib import Path
import numpy as np
import inject
import measure


def angular_stats(ref, n, mask):
    ok = mask & inject.finite_normals(ref) & inject.finite_normals(n)
    if not ok.any():
        raise ValueError("No valid normals for comparison")
    a, b = ref[ok], n[ok]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {"angular_std_deg": float(angle.std()), "angular_rms_deg": float(np.sqrt(np.mean(angle ** 2))), "normal_pixels": int(ok.sum())}


def profile(xyz, res, inner):
    selected = np.where(inner & np.isfinite(res), res, np.nan)
    if not np.isfinite(selected).any():
        raise ValueError("No finite dent interior")
    iy, ix = np.unravel_index(np.nanargmin(selected), selected.shape)
    x = np.linalg.norm(xyz[iy] - xyz[iy, ix], axis=-1) * np.sign(np.arange(xyz.shape[1]) - ix)
    return x, res[iy], float(-selected[iy, ix])


def profile_metrics(real, synth, spacing):
    grid = np.arange(-15, 15 + spacing / 2, spacing)
    def interpolate(p):
        x, y = p[:2]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        order = np.argsort(x)
        x, y = x[order], y[order]
        if len(x) < 3:
            raise ValueError("Insufficient profile points")
        z = np.interp(grid, x, y, left=np.nan, right=np.nan)
        for a, b in zip(x[:-1], x[1:]):
            if b - a > 2.5 * spacing:
                z[(grid > a) & (grid < b)] = np.nan
        return z
    a, b = interpolate(real), interpolate(synth)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        raise ValueError("Insufficient common profile support")
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if min(a[ok].std(), b[ok].std()) > 1e-8 else None
    return {"profile_correlation": corr, "profile_rmse_mm": float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))),
            "profile_common_points": int(ok.sum()), "profile_interval_mm": [-15, 15]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("parts.yaml"))
    ap.add_argument("--scan", default="bumper_cover")
    ap.add_argument("--real", nargs=2, type=int, required=True)
    ap.add_argument("--at", nargs=2, type=int, required=True)
    ap.add_argument("--depth", type=float, default=3.06)
    ap.add_argument("--radius", type=float, default=22.5, help="Zero-crossing major radius in mm, not half-depth radius")
    ap.add_argument("--rim", type=float, default=0.25, help="Peak outward rim height in mm")
    ap.add_argument("--aspect", type=float, default=1.2)
    ap.add_argument("--angle", type=float, default=0.4)
    ap.add_argument("--specular", type=float, default=None, help="Experimental, uncalibrated Blinn-Phong weight")
    ap.add_argument("--ring-inner", type=float, default=32.0)
    ap.add_argument("--ring-outer", type=float, default=42.0)
    ap.add_argument("--out", type=Path, default=Path("out/compare"))
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory: {args.out}")
    if not 0 < args.radius * 1.3 < args.ring_inner < args.ring_outer:
        ap.error("Fit ring must be outside the rim: 1.3 * radius < ring-inner < ring-outer")
    cfg = inject.load_config(args.config)
    entry = next(s for s in cfg["scans"] if s["name"] == args.scan)
    scan, spacing = inject.prepare_scan(entry)
    if args.specular is not None:
        scan["shading"] = {**scan["shading"], "specular": args.specular}
    H, W = scan["valid"].shape
    for y, x in (args.real, args.at):
        if not (0 <= y < H and 0 <= x < W) or not scan["valid"][y, x]:
            raise ValueError("Comparison centers must be valid scan pixels")
    field = lambda w, v, c, n: inject.dent_field(w, v, c, n, args.radius, args.depth, args.aspect, args.angle, args.rim)
    y, x, ys, xs, d = inject.place_field(scan, scan["allowed"], np.zeros((H, W), bool), np.random.default_rng(0),
                                      int(args.radius * 1.7 / spacing) + 4, field, args.at)
    xyz = scan["xyz"].copy()
    inject.apply_displacement(xyz, scan["n_smooth"], d, ys, xs)
    disp = np.zeros((H, W), np.float32)
    disp[ys, xs] = d
    rgb, normals, touched, shade = inject.render_changed(scan, xyz, scan["rgba"][..., :3].astype(np.float32), disp)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inject.configure_fonts()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    results, profiles = {}, []
    for i, (name, cloud, color, nrm, center) in enumerate((
        ("Real", scan["xyz"], scan["rgba"][..., :3], scan["normals"], args.real),
        ("Synthetic", xyz, rgb, normals, args.at),
    )):
        (sy, sx), res, ring, inner, _ = measure._quadric_residual(cloud, scan["valid"], *center, args.ring_inner, args.ring_outer, spacing)
        p = profile(cloud[sy, sx], res, inner)
        profiles.append(p)
        ref = inject.smooth_normals(nrm, scan["valid"])[sy, sx]
        results[name.lower()] = {"max_depth_mm": p[2], "ring_residual_rms_mm": float(np.sqrt(np.mean(res[ring] ** 2))),
                                **angular_stats(ref, nrm[sy, sx], inner & scan["valid"][sy, sx])}
        axes[i, 0].imshow(color[sy, sx]); axes[i, 0].set_title(name + " RGB")
        axes[i, 1].imshow(np.clip(np.nan_to_num(nrm[sy, sx]) * 127.5 + 127.5, 0, 255).astype(np.uint8))
        axes[i, 1].set_title(name + " normals")
        im = axes[i, 2].imshow(res, cmap="RdBu", vmin=-args.depth, vmax=args.depth)
        fig.colorbar(im, ax=axes[i, 2]); axes[i, 2].set_title(name + " residual (mm)")
        axes[i, 3].imshow(scan["rgba"][sy, sx, :3]); axes[i, 3].set_title("Original RGB / fitting ring")
        axes[i, 3].contour(ring, levels=[0.5], colors="cyan", linewidths=0.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    results.update(profile_metrics(*profiles, spacing))
    results["parameters"] = {"depth_mm": args.depth, "zero_crossing_radius_mm": args.radius, "rim_height_mm": args.rim,
                             "aspect": args.aspect, "angle_rad": args.angle, "shading": scan["shading"],
                             "real_center_px": args.real, "synthetic_center_px": args.at,
                             "fit_ring_mm": [args.ring_inner, args.ring_outer]}
    results["shade_range"] = [float(shade[touched].min()), float(shade[touched].max())]
    results["interpretation"] = "Exploratory comparison; angular residual includes surface structure. No similarity threshold calibrated."
    args.out.mkdir(parents=True)
    fig.tight_layout(); fig.savefig(args.out / "real_vs_synth.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, name in zip(profiles, ("Real", "Synthetic")):
        ax.plot(p[0], p[1], label=name)
    ax.set(xlim=(-30, 30), xlabel="Signed 3D distance along row (mm)", ylabel="Surface residual (mm)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "profile.png", dpi=110); plt.close(fig)
    (args.out / "stats.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
