"""SDXL img2img experiment; emits configs consumed by img2pcd.py in its own env."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

from PIL import Image, ImageOps
import yaml

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def prepare_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        # Preserve proportions and the entire part, including transparent inputs.
        return ImageOps.pad(background.convert("RGB"), (size, size),
                            method=Image.Resampling.LANCZOS, color="white")


def build_plan(args) -> tuple[dict, list[dict]]:
    if not 0 < args.strength <= 1 or not math.isfinite(args.strength):
        raise ValueError("strength must be in (0, 1]")
    if args.steps < 1 or int(args.steps * args.strength) < 1:
        raise ValueError("steps * strength must provide at least one denoising step")
    if args.size < 64 or args.size > 2048 or args.size % 8:
        raise ValueError("size must be a multiple of 8 between 64 and 2048")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    parts = raw["parts"]
    if args.only:
        parts = [part for part in parts if part["name"] == args.only]
    if not parts:
        raise ValueError("no matching parts")
    names = [part["name"] for part in parts]
    if len(set(names)) != len(names):
        raise ValueError("duplicate part names")
    jobs = []
    for part in parts:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", part["name"]):
            raise ValueError("part names must contain only letters, digits, _ or -")
        path = (args.config.resolve().parent / part["image"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        for seed in args.seeds:
            jobs.append({"part": dict(part), "source": str(path), "seed": seed})
    return raw, jobs


def generate(args, raw, jobs, pipeline=None) -> Path:
    import torch

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be empty to avoid mixed experiments: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SDXL spike; use --dry-run for CPU preflight")
    start = time.perf_counter()
    if pipeline is None:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_tiling()
    load_seconds = time.perf_counter() - start
    records, variants, baselines = [], [], []
    seen = set()
    for job in jobs:
        part, seed = job["part"], job["seed"]
        source = Path(job["source"])
        name = f"{part['name']}_sd_s{seed}"
        prepared = prepare_image(source, args.size)
        if part["name"] not in seen:
            prepared.save(out / f"{part['name']}_input.png")
            baseline = dict(part, image=str(source))
            baselines.append(baseline)
            seen.add(part["name"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipeline(
            prompt=args.prompt, negative_prompt=args.negative_prompt, image=prepared,
            strength=args.strength, num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        image_path = out / f"{name}.png"
        result.save(image_path)
        variants.append(dict(part, name=name, image=str(image_path)))
        records.append({"part": part["name"], "variant": name, "seed": seed,
                        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "image": str(image_path), "seconds": round(elapsed, 3),
                        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
        print(f"[sdxl] {name}: {elapsed:.2f}s -> {image_path}")
        # Preserve partial progress if a later job fails; completion is explicit below.
        write_manifest(out, args, records, load_seconds, complete=False)
    defaults = dict(raw.get("defaults") or {})
    defaults.update(backend=args.backend, model_id=None)
    for filename, parts in (("parts_sd.yaml", variants), ("parts_baseline.yaml", baselines)):
        (out / filename).write_text(yaml.safe_dump({"parts": parts, "defaults": defaults},
                                   allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_manifest(out, args, records, load_seconds, complete=True)
    return out / "parts_sd.yaml"


def write_manifest(out, args, records, load_seconds, complete):
    data = {"model": args.model, "prompt": args.prompt, "negative_prompt": args.negative_prompt,
            "strength": args.strength, "steps": args.steps, "guidance": args.guidance,
            "size": args.size, "load_seconds": load_seconds, "complete": complete,
            "downstream_backend": args.backend, "runs": records,
            "experiment": "SDXL image preprocessing followed by a separate 3D model"}
    (out / "sd_manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("parts.yaml"))
    ap.add_argument("--out", type=Path, default=Path("out_sd_images"))
    ap.add_argument("--only")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default="Product photograph of the same automotive body panel, unchanged shape and proportions, clean white background, neutral lighting, sharp details")
    ap.add_argument("--negative-prompt", default="deformed, inflated, extra parts, text, watermark")
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--backend", choices=["hunyuan3d", "trellis2", "sf3d"], default="hunyuan3d")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs/settings without loading torch or models")
    args = ap.parse_args(argv)
    try:
        raw, jobs = build_plan(args)
        if args.dry_run:
            for job in jobs:
                prepare_image(Path(job["source"]), args.size)
            print(json.dumps({"model": args.model, "jobs": jobs}, indent=2, ensure_ascii=False))
            return 0
        config = generate(args, raw, jobs)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    print(f'In the {args.backend} environment: python img2pcd.py --config "{config}" --out out_sd_pcd')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
