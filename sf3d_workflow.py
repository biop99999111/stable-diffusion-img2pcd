"""Reproducible original-photo SF3D run; --dry-run needs no model or GPU."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml
from PIL import Image
from img2pcd import load_config

ROOT = Path(__file__).resolve().parent
MODEL = "stabilityai/stable-fast-3d"


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_info(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}")
    return result.stdout.strip()


def build_config(config, only, resolution):
    parts, settings = load_config(config)
    selected = [p for p in parts if p.name == only]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one part named {only}")
    part = selected[0]
    if Path(part.name).name != part.name or part.name in {".", ".."} or "\\" in part.name:
        raise ValueError("Part name must be a single directory name")
    image = (config.resolve().parent / part.image).resolve()
    with Image.open(image) as source:
        source.verify()
    part.image = str(image)
    settings.backend = "sf3d"
    settings.texture_size = resolution
    settings.points = 300000
    settings.spacing_mm = None
    settings.single_view = False
    settings.smooth_iterations = 0
    settings.seed = 42
    settings.texture_only = False
    settings.run_kwargs = {"remesh": "none"}
    return {"parts": [asdict(part)], "defaults": asdict(settings)}


def preflight(repo):
    import torch
    if not (repo / "sf3d/system.py").is_file():
        raise FileNotFoundError(f"Missing SF3D source: {repo}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    sys.path.insert(0, str(repo))
    import rembg, texture_baker, uv_unwrapper  # noqa: F401
    from sf3d.system import SF3D  # noqa: F401
    command_info([sys.executable, "-m", "pip", "check"])
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "source_commit": command_info(["git", "rev-parse", "HEAD"], repo),
            "source_status": command_info(["git", "status", "--porcelain"], repo),
            "nvcc": command_info(["nvcc", "--version"]),
            "nvidia_smi": command_info(["nvidia-smi"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "parts.yaml")
    ap.add_argument("--only", default="bumper_cover")
    ap.add_argument("--texture-resolution", type=int, choices=[1024, 2048], default=1024)
    ap.add_argument("--repo-dir", type=Path, default=ROOT.parent / "stable-fast-3d")
    ap.add_argument("--revision", default="main", help="HF revision; resolved SHA is recorded")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config = build_config(args.config, args.only, args.texture_resolution)
    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error("Output must be empty; use a new --out for each attempt")
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "parts_sf3d.yaml"
    cfg.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {"status": "planned", "input_sha256": sha256(config["parts"][0]["image"]),
              "model_id": MODEL, "requested_revision": args.revision,
              "config": config, "visual_quality": "pending_review", "metrology": "unverified"}
    save(out / "workflow.json", report)
    if args.dry_run:
        print(f"Input/config verified; no model downloaded: {out}")
        return
    start = time.perf_counter()
    try:
        report["environment"] = preflight(args.repo_dir.resolve())
        (out / "freeze.txt").write_text(command_info([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
        report["experiment_commit"] = command_info(["git", "rev-parse", "HEAD"], ROOT)
        report["experiment_status"] = command_info(["git", "status", "--porcelain"], ROOT)
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(MODEL, revision=args.revision).sha
        report["model_revision"] = revision
        download_start = time.perf_counter()
        model_path = snapshot_download(MODEL, revision=revision,
                                       allow_patterns=["config.yaml", "model.safetensors"])
        report["snapshot_seconds"] = time.perf_counter() - download_start
        command = [sys.executable, str(ROOT / "img2pcd.py"), "--config", str(cfg),
                   "--backend", "sf3d", "--repo-dir", str(args.repo_dir.resolve()),
                   "--model", model_path, "--only", args.only, "--out", str(out)]
        report.update(command=command, status="running")
        save(out / "workflow.json", report)
        with (out / "inference.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", cwd=ROOT)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError("Inference failed; inspect inference.log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        from validate_sf3d import validate
        validation = validate(out / args.only, 300000)
        save(out / args.only / "validation.json", validation)
        report["validation"] = validation
        if not validation["passed"]:
            raise RuntimeError("Artifact validation failed; inspect validation.json")
        manifest = json.loads((out / args.only / "manifest.json").read_text(encoding="utf-8"))
        report["metrics"] = {k: manifest.get(k) for k in
                             ("bbox_mm", "shape_check", "t_load_s", "t_preprocess_s", "t_generate_s", "peak_vram_gib")}
        report["status"] = "artifacts_validated_video_pending"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        save(out / "workflow.json", report)


if __name__ == "__main__":
    main()
