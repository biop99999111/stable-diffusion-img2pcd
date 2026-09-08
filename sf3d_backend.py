"""Stable Fast 3D adapter. This model is distinct from Stable Diffusion/SDXL."""
from pathlib import Path
import sys
import time


class StableFast3DBackend:
    name = "sf3d"
    default_model = "stabilityai/stable-fast-3d"

    def __init__(self, st, repo_dir=None):
        self.st = st
        self.repo_dir = repo_dir

    def load(self):
        import torch
        from backends import _find_repo

        if not torch.cuda.is_available():
            raise RuntimeError("This SF3D spike requires CUDA. Use --skip-generate for CPU postprocessing.")
        root = _find_repo(["stable-fast-3d"], "SF3D_DIR", self.repo_dir, "sf3d/system.py")
        if root is None:
            raise FileNotFoundError("SF3D source not found. Run setup_sf3d.sh or specify --repo-dir /path/to/stable-fast-3d")
        sys.path.insert(0, str(root))
        from sf3d.system import SF3D

        self.model_id = self.st.model_id or self.default_model
        start = time.perf_counter()
        self.model = SF3D.from_pretrained(self.model_id, config_name="config.yaml",
                                          weight_name="model.safetensors").to("cuda").eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.session = None
        return self

    def generate(self, image_path: Path, out_glb: Path, st):
        import numpy as np
        import rembg
        import torch
        from PIL import Image, ImageOps
        from sf3d.utils import remove_background, resize_foreground

        allowed = {"remesh", "vertex_count", "estimate_illumination"}
        unknown = set(st.run_kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported SF3D run_kwargs: {sorted(unknown)}")
        if st.run_kwargs.get("remesh", "none") not in {"none", "triangle", "quad"}:
            raise ValueError("SF3D remesh must be none, triangle or quad")
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        if st.seed is not None:
            torch.manual_seed(st.seed)
            np.random.seed(st.seed)
        start = time.perf_counter()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        alpha = np.asarray(image.getchannel("A"))
        if not alpha.any():
            raise ValueError("Input image is fully transparent")
        if (alpha == 255).all():
            if self.session is None:
                self.session = rembg.new_session(providers=["CPUExecutionProvider"])
            image = remove_background(image, self.session)
        if not np.asarray(image.getchannel("A")).any():
            raise ValueError("Background removal produced an empty foreground")
        image = resize_foreground(image, 0.85)
        image.save(out_glb.parent / "input_nobg.png")
        preprocess_seconds = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            mesh, _ = self.model.run_image(image, bake_resolution=st.texture_size, **st.run_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError("SF3D returned an empty mesh")
        mesh.export(str(out_glb), include_normals=True)
        return {"backend": self.name, "model_id": self.model_id, "textured": True,
                "t_load_s": round(self.load_seconds, 3), "t_preprocess_s": round(preprocess_seconds, 3),
                "t_generate_s": round(elapsed, 3), "peak_vram_gib": round(peak, 3)}
