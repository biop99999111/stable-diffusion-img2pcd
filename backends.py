"""이미지 -> GLB 생성 백엔드. 여기만 GPU 를 쓴다.

img2pcd.py 의 3~6단계(mm 스케일·스무딩·샘플링·PCD·검증)는 백엔드와 무관하게
GLB 하나만 받으면 되므로, 모델을 갈아끼워도 그대로 재사용된다.

  trellis2   microsoft/TRELLIS.2-4B  — 품질 최상위. 단 이미지 인코더로 쓰는
             facebook/dinov3-... 가 gated: manual 이라 Meta 수동 승인이 필요하다.
  hunyuan3d  tencent/Hunyuan3D-2.1   — gated 의존성 없음. shape 10GB / texture 21GB.
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------- 공통


def _free_cuda() -> None:
    """다음 모델을 올리기 전에 VRAM 을 확실히 비운다.

    24GB 급에서는 이걸 안 하면 두 번째 부품이나 texture 단계에서 터진다.
    """
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_vram() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        return 0.0


def _find_repo(names: list[str], env_var: str, explicit: str | None, marker: str) -> Path | None:
    """모델 레포 위치를 찾는다(pip 패키지가 아니라 소스 트리를 쓰는 프로젝트들이라 필요)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]))
    here = Path(__file__).resolve().parent
    for name in names:
        candidates += [here.parent / name, Path("/workspace") / name, Path.home() / name, here / name]
    for c in candidates:
        if (c / marker).exists():
            return c.resolve()
    return None


def _accepted_kwargs(fn, wanted: dict) -> dict:
    """호출 대상이 실제로 받는 인자만 남긴다(버전차로 죽는 것 방지)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    ok = {k: v for k, v in wanted.items() if k in params}
    dropped = set(wanted) - set(ok)
    if dropped:
        print(f"[gen] 호출이 안 받는 인자 제외: {sorted(dropped)}")
    return ok


def _report_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"[gen] gpu: {p.name} {p.total_memory / 2**30:.1f} GiB")
    except Exception:
        pass


# ---------------------------------------------------------------- TRELLIS.2


class Trellis2Backend:
    name = "trellis2"
    default_model = "microsoft/TRELLIS.2-4B"

    def __init__(self, st, repo_dir: str | None = None):
        self.st = st
        self.repo_dir = repo_dir
        self.pipeline = None

    def load(self):
        try:
            import trellis2  # noqa: F401
        except ImportError:
            # trellis2 는 pip 패키지가 아니라 TRELLIS.2 레포 안의 소스 디렉터리다.
            found = _find_repo(["TRELLIS.2"], "TRELLIS_DIR", self.repo_dir, "trellis2")
            if found is None:
                raise SystemExit(
                    "trellis2 를 찾을 수 없습니다.\n"
                    "  --repo-dir /workspace/TRELLIS.2  또는  export TRELLIS_DIR=..."
                ) from None
            sys.path.insert(0, str(found))
            print(f"[gen] sys.path 에 TRELLIS.2 추가: {found}")

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        model_id = self.st.model_id or self.default_model
        print(f"[gen] loading {model_id} ...")
        t0 = time.time()
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.cuda()
        print(f"[gen] loaded in {time.time() - t0:.1f}s")
        print(f"[gen] run signature: {inspect.signature(self.pipeline.run)}")
        _report_gpu()
        return self

    def generate(self, image_path: Path, out_glb: Path, st) -> dict:
        import o_voxel
        from PIL import Image

        image = Image.open(image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        print(f"[gen] {image_path.name} {image.size} {image.mode}")

        wanted: dict = {}
        if st.seed is not None:
            wanted["seed"] = st.seed
        wanted.update(st.run_kwargs)
        kwargs = _accepted_kwargs(self.pipeline.run, wanted)

        _free_cuda()
        t0 = time.time()
        mesh = self.pipeline.run(image, **kwargs)[0]
        t_gen = time.time() - t0
        print(f"[gen] mesh in {t_gen:.1f}s")

        if st.simplify_target:
            mesh.simplify(st.simplify_target)

        t1 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=st.decimation_target,
            texture_size=st.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        t_glb = time.time() - t1
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(out_glb), extension_webp=True)
        print(f"[gen] glb in {t_glb:.1f}s -> {out_glb}")

        peak = _peak_vram()
        # 이슈 #188: 샘플링 중간 텐서가 남아 메시 후처리에서 24GB 급이 터진다.
        del mesh, glb
        _free_cuda()
        print(f"[gen] peak vram: {peak:.2f} GiB")
        return {
            "backend": self.name,
            "t_generate_s": round(t_gen, 2),
            "t_glb_s": round(t_glb, 2),
            "peak_vram_gib": round(peak, 2),
            "textured": True,
        }


# ---------------------------------------------------------------- Hunyuan3D-2.1


class Hunyuan3DBackend:
    name = "hunyuan3d"
    default_model = "tencent/Hunyuan3D-2.1"

    def __init__(self, st, repo_dir: str | None = None):
        self.st = st
        self.repo_dir = repo_dir
        self.shape_pipeline = None
        self.root: Path | None = None

    def _ensure_paths(self) -> Path:
        root = _find_repo(
            ["Hunyuan3D-2.1", "Hunyuan3D-2"], "HUNYUAN3D_DIR", self.repo_dir, "hy3dshape"
        )
        if root is None:
            raise SystemExit(
                "Hunyuan3D-2.1 레포를 찾을 수 없습니다.\n"
                "  bash setup_hunyuan3d.sh  로 설치하거나\n"
                "  --repo-dir /workspace/Hunyuan3D-2.1  또는  export HUNYUAN3D_DIR=..."
            )
        # 공식 demo.py 와 동일하게 소스 디렉터리를 path 에 넣는다.
        for sub in ("hy3dshape", "hy3dpaint"):
            p = str(root / sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        self.root = root
        print(f"[gen] Hunyuan3D 레포: {root}")
        return root

    def load(self):
        self._ensure_paths()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        model_id = self.st.model_id or self.default_model
        print(f"[gen] loading {model_id} (shape) ...")
        t0 = time.time()
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)
        print(f"[gen] loaded in {time.time() - t0:.1f}s")
        print(f"[gen] shape call signature: {inspect.signature(self.shape_pipeline.__call__)}")
        _report_gpu()
        return self

    @staticmethod
    def _prepare_image(image_path: Path, work_dir: Path) -> Path:
        """배경 제거. 공식 demo.py 와 같은 규칙(RGB 면 rembg, RGBA 면 그대로)."""
        from PIL import Image

        image = Image.open(image_path)
        if image.mode == "RGBA":
            # 알파가 전부 불투명하면 마스크가 없는 것과 같으므로 RGB 로 낮춰 rembg 를 태운다.
            import numpy as np

            alpha = np.array(image)[:, :, 3]
            if (alpha == 255).all():
                print("[gen] RGBA 지만 알파가 전부 불투명 -> RGB 로 변환해 배경 제거")
                image = image.convert("RGB")
            else:
                return image_path

        try:
            from hy3dshape.rembg import BackgroundRemover
        except ImportError:
            try:
                from hy3dshape.utils.rembg import BackgroundRemover  # 배치에 따라 경로가 다름
            except ImportError:
                print("[gen] BackgroundRemover 를 찾지 못함 — 원본 이미지를 그대로 사용")
                return image_path

        t0 = time.time()
        out = BackgroundRemover()(image.convert("RGB"))
        work_dir.mkdir(parents=True, exist_ok=True)
        prepared = work_dir / "input_nobg.png"
        out.save(prepared)
        print(f"[gen] 배경 제거 {time.time() - t0:.1f}s -> {prepared.name}")
        return prepared

    def generate(self, image_path: Path, out_glb: Path, st) -> dict:
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_image(image_path, out_glb.parent)

        wanted: dict = {}
        if st.seed is not None:
            wanted["seed"] = st.seed
        wanted.update(st.run_kwargs)
        kwargs = _accepted_kwargs(self.shape_pipeline.__call__, wanted)

        _free_cuda()
        t0 = time.time()
        mesh = self.shape_pipeline(image=str(prepared), **kwargs)[0]
        t_gen = time.time() - t0
        peak_shape = _peak_vram()
        print(f"[gen] mesh in {t_gen:.1f}s (peak {peak_shape:.2f} GiB)")

        shape_glb = out_glb.parent / "mesh_shape.glb"
        mesh.export(str(shape_glb))
        print(f"[gen] shape glb -> {shape_glb}")

        info = {
            "backend": self.name,
            "t_generate_s": round(t_gen, 2),
            "peak_vram_gib": round(peak_shape, 2),
            "textured": False,
        }

        if not st.texture:
            # 텍스처를 건너뛰면 색이 없다 -> PCD 의 rgb 가 회색으로 채워진다.
            import shutil

            shutil.copy(shape_glb, out_glb)
            print("[gen] texture 생략 (--texture 로 활성화). PCD 색은 회색이 된다")
            del mesh
            _free_cuda()
            return info

        # shape 파이프라인을 완전히 내려야 texture(21GB)가 24GB 안에 들어간다.
        # paint 가 메시를 '경로'로 받기 때문에 이렇게 나눌 수 있다.
        del mesh
        self.shape_pipeline = None
        _free_cuda()
        print("[gen] shape 파이프라인 해제 -> texture 로드")

        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        t1 = time.time()
        cfg = Hunyuan3DPaintConfig(max_num_view=st.paint_views, resolution=st.paint_resolution)
        paint = Hunyuan3DPaintPipeline(cfg)
        result = paint(str(shape_glb), image_path=str(prepared), output_mesh_path=str(out_glb))
        t_tex = time.time() - t1
        produced = Path(result) if isinstance(result, (str, Path)) else out_glb
        if produced != out_glb and produced.exists():
            produced.replace(out_glb)
        peak_tex = _peak_vram()
        print(f"[gen] texture in {t_tex:.1f}s (peak {peak_tex:.2f} GiB) -> {out_glb}")

        del paint
        _free_cuda()
        info.update(
            t_texture_s=round(t_tex, 2),
            peak_vram_texture_gib=round(peak_tex, 2),
            textured=True,
        )
        return info


# ---------------------------------------------------------------- 선택


BACKENDS = {"trellis2": Trellis2Backend, "hunyuan3d": Hunyuan3DBackend}


def load_backend(st, repo_dir: str | None = None):
    cls = BACKENDS.get(st.backend)
    if cls is None:
        raise SystemExit(f"모르는 백엔드: {st.backend} (가능: {', '.join(BACKENDS)})")
    return cls(st, repo_dir).load()
