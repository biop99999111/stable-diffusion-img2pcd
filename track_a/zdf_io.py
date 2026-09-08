"""Zivid .zdf -> organized numpy 배열 (xyz / rgba / snr / normals).

.zdf 는 SDK 2.18 부터 HDF5 가 아니라 zstd 컨테이너라 h5py 로는 못 읽는다.
Zivid SDK(C:/Program Files/Zivid) 와 같은 버전의 `zivid` pip 패키지가 필요하고,
Windows 에서는 SDK 의 bin 폴더가 PATH 에 있어야 pyd 가 ZividCore.dll 을 찾는다.

한 번 읽은 결과는 <name>_raw.npz 로 캐시한다 — SDK 없이도 뒤 단계가 돈다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ZIVID_BIN = Path(os.environ.get("ZIVID_BIN", r"C:\Program Files\Zivid\bin"))
FIELDS = ("xyz", "rgba", "snr", "normals")
_DLL_HANDLES = []


def _ensure_sdk_on_path() -> None:
    if sys.platform == "win32" and ZIVID_BIN.is_dir():
        os.environ["PATH"] = str(ZIVID_BIN) + os.pathsep + os.environ.get("PATH", "")
        try:
            if not _DLL_HANDLES:
                _DLL_HANDLES.append(os.add_dll_directory(str(ZIVID_BIN)))
        except (AttributeError, OSError):
            pass


def read_zdf(path: Path) -> dict[str, np.ndarray]:
    """SDK 로 읽는다. xyz(H,W,3 float32 mm, 결측=NaN) rgba(H,W,4 uint8) snr(H,W) normals(H,W,3)."""
    _ensure_sdk_on_path()
    import zivid

    app = zivid.Application()  # Keep the SDK application alive while frame data is copied.
    frame = zivid.Frame(str(path))
    pc = frame.point_cloud()
    data = {f: pc.copy_data(f) for f in FIELDS}
    data["meta"] = np.array(
        {
            "model": frame.camera_info.model_name,
            "serial": frame.camera_info.serial_number,
            "height": pc.height,
            "width": pc.width,
        },
        dtype=object,
    )
    return data


def load(path: Path, cache: bool = True) -> dict[str, np.ndarray]:
    """<name>_raw.npz 가 있으면 그것을, 없으면 .zdf 를 읽고 캐시한다."""
    path = Path(path)
    npz = path if path.suffix.lower() == ".npz" else path.with_name(path.stem + "_raw.npz")
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            return {k: z[k] for k in FIELDS}
    if path.suffix.lower() == ".npz":
        raise FileNotFoundError(path)
    data = read_zdf(path)
    if cache:
        np.savez_compressed(npz, **data)
    return data


def valid_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=-1)
