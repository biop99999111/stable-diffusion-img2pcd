"""vast.ai 인스턴스가 TRELLIS.2 를 돌릴 수 있는지 진단한다.

setup_vast.sh 를 돌리기 **전에** 실행한다. 표준 라이브러리만 쓰므로 아무것도
설치되지 않은 base 이미지에서도 돈다.

    python check_env.py

FAIL 이 하나라도 있으면 setup_vast.sh 가 도중에 깨진다. 특히 nvcc 가 없으면
CuMesh/o-voxel 소스 빌드에서 죽는다(vast.ai 에서 *-runtime 이미지를 고른 경우).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# TRELLIS.2 가 이미지 인코더로 쓰는 gated 모델. 여기 접근이 안 되면 파이프라인
# 로드가 401 로 죽는다 — 15GB 받고 나서가 아니라 여기서 먼저 걸러낸다.
GATED_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
GATED_URL = f"https://huggingface.co/{GATED_REPO}/resolve/main/config.json"
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
results: list[tuple[str, str, str]] = []


def add(level: str, name: str, detail: str) -> None:
    results.append((level, name, detail))


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def check_os() -> None:
    if platform.system() == "Linux":
        add(PASS, "OS", platform.platform())
    else:
        add(FAIL, "OS", f"{platform.system()} — TRELLIS.2 는 공식적으로 Linux 전용")


def check_gpu() -> None:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if not out:
        add(FAIL, "GPU (nvidia-smi)", "실패 — GPU 인스턴스가 맞는지 확인")
        return
    line = out.strip().splitlines()[0]
    add(PASS, "GPU", line)

    m = re.search(r"(\d+)\s*MiB", line)
    if not m:
        add(WARN, "VRAM", f"파싱 실패: {line}")
        return
    gib = int(m.group(1)) / 1024
    if gib >= 23.0:  # 24GB 카드는 23.x GiB 로 보고된다
        add(PASS, "VRAM", f"{gib:.1f} GiB (공식 최소 24GB 충족)")
    else:
        add(FAIL, "VRAM", f"{gib:.1f} GiB — TRELLIS.2 공식 최소는 24GB")


def check_nvcc() -> None:
    """가장 중요. 없으면 CuMesh/o-voxel/nvdiffrast 소스 빌드가 전부 실패한다."""
    out = run(["nvcc", "--version"])
    if out:
        m = re.search(r"release (\d+\.\d+)", out)
        add(PASS, "CUDA toolkit (nvcc)", f"CUDA {m.group(1)}" if m else "설치됨")
        return

    hint = "없음 — *-devel 이미지나 PyTorch 템플릿을 쓰거나 cuda-toolkit 설치 필요"
    for base in ("/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12.1"):
        if os.path.exists(f"{base}/bin/nvcc"):
            hint = f"PATH 에 없지만 {base}/bin/nvcc 는 있음 → export PATH={base}/bin:$PATH"
            add(WARN, "CUDA toolkit (nvcc)", hint)
            return
    add(FAIL, "CUDA toolkit (nvcc)", hint)


def check_cuda_home() -> None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        add(PASS if os.path.isdir(home) else WARN, "CUDA_HOME", home)
    elif os.path.isdir("/usr/local/cuda"):
        add(WARN, "CUDA_HOME", "미설정 — setup_vast.sh 가 /usr/local/cuda 로 잡는다 (이슈 #106)")
    else:
        add(WARN, "CUDA_HOME", "미설정 · /usr/local/cuda 도 없음 — 빌드 전 직접 지정 필요")


def check_tool(cmd: str, label: str, level_if_missing: str = FAIL, note: str = "") -> None:
    path = shutil.which(cmd)
    if path:
        add(PASS, label, path)
    else:
        add(level_if_missing, label, f"없음{' — ' + note if note else ''}")


def check_disk() -> None:
    usage = shutil.disk_usage(os.path.expanduser("~"))
    free = usage.free / 2**30
    total = usage.total / 2**30
    detail = f"{free:.0f} GiB free / {total:.0f} GiB"
    if free >= 100:
        add(PASS, "디스크", detail)
    elif free >= 60:
        add(WARN, "디스크", detail + " — 모델 15GB + 의존성 30GB 라 빠듯함")
    else:
        add(FAIL, "디스크", detail + " — 100GB 권장")


def check_python() -> None:
    v = sys.version_info
    add(PASS, "python(현재)", f"{v.major}.{v.minor}.{v.micro} @ {sys.executable}")
    # setup.sh 가 conda env 를 python=3.10 으로 새로 만들므로 현재 버전은 상관없다.


def check_torch() -> None:
    """지금은 없어도 된다 — setup.sh 가 torch 2.6.0(cu124)을 직접 설치한다."""
    try:
        import torch
    except ImportError:
        add(WARN, "torch(현재)", "없음 — 정상. setup.sh 가 torch 2.6.0+cu124 를 설치한다")
        return
    # 버전은 따지지 않는다 — setup.sh 가 conda env 'trellis2' 를 새로 만들고
    # 거기에 torch 2.6.0 을 따로 깔기 때문에 현재 env 의 torch 는 쓰이지 않는다.
    add(PASS, "torch(현재)", f"{torch.__version__} · cuda={torch.cuda.is_available()} (참고용)")


def check_hf_gated() -> None:
    """gated 모델(DINOv3) 접근 가능 여부를 실제 HTTP 요청으로 확인한다.

    TRELLIS.2 는 이미지 인코더로 이 모델을 쓴다. 승인/토큰이 없으면
    Trellis2ImageTo3DPipeline.from_pretrained 가 401 GatedRepoError 로 죽는다.
    """
    token = next((os.environ[v] for v in HF_TOKEN_VARS if os.environ.get(v)), None)
    add(
        PASS if token else WARN,
        "HF 토큰",
        f"{next(v for v in HF_TOKEN_VARS if os.environ.get(v))} 설정됨"
        if token
        else "미설정 — gated 모델 접근과 다운로드 속도에 필요",
    )

    req = urllib.request.Request(GATED_URL)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            ok = resp.status == 200
        add(PASS if ok else WARN, f"gated 접근 ({GATED_REPO})", f"HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            add(
                FAIL,
                f"gated 접근 ({GATED_REPO})",
                f"HTTP {e.code} — 약관 동의 + 토큰 필요 (아래 안내 참조)",
            )
        else:
            add(WARN, f"gated 접근 ({GATED_REPO})", f"HTTP {e.code} — 판정 보류")
    except Exception as e:  # 네트워크 차단·DNS 실패 등
        add(WARN, f"gated 접근 ({GATED_REPO})", f"확인 불가(네트워크): {type(e).__name__}")


def main() -> int:
    print("=" * 72)
    print("TRELLIS.2 실행 환경 진단")
    print("=" * 72)

    check_os()
    check_gpu()
    check_nvcc()
    check_cuda_home()
    check_disk()
    check_python()
    check_torch()
    check_tool("conda", "conda", FAIL, "setup.sh --new-env 가 conda 를 쓴다")
    check_tool("git", "git", FAIL)
    check_tool("gcc", "gcc", FAIL, "CUDA 확장 소스 빌드에 필요")
    check_tool("ninja", "ninja", WARN, "없어도 setup.sh --basic 이 설치한다")
    check_hf_gated()

    print()
    width = max(len(n) for _, n, _ in results)
    for level, name, detail in results:
        mark = {PASS: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}[level]
        print(f"[{mark}] {name:{width}s}  {detail}")

    failed_names = [n for lv, n, _ in results if lv == FAIL]
    failed_details = {n: d for lv, n, d in results if lv == FAIL}
    fails = len(failed_names)
    warns = sum(1 for lv, _, _ in results if lv == WARN)

    # 실패한 항목에 대한 안내만 찍는다. 전부 찍으면 멀쩡한 항목까지 문제로 보인다.
    hints: dict[str, list[str]] = {
        "CUDA toolkit": ["vast.ai 에서 *-devel 또는 PyTorch 템플릿 인스턴스로 다시 띄운다"],
        "VRAM": ["24GB+ (4090/A10G/L40S) 로 바꾼다"],
        "디스크": ["인스턴스 디스크를 100GB 이상으로 늘린다"],
        "conda": ["miniforge/miniconda 를 설치하거나 conda 가 있는 이미지를 쓴다"],
        "gcc": ["apt install -y build-essential"],
        "OS": ["TRELLIS.2 는 Linux 전용 — Linux 인스턴스가 필요하다"],
        "gated 접근": [],  # 아래에서 401/403 을 구분해 채운다
    }

    print("\n" + "=" * 72)
    if fails:
        print(f"FAIL {fails}건 · WARN {warns}건 — 이대로 진행하면 도중에 깨집니다.")
        for name in failed_names:
            key = next((k for k in hints if name.startswith(k)), None)
            if key is None:
                continue
            if key == "gated 접근":
                # 401(미인증) 과 403(인증됐으나 미승인) 은 대처가 다르다.
                detail = failed_details[name]
                print(f"  · {name}")
                if "403" in detail:
                    print("      토큰은 유효하지만 이 계정에 접근 권한이 없습니다.")
                    print(f"      1) 로그인 상태로 https://huggingface.co/{GATED_REPO} 접속")
                    print("         - 'Request access' 가 보이면 → 아직 신청 전. 약관에 동의한다")
                    print("         - 'pending' 이면 → 승인 대기. 기다려야 한다")
                    print("         - 이미 승인됐다면 → 토큰 권한 문제(아래)")
                    print("      2) fine-grained 토큰이면 'Read access to contents of all public")
                    print("         gated repos you can access' 체크. Classic → Read 가 확실하다")
                else:
                    print(f"      1) https://huggingface.co/{GATED_REPO} 에서 약관 동의")
                    print("      2) https://huggingface.co/settings/tokens 에서 read 토큰 발급")
                    print("      3) export HF_TOKEN=hf_...")
            else:
                for line in hints[key]:
                    print(f"  · {name} → {line}")
    elif warns:
        print(f"WARN {warns}건 — 진행 가능. 위 메모를 확인하세요.")
    else:
        print("전부 통과 — bash setup_vast.sh 로 진행하세요.")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
