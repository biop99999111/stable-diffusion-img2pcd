"""backends.py 의 TRELLIS.2 호환 패치를 GPU·모델 없이 검증한다.

    python test_compat.py

여기서 잡으려는 사고는 하나다: 빌린 GPU 에서 15GB 를 받고 파이프라인을 올린
뒤에야 `'DINOv3ViTModel' object has no attribute 'layer'` 로 죽는 것.
transformers 4.x / 5.x 의 DINOv3 레이아웃을 가짜 모듈로 재현해서,

  1) 패치가 두 레이아웃 모두에서 돌고
  2) 두 레이아웃의 결과가 **완전히 같고**
  3) 4.x 에서는 원본 구현과 결과가 같다 (조건 특징이 바뀌면 실험이 오염된다)

를 확인한다. 3) 이 핵심이다 — 돌아가기만 하는 패치는 안 된다.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 가짜 DINOv3

HIDDEN, PATCHES = 8, 5


class FakeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, HIDDEN, kernel_size=1)

    def forward(self, pixel_values, bool_masked_pos=None):
        b = pixel_values.shape[0]
        pooled = pixel_values.mean(dim=(2, 3))  # (B, 3)
        base = self.patch_embeddings(pooled[:, :, None, None]).reshape(b, 1, HIDDEN)
        return base.repeat(1, PATCHES, 1) + torch.arange(PATCHES).reshape(1, PATCHES, 1)


class FakeRope(nn.Module):
    def forward(self, pixel_values):
        return (torch.ones(PATCHES, HIDDEN), torch.zeros(PATCHES, HIDDEN))


class FakeLayer(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, HIDDEN)
        self.k = k

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        assert position_embeddings is not None, "position_embeddings 가 전달되지 않았다"
        cos, _ = position_embeddings
        return self.lin(hidden_states) + self.k * cos


def _make_layers(seed: int = 0) -> nn.ModuleList:
    torch.manual_seed(seed)
    return nn.ModuleList([FakeLayer(k) for k in range(3)])


class V4Model(nn.Module):
    """transformers 4.56~4.57: 블록이 최상위 .layer 에 있다."""

    def __init__(self, layers, embeddings, rope):
        super().__init__()
        self.embeddings = embeddings
        self.rope_embeddings = rope
        self.layer = layers
        self.norm = nn.LayerNorm(HIDDEN)


class V5Model(nn.Module):
    """transformers 5.x: 블록이 encoder(=self.model).layer 로 내려갔다."""

    def __init__(self, layers, embeddings, rope):
        super().__init__()
        self.embeddings = embeddings
        self.rope_embeddings = rope
        self.model = types.SimpleNamespace(layer=layers)
        self.norm = nn.LayerNorm(HIDDEN)


class NoLayerModel(nn.Module):
    """레이아웃이 또 바뀐 미래 버전 — 조용히 틀리지 말고 죽어야 한다."""

    def __init__(self, embeddings, rope):
        super().__init__()
        self.embeddings = embeddings
        self.rope_embeddings = rope


# ---------------------------------------------------------------- 가짜 trellis2 패키지


def original_extract_features(self, image):
    """TRELLIS.2 원본 구현 그대로(비교 기준)."""
    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = self.model.embeddings(image, bool_masked_pos=None)
    position_embeddings = self.model.rope_embeddings(image)
    for _, layer_module in enumerate(self.model.layer):
        hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


class FakeExtractor:
    def __init__(self, model):
        self.model = model
        self.image_size = 512

    extract_features = original_extract_features


class FakeBiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name


def install_fake_trellis2() -> None:
    """sys.modules 에 trellis2 소스 트리를 흉내낸 모듈을 심는다."""
    pkg = types.ModuleType("trellis2")
    modules = types.ModuleType("trellis2.modules")
    ife = types.ModuleType("trellis2.modules.image_feature_extractor")
    ife.DinoV3FeatureExtractor = FakeExtractor
    modules.image_feature_extractor = ife
    pipelines = types.ModuleType("trellis2.pipelines")
    rembg = types.ModuleType("trellis2.pipelines.rembg")
    rembg.BiRefNet = FakeBiRefNet
    pipelines.rembg = rembg
    pkg.modules, pkg.pipelines = modules, pipelines
    sys.modules.update(
        {
            "trellis2": pkg,
            "trellis2.modules": modules,
            "trellis2.modules.image_feature_extractor": ife,
            "trellis2.pipelines": pipelines,
            "trellis2.pipelines.rembg": rembg,
        }
    )


# ---------------------------------------------------------------- 검증


def main() -> int:
    install_fake_trellis2()
    import backends

    checks: list[tuple[str, bool, str]] = []
    image = torch.linspace(0, 1, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)

    embeddings, rope, layers = FakeEmbeddings(), FakeRope(), _make_layers()
    v4, v5 = V4Model(layers, embeddings, rope), V5Model(layers, embeddings, rope)

    # 패치 전 원본 결과(4.x 에서만 돈다) — 이게 정답지다.
    baseline = FakeExtractor(v4).extract_features(image)
    try:
        FakeExtractor(v5).extract_features(image)
        crashed = False
    except AttributeError:
        crashed = True
    checks.append(("패치 전 5.x 레이아웃은 AttributeError", crashed, "재현됨" if crashed else "재현 실패"))

    backends._patch_dinov3_layout()
    ife = sys.modules["trellis2.modules.image_feature_extractor"]

    out4 = ife.DinoV3FeatureExtractor(v4).extract_features(image)
    out5 = ife.DinoV3FeatureExtractor(v5).extract_features(image)
    checks.append(("4.x 결과 == 원본 구현", torch.allclose(out4, baseline), f"max diff {(out4 - baseline).abs().max():.2e}"))
    checks.append(("5.x 결과 == 4.x 결과", torch.allclose(out4, out5), f"max diff {(out4 - out5).abs().max():.2e}"))

    # 학습된 affine 이 붙는 model.norm 을 타면 값이 달라진다 = 우회 구현이 아님을 확인.
    with torch.no_grad():
        v4.norm.weight.fill_(2.0)
    checks.append(
        ("model.norm 을 타지 않음", torch.allclose(ife.DinoV3FeatureExtractor(v4).extract_features(image), baseline), "affine 무시")
    )

    try:
        ife.DinoV3FeatureExtractor(NoLayerModel(embeddings, rope)).extract_features(image)
        died = False
    except SystemExit:
        died = True
    checks.append(("모르는 레이아웃이면 SystemExit", died, "조용히 틀리지 않음"))

    # 두 번 패치해도 중첩되지 않아야 한다(부품마다 load 를 다시 부를 수 있다).
    backends._patch_dinov3_layout()
    checks.append(
        ("중복 패치 무해", torch.allclose(ife.DinoV3FeatureExtractor(v4).extract_features(image), baseline), "idempotent")
    )

    # rembg 교체: gated 인 RMBG-2.0 대신 MIT 모델로 강제된다.
    backends._patch_rembg("ZhengPeng7/BiRefNet")
    rembg = sys.modules["trellis2.pipelines.rembg"]
    forced = rembg.BiRefNet("briaai/RMBG-2.0").model_name
    checks.append(("rembg 강제 교체", forced == "ZhengPeng7/BiRefNet", forced))
    backends._patch_rembg("ZhengPeng7/BiRefNet")  # 중복 호출
    checks.append(("rembg 중복 패치 무해", rembg.BiRefNet("x").model_name == "ZhengPeng7/BiRefNet", "idempotent"))

    # OOM 판별: 문자열이든 예외 타입이든 잡아야 자동 재시도가 작동한다.
    checks.append(("OOM 문자열 인식", backends._is_oom(RuntimeError("CUDA out of memory. Tried...")), "ok"))
    checks.append(("일반 예외는 OOM 아님", not backends._is_oom(ValueError("bad shape")), "ok"))

    print()
    print("=" * 60)
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:32s} {detail}")
        failed += not ok
    print("=" * 60)
    print(f"{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
