"""Component registry: FLOAT's six inference components as data.

Single source of truth for the per-component matrix in
``docs/BUILD_RUNTIME_SESSIONS.md``. Phase 0 of the builder/runtime migration
records the matrix here; later phases drive ``apply_backends`` and the builders
from it instead of the current split between in-constructor wiring
(emotion/audio/fmt) and post-load wiring (direction/refenc/decoder).

Nothing here imports torch/onnx beyond ``fast_float.onnx.backends`` constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from fast_float.onnx.backends import COMPONENTS, HOT_COMPONENTS

BATCH_KINDS = ("single", "dynamic", "fixed", "micro")
BACKEND_STAGES = ("constructor", "post_load")
DEGRADE_MODES = ("sticky", "transient", "raise")


@dataclass(frozen=True)
class ComponentSpec:
    """Static description of one pipeline component (the matrix row)."""

    name: str
    backend_stage: str  # constructor | post_load
    torch_target: str  # dotted path of the module/function that runs
    install_attr: str  # where a runtime is installed on the model
    io_signature: str
    batch: str  # single | dynamic | fixed | micro
    replicable: bool
    tolerance: float = 2e-4
    min_cosine: float | None = None
    max_mean: float | None = None
    onnx_default: str | None = None
    degrade_on_error: str = "transient"
    notes: str = ""

    @property
    def hot(self) -> bool:
        return self.name in HOT_COMPONENTS


REGISTRY: dict[str, ComponentSpec] = {
    "emotion": ComponentSpec(
        name="emotion",
        backend_stage="constructor",
        torch_target="fast_float.models.float.condition_encoders.Audio2Emotion._predict_body",
        install_attr="_onnx_backend",
        io_signature="waveform[1,N:16000..160000] -> probs[1,7]",
        batch="single",
        replicable=False,
        onnx_default="./onnx/emotion_v1.onnx",
        notes="softmax baked into the graph; TRT min length 16000 (runtime pads)",
    ),
    "audio": ComponentSpec(
        name="audio",
        backend_stage="constructor",
        torch_target="fast_float.models.float.condition_encoders.AudioEncoder._inference_body",
        install_attr="_onnx_backend",
        io_signature="conv_feats[1,F:10..2000,512] -> audio_feats[1,F,dim_w]",
        batch="single",
        replicable=False,
        onnx_default="./onnx/wav2vec_v1.onnx",
        notes="conv front-end + interpolation stay in torch (Wav2VecModel.feature_extract)",
    ),
    "direction": ComponentSpec(
        name="direction",
        backend_stage="post_load",
        torch_target="fast_float.models.float.styledecoder.Direction.forward",
        install_attr="motion_autoencoder.dec.direction",
        io_signature="x[B,20] -> dir[B,512]",
        batch="dynamic",
        replicable=False,
        onnx_default="./onnx/direction_v1.onnx",
    ),
    "fmt": ComponentSpec(
        name="fmt",
        backend_stage="constructor",
        torch_target="fast_float.models.float.FMT.FlowMatchingTransformer",
        install_attr="_onnx_backend",
        io_signature=(
            "step(t, x[B,clip,512], wa[B,clip,512], wr[B,1,512], we[B,1,7], "
            "prev_x[B,prev,512], prev_wa[B,prev,512]) -> vec[B,clip+prev,512]"
        ),
        batch="fixed",
        replicable=False,
        degrade_on_error="sticky",
        onnx_default="./onnx/fmt_step_v1.onnx",
        notes="only the single vector-field step is exported; CFG b=3 is the hot shape; ODE loop stays in Python",
    ),
    "decoder": ComponentSpec(
        name="decoder",
        backend_stage="post_load",
        torch_target="fast_float.models.float.styledecoder.Synthesis.forward",
        install_attr="_decoder_onnx",
        io_signature="latent[m,512] + feats -> img[m,3,H,W]",
        batch="micro",
        replicable=True,
        tolerance=1.0,
        min_cosine=0.995,
        max_mean=0.05,
        onnx_default="./onnx/decoder_v1.onnx",
        notes="static batch baked into grouped convs; runtime pads/trims to it; only per-GPU-replicated component",
    ),
    "refenc": ComponentSpec(
        name="refenc",
        backend_stage="post_load",
        torch_target="fast_float.models.float.encoder.Encoder.forward",
        install_attr="motion_autoencoder.enc",
        io_signature="img[1,3,H,W] -> (x_r, lambda, *feats)",
        batch="single",
        replicable=False,
        onnx_default="./onnx/refenc_v1.onnx",
        notes="exposes .fc; the wrapper keeps the torch encoder as fallback for the non-deployed 3-arg path",
    ),
}


def get(name: str) -> ComponentSpec:
    return REGISTRY[name]


def names() -> tuple[str, ...]:
    return tuple(REGISTRY)


def hot() -> tuple[str, ...]:
    return tuple(n for n, c in REGISTRY.items() if c.hot)


def replicable() -> tuple[str, ...]:
    return tuple(n for n, c in REGISTRY.items() if c.replicable)


def post_load() -> tuple[str, ...]:
    """Components installed after the checkpoint is loaded (by ``apply_backends``)."""
    return tuple(n for n, c in REGISTRY.items() if c.backend_stage == "post_load")


def constructor() -> tuple[str, ...]:
    """Components that self-wire inside their own constructors."""
    return tuple(n for n, c in REGISTRY.items() if c.backend_stage == "constructor")


def validate() -> list[str]:
    """Consistency problems (empty when the registry agrees with the config).

    Intended for tests: catches drift between this matrix and
    ``fast_float.onnx.backends.COMPONENTS`` or an invalid enum.
    """
    problems: list[str] = []
    if set(REGISTRY) != set(COMPONENTS):
        problems.append(f"registry {sorted(REGISTRY)} != COMPONENTS {sorted(COMPONENTS)}")
    for name, comp in REGISTRY.items():
        if comp.name != name:
            problems.append(f"{name}: name field {comp.name!r} != key")
        if comp.batch not in BATCH_KINDS:
            problems.append(f"{name}: unknown batch kind {comp.batch!r}")
        if comp.backend_stage not in BACKEND_STAGES:
            problems.append(f"{name}: unknown backend_stage {comp.backend_stage!r}")
        if comp.degrade_on_error not in DEGRADE_MODES:
            problems.append(f"{name}: unknown degrade_on_error {comp.degrade_on_error!r}")
        if comp.replicable and comp.backend_stage != "post_load":
            problems.append(f"{name}: replicable components must be installed post_load")
    return problems


__all__ = ["REGISTRY", "ComponentSpec", "constructor", "get", "hot", "names", "post_load", "replicable", "validate"]


if __name__ == "__main__":  # pragma: no cover - manual check
    print(f"components: {names()}")
    print(f"hot: {hot()}")
    print(f"post_load: {post_load()}")
    print(f"replicable: {replicable()}")
    print(f"problems: {validate() or 'none'}")
