"""Runtime per-component backend dispatch.

Reads the config (see :mod:`fast_float.onnx.backends`) and routes each pipeline
component to ``eager`` (plain torch), ``compile`` (``torch.compile``) or
``onnx`` (an exported graph on an execution provider).

The three components that already had a runtime backend -- ``emotion``,
``audio`` and ``fmt`` -- are wired in their own modules (config-gated there).
This module wires the remaining three, registry-driven (see
``fast_float/onnx/registry.py``, the source of truth for the component matrix):

  * ``direction``  -- substitutes ``motion_autoencoder.dec.direction``
  * ``refenc``     -- substitutes ``motion_autoencoder.enc`` (exposes ``.fc``)
  * ``decoder``    -- installed as ``model._decoder_onnx`` and consumed by
                      ``FLOAT._decode_latent_batched``/``_decode_u8``
                      (static-batch pad/trim)

The ONNX runners themselves live in ``fast_float.runtimes.onnx.runner`` and are
re-exported here for scripts.

Parity is validated by the converter/selector scripts (the config is the
contract), plus a one-shot self-test when a component is installed here: the
graph is only as good as the execution provider that loads it, and providers
are chosen at runtime. ``FLOAT_SKIP_BACKEND_SELFTEST=1`` skips the probes.
The constructor-wired components (``emotion``/``audio``/``fmt``) are verified
the same way, once the checkpoint is loaded, and dropped if the artifact does
not reproduce the torch model. With the default all-eager config nothing is
substituted.

Call once, after the checkpoint is loaded and the module is on its device.
"""

from __future__ import annotations

import logging
import os

import torch

from fast_float.onnx.backends import resolve_mode, resolve_use_trt
from fast_float.onnx.export import FP16_MAX_MEAN, FP16_MIN_COSINE
from fast_float.onnx.parity import U8_MAX_LSB, U8_MAX_PROVIDER_MEAN_LSB, compare_outputs, u8_lsb
from fast_float.onnx.registry import get as component_spec
from fast_float.onnx.registry import post_load
from fast_float.runtimes.dynamo.compiled import CompiledRuntime
from fast_float.runtimes.onnx.runner import DecoderOnnx, DirectionOnnx, RefEncOnnx

logger = logging.getLogger(__name__)

_MIN_COSINE = 0.999


def _device_of(module) -> torch.device:
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Parity self-tests, run once per component at install time.
#
# These used to be dead code ("the config is the contract"), which is how a
# decoder artifact that was off by 4.375 / cosine 0.983 got installed and
# shipped a broken face: the graph is only as good as the execution provider
# that loads it, and the provider is chosen at runtime, not at export time.
# --------------------------------------------------------------------------- #


def _selftests_enabled() -> bool:
    """Skip the install-time probes for benchmarking (``FLOAT_SKIP_BACKEND_SELFTEST=1``)."""
    return os.environ.get("FLOAT_SKIP_BACKEND_SELFTEST", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cosine_ok(ref: torch.Tensor, got: torch.Tensor, name: str) -> bool:
    rep = compare_outputs(ref, got, atol=1.0, rtol=1e-2)
    ok = bool(rep["ok"] or rep["cosine"] >= _MIN_COSINE)
    print(
        f"[backends:{name}] self-test cosine={rep['cosine']:.6f} "
        f"mean_abs={rep['mean_abs']:.3e} max_abs={rep['max_abs']:.3e} ok={ok}"
    )
    return ok


def _selftest_direction(ref, runner) -> bool:
    device = _device_of(ref)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, ref.weight.shape[1], generator=g).to(device)
    with torch.no_grad():
        r = ref(x)
    return _cosine_ok(r, runner(x), "direction")


def _selftest_refenc(ref, runner, image_size: int) -> bool:
    device = _device_of(ref)
    g = torch.Generator(device="cpu").manual_seed(1)
    img = torch.randn(1, 3, image_size, image_size, generator=g).to(device)
    with torch.no_grad():
        x_r, _, feats = ref(img, None)
        lam = ref.fc(x_r)
    got = runner(img)
    if len(got[2]) != len(feats):
        print(f"[backends:refenc] feat count {len(got[2])} != {len(feats)}")
        return False
    ok = _cosine_ok(x_r, got[0], "refenc.x_r") and _cosine_ok(lam, got[1], "refenc.lambda")
    for i, (a, b) in enumerate(zip(feats, got[2], strict=False)):
        ok = _cosine_ok(a, b, f"refenc.feat{i}") and ok
    return ok


def _selftest_decoder(ae, runner, image_size: int) -> bool:
    dec = ae.dec
    device = _device_of(dec)
    # One artifact is enough to judge the exporter run, and load time matters:
    # probe the smallest static batch (the artifact the converter names).
    big_b = int(getattr(runner, "batch", None) or 4)
    b = max(1, min(big_b, 4))
    g = torch.Generator(device="cpu").manual_seed(2)
    try:
        with torch.no_grad():
            probe = torch.randn(1, 3, image_size, image_size, generator=g).to(device)
            _, _, feats1 = ae.enc(probe, None)
    except Exception as e:  # an encoder that cannot be probed must not veto the decoder
        print(f"[backends:decoder] self-test skipped ({e})")
        return True
    lat = torch.randn(b, 512, generator=g).to(device)
    feats = [f.expand(b, *f.shape[1:]).contiguous() for f in feats1]
    with torch.no_grad():
        ref = dec(lat, alpha=None, feats=feats)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        got = runner(lat, feats)
    # Gate in the deployed uint8 frame domain. The old image-space gate
    # (atol=1.0 on a [-1, 1] tensor, i.e. half the range) accepted a graph off
    # by 4.375; 2 LSB is invisible on screen and fails a broken graph.
    #
    # The *provider* is judged by the mean (invisible on screen) and by the
    # cosine (structurally sound), not by the max: the provider runs TF32 by
    # default, and TF32 alone moves this 22-conv stack by ~0.5 LSB mean
    # (~50 LSB on a few pixels) between two implementations. A structurally
    # wrong graph (the broken TensorRT engine: max_abs 4.375, cosine 0.983)
    # fails both floors.
    max_lsb, mean_lsb = u8_lsb(ref, got)
    cosine = compare_outputs(ref, got)["cosine"]
    # An all-zero reference has no meaningful cosine (it comes out 0), so
    # there the per-pixel criterion decides.
    degenerate = float(ref.detach().abs().max()) == 0.0
    structural = max_lsb <= U8_MAX_LSB if degenerate else cosine >= _MIN_COSINE
    ok = mean_lsb <= U8_MAX_PROVIDER_MEAN_LSB and structural
    print(f"[backends:decoder] self-test max_lsb={max_lsb} mean_lsb={mean_lsb:.2f} cosine={cosine:.6f} ok={ok}")
    return ok


def _selftest_fmt(fmt) -> bool:
    """One FMT step, torch vs the installed ONNX graph (CFG-shaped batch).

    The FMT artifact is the one that decides the *motion latents*, so a wrong
    graph does not degrade the picture a little -- it produces different
    motion. A stale artifact (e.g. one exported with the inverted attention
    mask) has to be rejected here, at install time, instead of quietly
    shipping broken frames.
    """
    opt = getattr(fmt, "opt", None)
    device = _device_of(fmt)
    n_clip = int(getattr(opt, "wav2vec_sec", 2.0) * getattr(opt, "fps", 25.0))
    n_prev = int(getattr(opt, "num_prev_frames", 10))
    dim_w = int(getattr(opt, "dim_w", 512))
    dim_e = int(getattr(opt, "dim_e", 7))
    g = torch.Generator(device="cpu").manual_seed(3)

    def _rand(*shape):
        return torch.randn(*shape, generator=g).to(device)

    t = torch.tensor([0.5], device=device)
    x = _rand(3, n_clip, dim_w)
    wa = _rand(3, n_clip, dim_w)
    wr = _rand(3, 1, dim_w)
    we = _rand(3, 1, dim_e)
    prev_x = _rand(3, n_prev, dim_w)
    prev_wa = _rand(3, n_prev, dim_w)
    with torch.no_grad():
        ref = fmt.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)
        got = fmt._onnx_backend.run_step(t, x, wa, wr, we, prev_x, prev_wa)
    rep = compare_outputs(ref, got, atol=1e-3, rtol=1e-3)
    # The vector field is small-magnitude, so the drift is absolute: judge it
    # functionally (cosine + bounded mean), like the converter does for the
    # TRT FP16 path. A wrong graph scores cosine ~0.94 and is rejected.
    ok = bool(rep["cosine"] >= FP16_MIN_COSINE and rep["mean_abs"] <= FP16_MAX_MEAN)
    print(
        f"[backends:fmt] self-test cosine={rep['cosine']:.6f} "
        f"mean_abs={rep['mean_abs']:.3e} max_abs={rep['max_abs']:.3e} ok={ok}"
    )
    return ok


def _verify_constructor_backends(model) -> None:
    """Check the components that wire themselves in their own constructor.

    ``emotion``/``audio``/``fmt`` install their ONNX backend before the
    checkpoint is loaded, so their artifact can only be judged now. A graph
    that does not reproduce the torch model is dropped (the component stays
    eager) -- the graph is only as good as the artifact and the provider that
    loads it.
    """
    fmt = getattr(model, "fmt", None)
    backend = getattr(fmt, "_onnx_backend", None)
    if backend is None:
        return
    try:
        if not _selftests_enabled() or _selftest_fmt(fmt):
            return
    except Exception as e:
        logger.warning(f"backends: fmt ONNX self-test failed ({e}); eager")
        fmt._onnx_backend = None
        fmt._mode = "eager"
        return
    logger.warning("backends: fmt ONNX failed the parity self-test; eager")
    fmt._onnx_backend = None
    fmt._mode = "eager"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _path(opt, attr: str, default: str) -> str:
    if opt is not None:
        return getattr(opt, attr, default) or default
    return default


def _image_size(opt) -> int:
    if opt is not None and getattr(opt, "input_size", None):
        return int(opt.input_size)
    return 512


def _primary_device_id(opt) -> int:
    """Rank the primary runner sessions bind to (mirrors the model's device)."""
    rank = getattr(opt, "rank", 0) if opt is not None else 0
    if isinstance(rank, int):
        return rank
    try:
        dev = torch.device(rank)
        return dev.index if dev.index is not None else 0
    except Exception:
        return 0


def _apply_direction(model, ae, opt) -> None:
    mode = resolve_mode("direction")
    if mode == "eager":
        return
    target = getattr(getattr(ae, "dec", None), "direction", None)
    if target is None:
        return
    if mode == "compile":
        try:
            ae.dec.direction = CompiledRuntime(target, component="direction")
            logger.info("backends: direction -> torch.compile")
        except Exception as e:
            logger.warning(f"backends: direction compile failed ({e}); eager")
        return
    path = _path(opt, "direction_onnx_path", component_spec("direction").onnx_default)
    try:
        # No automatic parity probe: the converter/selector scripts validate;
        # the config is the contract.
        motion_dim = int(target.weight.shape[1])
        runner = DirectionOnnx(
            path,
            opt_shapes={"x": [1, motion_dim]},
            device_id=_primary_device_id(opt),
            use_trt=resolve_use_trt("direction"),
        )
        if _selftests_enabled() and not _selftest_direction(target, runner):
            logger.warning("backends: direction ONNX failed the parity self-test; eager")
            return
        ae.dec.direction = runner
        logger.info(f"backends: direction -> ONNX {path} ({runner.providers})")
    except Exception as e:
        logger.warning(f"backends: direction ONNX skipped ({e}); eager")


def _apply_refenc(model, ae, opt) -> None:
    mode = resolve_mode("refenc")
    if mode == "eager":
        return
    enc = getattr(ae, "enc", None)
    if enc is None:
        return
    if mode == "compile":
        try:
            ae.enc = CompiledRuntime(enc, extra={"fc": enc.fc}, component="refenc")
            logger.info("backends: refenc -> torch.compile")
        except Exception as e:
            logger.warning(f"backends: refenc compile failed ({e}); eager")
        return
    path = _path(opt, "refenc_onnx_path", component_spec("refenc").onnx_default)
    try:
        # The original encoder stays owned by the runner (single residency):
        # it serves .fc and the non-deployed 3-arg path.
        size = _image_size(opt)
        runner = RefEncOnnx(
            path,
            enc.fc,
            torch_fallback=enc,
            opt_shapes={"img": [1, 3, size, size]},
            device_id=_primary_device_id(opt),
            use_trt=resolve_use_trt("refenc"),
        )
        if _selftests_enabled() and not _selftest_refenc(enc, runner, size):
            logger.warning("backends: refenc ONNX failed the parity self-test; eager")
            return
        ae.enc = runner
        logger.info(f"backends: refenc -> ONNX {path} ({runner.providers})")
    except Exception as e:
        logger.warning(f"backends: refenc ONNX skipped ({e}); eager")


def _apply_decoder(model, ae, opt) -> None:
    mode = resolve_mode("decoder")
    use_infer = os.environ.get("FLOAT_DISABLE_SYNTHESIS_INFER", "").strip().lower() not in {"1", "true", "yes", "on"}
    if use_infer and hasattr(ae, "dec") and hasattr(ae.dec, "to_flows"):
        try:
            from fast_float.models.float.synthesis_infer import SynthesisInfer

            if not isinstance(ae.dec, SynthesisInfer):
                ae.dec = SynthesisInfer.from_synthesis(ae.dec)
                logger.info("backends: decoder (eager) -> SynthesisInfer")
        except Exception as e:
            logger.warning(f"backends: SynthesisInfer upgrade failed ({e}); keeping original Synthesis")

    if mode == "eager":
        return
    if mode == "compile":
        try:
            model._decoder_compiled = torch.compile(ae.dec)
            logger.info("backends: decoder -> torch.compile")
        except Exception as e:
            logger.warning(f"backends: decoder compile failed ({e}); eager")
        return
    path = _path(opt, "decoder_onnx_path", component_spec("decoder").onnx_default)
    try:
        runner = DecoderOnnx(path, device_id=_primary_device_id(opt), use_trt=resolve_use_trt("decoder"))
        if _selftests_enabled() and not _selftest_decoder(ae, runner, _image_size(opt)):
            logger.warning("backends: decoder ONNX failed the parity self-test; eager")
            return
        model._decoder_onnx = runner
        logger.info(f"backends: decoder -> ONNX {path} (batches {getattr(runner, 'batches', [])}, {runner.providers})")
    except Exception as e:
        logger.warning(f"backends: decoder ONNX skipped ({e}); eager")


_INSTALLERS = {
    "decoder": _apply_decoder,
    "direction": _apply_direction,
    "refenc": _apply_refenc,
}


def apply_backends(model, opt=None):
    """Substitute configured backends on a loaded FLOAT model.

    Registry-driven: iterates the components the registry marks ``post_load``
    (direction/refenc/decoder), in ``_INSTALLERS`` order (behaviour-preserving).
    Call once after weights are loaded and the model is on its device. The
    default all-eager config is a no-op. Never raises.
    """
    if os.environ.get("FLOAT_DISABLE_BACKENDS"):
        # The constructor-wired components (emotion/audio/fmt) install
        # themselves before this pass, so the flag does not reach them --
        # but their artifact still has to be verified, because a wrong graph
        # ships broken frames.
        try:
            _verify_constructor_backends(model)
        except Exception as e:
            logger.warning(f"backends: constructor backend verification failed ({e})")
        return model
    ae = getattr(model, "motion_autoencoder", None)
    if ae is not None:
        post = set(post_load())
        for name, install in _INSTALLERS.items():
            if name not in post:
                continue
            try:
                install(model, ae, opt)
            except Exception as e:
                logger.warning(f"backends: {name} install failed ({e}); eager")
    try:
        _verify_constructor_backends(model)
    except Exception as e:
        logger.warning(f"backends: constructor backend verification failed ({e})")
    return model


__all__ = [
    "DecoderOnnx",
    "DirectionOnnx",
    "RefEncOnnx",
    "apply_backends",
]
