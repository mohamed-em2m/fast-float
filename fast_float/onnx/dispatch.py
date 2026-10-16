"""Runtime per-component backend dispatch.

Reads the config (see :mod:`fast_float.onnx.backends`) and routes each pipeline
component to ``eager`` (plain torch, the baseline/fallback), ``compile``
(``torch.compile``) or ``onnx`` (an exported graph on an execution provider).

The three components that already had a runtime backend -- ``emotion``,
``audio`` and ``fmt`` -- are wired in their own modules (config-gated there).
This module wires the remaining three, which had none:

  * ``direction``  -- substitutes ``motion_autoencoder.dec.direction``
  * ``refenc``     -- substitutes ``motion_autoencoder.enc`` (exposes ``.fc``)
  * ``decoder``    -- installed as ``model._decoder_onnx`` and consumed by
                      ``FLOAT._decode_latent_batched`` (static-batch pad/trim)

Every ONNX substitution runs an init-time parity self-test against the torch
module and falls back to eager on mismatch, so a missing, stale or
shape-incompatible artifact can never break inference. With the default
all-eager config nothing is substituted -- behaviour is unchanged.

Call once, after the checkpoint is loaded and the module is on its device.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from fast_float.onnx.backends import resolve_mode
from fast_float.onnx.export import make_ort_session
from fast_float.onnx.parity import compare_outputs

logger = logging.getLogger(__name__)

_ONNX_DIR = "./onnx"
_MIN_COSINE = 0.999


def _dynamo_disable(fn):
    try:
        from torch._dynamo import disable

        return disable(fn)
    except Exception:
        return fn


def _device_of(module) -> torch.device:
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")


class _CompiledLeaf(nn.Module):
    """``torch.compile`` around a leaf module, forwarding extra attributes.

    Used where a call site also reaches into the module (e.g. ``enc.fc``).
    """

    def __init__(self, module: nn.Module, extra: dict | None = None):
        super().__init__()
        self._orig = module
        self._fn = torch.compile(module)
        for k, v in (extra or {}).items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _OnnxRunner(nn.Module):
    """Base ONNX session wrapper: torch in (any device/dtype) -> torch out."""

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        super().__init__()
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.path = os.path.abspath(onnx_path)
        self.session = make_ort_session(self.path, use_gpu=use_gpu)

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    def _np(self, t: torch.Tensor):
        return t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()


class DirectionOnnx(_OnnxRunner):
    """``x [B, M] -> dir [B, 512]`` (dynamic batch)."""

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        super().__init__(onnx_path, use_gpu=use_gpu)
        self.input_name = self.session.get_inputs()[0].name

    @_dynamo_disable
    def _run(self, x_np):
        return self.session.run(None, {self.input_name: x_np})[0]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = self._run(self._np(x))
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)


class RefEncOnnx(_OnnxRunner):
    """``img [B,3,H,W] -> (x_r, x_r_lambda, *feats)``; exposes ``.fc``."""

    def __init__(self, onnx_path: str, fc: nn.Module, use_gpu: bool | None = None):
        super().__init__(onnx_path, use_gpu=use_gpu)
        self.fc = fc
        self.input_name = self.session.get_inputs()[0].name

    @_dynamo_disable
    def _run(self, img_np):
        return self.session.run(None, {self.input_name: img_np})

    @torch.no_grad()
    def forward(self, x: torch.Tensor, input_target=None, h_start=None):
        if input_target is not None:
            raise RuntimeError(
                "RefEncOnnx only serves the encode_image_into_latent path"
            )
        device = x.device
        if x.ndim == 3:
            x = x.unsqueeze(0)
        outs = self._run(self._np(x))
        outs = [
            torch.from_numpy(o).to(device=device, dtype=torch.float32) for o in outs
        ]
        return outs[0], outs[1], outs[2:]


class DecoderOnnx(_OnnxRunner):
    """Static-batch decode: ``latent [m,512] + feats [m,...] -> img [m,3,H,W]``.

    The graph bakes ``groups=batch`` into its convs, so it is valid for exactly
    one ``B`` (``self.batch``). Runtime pads micro-batches up to ``B``
    (replicate) and trims the output back to ``m``.
    """

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        super().__init__(onnx_path, use_gpu=use_gpu)
        ins = self.session.get_inputs()
        self.latent_name = ins[0].name
        self.feat_names = [i.name for i in ins[1:]]
        shape0 = ins[0].shape
        b = shape0[0] if shape0 else None
        self.batch = int(b) if isinstance(b, int) else None

    @_dynamo_disable
    def _run(self, feed):
        return self.session.run(None, feed)[0]

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, feats) -> torch.Tensor:
        device = latent.device
        m = latent.shape[0]
        big_b = self.batch or m
        if m > big_b:
            raise ValueError(f"decode batch {m} exceeds static ONNX batch {big_b}")
        fs = list(feats)
        fs = [f.expand(big_b, *f.shape[1:]) if f.shape[0] == 1 else f for f in fs]
        if m < big_b:
            pad = big_b - m
            latent = torch.cat(
                [latent, latent[-1:].expand(pad, *latent.shape[1:])], dim=0
            )
            fs = [
                torch.cat([f, f[-1:].expand(pad, *f.shape[1:])], dim=0)
                if f.shape[0] == m
                else f
                for f in fs
            ]
        feed = {self.latent_name: self._np(latent)}
        for name, f in zip(self.feat_names, fs):
            feed[name] = self._np(f)
        out = self._run(feed)
        return torch.from_numpy(out[:m]).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Self-tests (cosine-gated: robust to fp16 EP kernels and random weights)
# --------------------------------------------------------------------------- #

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
    ok = _cosine_ok(x_r, got[0], "refenc.x_r") and _cosine_ok(
        lam, got[1], "refenc.lambda"
    )
    for i, (a, b) in enumerate(zip(feats, got[2])):
        ok = _cosine_ok(a, b, f"refenc.feat{i}") and ok
    return ok


def _selftest_decoder(ae, runner, image_size: int) -> bool:
    dec = ae.dec
    device = _device_of(dec)
    big_b = runner.batch or 4
    g = torch.Generator(device="cpu").manual_seed(2)
    with torch.no_grad():
        probe = torch.randn(1, 3, image_size, image_size, generator=g).to(device)
        _, _, feats1 = ae.enc(probe, None)
    lat = torch.randn(big_b, 512, generator=g).to(device)
    feats = [f.expand(big_b, *f.shape[1:]).contiguous() for f in feats1]
    with torch.no_grad():
        ref = dec(lat, alpha=None, feats=feats)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    return _cosine_ok(ref, runner(lat, feats), "decoder")


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


def _apply_direction(model, ae, opt) -> None:
    mode = resolve_mode("direction")
    if mode == "eager":
        return
    target = getattr(getattr(ae, "dec", None), "direction", None)
    if target is None:
        return
    if mode == "compile":
        try:
            ae.dec.direction = _CompiledLeaf(target)
            logger.info("backends: direction -> torch.compile")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"backends: direction compile failed ({e}); eager")
        return
    path = _path(opt, "direction_onnx_path", f"{_ONNX_DIR}/direction_v1.onnx")
    try:
        runner = DirectionOnnx(path)
        if not _selftest_direction(target, runner):
            raise RuntimeError("parity self-test failed")
        ae.dec.direction = runner
        logger.info(f"backends: direction -> ONNX {path} ({runner.providers})")
    except Exception as e:  # noqa: BLE001
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
            ae.enc = _CompiledLeaf(enc, extra={"fc": enc.fc})
            logger.info("backends: refenc -> torch.compile")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"backends: refenc compile failed ({e}); eager")
        return
    path = _path(opt, "refenc_onnx_path", f"{_ONNX_DIR}/refenc_v1.onnx")
    try:
        runner = RefEncOnnx(path, enc.fc)
        if not _selftest_refenc(enc, runner, _image_size(opt)):
            raise RuntimeError("parity self-test failed")
        ae.enc = runner
        logger.info(f"backends: refenc -> ONNX {path} ({runner.providers})")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"backends: refenc ONNX skipped ({e}); eager")


def _apply_decoder(model, ae, opt) -> None:
    mode = resolve_mode("decoder")
    if mode == "eager":
        return
    if mode == "compile":
        try:
            model._decoder_compiled = torch.compile(ae.dec)
            logger.info("backends: decoder -> torch.compile")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"backends: decoder compile failed ({e}); eager")
        return
    path = _path(opt, "decoder_onnx_path", f"{_ONNX_DIR}/decoder_v1.onnx")
    try:
        runner = DecoderOnnx(path)
        if not _selftest_decoder(ae, runner, _image_size(opt)):
            raise RuntimeError("parity self-test failed")
        model._decoder_onnx = runner
        logger.info(
            f"backends: decoder -> ONNX {path} "
            f"(static batch {runner.batch}, {runner.providers})"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"backends: decoder ONNX skipped ({e}); eager")


def apply_backends(model, opt=None):
    """Substitute configured backends on a loaded FLOAT model.

    Call once after weights are loaded and the model is on its device. The
    default all-eager config is a no-op. Never raises.
    """
    if os.environ.get("FLOAT_DISABLE_BACKENDS"):
        return model
    ae = getattr(model, "motion_autoencoder", None)
    if ae is None:
        return model
    for fn in (_apply_direction, _apply_refenc, _apply_decoder):
        try:
            fn(model, ae, opt)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"backends: {fn.__name__} failed ({e}); eager")
    return model


__all__ = [
    "apply_backends",
    "DirectionOnnx",
    "RefEncOnnx",
    "DecoderOnnx",
]
