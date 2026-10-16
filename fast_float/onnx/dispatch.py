"""Runtime per-component backend dispatch.

Reads the config (see :mod:`fast_float.onnx.backends`) and routes each pipeline
component to ``eager`` (plain torch), ``compile`` (``torch.compile``) or
``onnx`` (an exported graph on an execution provider).

The three components that already had a runtime backend -- ``emotion``,
``audio`` and ``fmt`` -- are wired in their own modules (config-gated there).
This module wires the remaining three, which had none:

  * ``direction``  -- substitutes ``motion_autoencoder.dec.direction``
  * ``refenc``     -- substitutes ``motion_autoencoder.enc`` (exposes ``.fc``)
  * ``decoder``    -- installed as ``model._decoder_onnx`` and consumed by
                      ``FLOAT._decode_latent_batched``/``_decode_u8``
                      (static-batch pad/trim)

Parity is validated by the converter/selector scripts, never automatically
here -- the config is the contract (helpers below stay importable for
scripts). With the default all-eager config nothing is substituted.

ORT runners prefer CUDA IOBinding (device-resident, zero host copies) and
fall back to the numpy host path on any failure. Runners implement
``__deepcopy__`` so multi-GPU ``_get_ae`` replicas rebuild clean sessions
instead of copying live ones.

Call once, after the checkpoint is loaded and the module is on its device.
"""

from __future__ import annotations

import copy
import logging
import os

import torch
import torch.nn as nn

from fast_float.onnx.backends import resolve_mode
from fast_float.onnx.export import make_runtime_session
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
        self._extra_keys = list(extra or {})
        for k, v in (extra or {}).items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __deepcopy__(self, memo):
        # Recompile around a fresh copy; the compiled callable itself is not
        # copied. Weights are frozen at inference, so duplicated extras (e.g.
        # enc.fc alongside _orig.fc) cannot diverge.
        new_mod = copy.deepcopy(self._orig, memo)
        extra = {k: copy.deepcopy(getattr(self, k), memo) for k in self._extra_keys}
        return _CompiledLeaf(new_mod, extra=extra)


class _OnnxRunner(nn.Module):
    """Base ONNX session wrapper: torch in (any device/dtype) -> torch out.

    Prefers CUDA IOBinding (device-resident, zero host copies); falls back to
    the numpy host path on any failure (flag cached after the first failure so
    the hot path never pays repeated exception costs).
    """

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__()
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else bool(use_gpu)
        self.device_id = device_id
        self.path = os.path.abspath(onnx_path)
        self._opt_shapes = opt_shapes
        self._extra_dynamic = extra_dynamic
        self._max_batch = max_batch
        self.session, _ = make_runtime_session(
            self.path,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            use_gpu=self.use_gpu,
            device_id=self.device_id,
        )
        self._io_failed = False

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    def _np(self, t: torch.Tensor):
        return t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()

    def _gpu_active(self) -> bool:
        try:
            provs = self.session.get_providers()
        except Exception:
            return False
        return bool(provs) and provs[0] != "CPUExecutionProvider"

    def _io_shapes(self, *args, **kwargs):
        """Concrete [(output_name, shape)] for IOBinding, or None to skip it."""
        raise NotImplementedError

    def _io_run(self, feed, out_shapes):
        from fast_float.onnx.export import run_ort_iobinding

        dev = next(iter(feed.values())).device
        return run_ort_iobinding(self.session, feed, out_shapes, device=dev)

    def _try_io(self, feed, out_shapes):
        """IOBinding attempt; raises (and disables future attempts) on failure."""
        if self._io_failed or not out_shapes:
            raise RuntimeError("iobinding unavailable")
        try:
            return self._io_run(feed, out_shapes)
        except Exception as e:
            self._io_failed = True
            print(f"[backends:iobinding] disabled ({e}); using host path")
            raise

    def _recreate(self, device_id: int | None = None):
        """Fresh equivalent runner (used by __deepcopy__ for GPU replicas)."""
        raise NotImplementedError

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and device.type == "cuda":
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            if self.device_id != dev_idx:
                return self._recreate(device_id=dev_idx)
        return super().to(*args, **kwargs)

    def __deepcopy__(self, memo):
        # ORT sessions cannot be deep-copied; rebuild a clean session. The
        # replica's .to(dev) afterwards is a no-op (no params); the session
        # runs on its own provider device and outputs are moved by the caller.
        return self._recreate(device_id=self.device_id)


class DirectionOnnx(_OnnxRunner):
    """``x [B, M] -> dir [B, 512]`` (dynamic batch)."""

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(
            onnx_path,
            use_gpu=use_gpu,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            device_id=device_id,
        )
        self.input_name = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self.output_dim = shape[-1] if shape and isinstance(shape[-1], int) else None

    def _recreate(self, device_id: int | None = None):
        return DirectionOnnx(
            self.path,
            use_gpu=self.use_gpu,
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            device_id=device_id if device_id is not None else self.device_id,
        )

    def _io_shapes(self, b: int):
        if self.output_dim is None:
            return None
        return [(self.output_name, (int(b), int(self.output_dim)))]

    @_dynamo_disable
    def _run(self, x_np):
        return self.session.run(None, {self.input_name: x_np})[0]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if device.type == "cuda" and self._gpu_active():
            shapes = self._io_shapes(x.shape[0])
            if shapes is not None:
                try:
                    (out,) = self._try_io({self.input_name: x.to(dtype=torch.float32)}, shapes)
                    return out.to(device=device, dtype=torch.float32)
                except Exception:
                    pass
        out = self._run(self._np(x))
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)


class RefEncOnnx(_OnnxRunner):
    """``img [B,3,H,W] -> (x_r, x_r_lambda, *feats)``; exposes ``.fc``.

    ``torch_fallback`` is the original torch encoder: it is the single owner
    of the torch weights (no duplication), serves ``.fc`` for callers, and
    handles the non-deployed 3-arg path (``input_target is not None``) which
    the exported graph does not cover.
    """

    def __init__(
        self,
        onnx_path: str,
        fc: nn.Module,
        use_gpu: bool | None = None,
        torch_fallback=None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(
            onnx_path,
            use_gpu=use_gpu,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            device_id=device_id,
        )
        self.fc = fc
        self._torch_fallback = torch_fallback
        self.input_name = self.session.get_inputs()[0].name

    def _recreate(self, device_id: int | None = None):
        return RefEncOnnx(
            self.path,
            copy.deepcopy(self.fc),
            use_gpu=self.use_gpu,
            torch_fallback=(copy.deepcopy(self._torch_fallback) if self._torch_fallback is not None else None),
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            device_id=device_id if device_id is not None else self.device_id,
        )

    def _io_shapes(self, b: int):
        shapes = []
        for o in self.session.get_outputs():
            s = list(o.shape or [])
            if not s:
                return None
            s[0] = int(b)
            if any(not isinstance(d, int) for d in s):
                return None
            shapes.append((o.name, tuple(s)))
        return shapes or None

    @_dynamo_disable
    def _run(self, img_np):
        return self.session.run(None, {self.input_name: img_np})

    @torch.no_grad()
    def forward(self, x: torch.Tensor, input_target=None, h_start=None):
        if input_target is not None:
            if self._torch_fallback is not None:
                return self._torch_fallback(x, input_target, h_start)
            raise RuntimeError("RefEncOnnx only serves the encode_image_into_latent path")
        device = x.device
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if device.type == "cuda" and self._gpu_active():
            shapes = self._io_shapes(x.shape[0])
            if shapes is not None:
                try:
                    outs = self._try_io({self.input_name: x.to(dtype=torch.float32)}, shapes)
                    outs = [o.to(device=device, dtype=torch.float32) for o in outs]
                    return outs[0], outs[1], outs[2:]
                except Exception:
                    pass
        outs = self._run(self._np(x))
        outs = [torch.from_numpy(o).to(device=device, dtype=torch.float32) for o in outs]
        return outs[0], outs[1], outs[2:]


class DecoderOnnx(_OnnxRunner):
    """Static-batch decode: ``latent [m,512] + feats [m,...] -> img [m,3,H,W]``.

    The graph bakes ``groups=batch`` into its convs, so it is valid for exactly
    one ``B`` (``self.batch``). Runtime pads micro-batches up to ``B``
    (replicate) and trims the output back to ``m``.
    """

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(
            onnx_path,
            use_gpu=use_gpu,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            device_id=device_id,
        )
        ins = self.session.get_inputs()
        self.latent_name = ins[0].name
        self.feat_names = [i.name for i in ins[1:]]
        shape0 = ins[0].shape
        b = shape0[0] if shape0 else None
        self.batch = int(b) if isinstance(b, int) else None
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        out_shape = tuple(out.shape or ())
        self.output_shape = out_shape if len(out_shape) == 4 and all(isinstance(d, int) for d in out_shape) else None

    def _recreate(self, device_id: int | None = None):
        return DecoderOnnx(
            self.path,
            use_gpu=self.use_gpu,
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            device_id=device_id if device_id is not None else self.device_id,
        )

    def _io_shapes(self):
        if self.output_shape is None:
            return None
        return [(self.output_name, self.output_shape)]

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
            latent = torch.cat([latent, latent[-1:].expand(pad, *latent.shape[1:])], dim=0)
            fs = [torch.cat([f, f[-1:].expand(pad, *f.shape[1:])], dim=0) if f.shape[0] == m else f for f in fs]
        if device.type == "cuda" and self._gpu_active() and len(fs) == len(self.feat_names) and self.batch is not None:
            shapes = self._io_shapes()
            if shapes is not None:
                try:
                    feed = {self.latent_name: latent.to(dtype=torch.float32)}
                    for name, f in zip(self.feat_names, fs, strict=False):
                        feed[name] = f.to(dtype=torch.float32)
                    (out,) = self._try_io(feed, shapes)
                    return out[:m].to(device=device, dtype=torch.float32)
                except Exception:
                    pass
        feed = {self.latent_name: self._np(latent)}
        for name, f in zip(self.feat_names, fs, strict=False):
            feed[name] = self._np(f)
        out = self._run(feed)
        return torch.from_numpy(out[:m]).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Parity helpers for scripts/manual use only (never called automatically).
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
    ok = _cosine_ok(x_r, got[0], "refenc.x_r") and _cosine_ok(lam, got[1], "refenc.lambda")
    for i, (a, b) in enumerate(zip(feats, got[2], strict=False)):
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
        except Exception as e:
            logger.warning(f"backends: direction compile failed ({e}); eager")
        return
    path = _path(opt, "direction_onnx_path", f"{_ONNX_DIR}/direction_v1.onnx")
    try:
        # No automatic parity probe: the converter/selector scripts validate;
        # the config is the contract.
        motion_dim = int(target.weight.shape[1])
        runner = DirectionOnnx(path, opt_shapes={"x": [1, motion_dim]})
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
            ae.enc = _CompiledLeaf(enc, extra={"fc": enc.fc})
            logger.info("backends: refenc -> torch.compile")
        except Exception as e:
            logger.warning(f"backends: refenc compile failed ({e}); eager")
        return
    path = _path(opt, "refenc_onnx_path", f"{_ONNX_DIR}/refenc_v1.onnx")
    try:
        # The original encoder stays owned by the runner (single residency):
        # it serves .fc and the non-deployed 3-arg path.
        size = _image_size(opt)
        runner = RefEncOnnx(
            path,
            enc.fc,
            torch_fallback=enc,
            opt_shapes={"img": [1, 3, size, size]},
        )
        ae.enc = runner
        logger.info(f"backends: refenc -> ONNX {path} ({runner.providers})")
    except Exception as e:
        logger.warning(f"backends: refenc ONNX skipped ({e}); eager")


def _apply_decoder(model, ae, opt) -> None:
    mode = resolve_mode("decoder")
    if mode == "eager":
        return
    if mode == "compile":
        try:
            model._decoder_compiled = torch.compile(ae.dec)
            logger.info("backends: decoder -> torch.compile")
        except Exception as e:
            logger.warning(f"backends: decoder compile failed ({e}); eager")
        return
    path = _path(opt, "decoder_onnx_path", f"{_ONNX_DIR}/decoder_v1.onnx")
    try:
        runner = DecoderOnnx(path)
        model._decoder_onnx = runner
        logger.info(f"backends: decoder -> ONNX {path} (static batch {runner.batch}, {runner.providers})")
    except Exception as e:
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
        except Exception as e:
            logger.warning(f"backends: {fn.__name__} failed ({e}); eager")
    return model


__all__ = [
    "DecoderOnnx",
    "DirectionOnnx",
    "RefEncOnnx",
    "apply_backends",
]
