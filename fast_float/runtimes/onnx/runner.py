"""ONNX runtime sessions: torch in (any device/dtype) -> torch out.

The three post-load runners (``direction``/``refenc``/``decoder``). Moved here
from ``fast_float/onnx/dispatch.py`` (Phase 2 of
``docs/BUILD_RUNTIME_SESSIONS.md``); ``dispatch`` still re-exports them so
scripts keep importing from either place.

ORT sessions prefer CUDA IOBinding (device-resident, zero host copies) and fall
back to the numpy host path on any failure. Runners implement ``__deepcopy__``
so multi-GPU replicas rebuild clean sessions instead of copying live ones.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn

from fast_float.onnx.export import make_runtime_session
from fast_float.runtimes.base_runtime import RuntimeSession


def _dynamo_disable(fn):
    try:
        from torch._dynamo import disable

        return disable(fn)
    except Exception:
        return fn


class _OnnxRunner(RuntimeSession, nn.Module):
    """Base ONNX session wrapper: torch in (any device/dtype) -> torch out.

    Prefers CUDA IOBinding (device-resident, zero host copies); falls back to
    the numpy host path on any failure (flag cached after the first failure so
    the hot path never pays repeated exception costs).
    """

    backend = "onnx"

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(device_id=device_id)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else bool(use_gpu)
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

    def _io_run(self, feed, out_shapes, session=None):
        from fast_float.onnx.export import run_ort_iobinding

        dev = next(iter(feed.values())).device
        return run_ort_iobinding(session or self.session, feed, out_shapes, device=dev)

    def _try_io(self, feed, out_shapes, session=None):
        """IOBinding attempt; raises (and disables future attempts) on failure."""
        if self._io_failed or not out_shapes:
            raise RuntimeError("iobinding unavailable")
        try:
            out = self._io_run(feed, out_shapes, session)
            self.stats.io_bindings += 1
            return out
        except Exception as e:
            self._io_failed = True
            print(f"[backends:iobinding] disabled ({e}); using host path")
            raise

    def _recreate(self, device_id: int | None = None):
        """Fresh equivalent runner (used for GPU replicas / device rebind)."""
        raise NotImplementedError

    def _replicate(self, device_id: int) -> _OnnxRunner:
        """Registry-gated replicate: rebuild an equivalent session on ``device_id``."""
        return self._recreate(device_id)

    def _refresh_io(self) -> None:
        """Re-cache input/output names after the session is (re)built."""

    def _rebind(self, device_id: int) -> None:
        """Rebuild this runner's ORT session on ``device_id``, in place."""
        self.device_id = int(device_id)
        self.session, _ = make_runtime_session(
            self.path,
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            use_gpu=self.use_gpu,
            device_id=self.device_id,
        )
        self._io_failed = False
        self._refresh_io()

    def _apply(self, fn, recurse=True):
        # ``Module.to(device)`` recurses through ``_apply`` (never ``to``), so
        # this is the only hook that observes a parent module's device move;
        # rebind the ORT session so replicas run on their own GPU.
        super()._apply(fn, recurse)
        try:
            target = fn(torch.empty(0)).device
        except Exception:
            return self
        if target.type == "cuda":
            idx = target.index if target.index is not None else torch.cuda.current_device()
            if self.device_id != idx:
                self._rebind(idx)
        return self

    def __deepcopy__(self, memo):
        # ORT sessions cannot be deep-copied; rebuild a clean session. Replicas
        # are moved with ``.to(dev)`` afterwards, which rebinds via ``_apply``.
        return self._recreate(device_id=self.device_id)


class DirectionOnnx(_OnnxRunner):
    """``x [B, M] -> dir [B, 512]`` (dynamic batch)."""

    component = "direction"

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
        self._refresh_io()

    def _refresh_io(self) -> None:
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

    component = "refenc"

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
        self._refresh_io()

    def _refresh_io(self) -> None:
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


def _discover_decoder_batches(onnx_path: str) -> dict[int, str]:
    """Map ``batch -> path`` for ``<stem>_b<N>.onnx`` siblings of ``onnx_path``.

    ``convert_decoder`` writes the primary batch to ``--out`` and every other
    batch to a ``<stem>_b<N>.onnx`` sibling (see its docstring); the runtime
    discovers them by filename so the selector/config only need to name the
    primary artifact. Returns ``{}`` when there are no siblings (legacy single
    graph), in which case the caller keys everything off the primary file.
    """
    import glob
    import re

    abs_path = os.path.abspath(onnx_path)
    directory = os.path.dirname(abs_path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    pattern = re.compile(re.escape(stem) + r"_b(\d+)\.onnx$")
    found: dict[int, str] = {}
    for p in glob.glob(os.path.join(directory, f"{stem}_b*.onnx")):
        m = pattern.search(os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    return found


class DecoderOnnx(_OnnxRunner):
    """Static-batch decode: ``latent [m,512] + feats [m,...] -> img [m,3,H,W]``.

    The graph bakes ``groups=batch`` into its convs, so each artifact is valid
    for exactly one ``B``. ``convert_decoder`` exports a *set* of them; this
    runner loads them lazily and, per call, picks the smallest session covering
    the incoming micro-batch ``m`` (replicate-pads up to it and trims the
    output back to ``m``). ``self.batch`` is the largest available batch, which
    lifts the decode chunk-size cap from the single artifact's ``B``.
    """

    component = "decoder"

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        RuntimeSession.__init__(self, device_id=device_id)
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else bool(use_gpu)
        self.path = os.path.abspath(onnx_path)
        self._opt_shapes = opt_shapes
        self._extra_dynamic = extra_dynamic
        self._max_batch = max_batch
        self._io_failed = False
        self._sessions: dict = {}
        self._io: dict = {}
        self._paths = _discover_decoder_batches(onnx_path)
        # Load the primary artifact eagerly: gives ``providers`` a session and
        # anchors the no-siblings (legacy single-graph) case.
        session, meta = self._load(self.path)
        key = meta["batch"]
        self._primary_key = key
        self._paths[key] = self.path
        self._sessions[key] = session
        self._io[key] = meta
        self.session = session
        self._refresh_io()
        int_keys = sorted(k for k in self._paths if isinstance(k, int))
        self.batches = int_keys
        self.batch = int_keys[-1] if int_keys else None

    @staticmethod
    def _read_io(session, path: str) -> dict:
        ins = session.get_inputs()
        shape0 = ins[0].shape
        b = shape0[0] if shape0 else None
        out = session.get_outputs()[0]
        out_shape = tuple(out.shape or ())
        return {
            "path": path,
            "latent_name": ins[0].name,
            "feat_names": [i.name for i in ins[1:]],
            "batch": int(b) if isinstance(b, int) else None,
            "output_name": out.name,
            "output_shape": out_shape if len(out_shape) == 4 and all(isinstance(d, int) for d in out_shape) else None,
        }

    def _load(self, path: str):
        session, _ = make_runtime_session(
            path,
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            use_gpu=self.use_gpu,
            device_id=self.device_id,
        )
        return session, self._read_io(session, path)

    def _ensure(self, key):
        if key not in self._sessions:
            path = self._paths.get(key)
            if path is None:
                raise KeyError(f"no ONNX decoder artifact for batch {key}")
            session, meta = self._load(path)
            self._sessions[key] = session
            self._io[key] = meta
        return self._sessions[key], self._io[key]

    def _select_batch(self, m: int):
        for b in self.batches:
            if b >= m:
                return b
        return self.batches[-1] if self.batches else self._primary_key

    def _refresh_io(self) -> None:
        # Primary-session metadata (self.session); keeps ``providers`` and the
        # legacy single-artifact attributes working.
        meta = self._read_io(self.session, self.path)
        self.latent_name = meta["latent_name"]
        self.feat_names = meta["feat_names"]
        self.output_name = meta["output_name"]
        self.output_shape = meta["output_shape"]

    def _recreate(self, device_id: int | None = None):
        return DecoderOnnx(
            self.path,
            use_gpu=self.use_gpu,
            opt_shapes=self._opt_shapes,
            extra_dynamic=self._extra_dynamic,
            max_batch=self._max_batch,
            device_id=device_id if device_id is not None else self.device_id,
        )

    def _rebind(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self._sessions = {}
        self._io = {}
        session, meta = self._load(self.path)
        key = meta["batch"]
        self._primary_key = key
        self._sessions[key] = session
        self._io[key] = meta
        self.session = session
        self._io_failed = False
        self._refresh_io()

    def _io_shapes(self, meta: dict | None = None):
        meta = meta or self._io.get(self._primary_key) or {}
        out_shape = meta.get("output_shape")
        out_name = meta.get("output_name") or getattr(self, "output_name", "img")
        return [(out_name, out_shape)] if out_shape else None

    @_dynamo_disable
    def _run(self, feed):
        return self.session.run(None, feed)[0]

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, feats) -> torch.Tensor:
        device = latent.device
        m = latent.shape[0]
        key = self._select_batch(m)
        session, meta = self._ensure(key)
        big_b = meta["batch"]
        if big_b is not None and m > big_b:
            raise ValueError(f"decode batch {m} exceeds static ONNX batch {big_b}")
        big_b = big_b or m
        fs = list(feats)
        fs = [f.expand(big_b, *f.shape[1:]) if f.shape[0] == 1 else f for f in fs]
        if m < big_b:
            pad = big_b - m
            latent = torch.cat([latent, latent[-1:].expand(pad, *latent.shape[1:])], dim=0)
            fs = [torch.cat([f, f[-1:].expand(pad, *f.shape[1:])], dim=0) if f.shape[0] == m else f for f in fs]
        feat_names = meta["feat_names"]
        if device.type == "cuda" and self._gpu_active() and len(fs) == len(feat_names) and meta["batch"] is not None:
            shapes = self._io_shapes(meta)
            if shapes is not None:
                try:
                    feed = {meta["latent_name"]: latent.to(dtype=torch.float32)}
                    for name, f in zip(feat_names, fs, strict=False):
                        feed[name] = f.to(dtype=torch.float32)
                    (out,) = self._try_io(feed, shapes, session)
                    return out[:m].to(device=device, dtype=torch.float32)
                except Exception:
                    pass
        feed = {meta["latent_name"]: self._np(latent)}
        for name, f in zip(feat_names, fs, strict=False):
            feed[name] = self._np(f)
        out = session.run(None, feed)[0]
        return torch.from_numpy(out[:m]).to(device=device, dtype=torch.float32)


class OrtRuntime(RuntimeSession, nn.Module):
    """RuntimeSession over an already-built ORT ``InferenceSession``.

    For scripts (the backend selector) that construct the session themselves --
    with their own provider/precision knobs -- but want to measure and report
    through the runtime contract (torch in on any device -> torch out, provider
    listing, per-call timing/stats). Uses the numpy host path only; there is no
    device rebinding, so ``device_id`` is metadata for ``inspect``.
    """

    backend = "onnx"

    def __init__(
        self,
        session,
        input_names,
        component: str = "",
        device_id: int | None = None,
        degrade_on_error: str = "transient",
    ):
        super().__init__(device_id=device_id, degrade_on_error=degrade_on_error)
        self.session = session
        self.input_names = tuple(input_names)
        self.component = component

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.session.get_outputs()]

    @_dynamo_disable
    def _run(self, feed):
        return self.session.run(None, feed)

    @torch.no_grad()
    def forward(self, *inputs):
        device = inputs[0].device if inputs else torch.device("cpu")
        feed = {
            name: t.detach().to("cpu", dtype=torch.float32).numpy()
            for name, t in zip(self.input_names, inputs, strict=False)
        }
        outs = self._run(feed)
        return [torch.from_numpy(np.asarray(o)).to(device=device, dtype=torch.float32) for o in outs]


__all__ = ["DecoderOnnx", "DirectionOnnx", "OrtRuntime", "RefEncOnnx"]
