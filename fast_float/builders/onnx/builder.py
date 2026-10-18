"""OnnxBuilder: build-time driver for the ``onnx`` backend.

Wraps the existing export machinery (``fast_float.onnx.export.export_module``,
``make_runtime_session``, ``trt_optimization_profiles``) behind
``BuilderSession`` plus an ``ArtifactCache`` manifest keyed by the spec
fingerprint.

Two sub-steps (see ``docs/BUILD_RUNTIME_SESSIONS.md`` §5):

  (a) export the graph -- deterministic torch -> ONNX;
  (b) optional pre-warm of the ORT-TensorRT engine into the TRT cache dir.

Component-specific model building (checkpoint loading, export-safe wrappers)
stays in the ``convert_*.py`` scripts; the builder takes an already
export-ready module via ``spec.extra['module']`` and its ``sample_inputs``.
Parity is opt-in via ``spec.extra['reference']`` (a torch callable matching the
module's IO) and is recorded on the artifact, never enforced here.
"""

from __future__ import annotations

import time
from pathlib import Path

from fast_float.builders.base_builder import BuilderSession
from fast_float.builders.spec import ArtifactCache, BuildArtifact, BuildSpec
from fast_float.onnx.registry import get as component_spec

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = _REPO_ROOT / "onnx" / ".build_manifest.json"


def _entry_from_artifact(a: BuildArtifact) -> dict:
    return {
        "component": a.component,
        "kind": a.kind,
        "path": a.path,
        "provider": a.provider,
        "precision": a.precision,
        "batch": a.batch,
        "fingerprint": a.fingerprint,
        "metrics": a.metrics,
        "parity": a.parity,
        "built_at": a.built_at,
    }


def _artifact_from_entry(entry: dict, component: str, fingerprint: str) -> BuildArtifact:
    return BuildArtifact(
        component=entry.get("component", component),
        kind=entry.get("kind", "onnx"),
        path=entry.get("path"),
        provider=entry.get("provider"),
        precision=entry.get("precision", "fp32"),
        batch=entry.get("batch"),
        fingerprint=fingerprint,
        metrics=dict(entry.get("metrics") or {}),
        parity=entry.get("parity"),
        built_at=entry.get("built_at"),
    )


class OnnxBuilder(BuilderSession):
    """Export a component module to ONNX and record it in the artifact cache."""

    backend = "onnx"

    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_MANIFEST
        self._cache: ArtifactCache | None = None

    @property
    def cache(self) -> ArtifactCache:
        if self._cache is None:
            self._cache = ArtifactCache(self.cache_path)
        return self._cache

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            import torch  # noqa: F401
            from torch.export import export  # noqa: F401
        except Exception:
            return False
        return True

    def out_path(self, spec: BuildSpec) -> str:
        return spec.extra.get("out_path") or component_spec(spec.component).onnx_default

    def fingerprint(self, spec: BuildSpec, env: dict | None = None, *, module_class: str | None = None) -> str:
        if module_class is None:
            module_class = spec.extra.get("module_class")
            module = spec.extra.get("module")
            if module_class is None and module is not None:
                module_class = type(module).__name__
        return super().fingerprint(spec, env, module_class=module_class)

    def lookup(self, spec: BuildSpec, env: dict | None = None) -> BuildArtifact | None:
        """Return the cached artifact for ``spec`` or ``None`` on a miss."""
        fp = self.fingerprint(spec, env)
        entry = self.cache.get(fp)
        return _artifact_from_entry(entry, spec.component, fp) if entry else None

    def record(self, artifact: BuildArtifact) -> None:
        self.cache.put(artifact.fingerprint, _entry_from_artifact(artifact))

    def runtime(self, spec: BuildSpec, device_id: int | None = None):
        """Serve ``spec``'s artifact through a :class:`RuntimeSession`.

        Builds the ORT session from the spec's provider chain / precision /
        TRT knobs (the same knobs ``make_session_from_args`` exposes), so the
        selector measures the runtime contract instead of a raw session.
        """
        import torch

        from fast_float.onnx.export import make_ort_session, trt_optimization_profiles
        from fast_float.runtimes.onnx.runner import OrtRuntime

        chain = tuple(spec.provider_chain or ())
        # A GPU chain on a box without CUDA must fall back to CPU, not request a
        # GPU EP and raise (mirrors ``make_session_from_args``).
        use_gpu = ("cuda" in chain or "trt" in chain) and torch.cuda.is_available()
        use_trt = use_gpu and "trt" in chain
        mn = op = mx = None
        if use_trt and spec.opt_shapes:
            mn, op, mx = trt_optimization_profiles(
                spec.opt_shapes,
                min_batch=spec.trt_min_batch,
                max_batch=spec.trt_max_batch,
                extra=spec.extra_dynamic,
            )
        session = make_ort_session(
            self.out_path(spec),
            use_gpu=use_gpu,
            require_gpu=use_gpu,
            use_trt=use_trt,
            trt_fp16=(spec.precision == "fp16"),
            trt_cache_dir=spec.trt_cache_dir,
            trt_min_shapes=mn,
            trt_opt_shapes=op,
            trt_max_shapes=mx,
            device_id=device_id,
        )
        return OrtRuntime(session, spec.input_names, component=spec.component, device_id=device_id)

    def build(self, spec: BuildSpec, env: dict | None = None) -> BuildArtifact:
        module = spec.extra.get("module")
        sample = spec.extra.get("sample_inputs")
        if module is None or sample is None:
            raise ValueError("onnx build requires spec.extra['module'] and spec.extra['sample_inputs']")
        if not (spec.input_names and spec.output_names):
            raise ValueError("onnx build requires spec.input_names and spec.output_names")
        if not self.available():
            raise RuntimeError("ONNX export requires onnxruntime and torch.export (torch>=2.1)")

        from fast_float.onnx.export import export_module

        out_path = self.out_path(spec)
        info = export_module(
            module,
            sample,
            out_path,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            dynamic_axes=spec.dynamic_axes,
            opset=spec.opset,
            fixed_batch=spec.fixed_batch,
            mark_dims=tuple(spec.extra.get("mark_dims", (0, 1))),
        )
        parity = self._parity(spec, info["path"]) if spec.extra.get("reference") is not None else None
        artifact = BuildArtifact(
            component=spec.component,
            kind="onnx",
            path=info["path"],
            precision=spec.precision,
            batch=spec.batch,
            fingerprint=self.fingerprint(spec, env),
            built_at=time.time(),
            metrics={"exporter": info.get("exporter"), "notes": list(info.get("notes") or [])},
            parity=parity,
        )
        self.record(artifact)
        return artifact

    def _parity(self, spec: BuildSpec, onnx_path: str) -> dict:
        import torch

        from fast_float.onnx.export import make_runtime_session
        from fast_float.onnx.parity import compare_outputs

        reference = spec.extra["reference"]
        sample = spec.extra["sample_inputs"]
        sample = sample if isinstance(sample, tuple) else (sample,)
        sess, _ = make_runtime_session(
            onnx_path,
            opt_shapes=spec.opt_shapes,
            use_gpu=bool(spec.extra.get("parity_use_gpu", False)),
        )
        with torch.no_grad():
            ref = reference(*sample)
        feed = {n: t.detach().cpu().numpy() for n, t in zip(spec.input_names, sample, strict=False)}
        got = sess.run(None, feed)
        got0 = got[0] if isinstance(got, list) else got
        ref0 = ref[0] if isinstance(ref, (tuple, list)) else ref
        return compare_outputs(ref0, got0, atol=spec.tolerance, rtol=spec.tolerance)


__all__ = ["DEFAULT_MANIFEST", "OnnxBuilder"]
