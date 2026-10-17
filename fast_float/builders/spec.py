"""Declarative build/run specifications for the builder/runtime migration.

See ``docs/BUILD_RUNTIME_SESSIONS.md``. These dataclasses describe *what* to
build and *how* to run a component backend. They import no torch/onnx so they
stay cheap to import (models, scripts, tests).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKENDS = ("eager", "compile", "onnx")
PROVIDERS = ("trt", "cuda", "cpu")
PRECISIONS = ("fp16", "fp32")
DEGRADE_MODES = ("sticky", "transient", "raise")
DEFAULT_PROVIDER_CHAIN = ("trt", "cuda", "cpu")

# Subset of ``fast_float.onnx.backends.env_fingerprint()`` that changes the
# bytes of a built artifact / cached TensorRT engine.
_ENV_KEYS = ("torch", "cuda", "cc", "ort", "trt", "cuda_ep")


@dataclass(frozen=True)
class BuildSpec:
    """Inputs a builder needs to produce one artifact for one component."""

    component: str
    backend: str = "eager"
    source: str | None = None
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    opt_shapes: dict[str, tuple[int, ...]] | None = None
    extra_dynamic: dict[str, dict[int, tuple[int, int]]] | None = None
    dynamic_axes: dict[str, dict[int, str]] | None = None
    precision: str = "fp16"
    provider_chain: tuple[str, ...] = DEFAULT_PROVIDER_CHAIN
    trt_min_batch: int = 1
    trt_max_batch: int | None = None
    trt_cache_dir: str | None = None
    fixed_batch: bool = False
    batch: int | None = None
    opset: int = 17
    tolerance: float = 2e-4
    min_cosine: float | None = None
    max_mean: float | None = None
    ckpt_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r} (known: {BACKENDS})")
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision!r} (known: {PRECISIONS})")


@dataclass(frozen=True)
class BuildArtifact:
    """Result of a build: where the artifact is and how it was produced."""

    component: str
    kind: str  # onnx | dynamo | engine | eager
    path: str | None = None
    provider: str | None = None
    precision: str = "fp32"
    batch: int | None = None
    fingerprint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] | None = None
    built_at: float | None = None


@dataclass(frozen=True)
class RunSpec:
    """Everything a runtime needs to serve one component on one device."""

    component: str
    backend: str = "eager"
    artifact: BuildArtifact | None = None
    device_id: int | None = None
    provider_chain: tuple[str, ...] = DEFAULT_PROVIDER_CHAIN
    fallback_backend: str = "eager"
    tolerance: float = 2e-4
    degrade_on_error: str = "sticky"
    replicable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.degrade_on_error not in DEGRADE_MODES:
            raise ValueError(f"unknown degrade_on_error {self.degrade_on_error!r} (known: {DEGRADE_MODES})")


def _ckpt_signature(path: str | None) -> str | None:
    """Cheap weights fingerprint: size + mtime (never a content hash)."""
    if not path:
        return None
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return f"{int(st.st_size)}:{int(st.st_mtime)}"


def fingerprint_env_fields(env: dict[str, Any] | None) -> dict[str, Any]:
    """Stable subset of ``env_fingerprint()`` that affects a built artifact."""
    if not env:
        return {}
    return {k: env.get(k) for k in _ENV_KEYS if k in env}


def build_fingerprint(
    spec: BuildSpec,
    env: dict[str, Any] | None = None,
    *,
    module_class: str | None = None,
) -> str:
    """Stable 32-hex cache key for a (spec, environment) pair."""
    payload = {
        "component": spec.component,
        "backend": spec.backend,
        "module_class": module_class,
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "opt_shapes": spec.opt_shapes,
        "extra_dynamic": spec.extra_dynamic,
        "dynamic_axes": spec.dynamic_axes,
        "precision": spec.precision,
        "fixed_batch": spec.fixed_batch,
        "batch": spec.batch,
        "opset": spec.opset,
        "provider_chain": list(spec.provider_chain),
        "ckpt": _ckpt_signature(spec.ckpt_path),
        "env": fingerprint_env_fields(env),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class ArtifactCache:
    """JSON manifest of built artifacts keyed by fingerprint.

    Lives at ``onnx/.build_manifest.json`` by default. Misses are cheap (one
    dict lookup); the runtime must not build on a cache miss unless the backend
    allows it (compile may; onnx/trt must not on a served box).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        return self._entries.get(fingerprint)

    def put(self, fingerprint: str, entry: dict[str, Any]) -> None:
        self._entries[fingerprint] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def __contains__(self, fingerprint: str) -> bool:
        return fingerprint in self._entries

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "BACKENDS",
    "DEFAULT_PROVIDER_CHAIN",
    "DEGRADE_MODES",
    "PRECISIONS",
    "PROVIDERS",
    "ArtifactCache",
    "BuildArtifact",
    "BuildSpec",
    "RunSpec",
    "build_fingerprint",
    "fingerprint_env_fields",
]
