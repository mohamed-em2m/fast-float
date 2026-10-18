"""DynamoBuilder: build-time driver for the ``compile`` backend.

Nothing is serialized for ``torch.compile`` -- inductor caches kernels on disk,
keyed by the graph -- so a "build" is a warm-up: compile the module once (and
optionally run it on sample inputs) to populate the cache, and record where the
cache lives. The served process recompiles on load (a cache hit) via
``fast_float.runtimes.dynamo.CompiledRuntime``.
"""

from __future__ import annotations

import os
import time

from fast_float.builders.base_builder import BuilderSession
from fast_float.builders.spec import BuildArtifact, BuildSpec

_DEFAULT_INDUCTOR_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch", "inductor")


def _inductor_cache_dir() -> str:
    return os.environ.get("TORCHINDUCTOR_CACHE_DIR") or _DEFAULT_INDUCTOR_CACHE


class DynamoBuilder(BuilderSession):
    """Warm the inductor cache for one component's module."""

    backend = "compile"

    def available(self) -> bool:
        try:
            import torch
        except Exception:
            return False
        if not hasattr(torch, "compile"):
            return False
        try:
            # Presence is not usability: torch.compile raises on some platforms
            # (e.g. Windows), so probe with a trivial function.
            torch.compile(lambda x: x + 1)(torch.zeros(1))
            return True
        except Exception:
            return False

    def build(self, spec: BuildSpec) -> BuildArtifact:
        module = spec.extra.get("module")
        if module is None:
            raise ValueError("dynamo build requires spec.extra['module'] (a live nn.Module)")
        if not self.available():
            raise RuntimeError("torch.compile unavailable on this platform")
        import torch

        started = time.perf_counter()
        fn = torch.compile(module)
        sample_inputs = spec.extra.get("sample_inputs")
        warmed = False
        if sample_inputs is not None:
            with torch.no_grad():
                fn(*sample_inputs)
            warmed = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return BuildArtifact(
            component=spec.component,
            kind="dynamo",
            precision=spec.precision,
            fingerprint=self.fingerprint(spec, None, module_class=type(module).__name__),
            built_at=time.time(),
            metrics={
                "warmed": warmed,
                "compile_ms": round(elapsed_ms, 3),
                "cache_dir": _inductor_cache_dir(),
            },
        )


__all__ = ["DynamoBuilder"]
