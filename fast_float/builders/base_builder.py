"""BuilderSession: build-time driver producing a BuildArtifact per component.

One builder per backend family (``onnx``, ``dynamo``, native ``tensorrt``).
Builders are heavy and run on the target box (offline); a served process only
ever *loads* artifacts into runtimes. See ``docs/BUILD_RUNTIME_SESSIONS.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fast_float.builders.spec import BuildArtifact, BuildSpec, build_fingerprint


class BuilderSession(ABC):
    """Base class for one backend family's build-time driver."""

    backend: str = ""

    @abstractmethod
    def available(self) -> bool:
        """True when this builder's toolchain is importable/usable on this box."""

    @abstractmethod
    def build(self, spec: BuildSpec) -> BuildArtifact:
        """Produce an artifact (an ``eager`` builder is a no-op artifact)."""

    def fingerprint(
        self, spec: BuildSpec, env: dict[str, Any] | None = None, *, module_class: str | None = None
    ) -> str:
        """Cache key for ``spec`` under the given environment fingerprint."""
        return build_fingerprint(spec, env, module_class=module_class)


__all__ = ["BuilderSession"]
