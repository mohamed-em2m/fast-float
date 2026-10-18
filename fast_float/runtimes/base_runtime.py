"""RuntimeSession: per-device serving wrapper for one component backend.

Implementations wrap eager modules, ``torch.compile`` callables, or ORT
sessions. The contract, so call sites can stay backend-agnostic:

* torch in (any device/dtype) -> torch out on the caller's device;
* never let a backend failure crash the pipeline -- follow the
  ``degrade_on_error`` policy (sticky / transient / raise);
* ``replicate`` for the components that run on more than one GPU (only
  ``decoder`` today, per the registry).

See ``docs/BUILD_RUNTIME_SESSIONS.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeStats:
    """Per-session counters surfaced by the runtime ``inspect`` CLI."""

    calls: int = 0
    errors: int = 0
    io_bindings: int = 0
    degraded: bool = False
    reason: str | None = None


class RuntimeSession(ABC):
    """Serves one component's forward pass on one device."""

    component: str = ""
    backend: str = "eager"

    def __init__(self, device_id: int | None = None, degrade_on_error: str = "transient"):
        self.device_id = device_id
        self.degrade_on_error = degrade_on_error
        self.stats = RuntimeStats()
        self._degraded = False
        self._closed = False
        # Cooperative: in the ``(RuntimeSession, nn.Module)`` mixins this chains
        # to ``nn.Module.__init__`` so torch bookkeeping is initialised.
        super().__init__()

    @property
    def providers(self) -> list:
        """Active ORT providers (empty for eager/compile runtimes)."""
        return []

    @property
    def degraded(self) -> bool:
        return self._degraded

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run one inference call."""

    def replicate(self, device_id: int) -> RuntimeSession:
        """Return an equivalent session bound to ``device_id``.

        Only registry-``replicable`` components (``decoder`` today) may
        replicate; the gate fails loudly instead of silently sharing a session
        across GPUs.
        """
        if not self._replicable():
            raise NotImplementedError(f"{self._label()} is not replicable")
        return self._replicate(device_id)

    def _replicable(self) -> bool:
        if not self.component:
            return False
        try:
            from fast_float.onnx.registry import get

            return get(self.component).replicable
        except Exception:
            return False

    def _replicate(self, device_id: int) -> RuntimeSession:
        raise NotImplementedError(f"{type(self).__name__} has no replicate implementation")

    def _label(self) -> str:
        return self.component or type(self).__name__

    def degrade(self, reason: str) -> None:
        """Mark the session degraded (sticky) after a backend failure."""
        self._degraded = True
        self.stats.degraded = True
        self.stats.reason = reason

    def close(self) -> None:
        """Release resources (ORT sessions, CUDA memory). Best-effort."""
        self._closed = True


__all__ = ["RuntimeSession", "RuntimeStats"]
