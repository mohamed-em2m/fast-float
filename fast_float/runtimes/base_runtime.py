"""RuntimeSession: per-device serving wrapper for one component backend.

Implementations wrap eager modules, ``torch.compile`` callables, or ORT
sessions. The contract, so call sites can stay backend-agnostic:

* torch in (any device/dtype) -> torch out on the caller's device;
* never let a backend failure crash the pipeline -- follow the
  ``degrade_on_error`` policy (sticky / transient / raise);
* ``replicate`` for the components that run on more than one GPU (only
  ``decoder`` today).

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

    def __init__(self, device_id: int | None = None, degrade_on_error: str = "sticky"):
        self.device_id = device_id
        self.degrade_on_error = degrade_on_error
        self.stats = RuntimeStats()
        self._degraded = False
        self._closed = False

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

        Only components flagged ``replicable`` (``decoder``) need this; the
        default raises so an accidental replicate on a primary-only component
        fails loudly instead of silently sharing a session across GPUs.
        """
        raise NotImplementedError(f"{type(self).__name__} is not replicable")

    def degrade(self, reason: str) -> None:
        """Mark the session degraded (sticky) after a backend failure."""
        self._degraded = True
        self.stats.degraded = True
        self.stats.reason = reason

    def close(self) -> None:
        """Release resources (ORT sessions, CUDA memory). Best-effort."""
        self._closed = True


__all__ = ["RuntimeSession", "RuntimeStats"]
