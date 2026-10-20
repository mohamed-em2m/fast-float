"""CompiledRuntime: a ``torch.compile`` leaf under RuntimeSession.

Replaces the private ``_CompiledLeaf`` previously in ``onnx/dispatch.py``
(Phase 3 of ``docs/BUILD_RUNTIME_SESSIONS.md``). Unlike the ONNX runners the
compiled callable cannot be deep-copied, so ``__deepcopy__`` recompiles around
a fresh copy of the wrapped module and its forwarded extra attributes.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from fast_float.runtimes.base_runtime import RuntimeSession


class CompiledRuntime(RuntimeSession, nn.Module):
    """``torch.compile`` around a leaf module, forwarding extra attributes.

    Used where a call site also reaches into the module (e.g. ``enc.fc``); the
    extra attributes (``extra``) are duplicated onto this wrapper so the call
    site keeps working through the substitution.
    """

    backend = "compile"

    def __init__(
        self,
        module: nn.Module,
        extra: dict | None = None,
        component: str = "",
        device_id: int | None = None,
    ):
        super().__init__(device_id=device_id)
        if component:
            self.component = component
        self._orig = module
        self._fn = torch.compile(module)
        self._extra_keys = list(extra or {})
        for k, v in (extra or {}).items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def _recreate(self, device_id: int | None = None) -> CompiledRuntime:
        # Recompile around a fresh copy; the compiled callable itself is not
        # copied. Weights are frozen at inference, so duplicated extras (e.g.
        # enc.fc alongside _orig.fc) cannot diverge.
        new_mod = copy.deepcopy(self._orig)
        extra = {k: copy.deepcopy(getattr(self, k)) for k in self._extra_keys}
        return CompiledRuntime(new_mod, extra=extra, component=self.component, device_id=device_id)

    def _replicate(self, device_id: int) -> CompiledRuntime:
        return self._recreate(device_id)

    def __deepcopy__(self, memo):
        return self._recreate(self.device_id)


__all__ = ["CompiledRuntime"]
