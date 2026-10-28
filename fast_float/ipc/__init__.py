"""IPC package initialization."""

from __future__ import annotations

import os
from typing import Any

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest, IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer, WorkerUnavailable, segment_exists

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def ipc_enabled(ipc_cfg: Any = None) -> bool:
    """Whether live rendering should go through the IPC worker.

    ``[ipc] enabled`` in ``settings.toml`` is the operator default, but
    ``FLOAT_IPC_ENABLED`` wins: the launchers already export it
    (``run_dual_process.sh`` → true, ``run_all.sh`` / ``serve --no-dual-process``
    → false). Without that override a single-process backend still built a
    ``ZmqFloatVideoGenerator`` and then died attaching the worker's segment.
    """
    override = os.environ.get("FLOAT_IPC_ENABLED", "").strip().lower()
    if override in _TRUTHY:
        return True
    if override in _FALSY:
        return False
    return isinstance(ipc_cfg, dict) and bool(ipc_cfg.get("enabled", False))


__all__ = [
    "FrameRingBuffer",
    "IPCFrameNotification",
    "IPCRequest",
    "IPCResponse",
    "WorkerUnavailable",
    "ipc_enabled",
    "segment_exists",
]
