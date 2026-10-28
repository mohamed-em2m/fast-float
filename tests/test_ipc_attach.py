"""Consumer-side attach guardrails for the dual-process stack.

The LiveKit renderer attaches to the worker's ``FrameRingBuffer`` with a bare
``SharedMemory(name=...)``. When the worker was not running (single-process
launcher) or still loading its weights, that raised

    FileNotFoundError: [Errno 2] No such file or directory: '/float_frame_ring'

from inside ``POST /livekit/conversation/start`` — an ASGI 500 with a traceback
instead of an actionable message. Worse, ``_ensure_connected`` stored the ZMQ
context *before* the attach, so the next attempt returned early "connected"
with no ring, sockets or listener.

These tests pin the three properties that keep that failure legible:
a bounded wait for a booting worker, a ``RuntimeError`` (→ HTTP 503) with a
remedy, and no half-initialised generator.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np
import pytest

from fast_float.ipc import ipc_enabled, segment_exists
from fast_float.ipc.shm_buffer import FrameRingBuffer, WorkerUnavailable


def test_attach_reports_missing_worker_with_remedy():
    """A missing segment is a RuntimeError carrying the fix, not FileNotFoundError."""
    with pytest.raises(WorkerUnavailable) as excinfo:
        FrameRingBuffer.attach("test_attach_absent", num_slots=4, timeout=0.2)

    # The API maps RuntimeError -> 503; anything else becomes a 500 traceback.
    assert isinstance(excinfo.value, RuntimeError)
    message = str(excinfo.value)
    assert "test_attach_absent" in message
    assert "run_dual_process.sh" in message
    assert "FLOAT_IPC_ENABLED" in message


def test_attach_waits_for_worker_that_is_still_booting():
    """The worker publishes the ring only after loading; the consumer waits."""
    name = "test_attach_late"
    ring_owner = {}

    def _publish_later() -> None:
        time.sleep(0.4)
        ring_owner["ring"] = FrameRingBuffer(name=name, num_slots=4, create=True)

    thread = threading.Thread(target=_publish_later, daemon=True)
    thread.start()
    try:
        time.sleep(0.05)
        consumer = FrameRingBuffer.attach(name, num_slots=4, timeout=5.0)
        try:
            owner = ring_owner["ring"]
            owner.write_frame(0, frame_idx=7, pts_ms=280.0, image_array=np.zeros((4, 4, 3), dtype=np.uint8))
            frame_idx, _pts, _img, _audio = consumer.read_frame(0)
            assert frame_idx == 7
        finally:
            consumer.close()
    finally:
        thread.join(timeout=5.0)
        owner = ring_owner.get("ring")
        if owner is not None:
            owner.close()
            owner.unlink()


def test_segment_exists_tracks_the_worker_lifetime():
    name = "test_segment_exists"
    assert segment_exists(name) is False
    ring = FrameRingBuffer(name=name, num_slots=4, create=True)
    try:
        assert segment_exists(name) is True
    finally:
        ring.close()
        ring.unlink()
    assert segment_exists(name) is False


def test_ipc_enabled_lets_the_launcher_override_settings():
    """`run_all.sh` / `serve --no-dual-process` must be able to turn IPC off."""
    settings_on = {"enabled": True}
    settings_off = {"enabled": False}

    os.environ.pop("FLOAT_IPC_ENABLED", None)
    try:
        # No override: settings.toml decides.
        assert ipc_enabled(settings_on) is True
        assert ipc_enabled(settings_off) is False
        assert ipc_enabled(None) is False

        os.environ["FLOAT_IPC_ENABLED"] = "false"
        assert ipc_enabled(settings_on) is False

        os.environ["FLOAT_IPC_ENABLED"] = "1"
        assert ipc_enabled(settings_off) is True
    finally:
        os.environ.pop("FLOAT_IPC_ENABLED", None)


def test_generator_attach_failure_is_retry_safe(monkeypatch):
    """A failed attach leaves nothing behind: the retry connects for real."""
    pytest.importorskip("zmq")
    from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

    name = "test_retry_attach"

    async def _test() -> None:
        gen = ZmqFloatVideoGenerator(
            shm_name=name,
            shm_slots=4,
            control_endpoint="tcp://127.0.0.1:6555",
            frames_endpoint="tcp://127.0.0.1:6556",
            shm_wait_s=0.0,
        )
        with pytest.raises(RuntimeError):
            await gen._ensure_connected()

        # Nothing half-open: a second call must not short-circuit on _zmq_ctx.
        assert gen._zmq_ctx is None
        assert gen._req_sock is None
        assert gen._sub_sock is None
        assert gen._ring is None
        assert gen._listener_task is None

        ring = FrameRingBuffer(name=name, num_slots=4, create=True)
        try:
            await gen._ensure_connected()
            assert gen._ring is not None
            assert gen._listener_task is not None
        finally:
            await gen.close()
            ring.close()
            ring.unlink()

    asyncio.run(_test())
