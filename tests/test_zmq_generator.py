"""Tests for ZmqFloatVideoGenerator."""

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:
    RTC_AVAILABLE = False


def test_zmq_generator_initialization():
    async def _test():
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=2.0,
            first_chunk_sec=2.0,
            shm_name="test_shm_init",
            shm_slots=8,
        )
        assert gen._nfe == 5
        assert gen._chunk_sec == 2.0
        assert gen._audio_samples == 0

    asyncio.run(_test())


def test_zmq_generator_push_audio_and_clear():
    async def _test():
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=2.0,
            first_chunk_sec=2.0,
            shm_name="test_shm_push",
            shm_slots=8,
        )

        # Mock _dispatch_infer so it does not send actual network requests
        with patch.object(gen, "_dispatch_infer", new_callable=AsyncMock) as mock_dispatch:
            # Push 1 second of 16kHz audio (16000 samples = 32000 bytes)
            one_sec_pcm = b"\x00" * 32000
            gen.push_audio(one_sec_pcm)
            assert gen._audio_samples == 16000
            assert mock_dispatch.call_count == 0

            # Push another 1 second to reach chunk threshold (32000 samples)
            gen.push_audio(one_sec_pcm)
            # Buffer should be drained and dispatched
            assert gen._audio_samples == 0
            assert gen._first_chunk_done is True
            assert gen._gen == 1

        # Test clear_buffer
        with patch.object(gen, "_send_interrupt", new_callable=AsyncMock) as mock_interrupt:
            gen.push_audio(b"\x00" * 1000)
            assert gen._audio_samples > 0
            gen.clear_buffer()
            assert gen._audio_samples == 0
            assert gen._first_chunk_done is False

    asyncio.run(_test())


def test_prepare_silence_accepts_duration_kwarg():
    """Regression: session.start calls prepare_silence(duration_sec=...) on
    every generator. ZmqFloatVideoGenerator used to lack the kwarg, and the
    TypeError fell back to a bare retry that hung the session."""

    from fast_float.livekit.generator import FloatVideoGenerator

    for cls in (FloatVideoGenerator, ZmqFloatVideoGenerator):
        params = inspect.signature(cls.prepare_silence).parameters
        assert "duration_sec" in params, f"{cls.__name__}.prepare_silence lacks duration_sec"


def test_prepare_silence_collects_worker_frames():
    """prepare_silence caches the frames published by the worker.

    The SUB socket must have a single reader: the frame listener routes the
    silence notifications into the collection bucket while the prepare request
    is in flight (a second inline receiver used to steal the ``is_last``
    sentinel and starve the collector into a 30 s timeout).
    """
    pytest.importorskip("zmq")
    import threading
    import time

    import zmq

    from fast_float.ipc.messages import IPCFrameNotification, IPCResponse
    from fast_float.livekit.generator import FloatVideoGenerator  # noqa: F401  (import sanity)
    from fast_float.ipc.shm_buffer import FrameRingBuffer

    async def _test():
        ring = FrameRingBuffer("test_zmq_silence_ring", num_slots=8, create=True)
        pub_ctx = zmq.Context()
        pub = pub_ctx.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")

        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            control_endpoint="tcp://127.0.0.1:6555",  # never dialed — request is stubbed
            frames_endpoint=f"tcp://127.0.0.1:{port}",
            shm_name="test_zmq_silence_ring",
            shm_slots=8,
            session_id="sess-sil-test",
        )

        captured_reqs: list = []

        async def fake_send_request(req):
            # The request is sent once, with collection mode already on.
            assert gen._collecting is True
            captured_reqs.append(req)
            return IPCResponse(status="ok", req_id=req.req_id, session_id=req.session_id, total_frames=4)

        gen._send_request = fake_send_request  # type: ignore[method-assign]

        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        audio = b"\x00" * 1280

        def publisher() -> None:
            # PUB/SUB slow joiner: the connect side (created by
            # _ensure_connected) needs a moment before its handshake is up.
            time.sleep(0.25)
            for i in range(4):
                ring.write_frame(
                    slot_idx=i,
                    frame_idx=i,
                    pts_ms=i * 40.0,
                    image_array=img,
                    audio_bytes=audio,
                )
                pub.send_string(
                    IPCFrameNotification(
                        session_id="sess-sil-test",
                        gen=0,
                        slot_idx=i,
                        frame_idx=i,
                        pts_ms=i * 40.0,
                        width=16,
                        height=16,
                        audio_bytes_len=len(audio),
                    ).to_json()
                )
            pub.send_string(
                IPCFrameNotification(
                    session_id="sess-sil-test",
                    gen=0,
                    slot_idx=-1,
                    frame_idx=4,
                    pts_ms=0.0,
                    width=0,
                    height=0,
                    is_last=True,
                ).to_json()
            )

        try:
            await gen._ensure_connected()
            publisher_thread = threading.Thread(target=publisher)
            publisher_thread.start()

            await gen.prepare_silence(duration_sec=2.0)

            assert captured_reqs and captured_reqs[0].duration_sec == 2.0
            assert gen._collecting is False
            assert gen._silence_ready.is_set()
            expected = 4 if RTC_AVAILABLE else 0  # frames need livekit RTC to wrap
            assert len(gen._silence_frames) == expected

            # Idempotency: a repeated call is a no-op (matches FloatVideoGenerator).
            snapshot = list(gen._silence_frames)
            await gen.prepare_silence(duration_sec=2.0)
            assert gen._silence_frames == snapshot
        finally:
            publisher_thread.join(timeout=5.0)
            await gen.close()
            pub.close()
            pub_ctx.term()
            ring.close()
            ring.unlink()

    asyncio.run(_test())
