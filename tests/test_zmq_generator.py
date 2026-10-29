"""Tests for ZmqFloatVideoGenerator."""

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest
from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

try:
    from livekit import rtc  # noqa: F401  (probe: the generator needs it at runtime)

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
    from fast_float.ipc.shm_buffer import FrameRingBuffer
    from fast_float.livekit.generator import FloatVideoGenerator  # noqa: F401  (import sanity)

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

            assert captured_reqs
            assert captured_reqs[0].duration_sec == 2.0
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


def test_push_audio_accepts_bytes_and_frames():
    """Regression for the Gemini avatar path: DataStreamAudioReceiver yields
    rtc.AudioFrame objects; a bare len() on one used to kill the runner's
    audio loop ('object of type AudioFrame has no len()').
    """

    if not RTC_AVAILABLE:
        pytest.skip("RTC SDK not available")

    async def _test():
        pytest.importorskip("zmq")
        from livekit import rtc

        from fast_float.livekit.zmq_generator import SAMPLE_RATE

        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=999.0,  # thresholds never crossed: no dispatch, no sockets
            first_chunk_sec=999.0,
            shm_name="test_shm_norm",
            shm_slots=8,
        )

        # Raw 16 kHz mono bytes pass through unchanged: 0.5 s.
        gen.push_audio(b"\x00\x00" * 8000)
        assert gen._audio_samples == 8000
        gen._audio_buf.clear()
        gen._audio_samples = 0

        # An rtc.AudioFrame at the agent's rate (24 kHz, the Gemini Live TTS
        # rate) is resampled down to 16 kHz: 24000 -> 16000 samples.
        gen.push_audio(rtc.AudioFrame(b"\x00\x00" * 24000, 24000, 1, 24000))
        assert gen._audio_samples == SAMPLE_RATE * 1  # one second
        gen._audio_buf.clear()
        gen._audio_samples = 0

        # A stereo frame is downmixed to mono before counting: 3000 stereo
        # samples = 3000 mono samples.
        gen.push_audio(rtc.AudioFrame(b"\x00\x00" * 6000, 16000, 2, 3000))
        assert gen._audio_samples == 3000
        gen._audio_buf.clear()
        gen._audio_samples = 0

        # float arrays are quantized to int16.
        gen.push_audio(np.zeros(1600, dtype=np.float32))
        assert gen._audio_samples == 1600
        gen._audio_buf.clear()
        gen._audio_samples = 0

        # Unsupported payload still raises TypeError (same contract as the
        # builtin generator), which the runner now skips per-item.
        with pytest.raises(TypeError):
            gen.push_audio(123)

    asyncio.run(_test())


def test_flush_dispatches_buffered_tail():
    """Segment end must render the sub-threshold tail instead of stranding it."""

    async def _test():
        pytest.importorskip("zmq")
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=1.0,
            first_chunk_sec=1.0,
            shm_name="test_shm_flush",
            shm_slots=8,
        )
        gen._ensure_connected = AsyncMock()
        gen._dispatch_infer = AsyncMock()

        # 0.5 s: below the 1 s chunk threshold, nothing dispatched yet.
        gen.push_audio(b"\x00\x00" * 8000)
        assert gen._dispatch_infer.call_count == 0

        gen.flush()

        # The dispatch is scheduled on the loop; give it time to run (and be
        # awaited — that is when the mock records await_args).
        for _ in range(50):
            if gen._dispatch_infer.await_args is not None:
                break
            await asyncio.sleep(0.01)

        assert gen._dispatch_infer.await_args is not None
        args = gen._dispatch_infer.await_args.args
        assert args[0] == b"\x00\x00" * 8000
        assert args[1] == 1  # generation 1
        assert gen._audio_samples == 0
        assert not gen._audio_buf
        assert gen._first_chunk_done is True
        # Work is outstanding: idle cleared until the worker's sentinel…
        assert not gen._idle_event.is_set()
        # …which resolves idleness once it arrives and the queue drains.
        gen._sentinel_gen = gen._gen
        gen._refresh_idle()
        assert gen._idle_event.is_set()

        # A flush with an empty buffer is a no-op.
        gen.flush()
        assert gen._dispatch_infer.call_count == 1

    asyncio.run(_test())


def test_wait_until_idle_resolves_on_sentinel_and_drain():
    async def _test():
        pytest.importorskip("zmq")
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=1.0,
            first_chunk_sec=1.0,
            shm_name="test_shm_idle",
            shm_slots=8,
        )
        gen._ensure_connected = AsyncMock()
        gen._dispatch_infer = AsyncMock()

        # Nothing outstanding initially.
        assert gen._idle_event.is_set()
        assert await gen.wait_until_idle(timeout=0.1) is True

        # A whole chunk (1 s) crosses the threshold -> generation submitted.
        gen.push_audio(b"\x00\x00" * 32000)
        assert gen._gen == 1
        assert not gen._idle_event.is_set()
        assert await gen.wait_until_idle(timeout=0.05) is False

        # The worker's is_last sentinel for gen 1 arrived, but the queue still
        # holds frames -> not idle yet.
        gen._sentinel_gen = 1
        await gen._frame_queue.put(object())
        gen._refresh_idle()
        assert not gen._idle_event.is_set()

        # Consumer drains -> idle resolves and the runner's wait returns True.
        gen._frame_queue.get_nowait()
        gen._refresh_idle()
        assert gen._idle_event.is_set()
        assert await gen.wait_until_idle() is True

    asyncio.run(_test())


def test_concurrent_control_requests_are_serialized():
    """Overlapping infer/interrupt requests must not hit the REQ socket twice.

    A REQ socket forbids a second ``send`` while a request is in flight
    (EFSM). Gemini audio streams continuously, so gen N+1 can be submitted
    while gen N still awaits its reply — the unserialized version raised and
    silently dropped the chunk (avatar froze after the first chunk).
    """

    async def _test():
        pytest.importorskip("zmq")
        gen = ZmqFloatVideoGenerator(shm_name="test_shm_serial", shm_slots=4)
        gen._req_lock = asyncio.Lock()

        inflight = set()
        sends = {"count": 0}

        class _Sock:
            async def send_string(self, *_a, **_k):
                # Mimic the REQ protocol: reject any overlapping send.
                assert not inflight, "second send while a request is in flight"
                inflight.add(object())
                sends["count"] += 1

            async def recv(self):
                await asyncio.sleep(0.05)  # in-flight generation
                inflight.clear()
                return b'{"status": "ok"}'

        gen._req_sock = _Sock()

        # Fire three requests overlapping in time before awaiting them.
        await asyncio.gather(
            gen._send_request(IPCRequest(cmd="infer")),
            gen._send_request(IPCRequest(cmd="infer")),
            gen._send_request(IPCRequest(cmd="infer")),
        )
        assert sends["count"] == 3

    asyncio.run(_test())


def test_failed_infer_keeps_idle_accounting_consistent():
    """A dead dispatch must not wedge wait_until_idle until the long timeout."""

    async def _test():
        pytest.importorskip("zmq")
        gen = ZmqFloatVideoGenerator(shm_name="test_shm_fail", shm_slots=4)
        gen._ensure_connected = AsyncMock()

        async def _boom(_pcm, _gen):
            raise RuntimeError("EFSM: overlapping send")

        gen._send_request = _boom
        gen._gen = 1  # _submit_chunk incremented it before scheduling the task

        task = asyncio.create_task(gen._dispatch_infer(b"\x00\x00" * 100, 1))
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        # The failed generation is accounted as completed: idle resolves.
        assert gen._sentinel_gen == 1
        assert gen._idle_event.is_set()
        assert await gen.wait_until_idle(timeout=0.1) is True

    asyncio.run(_test())


def _frame_notif(gen: int, frame_idx: int, *, is_last: bool = False) -> IPCFrameNotification:
    return IPCFrameNotification(
        session_id="sess",
        gen=gen,
        slot_idx=0,
        frame_idx=frame_idx,
        pts_ms=frame_idx * 40.0,
        width=2,
        height=2,
        audio_bytes_len=1280,
        is_last=is_last,
    )


def test_clear_buffer_abandons_in_flight_frames():
    """A barge-in must drop the interrupted chunk, queued and still in flight.

    The worker only reads ``interrupt`` after the chunk it is already rendering
    is done (its REP loop is blocked inside that infer), so a cleared chunk's
    frames keep arriving. Without a generation watermark they were played over
    the next reply — "a chunk comes over the old one".
    """

    async def _test():
        gen = ZmqFloatVideoGenerator(shm_name="test_shm_clear", shm_slots=4)
        gen._send_interrupt = AsyncMock()

        # Two chunks submitted; the first one's frames are already queued.
        gen._gen = 2
        gen._valid_frames = {1: 50, 2: 50}
        gen._frame_queue.put_nowait(("video", "audio"))
        gen._frame_queue.put_nowait(("video", "audio"))
        gen._idle_event.clear()  # work outstanding
        assert not gen._idle_event.is_set()

        gen.clear_buffer()

        # Queued frames of the interrupted sentence are gone.
        assert gen._frame_queue.empty()
        # Frames of both submitted generations are refused from now on.
        assert gen._cleared_gen == 2
        assert not gen._is_live_frame(_frame_notif(1, 0))
        assert not gen._is_live_frame(_frame_notif(2, 49))
        # The next chunk (gen 3) still plays.
        gen._valid_frames[3] = 50
        assert gen._is_live_frame(_frame_notif(3, 0))

        # Cancelled generations are accounted as finished, so the segment
        # watcher does not stall on wait_until_idle after every barge-in.
        assert gen._sentinel_gen == 2
        assert gen._idle_event.is_set()
        assert await gen.wait_until_idle(timeout=0.1) is True

    asyncio.run(_test())


def test_padded_tail_frames_are_trimmed():
    """FLOAT renders whole 2 s windows: the silent tail must not be played."""

    async def _test():
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=1.0,
            first_chunk_sec=1.0,
            shm_name="test_shm_tail",
            shm_slots=4,
        )
        gen._ensure_connected = AsyncMock()
        gen._dispatch_infer = AsyncMock()

        # 1.76 s of audio = 44 frames, padded by the sampler to a 2 s window.
        n_samples = 44 * 640
        gen.push_audio(b"\x00\x00" * n_samples)
        assert gen._gen == 1
        assert gen._valid_frames[1] == 44

        assert gen._is_live_frame(_frame_notif(1, 0))
        assert gen._is_live_frame(_frame_notif(1, 43))
        assert not gen._is_live_frame(_frame_notif(1, 44))
        assert not gen._is_live_frame(_frame_notif(1, 49))

        # The sentinel releases the bookkeeping for that generation.
        gen._valid_frames.pop(1, None)
        assert 1 not in gen._valid_frames

    asyncio.run(_test())


def test_dispatch_infer_skips_cleared_generation():
    """A chunk that lost the race against a barge-in must not be rendered."""

    async def _test():
        gen = ZmqFloatVideoGenerator(shm_name="test_shm_stale", shm_slots=4)
        gen._ensure_connected = AsyncMock()
        gen._send_request = AsyncMock()
        gen._gen = 2
        gen.clear_buffer()  # abandon gen 1 and 2
        gen._send_interrupt = AsyncMock()

        await gen._dispatch_infer(b"\x00\x00" * 640, 2)
        gen._send_request.assert_not_called()

        await gen._dispatch_infer(b"\x00\x00" * 640, 3)
        gen._send_request.assert_called_once()

    asyncio.run(_test())


def test_production_is_paced_to_playout():
    """Chunks must wait for playout headroom instead of flooding the queue.

    The worker renders faster than real time, so dispatching every chunk as
    soon as its audio is cut queues the whole reply; the bounded frame queue
    then drops the OLDEST pairs and the middle of the answer jumps forward
    ("a chunk comes over the old one"). ``FloatVideoGenerator`` paces
    production with ``_pump`` — the ZMQ path has to do the same.
    """

    async def _test():
        gen = ZmqFloatVideoGenerator(
            ref_image_path="ref.png",
            nfe=5,
            chunk_sec=1.0,
            first_chunk_sec=1.0,
            shm_name="test_shm_paced",
            shm_slots=4,
        )
        gen._ensure_connected = AsyncMock()
        gen._dispatch_infer = AsyncMock()
        assert gen._max_chunk_frames == 25  # 1 s of audio at 25 fps

        # Playout is behind: only a few slots are free, so the chunk is held.
        for _ in range(250):
            gen._frame_queue.put_nowait(("video", "audio"))
        gen._queue_chunk(b"\x00\x00" * 16000)
        assert len(gen._pending) == 1
        assert gen._dispatch_infer.call_count == 0
        # Outstanding work must keep the runner's wait_until_idle pending.
        assert not gen._idle_event.is_set()
        assert gen._pump_retry is not None

        # The consumer drains: the held chunk is dispatched on the next pump.
        for _ in range(40):
            gen._frame_queue.get_nowait()
        gen._pump()
        assert not gen._pending
        assert gen._dispatch_infer.call_count == 1
        assert gen._gen == 1
        assert gen._valid_frames[1] == 25

        if gen._pump_retry is not None:
            gen._pump_retry.cancel()
            gen._pump_retry = None

    asyncio.run(_test())


def test_clear_buffer_drops_pending_chunks():
    """A barge-in must also drop chunks that were cut but not yet dispatched."""

    async def _test():
        gen = ZmqFloatVideoGenerator(shm_name="test_shm_pending", shm_slots=4)
        gen._send_interrupt = AsyncMock()
        gen._ensure_connected = AsyncMock()
        gen._dispatch_infer = AsyncMock()

        for _ in range(250):
            gen._frame_queue.put_nowait(("video", "audio"))
        gen._queue_chunk(b"\x00\x00" * 16000)
        assert gen._pending

        gen.clear_buffer()

        assert not gen._pending
        assert gen._pump_retry is None
        assert gen._frame_queue.empty()
        # Nothing was rendered for the interrupted sentence.
        gen._dispatch_infer.assert_not_called()

    asyncio.run(_test())


def test_stale_slot_read_is_skipped_not_played():
    """A slot the writer lapped must never be played under the old index.

    The SHM ring is round-robin, so a writer that outruns the reader replaces
    unread frames. Reading it anyway paired an older frame index with a *newer*
    picture and audio — heard as the count jumping forward mid-sentence, and,
    because every chunk restarts the slot counter at 0, as the next chunk's
    audio landing on the previous chunk's unread tail.
    """

    async def _test():
        from fast_float.ipc.shm_buffer import FrameRingBuffer

        ring = FrameRingBuffer(name="test_shm_stale_read", num_slots=4, width=2, height=2, create=True)
        try:
            gen = ZmqFloatVideoGenerator(shm_name="test_shm_stale_read", shm_slots=4)
            gen._ring = ring

            img = np.zeros((2, 2, 3), dtype=np.uint8)
            # The worker wrote frame 7 into slot 1 after the notification for
            # frame 1 was sent: slot 1 no longer holds what notif announced.
            ring.write_frame(1, 7, 280.0, img, b"\x00\x00" * 640)

            notif = IPCFrameNotification(
                session_id="sess",
                gen=1,
                slot_idx=1,
                frame_idx=1,
                pts_ms=40.0,
                width=2,
                height=2,
                audio_bytes_len=1280,
            )
            assert gen._read_frame_pair(notif) is None
            assert gen._stale_reads == 1
            # The slot is handed back, so the writer can never stall on it.
            assert ring._slot_state(1) == 0
        finally:
            ring.close()
            ring.unlink()

    asyncio.run(_test())


def test_skipped_frames_release_their_slots():
    """Trimmed/stale frames must free their ring slot, or the writer stalls."""

    async def _test():
        from fast_float.ipc.shm_buffer import FrameRingBuffer

        ring = FrameRingBuffer(name="test_shm_release", num_slots=4, width=2, height=2, create=True)
        try:
            gen = ZmqFloatVideoGenerator(shm_name="test_shm_release", shm_slots=4)
            gen._ring = ring
            gen._valid_frames[1] = 2  # only 2 real frames of audio
            gen._cleared_gen = 0

            img = np.zeros((2, 2, 3), dtype=np.uint8)
            ring.write_frame(0, 0, 0.0, img, b"")
            ring.write_frame(1, 5, 200.0, img, b"")  # padded tail
            assert ring._slot_state(1) == 1

            # The trimmed frame is refused and its slot released.
            assert not gen._is_live_frame(
                IPCFrameNotification(
                    session_id="sess",
                    gen=1,
                    slot_idx=1,
                    frame_idx=5,
                    pts_ms=200.0,
                    width=2,
                    height=2,
                    audio_bytes_len=0,
                )
            )
            gen._release_slot(1)
            assert ring._slot_state(1) == 0
        finally:
            ring.close()
            ring.unlink()

    asyncio.run(_test())
