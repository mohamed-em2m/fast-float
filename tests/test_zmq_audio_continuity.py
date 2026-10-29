"""End-to-end continuity check of the ZMQ live path — no GPU, no sockets.

The live avatar's worst failure mode is silent audio loss: the consumer plays
*most* of a reply, skips a slice, and the listener hears the speaker jump
forward mid-sentence (or a chunk of the previous sentence talking over the new
one). Every queue on that path can lose a frame, and a lost frame is a lost
40 ms of speech, so the symptom is indistinguishable from a render bug without
accounting.

This test drives the real objects on both sides of the IPC boundary —
``_ShmFrameDispatcher`` (worker) writing into a real ``FrameRingBuffer`` and
``ZmqFloatVideoGenerator`` (backend) reading it — with the pipeline replaced by
a synthetic renderer, and asserts the strict property:

    the audio the consumer plays == the audio that was pushed, in order.

A random PCM signal makes any dropped, duplicated or reordered frame show up as
a byte mismatch, and the failure message reports which frame indices differ.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from fast_float.ipc.messages import IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer
from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator
from fast_float.streaming.types import StreamMeta
from fast_float.workers.inference_worker import FPS, SAMPLE_RATE, SAMPLES_PER_FRAME, _ShmFrameDispatcher

W, H = 8, 8
SID = "sess-continuity"


class _FakeCtx:
    """Stand-in for the ZMQ context so ``close()`` has something to term."""

    def term(self) -> None:
        pass


class _FakePub:
    """Worker-side PUB: forwards notifications to the consumer's fake SUB."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]) -> None:
        self._loop = loop
        self._queue = queue

    def send_string(self, payload: str) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)


class _FakeSub:
    """Consumer-side SUB: just an async queue of notification payloads."""

    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self._queue = queue

    async def recv_string(self) -> str:
        return await self._queue.get()

    def close(self) -> None:
        pass


class _FakeWorker:
    """Renders a chunk the way the pipeline does: batches of frames into SHM."""

    def __init__(self, ring: FrameRingBuffer, loop: asyncio.AbstractEventLoop) -> None:
        self._ring = ring
        self._loop = loop
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self.pub = _FakePub(loop, self._queue)
        self.sub = _FakeSub(self._queue)
        self.rendered: list[int] = []
        self.dispatchers: list[_ShmFrameDispatcher] = []
        # Wall-clock seconds of render per output frame (0 = instant).
        self.frame_cost_s = 0.0
        # FLOAT renders whole 2 s windows (50 frames at 25 fps); the padded tail
        # is trimmed by the consumer's ``_valid_frames``.
        self.window = 50

    def render(self, pcm: np.ndarray, gen: int) -> IPCResponse:
        """Mirror ``_handle_infer``: dispatch every decoded frame, then close."""
        valid = int(np.ceil(len(pcm) / SAMPLES_PER_FRAME))
        total = max(self.window, int(np.ceil(valid / self.window)) * self.window)

        disp = _ShmFrameDispatcher(
            ring=self._ring,
            pub_socket=self.pub,
            session_id=SID,
            req_id=f"r{gen}",
            gen=gen,
            interrupt_event=threading.Event(),
            audio_pcm_int16=pcm,
        )
        self.dispatchers.append(disp)
        disp.bind(StreamMeta(fps=float(FPS), width=W, height=H, request_id=f"r{gen}"))
        try:
            for start in range(0, total, 8):
                n = min(8, total - start)
                batch = np.zeros((n, H, W, 3), dtype=np.uint8)
                batch[:] = (start % 251, gen % 251, 0)
                disp.submit(batch, idx=start)
                # Emulate FLOAT at ~6x real time (40 ms of audio per 6.7 ms).
                if self.frame_cost_s:
                    time.sleep(n * self.frame_cost_s)
        finally:
            disp.shutdown(timeout=30.0)
        self.rendered.append(total)
        return IPCResponse(status="ok", req_id=f"r{gen}", total_frames=total, rtf=6.0)


def _make_generator(
    ring: FrameRingBuffer,
    worker: _FakeWorker,
    *,
    chunk_sec: float = 5.0,
    first_chunk_sec: float = 2.0,
) -> ZmqFloatVideoGenerator:
    gen = ZmqFloatVideoGenerator(
        ref_image_path="ref.png",
        nfe=5,
        chunk_sec=chunk_sec,
        first_chunk_sec=first_chunk_sec,
        shm_name=ring.name,
        shm_slots=ring.num_slots,
        session_id=SID,
    )
    # Bypass the socket/shm attach: hand it the real ring and a fake SUB.
    gen._zmq_ctx = _FakeCtx()
    gen._ring = ring
    gen._sub_sock = worker.sub
    gen._req_lock = asyncio.Lock()
    gen._listener_task = asyncio.create_task(gen._frame_listener(), name="zmq-frame-listener")

    async def _send_request(req):
        pcm = np.frombuffer(__import__("base64").b64decode(req.audio_pcm), dtype="<i2")
        return await asyncio.to_thread(worker.render, pcm, req.gen)

    gen._send_request = _send_request  # type: ignore[method-assign]
    return gen


async def _consume(gen: ZmqFloatVideoGenerator, expected_frames: int, frame_delay_s: float) -> list[bytes]:
    """Pull up to ``expected_frames`` pairs off the generator, pacing like the runner."""
    out: list[bytes] = []
    it = gen.__aiter__()
    while len(out) < expected_frames:
        try:
            video, audio = await asyncio.wait_for(it.__anext__(), timeout=5.0)
        except asyncio.TimeoutError:
            break  # the stream stalled: report what was played
        assert video is not None
        assert audio is not None
        out.append(bytes(audio.data))
        if frame_delay_s > 0:
            await asyncio.sleep(frame_delay_s)
    return out


def _expected(pcm: np.ndarray) -> bytes:
    """The PCM the consumer should play: every frame slice, last one padded."""
    valid = int(np.ceil(len(pcm) / SAMPLES_PER_FRAME))
    padded = np.zeros(valid * SAMPLES_PER_FRAME, dtype="<i2")
    padded[: len(pcm)] = pcm
    return padded.astype("<i2").tobytes()


def _first_diff(played: bytes, expected: bytes) -> str:
    """Locate the first differing frame so a failure names the index."""
    n = min(len(played), len(expected)) // (SAMPLES_PER_FRAME * 2)
    for i in range(n):
        a = played[i * 1280 : (i + 1) * 1280]
        b = expected[i * 1280 : (i + 1) * 1280]
        if a != b:
            return f"frame {i} differs (played {len(played) // 1280} frames, expected {len(expected) // 1280})"
    return f"length differs: played {len(played) // 1280} frames, expected {len(expected) // 1280}"


@pytest.mark.parametrize("frame_delay_s", [0.0, 0.002, 0.04])
def test_zmq_audio_is_played_exactly_once_in_order(frame_delay_s: float) -> None:
    """No frame may be dropped, duplicated or reordered anywhere on the path."""

    async def _test() -> None:
        ring = FrameRingBuffer(
            name=f"test_shm_cont_{int(frame_delay_s * 1000)}", num_slots=64, width=W, height=H, create=True
        )
        try:
            worker = _FakeWorker(ring, asyncio.get_running_loop())
            gen = _make_generator(ring, worker)

            rng = np.random.RandomState(7)
            # Two TTS segments, each flushed below chunk_sec => exactly one chunk
            # of 1.0 s (25 frames) per segment.
            seg_a = rng.randint(-30000, 30000, SAMPLE_RATE, dtype=np.int16)
            seg_b = rng.randint(-30000, 30000, SAMPLE_RATE, dtype=np.int16)
            pcm = np.concatenate([seg_a, seg_b])

            expected_frames = len(pcm) // SAMPLES_PER_FRAME
            consume_task = asyncio.create_task(_consume(gen, expected_frames, frame_delay_s))

            for seg in (seg_a, seg_b):
                for start in range(0, len(seg), SAMPLE_RATE // 10):
                    gen.push_audio(seg[start : start + SAMPLE_RATE // 10].tobytes())
                gen.flush()
                await asyncio.sleep(0.05)

            played = await asyncio.wait_for(consume_task, timeout=60.0)
            gen.stop()
            # The ring is shared by both "processes" here; the worker owns it in
            # production, so the consumer must not close it.
            gen._ring = None
            await gen.close()

            got = b"".join(played)
            want = _expected(pcm)
            assert got == want, _first_diff(got, want)
        finally:
            ring.close()
            ring.unlink()

    asyncio.run(_test())


def test_zmq_audio_survives_a_full_size_chunk() -> None:
    """A chunk as large as production allows must still play every frame.

    ``[float].livekit_chunk_sec = 5.0`` means one chunk can carry 125 valid
    frames, and FLOAT renders whole 2 s windows, so the worker actually emits
    ~150 frames for it. Both worker-side buffers must be able to hold that:

    * the ``FrameDispatcher`` delivery queue (``max_queue_frames``), because a
      ``live`` tap drops the OLDEST frames when it is full — a hole in the
      middle of the chunk, i.e. the speaker jumping forward;
    * the SHM ring, because with flow control the delivery thread blocks until
      the consumer frees a slot, which is what fills that queue.

    A consumer that paces at playout (40 ms/frame) is the real-world case: the
    renderer is ~6x faster than real time, so it is always ahead of the reader.
    """

    async def _test() -> None:
        ring = FrameRingBuffer(name="test_shm_cont_big", num_slots=64, width=W, height=H, create=True)
        try:
            worker = _FakeWorker(ring, asyncio.get_running_loop())
            gen = _make_generator(ring, worker, chunk_sec=5.0, first_chunk_sec=5.0)

            rng = np.random.RandomState(11)
            pcm = rng.randint(-30000, 30000, 5 * SAMPLE_RATE, dtype=np.int16)
            expected_frames = len(pcm) // SAMPLES_PER_FRAME

            consume_task = asyncio.create_task(_consume(gen, expected_frames, 0.04))
            for start in range(0, len(pcm), SAMPLE_RATE // 10):
                gen.push_audio(pcm[start : start + SAMPLE_RATE // 10].tobytes())
            gen.flush()

            played = await asyncio.wait_for(consume_task, timeout=120.0)
            gen.stop()
            gen._ring = None
            await gen.close()

            stats = [d.stats() for d in worker.dispatchers]
            got = b"".join(played)
            want = _expected(pcm)
            assert len(played) == expected_frames, (
                f"consumer stalled after {len(played)}/{expected_frames} frames; "
                f"forced_overwrites={ring.forced_overwrites}, dispatch stats: {stats}"
            )
            assert got == want, f"{_first_diff(got, want)}; worker dispatch stats: {stats}"
        finally:
            ring.close()
            ring.unlink()

    asyncio.run(_test())


def test_zmq_production_is_paced_to_playout() -> None:
    """A reply streamed faster than real time must not overflow the frame queue.

    The agent delivers a whole answer in a fraction of a second, so many chunks
    are cut before the first one has been played. If ``_pump`` dispatches each
    of them as it is cut, it decides on a *nearly empty* queue — the frames of
    the chunks already rendering have not landed yet — and the queue overflows.
    Overflow drops the OLDEST pairs, i.e. a hole in the middle of the sentence:
    the avatar skips from "3" to "13". Production must be paced to playout the
    way ``FloatVideoGenerator`` does with its ``_inference_running`` guard.
    """

    async def _test() -> None:
        ring = FrameRingBuffer(name="test_shm_cont_pace", num_slots=64, width=W, height=H, create=True)
        try:
            worker = _FakeWorker(ring, asyncio.get_running_loop())
            # Slow the renderer down a little so production outruns playout even
            # for the small chunks used here.
            worker.frame_cost_s = 0.001
            worker.window = 5
            gen = _make_generator(ring, worker, chunk_sec=0.2, first_chunk_sec=0.2)

            rng = np.random.RandomState(13)
            pcm = rng.randint(-30000, 30000, 15 * SAMPLE_RATE, dtype=np.int16)
            expected_frames = len(pcm) // SAMPLES_PER_FRAME

            consume_task = asyncio.create_task(_consume(gen, expected_frames, 0.005))
            # The agent streams the whole reply at once (faster than real time).
            for start in range(0, len(pcm), SAMPLE_RATE // 10):
                gen.push_audio(pcm[start : start + SAMPLE_RATE // 10].tobytes())
            gen.flush()

            played = await asyncio.wait_for(consume_task, timeout=120.0)
            gen.stop()
            gen._ring = None
            await gen.close()

            assert gen._dropped_frames == 0, (
                f"frame queue overflowed: {gen._dropped_frames} pair(s) dropped — production was not paced to playout"
            )
            got = b"".join(played)
            want = _expected(pcm)
            assert got == want, _first_diff(got, want)
        finally:
            ring.close()
            ring.unlink()

    asyncio.run(_test())
