"""A/V sync regressions for the live transports.

These cover the timeline helpers and the two places where audio used to be
paired with video by *count* instead of media time: the raw WebRTC audio track
and the finished-file LiveKit publisher.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from fast_float.utils.audio import (
    frame_sample_bounds,
    load_pcm16_mono,
    slice_audio_at_time,
    slice_audio_for_frame,
)

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RTC_AVAILABLE = False

try:
    import av  # noqa: F401
    from aiortc.mediastreams import MediaStreamError

    from fast_float.routers import webrtc as webrtc_router

    AIORTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    AIORTC_AVAILABLE = False


def test_frame_sample_bounds_cover_the_timeline_without_drift():
    """Non-integer sample rates must not accumulate fractional-sample drift."""
    rate = 16000
    fps = 30.0
    bounds = [frame_sample_bounds(i, fps, rate) for i in range(30)]
    assert bounds[0] == (0, 533)
    assert bounds[1] == (533, 1067)
    # No gaps or overlaps between consecutive frames...
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
    # ...and 30 frames cover exactly one second of audio.
    assert bounds[-1][1] == rate


def test_slice_audio_for_frame_returns_that_frame_and_pads_the_tail():
    audio = np.arange(1000, dtype=np.int16)
    chunk, samples = slice_audio_for_frame(audio, 0, 25.0, 16000)
    assert samples == 640
    assert np.frombuffer(chunk, dtype=np.int16).tolist() == audio[:640].tolist()

    # Past the end of the audio: silence, same frame length.
    chunk, samples = slice_audio_for_frame(audio, 5, 25.0, 16000)
    assert samples == 640
    assert np.frombuffer(chunk, dtype=np.int16).tolist() == [0] * 640


def test_slice_audio_at_time_follows_video_pts_not_decode_order():
    audio = np.arange(16000, dtype=np.int16)
    # 10 frames into a 25 fps video -> 0.4 s -> samples 6400..7040.
    chunk, samples = slice_audio_at_time(audio, 0.4, 25.0, 16000)
    assert samples == 640
    assert np.frombuffer(chunk, dtype=np.int16).tolist() == audio[6400:7040].tolist()

    # An audio stream that starts later than the container: the first video
    # frames get silence instead of shifting the whole soundtrack earlier.
    chunk, samples = slice_audio_at_time(audio, 0.0, 25.0, 16000, audio_start_s=0.2)
    assert np.frombuffer(chunk, dtype=np.int16).tolist() == [0] * samples
    chunk, _ = slice_audio_at_time(audio, 0.2, 25.0, 16000, audio_start_s=0.2)
    assert np.frombuffer(chunk, dtype=np.int16).tolist() == audio[:640].tolist()


def test_load_pcm16_mono_resamples_to_the_requested_rate(tmp_path):
    import soundfile as sf

    path = tmp_path / "audio.wav"
    data = np.arange(4800, dtype=np.int16)
    sf.write(path, data, 48000, subtype="PCM_16")

    loaded = load_pcm16_mono(str(path), 16000)
    assert loaded is not None
    assert len(loaded) == 1600
    assert load_pcm16_mono(None) is None
    assert load_pcm16_mono(str(tmp_path / "missing.wav")) is None


@pytest.mark.skipif(not AIORTC_AVAILABLE, reason="aiortc/PyAV not installed")
def test_live_audio_feed_pairs_audio_with_the_video_frame_index(tmp_path):
    import soundfile as sf

    path = tmp_path / "audio.wav"
    audio = np.arange(16000, dtype=np.int16)
    sf.write(path, audio, 16000, subtype="PCM_16")

    async def _test():
        feed = webrtc_router.LiveAudioFeed(str(path), fps=25.0)
        assert feed.available
        # Frame 0 then frame 10: the second slice must be audio[6400:7040],
        # not the next 640 samples in FIFO order.
        feed.publish(0)
        feed.publish(10)
        first = await feed.next_frame()
        second = await feed.next_frame()
        assert first is not None
        assert second is not None
        assert np.frombuffer(first[0], dtype=np.int16).tolist() == audio[:640].tolist()
        assert np.frombuffer(second[0], dtype=np.int16).tolist() == audio[6400:7040].tolist()

        feed.close()
        assert await feed.next_frame() is None

    asyncio.run(_test())


@pytest.mark.skipif(not AIORTC_AVAILABLE, reason="aiortc/PyAV not installed")
def test_live_video_track_hands_its_index_to_the_audio_feed(tmp_path):
    import soundfile as sf

    path = tmp_path / "audio.wav"
    audio = np.arange(16000, dtype=np.int16)
    sf.write(path, audio, 16000, subtype="PCM_16")

    async def _test():
        queue: asyncio.Queue = asyncio.Queue()
        feed = webrtc_router.LiveAudioFeed(str(path), fps=25.0)
        video = webrtc_router.LiveVideoTrack(
            queue,
            MagicMock(fps=25.0, started_at_s=None, first_idx=0),
            feed,
        )
        audio_track = webrtc_router.LiveAudioTrack(feed)

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        queue.put_nowait(("frame", 0, 0.0, frame))
        queue.put_nowait(("frame", 10, 400.0, frame))

        await video.recv()
        first = await audio_track.recv()
        assert np.frombuffer(first.planes[0], dtype=np.int16).tolist() == audio[:640].tolist()

        await video.recv()
        second = await audio_track.recv()
        assert np.frombuffer(second.planes[0], dtype=np.int16).tolist() == audio[6400:7040].tolist()
        # Audio PTS is a continuous sample clock, independent of the video idx.
        assert second.pts == 640

        from fast_float.streaming.live import CLOSE

        queue.put_nowait(CLOSE)
        with pytest.raises(MediaStreamError):
            await video.recv()
        with pytest.raises(MediaStreamError):
            await audio_track.recv()

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_avatar_runner_paces_the_forward_loop():
    """Audio must not be pushed seconds ahead of the video it belongs to.

    A bursty generator (inference faster than real time) used to fill
    AVSynchronizer's video queue while audio was captured immediately, so the
    soundtrack led the picture by up to two frames. The forward loop now paces
    before pushing both.
    """

    async def _test():
        from fast_float.livekit.worker import AvatarOptions, AvatarRunner

        class Gen:
            def __init__(self, n: int) -> None:
                self.n = n
                self.i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.i >= self.n:
                    raise StopAsyncIteration
                self.i += 1
                return (
                    rtc.VideoFrame(8, 8, rtc.VideoBufferType.RGB24, bytes(8 * 8 * 3)),
                    rtc.AudioFrame(bytes(640 * 2), 16000, 1, 640),
                )

        class Sync:
            def __init__(self) -> None:
                self.times: list[float] = []

            async def push(self, _frame) -> None:
                self.times.append(asyncio.get_running_loop().time())

            async def wait_for_playout(self) -> None:
                pass

        runner = AvatarRunner(MagicMock(), Gen(5), options=AvatarOptions(video_fps=50.0))
        runner._av_sync = Sync()
        await runner._forward_video_loop()

        times = runner._av_sync.times
        assert len(times) == 10  # 5 video + 5 audio pushes
        spread = times[-1] - times[0]
        # 4 frame intervals at 50 fps = 80 ms. No pacing would be ~0 ms.
        assert spread >= 0.06

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_dispatcher_bridge_pairs_agent_audio_at_the_meta_fps():
    """The conversational agent's bridge must follow meta.fps, not a fixed 640."""
    from fast_float.livekit.generator import _LiveKitDispatcherBridge
    from fast_float.streaming.types import StreamMeta

    audio = np.arange(16000, dtype=np.int16)

    async def _test():
        pushed = []
        bridge = _LiveKitDispatcherBridge(
            asyncio.get_running_loop(),
            lambda _gen, item: pushed.append(item),
            audio.tobytes(),
            len(audio),
            0,
            lambda _g: True,
            sample_rate=16000,
        )
        bridge.open(StreamMeta(fps=30.0, width=8, height=8, channels=3))
        bridge.dispatch(0, np.zeros((2, 8, 8, 3), dtype=np.uint8), 0.0)
        await asyncio.sleep(0)
        assert [len(item[3]) // 2 for item in pushed] == [533, 534]
        assert np.frombuffer(pushed[0][3], dtype=np.int16).tolist() == audio[:533].tolist()
        assert np.frombuffer(pushed[1][3], dtype=np.int16).tolist() == audio[533:1067].tolist()

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_livestream_generator_slices_audio_at_the_stream_fps(tmp_path):
    import soundfile as sf

    from fast_float.livekit.generator import LiveStreamVideoGenerator
    from fast_float.streaming.live import LiveStream
    from fast_float.streaming.types import StreamMeta

    path = tmp_path / "audio.wav"
    audio = np.arange(16000, dtype=np.int16)
    sf.write(path, audio, 16000, subtype="PCM_16")

    async def _test():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-fps", loop)
        stream.audio_path = str(path)
        stream.start(StreamMeta(fps=30.0, width=8, height=8, channels=3, request_id="req-fps"))
        gen = LiveStreamVideoGenerator(stream)

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        stream.publish(("frame", 0, 0.0, frame))
        stream.publish(("frame", 1, 1000.0 / 30.0, frame))
        stream.finish()

        out = [item async for item in gen]
        assert len(out) == 2
        first = out[0][1]
        second = out[1][1]
        # Boundaries at 30 fps: 0..533, 533..1067 (533 then 534 samples).
        assert first.samples_per_channel == 533
        assert second.samples_per_channel == 534
        assert np.frombuffer(first.data, dtype=np.int16).tolist() == audio[:533].tolist()
        assert np.frombuffer(second.data, dtype=np.int16).tolist() == audio[533:1067].tolist()

    asyncio.run(_test())
