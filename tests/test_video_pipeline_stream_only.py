"""Streaming (dispatcher-only) runs must not try to mux a file.

Regression: the LiveKit generator calls ``run_inference_blocking(agent, None, ...)``
because frames go straight to the dispatcher. ``inference_and_save`` still ran
the FFmpeg writer, which crashed on every frame with
``TypeError: expected str, bytes or os.PathLike object, not NoneType``.
"""

import queue
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fast_float.models.float.video_pipeline import (  # noqa: E402
    SaveVideoPipeline,
    StreamingVideoSaver,
    _drain_ordered,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


class _Buf:
    def __init__(self, tag: int) -> None:
        self.tag = tag


class _Done:
    def __init__(self) -> None:
        self.synced = False

    def synchronize(self) -> None:
        self.synced = True


class _SilentDispatcher:
    """Minimal stand-in: the guards reject a run before any hook is called."""


def test_start_ffmpeg_rejects_missing_path():
    saver = StreamingVideoSaver()
    with pytest.raises(ValueError, match="video_path"):
        saver._start_ffmpeg(None, None, 512, 512, 3, 25)


@pytest.mark.parametrize("path", [None, ""])
def test_inference_and_save_rejects_no_sink(path):
    saver = StreamingVideoSaver()
    saver.opt = SimpleNamespace(fps=25, video_encoder="FFmpeg NVENC")

    with pytest.raises(ValueError, match="dispatcher"):
        saver.inference_and_save({"s": None}, path)


def test_stream_only_does_not_require_ffmpeg(monkeypatch):
    """A dispatcher-only run bails out before the encoder/GPU path is touched."""
    from fast_float.models.float import video_pipeline

    monkeypatch.setattr(video_pipeline.shutil, "which", lambda _name: None)
    saver = StreamingVideoSaver()
    saver.opt = SimpleNamespace(fps=25, video_encoder="FFmpeg NVENC")
    saver._decode_devices = lambda: (_ for _ in ()).throw(AssertionError("GPU path reached"))

    with pytest.raises(AssertionError, match="GPU path reached"):
        saver.inference_and_save({"s": None}, None, dispatcher=_SilentDispatcher())


def test_drain_ordered_reorders_and_recycles():
    """Batches finish out of order: delivery is in order, buffers return to the pool."""
    done_q, pool = queue.Queue(), queue.Queue()
    state = {"total": 3, "frames": 0}
    bufs = [_Buf(i) for i in range(3)]
    dones = [_Done() for _ in range(3)]
    for seq in (1, 2, 0):  # decode completion order
        done_q.put((seq, bufs[seq], 2, dones[seq]))

    delivered = []
    _drain_ordered(done_q, state, threading.Event(), pool, lambda b, n: delivered.append((b.tag, n)))

    assert delivered == [(0, 2), (1, 2), (2, 2)]
    assert state["frames"] == 6
    assert pool.qsize() == 3
    assert all(d.synced for d in dones)


def test_drain_ordered_stops_at_total():
    done_q, pool = queue.Queue(), queue.Queue()
    state = {"total": 1, "frames": 0}
    for seq in range(3):
        done_q.put((seq, _Buf(seq), 1, None))

    delivered = []
    _drain_ordered(done_q, state, threading.Event(), pool, lambda b, n: delivered.append(b.tag))

    assert delivered == [0]
    assert state["frames"] == 1
    assert pool.qsize() == 1


def test_drain_ordered_honours_stop():
    done_q, pool = queue.Queue(), queue.Queue()
    state = {"total": None, "frames": 0}
    stop = threading.Event()
    stop.set()
    done_q.put((0, _Buf(0), 1, None))

    delivered = []
    _drain_ordered(done_q, state, stop, pool, lambda b, n: delivered.append(b.tag), timeout=0.01)

    assert delivered == []
    assert state["frames"] == 0


class _RecordingTap:
    """Stands in for FrameDispatcher: records what the pipeline would deliver."""

    def __init__(self) -> None:
        self.meta = None
        self.batches: list = []
        self.errors: list = []

    def bind(self, meta, stop_event=None):
        self.meta = meta

    def submit(self, frames, idx=None):
        self.batches.append(frames.copy())
        return True

    def error(self, exc):
        self.errors.append(exc)

    def shutdown(self):
        pass


def _fake_model(n_frames: int = 5, batch_size: int = 2) -> SaveVideoPipeline:
    """Cheap stand-in for FLOAT: same threading/queue wiring, no real networks."""
    opt = SimpleNamespace(
        fps=25,
        a_cfg_scale=2.0,
        r_cfg_scale=1.0,
        e_cfg_scale=1.0,
        decode_streams=1,
        decode_coalesce_ms=5.0,
        video_encoder="FFmpeg NVENC",
        use_nvenc=False,
    )
    model = SaveVideoPipeline(opt=opt)
    dev = torch.device("cuda:0")
    prof = SimpleNamespace(height=4, width=4, channels=3, batch_size=batch_size)

    model._decode_devices = lambda: (dev, [dev])
    model.encode_image_into_latent = lambda s: (
        torch.zeros(1, 1, device=dev),
        torch.zeros(1, 1, device=dev),
        [torch.zeros(1, 1, device=dev)],
    )
    model.encode_identity_into_motion = lambda s_r: torch.zeros(1, 1, device=dev)
    model._get_ae = lambda d, primary: object()
    model._get_decode_profile = lambda *a, **k: (object(), prof)
    model._decoder_static_batch = lambda: None

    def sample(*args, **kwargs):
        for i in range(n_frames):
            yield torch.tensor([[float(i)]], device=dev)

    model.sample = sample

    def _decode_u8(ae, feats, lat):
        # Encode each frame's index as the pixel value so order is observable.
        values = lat[:, 0].to(torch.uint8).view(-1, 1, 1, 1)
        return values.expand(-1, prof.height, prof.width, prof.channels).contiguous()

    model._decode_u8 = _decode_u8
    return model


@requires_cuda
def test_stream_only_pipeline_delivers_frames_without_encoder(monkeypatch):
    """video_path=None: frames reach the dispatcher in order, no encoder runs."""
    model = _fake_model()
    tap = _RecordingTap()

    def _no_encoder(*args, **kwargs):
        raise AssertionError("encoder must not be started for a stream-only run")

    monkeypatch.setattr(model, "_start_ffmpeg", _no_encoder)

    result = model.inference_and_save(
        {"s": torch.zeros(1, 3, 4, 4), "s_r": torch.zeros(1, 3, 4, 4)},
        None,
        dispatcher=tap,
    )

    assert result is None
    assert tap.errors == []
    assert tap.meta.fps == 25.0
    assert (tap.meta.height, tap.meta.width, tap.meta.channels) == (4, 4, 3)
    delivered = [int(batch[i, 0, 0, 0]) for batch in tap.batches for i in range(len(batch))]
    assert delivered == [0, 1, 2, 3, 4]  # generation order, nothing dropped
    assert model.last_stats["frames"] == 5
