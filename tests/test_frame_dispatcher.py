import asyncio
import threading
import time

import numpy as np
import pytest

from fast_float.streaming import FrameDispatcher, FrameQueue, StreamMeta


class Collector(FrameDispatcher):
    def __init__(self):
        super().__init__()
        self.received = []
        self.gaps = []
        self.errors = []
        self.opened = 0
        self.closed = 0
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def open(self, meta):
        self.opened += 1
        self.meta = meta

    def dispatch(self, idx, frames, pts_ms):
        self.entered.set()
        self.gate.wait(5.0)
        self.received.append((idx, np.array(frames, copy=True), pts_ms))

    def gap(self, start, end, reason):
        self.gaps.append((start, end, reason))

    def error(self, exc):
        self.errors.append(exc)

    def close(self):
        self.closed += 1


def frames(n):
    return np.zeros((n, 4, 4, 3), dtype=np.uint8)


def meta(first_idx=0, fps=25.0):
    return StreamMeta(fps=fps, width=4, height=4, channels=3, first_idx=first_idx)


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_delivers_in_order_with_pts():
    d = Collector()
    d.bind(meta(fps=20.0))
    assert d.submit(frames(3))
    assert d.submit(frames(2))
    assert wait_until(lambda: len(d.received) == 2)
    d.shutdown()

    assert [idx for idx, _, _ in d.received] == [0, 3]
    assert [pts for _, _, pts in d.received] == [0.0, pytest.approx(150.0)]
    assert d.stats()["delivered"] == 5
    assert d.closed == 1


def test_reorders_explicit_indices():
    d = Collector()
    d.bind(meta())
    assert d.submit(frames(2), idx=4)
    time.sleep(0.05)
    assert d.received == []
    assert d.submit(frames(4), idx=0)
    assert wait_until(lambda: len(d.received) == 2)
    d.shutdown()
    assert [idx for idx, _, _ in d.received] == [0, 4]


def test_live_overflow_drops_oldest_and_reports_gap():
    d = Collector()
    d.max_queue_frames = 4
    d.gate.clear()
    d.bind(meta())

    assert d.submit(frames(4))
    assert d.entered.wait(2.0)
    for _ in range(5):
        d.submit(frames(4))
    d.gate.set()
    assert wait_until(lambda: d.stats()["delivered"] == 8)
    d.shutdown()

    assert [idx for idx, _, _ in d.received] == [0, 20]
    assert d.gaps == [(4, 20, "dropped")]
    stats = d.stats()
    assert stats["dropped_overflow"] == 16
    assert stats["skipped_gap"] == 16
    assert stats["last_idx"] == 23


def test_live_drops_frames_beyond_media_lag():
    d = Collector()
    d.max_lag_ms = 100
    d.gate.clear()
    d.bind(meta(fps=25.0))

    assert d.submit(frames(4))
    assert d.entered.wait(2.0)
    assert d.submit(frames(4))
    assert d.submit(frames(4))
    d.gate.set()
    assert wait_until(lambda: d.stats()["delivered"] == 8)
    d.shutdown()

    assert [idx for idx, _, _ in d.received] == [0, 8]
    assert d.gaps == [(4, 8, "dropped")]
    assert d.stats()["dropped_lag"] == 4


def test_slow_generation_is_not_treated_as_stale():
    d = Collector()
    d.bind(meta(fps=25.0))
    time.sleep(0.2)
    assert d.submit(frames(20))
    assert wait_until(lambda: d.stats()["delivered"] == 20)
    d.shutdown()
    assert d.gaps == []


def test_lossless_blocks_then_delivers_everything():
    d = Collector()
    d.mode = "lossless"
    d.max_queue_frames = 4
    d.gate.clear()
    d.bind(meta())

    assert d.submit(frames(4))
    assert d.entered.wait(2.0)
    assert d.submit(frames(4))

    third = threading.Thread(target=lambda: d.submit(frames(4)))
    third.start()
    time.sleep(0.05)
    assert third.is_alive(), "lossless submit should wait for space"

    d.gate.set()
    third.join(2.0)
    assert not third.is_alive()
    assert wait_until(lambda: d.stats()["delivered"] == 12)
    d.shutdown()

    assert [idx for idx, _, _ in d.received] == [0, 4, 8]
    assert d.gaps == []


def test_lossless_timeout_disables_instead_of_reordering():
    d = Collector()
    d.mode = "lossless"
    d.max_queue_frames = 4
    d.submit_timeout_ms = 50
    d.gate.clear()
    d.bind(meta())

    assert d.submit(frames(4))
    assert d.entered.wait(2.0)
    assert d.submit(frames(4))
    assert not d.submit(frames(4))

    stats = d.stats()
    assert stats["disabled_reason"] == "submit_timeout"
    assert stats["delivered"] == 0

    d.gate.set()
    d.shutdown()
    assert d.closed == 1


def test_dispatch_error_disables_and_reports():
    class Boom(Collector):
        def dispatch(self, idx, frames, pts_ms):
            raise RuntimeError("sink died")

    d = Boom()
    d.bind(meta())
    d.submit(frames(2))
    assert wait_until(lambda: d.disabled)
    assert isinstance(d.errors[0], RuntimeError)
    assert not d.submit(frames(2))
    d.shutdown()
    assert d.stats()["disabled_reason"] == "dispatch_error"


def test_open_once_and_open_failure():
    d = Collector()
    d.bind(meta())
    d.bind(meta(fps=60.0))
    assert d.opened == 1
    assert d.meta.fps == 25.0
    d.shutdown()

    class BadOpen(Collector):
        def open(self, meta):
            raise OSError("no transport")

    bad = BadOpen()
    bad.bind(meta())
    assert bad.disabled
    assert isinstance(bad.errors[0], OSError)
    assert not bad.submit(frames(1))
    bad.shutdown()
    assert bad.closed == 0, "close() must not run without a successful open()"

    unbound = Collector()
    unbound.shutdown()
    assert unbound.closed == 0


def test_shutdown_reports_unfilled_hole_before_draining():
    d = Collector()
    d.bind(meta())
    assert d.submit(frames(2), idx=10)
    d.shutdown()
    assert d.gaps == [(0, 10, "missing")]
    assert [idx for idx, _, _ in d.received] == [10]


def test_frame_granularity():
    d = Collector()
    d.dispatch_granularity = "frame"
    d.bind(meta())
    d.submit(frames(3))
    assert wait_until(lambda: len(d.received) == 3)
    d.shutdown()
    assert [idx for idx, _, _ in d.received] == [0, 1, 2]


def test_shutdown_drains_pending_and_is_idempotent():
    d = Collector()
    d.bind(meta())
    d.submit(frames(2))
    d.submit(frames(2))
    d.shutdown()
    assert [idx for idx, _, _ in d.received] == [0, 2]
    d.shutdown()
    assert d.closed == 1
    assert not d.submit(frames(2))


def test_stop_sets_pipeline_event():
    stop = threading.Event()
    d = Collector()
    d.bind(meta(), stop_event=stop)
    d.stop()
    assert stop.is_set()
    d.shutdown()


def test_frame_queue_adapter():
    async def run():
        loop = asyncio.get_running_loop()
        q = FrameQueue(loop, maxsize=2)
        q.bind(meta(fps=10.0))
        assert q.submit(frames(3))
        idx, data, pts = await asyncio.wait_for(q.get(), 1.0)
        assert idx == 0
        assert data.shape == (3, 4, 4, 3)
        assert pts == 0.0
        q.shutdown()

    asyncio.run(run())
