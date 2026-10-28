import asyncio
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fast_float.core.perf_metrics import RequestAccumulator, RequestSample


class AppState:
    """Shared application state containing runtime models, locks, and counters."""

    # Cap on retained progress entries so a long-lived server doesn't grow
    # the registry unbounded (progress is only needed while a request runs).
    MAX_PROGRESS_ENTRIES = 200

    def __init__(self, max_concurrent_requests: int = 10, max_workers: int = 2):
        # NOTE: inference is serialized by inference_lock (one GPU), so a
        # large thread pool only wastes RAM. 2 workers cover inference +
        # lifespan warmup overlap.
        self.agent: Any | None = None
        self.opt: Any | None = None
        self.video_optimizer: Any | None = None
        self.public_url: str | None = None
        self.public_backup_url: str | None = None
        self.pcs: set[Any] = set()
        self.active_streams: dict[str, str] = {}
        # UDP port of the built-in STUN responder, or None when it is disabled
        # or failed to bind (see fast_float/utils/stun.py).
        self.stun_port: int | None = None
        # Live (in-flight) streams keyed by request id; consumed by the
        # WebSocket and WebRTC transports.
        self.live_streams: dict[str, Any] = {}
        self.queue_counter: int = 0
        self.progress: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_concurrent_requests: int = max_concurrent_requests
        self.max_workers: int = max_workers
        self.executor: ThreadPoolExecutor | None = None
        self.request_semaphore: asyncio.Semaphore | None = None
        self.inference_lock: asyncio.Lock | None = None
        # True while background boot warmup runs. Requests that arrive while it
        # is set wait for ``warmup_event`` (see wait_for_warmup) instead of
        # being rejected, so a client that fires one request right after a
        # restart is served rather than 503'd for the whole warmup window.
        self.warming_up: bool = False
        self.warmup_event: asyncio.Event | None = None
        self.warmup_stage: str = ""
        self.warmup_started_at: float = 0.0
        self.warmup_finished_at: float = 0.0
        self.warmup_error: str | None = None
        self.performance_stats: dict[str, Any] = {
            "total_requests": 0,
            "total_inference_time": 0.0,
            "total_preprocess_time": 0.0,
            "total_postprocess_time": 0.0,
            "average_fps": 0.0,
        }
        # Rolling window of per-request measurements (model fps, e2e fps, RTF,
        # TTFF, latency percentiles, jitter, drops). Bounded, so a long-lived
        # server does not grow it without limit.
        self.perf_requests: RequestAccumulator = RequestAccumulator(maxlen=200)
        # Real-time factor measured by the live (IPC) avatar renderer: render
        # seconds per second of audio. ``None`` until the first chunk finishes.
        # Used by GET /gpu to estimate the sustainable stream count.
        self.live_rtf: float | None = None

    def record_request_sample(self, sample: RequestSample) -> None:
        """Add one finished request to the rolling performance window."""
        self.perf_requests.add(sample)

    def initialize_runtime(self):
        """Initialize async primitives and executors."""
        if self.request_semaphore is None:
            self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        if self.inference_lock is None:
            self.inference_lock = asyncio.Lock()
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    # ------------------------------------------------------------- boot warmup
    def begin_warmup(self) -> None:
        """Mark the boot warmup as running (call from inside the event loop)."""
        self.warmup_event = asyncio.Event()
        self.warmup_started_at = time.time()
        self.warmup_finished_at = 0.0
        self.warmup_stage = "starting"
        self.warmup_error = None
        self.warming_up = True

    def set_warmup_stage(self, stage: str) -> None:
        self.warmup_stage = stage

    def finish_warmup(self, error: str | None = None) -> None:
        """Clear ``warming_up`` and release every waiting request.

        Always called from the warmup task's ``finally``: a failed warmup is
        not fatal (sequential inference still works), but leaving the flag set
        would 503 the server forever.
        """
        self.warming_up = False
        self.warmup_stage = "failed" if error else "done"
        self.warmup_error = error
        self.warmup_finished_at = time.time()
        if self.warmup_event is not None:
            self.warmup_event.set()

    def warmup_elapsed(self) -> float:
        """Seconds since warmup began, frozen once it finished (health metric)."""
        if not self.warmup_started_at:
            return 0.0
        end = self.warmup_finished_at or time.time()
        return max(0.0, end - self.warmup_started_at)

    async def wait_for_warmup(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for warmup to finish.

        Returns ``True`` when the server is open (or was never closed), and
        ``False`` on timeout. Requests used to hard-fail with a 503 during
        warmup, which surfaced to users as "Model is warming up, try again
        shortly" on *every* call: a retrying client was the only way in.
        """
        if not self.warming_up:
            return True
        event = self.warmup_event
        if event is None:
            # begin_warmup() hasn't run yet (or we're on another loop): don't
            # block a request on a signal that can never arrive.
            return not self.warming_up
        try:
            await asyncio.wait_for(event.wait(), timeout=max(0.0, timeout))
            return True
        except asyncio.TimeoutError:
            return not self.warming_up

    def set_progress(self, request_id: str, **fields: Any) -> None:
        """Create or update the progress record for a request (LRU-capped)."""
        record = self.progress.get(request_id)
        if record is None:
            record = {"request_id": request_id}
            self.progress[request_id] = record
        record.update(fields)
        self.progress.move_to_end(request_id)
        while len(self.progress) > self.MAX_PROGRESS_ENTRIES:
            self.progress.popitem(last=False)

    def get_progress(self, request_id: str) -> dict[str, Any] | None:
        record = self.progress.get(request_id)
        return dict(record) if record is not None else None

    async def wait_for_live_stream(self, request_id: str, timeout: float = 15.0) -> Any | None:
        """Wait for a live stream to register (the POST and the WS race)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            stream = self.live_streams.get(request_id)
            if stream is not None:
                return stream
            await asyncio.sleep(0.05)
        return None

    def clear_progress(self, request_id: str) -> None:
        self.progress.pop(request_id, None)

    def shutdown(self):
        """Release resources on application shutdown."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)


# Global singleton instance
app_state = AppState()
