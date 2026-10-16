import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any


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
        self.queue_counter: int = 0
        self.progress: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_concurrent_requests: int = max_concurrent_requests
        self.max_workers: int = max_workers
        self.executor: ThreadPoolExecutor | None = None
        self.request_semaphore: asyncio.Semaphore | None = None
        self.inference_lock: asyncio.Lock | None = None
        # True while background boot warmup runs. /generate answers 503 until
        # it clears so a real request never contends with warmup on one GPU.
        self.warming_up: bool = False
        self.performance_stats: dict[str, Any] = {
            "total_requests": 0,
            "total_inference_time": 0.0,
            "total_preprocess_time": 0.0,
            "total_postprocess_time": 0.0,
            "average_fps": 0.0,
        }

    def initialize_runtime(self):
        """Initialize async primitives and executors."""
        if self.request_semaphore is None:
            self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        if self.inference_lock is None:
            self.inference_lock = asyncio.Lock()
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

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

    def clear_progress(self, request_id: str) -> None:
        self.progress.pop(request_id, None)

    def shutdown(self):
        """Release resources on application shutdown."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)


# Global singleton instance
app_state = AppState()
