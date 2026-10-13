import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Set, Any


class AppState:
    """Shared application state containing runtime models, locks, and counters."""

    def __init__(self, max_concurrent_requests: int = 10, max_workers: int = 2):
        # NOTE: inference is serialized by inference_lock (one GPU), so a
        # large thread pool only wastes RAM. 2 workers cover inference +
        # lifespan warmup overlap.
        self.agent: Optional[Any] = None
        self.opt: Optional[Any] = None
        self.video_optimizer: Optional[Any] = None
        self.public_url: Optional[str] = None
        self.public_backup_url: Optional[str] = None
        self.pcs: Set[Any] = set()
        self.active_streams: Dict[str, str] = {}
        self.queue_counter: int = 0
        self.max_concurrent_requests: int = max_concurrent_requests
        self.max_workers: int = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.request_semaphore: Optional[asyncio.Semaphore] = None
        self.inference_lock: Optional[asyncio.Lock] = None
        self.performance_stats: Dict[str, Any] = {
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

    def shutdown(self):
        """Release resources on application shutdown."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)


# Global singleton instance
app_state = AppState()
