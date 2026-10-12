from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Dict, List
import threading

from fast_float.sched.types import FloatOmniRequest, RequestStatus

logger = logging.getLogger(__name__)


class FloatOmniScheduler:
    """Fair-Share Round-Robin chunk scheduler for multi-stream diffusion inference.

    Inspired by vLLM-Omni's StepScheduler:
    - Slices video generation into fine-grained chunks (e.g. 50 frames / 2 seconds).
    - Interleaves chunk scheduling across all concurrent requests so that every user receives
      their first video chunk rapidly (TTFF < 500ms) without waiting behind long-duration jobs.
    - Supports immediate cancellation/aborts to prevent wasting GPU compute when clients disconnect.
    """

    def __init__(self, max_running_requests: int = 8):
        self.max_running_requests = max_running_requests
        self._lock = threading.Lock()
        self._requests: Dict[str, FloatOmniRequest] = {}
        self._waiting: deque[str] = deque()
        self._running: List[str] = []
        self._step_counter: int = 0

    def add_request(self, request: FloatOmniRequest) -> str:
        """Enqueue a new generation request into the scheduler."""
        with self._lock:
            self._requests[request.request_id] = request
            request.status = RequestStatus.WAITING
            self._waiting.append(request.request_id)
            logger.info(
                f"[Scheduler] Added request {request.request_id} "
                f"({request.total_chunks} chunks). Waiting: {len(self._waiting)}, Running: {len(self._running)}"
            )
        return request.request_id

    def abort_request(self, request_id: str, reason: str = "Client disconnected") -> bool:
        """Cancel an in-flight or waiting request immediately."""
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.is_finished:
                return False
            req.status = RequestStatus.FINISHED_ABORTED
            req.error = reason
            logger.info(f"[Scheduler] Aborted request {request_id}: {reason}")
            return True

    def get_request(self, request_id: str) -> Optional[FloatOmniRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def schedule_next_chunk(self) -> Optional[FloatOmniRequest]:
        """Pulls the next request eligible for one ODE latent chunk generation.

        Implements Round-Robin Fair Share:
        1. Purges terminal requests from the running pool.
        2. Admits waiting requests up to max_running_requests.
        3. Rotates the running pool so each active stream gets an interleaved turn.
        If a popped request was concurrently aborted, iterates to the next instead
        of returning None (avoids starving the pipeline on burst aborts).
        """
        with self._lock:
            # 1. Clean up running pool
            active_running: List[str] = []
            for req_id in self._running:
                req = self._requests.get(req_id)
                if req is not None and req.status == RequestStatus.RUNNING:
                    active_running.append(req_id)
            self._running = active_running

            # 2. Admit waiting requests
            while self._waiting and len(self._running) < self.max_running_requests:
                candidate_id = self._waiting.popleft()
                candidate = self._requests.get(candidate_id)
                if candidate is not None and candidate.status == RequestStatus.WAITING:
                    candidate.status = RequestStatus.RUNNING
                    self._running.append(candidate_id)
                    logger.debug(f"[Scheduler] Admitted request {candidate_id} to RUNNING pool")

            if not self._running:
                return None

            # 3. Round-robin rotation — skip aborted requests iteratively (no recursion)
            attempts = len(self._running)
            for _ in range(attempts):
                selected_id = self._running.pop(0)
                selected_req = self._requests.get(selected_id)

                if selected_req is None or selected_req.status != RequestStatus.RUNNING:
                    # This slot was aborted between step-1 cleanup and now; skip it
                    continue

                # Re-enqueue in running pool if it has more chunks remaining
                if selected_req.current_chunk + 1 < selected_req.total_chunks:
                    self._running.append(selected_id)

                self._step_counter += 1
                return selected_req

            # All candidates were aborted
            return None

    @property
    def num_waiting(self) -> int:
        with self._lock:
            return len(self._waiting)

    @property
    def num_running(self) -> int:
        with self._lock:
            return len(self._running)

    @property
    def total_active(self) -> int:
        with self._lock:
            return len(self._waiting) + len(self._running)
