"""stream_utils.py - Queue and frame streaming utilities for FLOAT video pipelines."""

import queue
import threading
from dataclasses import dataclass

import numpy as np

_ABORT = object()


def _qget(q: queue.Queue, stop: threading.Event, timeout: float = 0.1):
    """Retrieve an item from a queue with periodic timeout checks on a stop event.

    Returns the item or _ABORT if the stop event was set.
    """
    while not stop.is_set():
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            pass
    return _ABORT


@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer."""

    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False
