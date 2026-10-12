from __future__ import annotations

import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Any, Awaitable, Protocol
import torch
import numpy as np


class RequestStatus(IntEnum):
    """Lifecycle states of a streaming video generation request."""
    WAITING = 0
    RUNNING = 1
    FINISHED_COMPLETED = 2
    FINISHED_ABORTED = 3
    FINISHED_ERROR = 4

    @staticmethod
    def is_terminal(status: RequestStatus) -> bool:
        return status >= RequestStatus.FINISHED_COMPLETED


class ClipCallback(Protocol):
    """Protocol for per-request streaming chunk delivery."""

    def __call__(
        self,
        request_id: str,
        clip_idx: int,
        clip: Union[torch.Tensor, np.ndarray],
        is_last: bool,
    ) -> Union[None, Awaitable[None]]:
        ...


@dataclass
class FloatOmniRequest:
    """State tracking for a single active generation stream in the Omni pipeline."""

    request_id: str
    ref_image: torch.Tensor
    audio: torch.Tensor
    ref_latent: torch.Tensor
    ref_feats: list[torch.Tensor]
    wa: torch.Tensor
    we: torch.Tensor
    total_chunks: int
    total_frames: int = 0         # exact number of video frames to generate (for last-chunk trim)
    current_chunk: int = 0
    decoded_frames: int = 0       # running count of frames already emitted
    callback: Optional[Any] = None
    status: RequestStatus = RequestStatus.WAITING
    error: Optional[str] = None
    prev_x: Optional[torch.Tensor] = None
    prev_wa: Optional[torch.Tensor] = None
    wa_t_last: Optional[torch.Tensor] = None
    a_cfg_scale: float = 2.0
    r_cfg_scale: float = 1.0
    e_cfg_scale: float = 1.0
    nfe: int = 10
    created_at: float = field(default_factory=time.time)
    output_format: str = "tensor"  # 'tensor' (torch.Tensor CPU uint8) or 'numpy' (np.ndarray uint8)

    @property
    def is_finished(self) -> bool:
        return RequestStatus.is_terminal(self.status)

    @property
    def is_last_chunk(self) -> bool:
        return (self.current_chunk + 1) >= self.total_chunks


@dataclass
class LatentChunkTask:
    """Cross-stage payload passed from the ODE Sampler to the StyleDecoder."""

    request_id: str
    chunk_idx: int
    latents: torch.Tensor
    ref_feats: list[torch.Tensor]
    is_last: bool
    callback: Optional[Any]
    output_format: str = "tensor"
    enqueue_time: float = field(default_factory=time.time)
