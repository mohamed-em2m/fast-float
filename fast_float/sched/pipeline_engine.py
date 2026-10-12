from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from typing import Optional, Any, Callable, Union
import torch
import numpy as np

from fast_float.sched.types import FloatOmniRequest, LatentChunkTask, RequestStatus
from fast_float.sched.omni_scheduler import FloatOmniScheduler

logger = logging.getLogger(__name__)


class FloatOmniPipelineEngine:
    """Decoupled asynchronous pipeline engine coordinating the ODE Sampler,

    cross-stage Latent Queue, StyleDecoder workers, and Output Dispatcher.

    Architecture:
      Client Request -> Scheduler -> [Sampler Worker Loop]
                                            | (LatentChunkTask)
                                            v
                                   [Latent Queue (Async)]
                                            |
                                            v
                                   [Decoder Worker Loop]
                                            |
                                            v
                             [Output Callback Dispatcher]
    """

    def __init__(
        self,
        model: Any,
        max_running_requests: int = 8,
        latent_queue_size: int = 16,
        num_decoder_workers: int = 1,
    ):
        self.model = model
        self.scheduler = FloatOmniScheduler(max_running_requests=max_running_requests)
        self.latent_queue: Optional[asyncio.Queue[LatentChunkTask]] = None
        self.latent_queue_size = latent_queue_size
        self.num_decoder_workers = num_decoder_workers

        self._running = False
        self._sampler_task: Optional[asyncio.Task] = None
        self._decoder_tasks: list[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Dedicated single-threaded executor for GPU work (Bug 7):
        # Serialises all CUDA kernel dispatches on one OS thread so CUDA context
        # synchronisation is never blocked by ThreadPoolExecutor contention.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="float-gpu"
        )

    async def start(self):
        """Starts the background sampler and decoder worker loops."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self.latent_queue = asyncio.Queue(maxsize=self.latent_queue_size)
        self._running = True

        self._sampler_task = asyncio.create_task(self._sampler_loop(), name="omni-sampler-worker")
        self._decoder_tasks = [
            asyncio.create_task(self._decoder_loop(worker_id=i), name=f"omni-decoder-worker-{i}")
            for i in range(self.num_decoder_workers)
        ]
        logger.info(
            f"✓ FloatOmniPipelineEngine started (1 Sampler, {self.num_decoder_workers} Decoders, "
            f"queue capacity={self.latent_queue_size})"
        )

    async def stop(self):
        """Gracefully terminates worker loops."""
        self._running = False
        if self._sampler_task:
            self._sampler_task.cancel()
        for task in self._decoder_tasks:
            task.cancel()
        self._sampler_task = None
        self._decoder_tasks.clear()
        self._gpu_executor.shutdown(wait=False)
        logger.info("FloatOmniPipelineEngine stopped")

    async def submit_request(
        self,
        ref_image: torch.Tensor,
        audio: torch.Tensor,
        emotion: Optional[str] = None,
        request_id: Optional[str] = None,
        callback: Optional[Callable[[str, int, Union[torch.Tensor, np.ndarray], bool], Any]] = None,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        nfe: int = 10,
        output_format: str = "tensor",
    ) -> str:
        """Pre-extracts invariant conditions and enqueues request into the continuous scheduler."""
        if not self._running:
            await self.start()

        # Run invariant feature prefill (Wav2Vec2 + image identity) in threadpool to keep event loop free
        req: FloatOmniRequest = await asyncio.to_thread(
            self.model.prepare_omni_conditions,
            ref_image=ref_image,
            audio=audio,
            emotion=emotion,
            request_id=request_id,
            callback=callback,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            nfe=nfe,
            output_format=output_format,
        )

        self.scheduler.add_request(req)
        return req.request_id

    def cancel_request(self, request_id: str, reason: str = "Client cancelled") -> bool:
        """Signals the scheduler to immediately drop subsequent chunks for request_id."""
        return self.scheduler.abort_request(request_id, reason=reason)

    # ---------------- Worker Loops ---------------- #

    async def _sampler_loop(self):
        """Worker loop that continuously executes ODE Flow Matching sampling for scheduled chunks."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                # 1. Schedule next chunk across all active requests
                req = self.scheduler.schedule_next_chunk()
                if req is None:
                    # No active requests; sleep briefly to yield control
                    await asyncio.sleep(0.005)
                    continue

                if req.status == RequestStatus.FINISHED_ABORTED:
                    continue

                # 2. Execute one ODE chunk in the dedicated GPU thread (Bug 7)
                task: LatentChunkTask = await loop.run_in_executor(
                    self._gpu_executor, self.model.sample_omni_chunk, req
                )

                # Check if aborted while sampling was running
                if req.status == RequestStatus.FINISHED_ABORTED:
                    continue

                # 3. Push to cross-stage latent queue
                assert self.latent_queue is not None
                await self.latent_queue.put(task)
                # NOTE: FINISHED_COMPLETED is now set by the decoder loop AFTER
                # the callback fires for the last chunk (Bug 6 fix).

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Sampler] Error during chunk sampling: {e}", exc_info=True)
                await asyncio.sleep(0.01)

    async def _decoder_loop(self, worker_id: int):
        """Worker loop that synthesizes RGB pixel frames from latent tasks."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                assert self.latent_queue is not None
                task = await self.latent_queue.get()

                # Check if request was aborted while waiting in the decoder queue
                req = self.scheduler.get_request(task.request_id)
                if req is not None and req.status == RequestStatus.FINISHED_ABORTED:
                    self.latent_queue.task_done()
                    continue

                # Execute StyleDecoder synthesis in the dedicated GPU thread (Bug 7)
                clip = await loop.run_in_executor(
                    self._gpu_executor, self.model.decode_omni_chunk, task
                )

                # Invoke per-request callback if attached
                if task.callback is not None:
                    try:
                        res = task.callback(
                            task.request_id, task.chunk_idx, clip, task.is_last
                        )
                        if inspect.isawaitable(res):
                            await res
                    except Exception as cb_err:
                        logger.warning(
                            f"[Decoder {worker_id}] Callback error on {task.request_id} chunk {task.chunk_idx}: {cb_err}"
                        )

                # Bug 6 fix: mark COMPLETED only after callback fires for the last chunk
                if task.is_last and req is not None and req.status == RequestStatus.RUNNING:
                    req.status = RequestStatus.FINISHED_COMPLETED
                    logger.info(f"[Decoder {worker_id}] Completed final chunk for {task.request_id}")

                self.latent_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Decoder {worker_id}] Error during decoding: {e}", exc_info=True)
                await asyncio.sleep(0.01)
