"""decode_tuner.py - Auto-tuner for GPU video decode micro-batch size and streams."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

GiB = 1024**3


@dataclass
class DecodeProfile:
    batch_size: int
    transient_bytes: int  # peak extra GPU bytes of ONE batch at batch_size
    height: int
    width: int
    channels: int
    transient_by_batch: dict[int, int] = field(default_factory=dict)
    fps_by_batch: dict[int, float] = field(default_factory=dict)


class DecodeAutoTuner:
    """Auto-tuner (one instance per GPU) to select optimal micro-batch size and stream count."""

    def __init__(
        self,
        device,
        primary=True,
        max_batch=16,
        max_streams=1,
        reserve_gib=None,
        plateau=0.93,
        safety=1.3,
        repeats=1,
        verbose=True,
        opt=None,
    ):
        self.device = device
        self.primary = primary  # primary GPU also hosts sampler + NVENC -> bigger reserve
        self.max_batch = max_batch
        self.max_streams = max_streams
        self.reserve_gib = reserve_gib
        self.plateau = plateau  # "good enough" fraction of best throughput
        self.safety = safety  # headroom for allocator fragmentation
        self.repeats = repeats
        self.verbose = verbose
        self.opt = opt

    def _budget(self) -> int:
        """Bytes we may spend on decode streams right now (call with this device current)."""
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        if self.reserve_gib is not None:
            reserve = int(self.reserve_gib * GiB)
        elif self.primary:
            reserve = max(int(0.8 * GiB), int(0.08 * total))  # sampler + NVENC + context
        else:
            reserve = max(int(0.35 * GiB), int(0.03 * total))  # decode only
        return int(free) - reserve

    def _measure(self, fn: Callable[[int], torch.Tensor], b: int):
        out = fn(b)  # warm-up (cudnn.benchmark picks algos)
        torch.cuda.synchronize(self.device)
        del out
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(self.device)
        best_dt, peak, shape = float("inf"), 0, None
        for _ in range(self.repeats):
            torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.perf_counter()
            out = fn(b)
            torch.cuda.synchronize(self.device)
            best_dt = min(best_dt, time.perf_counter() - t0)
            peak = max(peak, torch.cuda.max_memory_allocated(self.device))
            shape = tuple(out.shape[1:])  # (H, W, C)
            del out
        return peak - base, b / best_dt, shape

    @torch.no_grad()
    def tune(self, decode_fn: Callable[[int], torch.Tensor]) -> DecodeProfile:
        """decode_fn(b) -> uint8 [b, H, W, C] tensor on self.device (no OOM fallback inside)."""
        with torch.cuda.device(self.device):
            results, shape, b = {}, None, 1
            while b <= self.max_batch:
                if results:
                    prev = max(results)
                    projected = results[prev][0] * (b / prev) * self.safety
                    if projected > self._budget():
                        break
                try:
                    transient, fps, shape = self._measure(decode_fn, b)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                results[b] = (transient, fps)
                b *= 2
            if not results:
                raise RuntimeError(f"{self.device}: decoding even 1 frame does not fit in free VRAM")

            opt_bs = getattr(self.opt, "decode_batch_size", None) if self.opt else None
            if opt_bs is not None and int(opt_bs) > 0:
                req_bs = int(opt_bs)
                if req_bs not in results and req_bs <= self.max_batch:
                    # Explicit user override not visited during powers-of-two loop: measure it directly
                    try:
                        transient, fps, s = self._measure(decode_fn, req_bs)
                        results[req_bs] = (transient, fps)
                        if shape is None:
                            shape = s
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

            budget = self._budget()
            feasible = {k: v for k, v in results.items() if v[0] * self.safety <= budget}
            if not feasible:
                k = min(results)
                feasible = {k: results[k]}

            if opt_bs is not None and int(opt_bs) > 0:
                req_bs = int(opt_bs)
                if req_bs in feasible:
                    chosen = req_bs
                else:
                    # Fallback to closest feasible batch size smaller than or equal to req_bs
                    candidates = [k for k in feasible if k <= req_bs]
                    chosen = max(candidates) if candidates else min(feasible)
            else:
                best = max(f for _, f in feasible.values())
                # Choose the smallest batch size that achieves plateau throughput
                # to minimize VRAM usage, lower latency, and prevent cuDNN grouped-conv alignment issues.
                chosen = min(k for k, (_, f) in feasible.items() if f >= self.plateau * best)

            h, w, c = shape
            prof = DecodeProfile(
                chosen,
                results[chosen][0],
                h,
                w,
                c,
                {k: v[0] for k, v in results.items()},
                {k: round(v[1], 1) for k, v in results.items()},
            )
            if self.verbose:
                print(
                    f"[TUNER {self.device}] frames/s by batch {prof.fps_by_batch} -> batch={chosen}, "
                    f"{prof.transient_bytes / GiB:.2f} GiB per in-flight batch"
                )
            return prof

    def num_streams(self, prof: DecodeProfile, batch: int | None = None) -> int:
        t = prof.transient_by_batch.get(batch, prof.transient_bytes) if batch else prof.transient_bytes
        per_stream = max(1, int(t * self.safety))
        with torch.cuda.device(self.device):
            n = self._budget() // per_stream
        return int(max(1, min(self.max_streams, n)))
