"""decode_tuner.py - Auto-tuner for GPU video decode micro-batch size and streams."""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

GiB = 1024**3

# The tuning sweep runs the real decoder at batch 1/2/4/8/16 and costs ~30 s on
# an A6000, entirely before the first request. The answer is a property of the
# box (GPU + torch/CUDA + frame shape + tuner settings), so persist it and only
# re-measure when one of those changes or when VRAM is tighter than last time.
#   FLOAT_DECODE_TUNE_CACHE   path to the json file, or 0/false/off to disable
CACHE_ENV = "FLOAT_DECODE_TUNE_CACHE"


def cache_path() -> str | None:
    """Location of the persisted tuning profiles (``None`` when disabled)."""
    setting = os.getenv(CACHE_ENV, "").strip()
    if setting.lower() in {"0", "off", "false", "no"}:
        return None
    if setting:
        return os.path.expanduser(setting)
    base = os.getenv("FLOAT_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "fast_float")
    return os.path.join(base, "decode_tune.json")


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

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _profile_to_dict(prof: DecodeProfile) -> dict:
        return {
            "batch_size": prof.batch_size,
            "transient_bytes": prof.transient_bytes,
            "height": prof.height,
            "width": prof.width,
            "channels": prof.channels,
            "transient_by_batch": {str(k): int(v) for k, v in prof.transient_by_batch.items()},
            "fps_by_batch": {str(k): float(v) for k, v in prof.fps_by_batch.items()},
        }

    @staticmethod
    def _profile_from_dict(entry: dict) -> DecodeProfile | None:
        try:
            return DecodeProfile(
                batch_size=int(entry["batch_size"]),
                transient_bytes=int(entry["transient_bytes"]),
                height=int(entry["height"]),
                width=int(entry["width"]),
                channels=int(entry["channels"]),
                transient_by_batch={int(k): int(v) for k, v in entry["transient_by_batch"].items()},
                fps_by_batch={int(k): float(v) for k, v in entry.get("fps_by_batch", {}).items()},
            )
        except (KeyError, TypeError, ValueError):
            return None

    def load_cached(self, key: str) -> DecodeProfile | None:
        """Read a previously measured profile for ``key`` (``None`` on a miss)."""
        path = cache_path()
        if not path or not key:
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        entry = data.get(key) if isinstance(data, dict) else None
        prof = self._profile_from_dict(entry) if isinstance(entry, dict) else None
        if prof is None or not 1 <= prof.batch_size <= self.max_batch:
            return None
        if not prof.transient_by_batch:
            return None
        return prof

    def save_cached(self, key: str, prof: DecodeProfile) -> None:
        """Persist ``prof`` for ``key``; best-effort (a read-only FS is fine)."""
        path = cache_path()
        if not path or not key:
            return
        data: dict = {}
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
        data[key] = self._profile_to_dict(prof)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, sort_keys=True)
            os.replace(tmp, path)
        except OSError:
            pass

    def fit_cached(self, prof: DecodeProfile) -> DecodeProfile | None:
        """Adapt a cached profile to the VRAM available *now*.

        The cached answer was measured on a possibly idle GPU; if something else
        now holds VRAM, drop to the largest measured batch that still fits. The
        tuner never *raises* the batch above what it measured, because the
        plateau rule deliberately picks the smallest near-peak size.
        """
        budget = self._budget()
        fits = {b: t for b, t in prof.transient_by_batch.items() if b <= self.max_batch and t * self.safety <= budget}
        if not fits:
            return None
        if prof.batch_size in fits:
            chosen = prof.batch_size
        else:
            chosen = (
                max(b for b in fits if b <= prof.batch_size) if any(b <= prof.batch_size for b in fits) else min(fits)
            )
        return DecodeProfile(
            chosen,
            fits[chosen],
            prof.height,
            prof.width,
            prof.channels,
            prof.transient_by_batch,
            prof.fps_by_batch,
        )

    @torch.no_grad()
    def tune(self, decode_fn: Callable[[int], torch.Tensor], cache_key: str | None = None) -> DecodeProfile:
        """decode_fn(b) -> uint8 [b, H, W, C] tensor on self.device (no OOM fallback inside).

        With ``cache_key`` a matching profile from a previous process is reused
        (validated against the current free VRAM) instead of re-running the
        sweep, which is the difference between a ~2 s and a ~35 s boot warmup.
        """
        if cache_key:
            cached = self.load_cached(cache_key)
            if cached is not None:
                fitted = self.fit_cached(cached)
                if fitted is not None:
                    if self.verbose:
                        print(
                            f"[TUNER {self.device}] cached -> batch={fitted.batch_size}, "
                            f"{fitted.transient_bytes / GiB:.2f} GiB per in-flight batch "
                            f"(set {CACHE_ENV}=0 to re-measure)"
                        )
                    return fitted
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
            # A transient VRAM spike (another process still releasing its
            # context, a training job) can empty this set even though the
            # measurements show the batches clearly fit. The old fallback then
            # picked batch 1, the slowest possible decode (0.5 vs ~180 frames/s),
            # and — worse — that answer was persisted for every later boot.
            degraded = not feasible
            if degraded:
                # Every batch in ``results`` was actually executed during the
                # sweep, so its transient was allocatable then; only this final
                # budget re-read collapsed. Pick by the normal plateau rule
                # instead of falling back to batch 1 (0.5 vs ~180 frames/s).
                best = max(f for _, f in results.values())
                k = min(b for b, (_, f) in results.items() if f >= self.plateau * best)
                feasible = {k: results[k]}
                if self.verbose:
                    gib = {b: round(v[0] / GiB, 2) for b, v in results.items()}
                    print(
                        f"[TUNER {self.device}] WARNING: nothing fits the current VRAM budget "
                        f"({budget / GiB:.1f} GiB; measured {gib}); using batch={k} and NOT caching "
                        f"this result. Re-tune with FLOAT_DECODE_TUNE_CACHE=0 once VRAM is free."
                    )

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
                gib = {k: round(v / GiB, 2) for k, v in prof.transient_by_batch.items()}
                print(
                    f"[TUNER {self.device}] frames/s by batch {prof.fps_by_batch} -> batch={chosen}, "
                    f"{prof.transient_bytes / GiB:.2f} GiB per in-flight batch "
                    f"(gib/batch {gib}, budget {budget / GiB:.1f} GiB)"
                )
            if cache_key and not degraded:
                self.save_cached(cache_key, prof)
            return prof

    def num_streams(self, prof: DecodeProfile, batch: int | None = None) -> int:
        t = prof.transient_by_batch.get(batch, prof.transient_bytes) if batch else prof.transient_bytes
        per_stream = max(1, int(t * self.safety))
        with torch.cuda.device(self.device):
            n = self._budget() // per_stream
        return int(max(1, min(self.max_streams, n)))
