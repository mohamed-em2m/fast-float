"""Stream decode auto-tuner and multi-GPU streaming video saver pipeline for FLOAT.

Contains:
  - DecodeProfile: Profiling statistics for decoder batch sizes and VRAM transients
  - DecodeAutoTuner: Per-GPU auto-tuner optimizing batch sizes and stream counts
  - _qget: Thread-safe queue polling with cooperative cancellation
  - StreamingVideoSaver: Mixin coordinating ODE sampling (producer), parallel decode
    workers (multi-GPU / multi-stream), and real-time video encoders (FFmpeg NVENC / PyAV / Torchcodec).
"""

from __future__ import annotations

import copy
import contextlib
import logging
import queue
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import torch

from fast_float.models.float.generator import Generator

logger = logging.getLogger(__name__)
GiB = 1024**3
_ABORT = object()


@dataclass
class DecodeProfile:
    batch_size: int
    transient_bytes: int  # peak extra GPU bytes of ONE batch at batch_size
    height: int
    width: int
    channels: int
    transient_by_batch: Dict[int, int] = field(default_factory=dict)
    fps_by_batch: Dict[int, float] = field(default_factory=dict)


class DecodeAutoTuner:
    """Auto-tuner for optimizing GPU memory allocation and batch decoding parameters."""

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
        self.primary = primary
        self.max_batch = max_batch
        self.max_streams = max_streams
        self.reserve_gib = reserve_gib
        self.plateau = plateau
        self.safety = safety
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
            reserve = max(int(0.8 * GiB), int(0.08 * total))
        else:
            reserve = max(int(0.35 * GiB), int(0.03 * total))
        return int(free) - reserve

    def _measure(self, fn: Callable[[int], torch.Tensor], b: int):
        out = fn(b)
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
            shape = tuple(out.shape[1:])
            del out
        return peak - base, b / best_dt, shape

    @torch.no_grad()
    def tune(self, decode_fn: Callable[[int], torch.Tensor]) -> DecodeProfile:
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
                raise RuntimeError(
                    f"{self.device}: decoding even 1 frame does not fit in free VRAM"
                )

            budget = self._budget()
            feasible = {
                k: v for k, v in results.items() if v[0] * self.safety <= budget
            }
            if not feasible:
                k = min(results)
                feasible = {k: results[k]}

            opt_bs = getattr(self.opt, "decode_batch_size", None) if self.opt else None
            if opt_bs is not None and int(opt_bs) > 0:
                req_bs = int(opt_bs)
                if req_bs in results:
                    chosen = req_bs
                else:
                    chosen = min(req_bs, max(feasible.keys()))
            else:
                best = max(f for _, f in feasible.values())
                chosen = min(
                    k for k, (_, f) in feasible.items() if f >= self.plateau * best
                )

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

    def num_streams(self, prof: DecodeProfile, batch: int = None) -> int:
        t = (
            prof.transient_by_batch.get(batch, prof.transient_bytes)
            if batch
            else prof.transient_bytes
        )
        per_stream = max(1, int(t * self.safety))
        with torch.cuda.device(self.device):
            n = self._budget() // per_stream
        return int(max(1, min(self.max_streams, n)))


def _qget(q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return _ABORT


def _torchcodec_usable() -> bool:
    """True only if torchcodec imports AND its native libs load.

    A binary-incompatible install raises OSError at import (broken .so,
    e.g. built for a newer torch) — not just ImportError — so catch
    broadly. Never raises.
    """
    try:
        from fast_float.utils.pyav_writer import is_torchcodec_available
    except ImportError:
        try:
            from pyav_writer import is_torchcodec_available
        except ImportError:
            return False
    try:
        return bool(is_torchcodec_available())
    except Exception:
        return False


class StreamingVideoSaver:
    """Mixin for multi-stream, multi-GPU decode execution and video encoding pipeline."""

    def _cuda_device(self) -> torch.device:
        r = getattr(self.opt, "rank", 0)
        if not torch.cuda.is_available():
            return torch.device("cpu")
        dev = torch.device(f"cuda:{r}") if isinstance(r, int) else torch.device(r)
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        return dev

    def _decode_devices(self):
        primary = self._cuda_device()
        visible = torch.cuda.device_count()
        want = max(1, int(getattr(self.opt, "ngpus", 1)))
        n = min(want, visible)
        p_idx = primary.index if primary.index is not None else 0
        if want > visible:
            logger.warning(
                f"[MULTI-GPU] ngpus={want} requested but only {visible} visible; using {n}"
            )
        else:
            logger.info(
                f"[MULTI-GPU] using {n}/{visible} GPU(s) (primary=cuda:{p_idx})"
            )
        others = [
            torch.device("cuda", i) for i in range(visible) if i != primary.index
        ][: max(0, n - 1)]
        devs = [primary] + others
        if n > 1 and not getattr(self.opt, "decode_on_primary", True):
            devs = others
        return primary, devs

    def _get_ae(self, dev, primary):
        """Motion autoencoder on dev: original on primary GPU, cached replicas elsewhere.

        Avoids copy.deepcopy on torch.compile()'d modules (which breaks due to _orig_mod
        wrappers) by always using state_dict() + fresh Generator construction.
        """
        if dev == primary or (dev.index is not None and primary.index is not None and dev.index == primary.index):
            return self.motion_autoencoder
        reps = getattr(self, "_ae_replicas", None)
        if reps is None:
            reps = self._ae_replicas = {}
        idx = dev.index if dev.index is not None else str(dev)
        if idx not in reps:
            # Build a clean (uncompiled) Generator replica and transfer weights.
            # We cannot use copy.deepcopy on torch.compile()'d modules - the _orig_mod
            # wrapper raises errors and the state_dict keys will have _orig_mod. prefixes.
            ae = Generator(
                size=self.opt.input_size, style_dim=self.opt.dim_w, motion_dim=self.opt.dim_m
            ).to(dev).eval()
            try:
                # Strip ._orig_mod prefixes from compiled module state_dict keys
                raw_sd = self.motion_autoencoder.state_dict()
                clean_sd = {
                    k.replace("._orig_mod", "").replace("_orig_mod.", ""): v
                    for k, v in raw_sd.items()
                }
                ae.load_state_dict(clean_sd, strict=False)
            except Exception as e:
                logger.warning(f"[_get_ae] state_dict copy failed for {dev}: {e}, replica may use random weights")
            ae.requires_grad_(False)
            reps[idx] = ae
        return reps[idx]

    def _decode_u8(self, ae, feats_in, lat, allow_split=True) -> torch.Tensor:
        """lat [b, dim] -> uint8 [b, H, W, C] on lat's device/current stream. Halves batch on OOM."""
        b = lat.shape[0]
        feats = [
            f.expand(b, *f.shape[1:]) if (f.shape[0] == 1 and b > 1) else f
            for f in feats_in
        ]
        try:
            out = ae.dec(lat, alpha=None, feats=feats)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.float()
            out = out.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
            return out.permute(0, 2, 3, 1).contiguous()
        except torch.cuda.OutOfMemoryError:
            if not allow_split or b <= 1:
                raise
        torch.cuda.empty_cache()
        mid = b // 2
        return torch.cat(
            [
                self._decode_u8(ae, feats_in, lat[:mid], allow_split=True),
                self._decode_u8(ae, feats_in, lat[mid:], allow_split=True),
            ],
            dim=0,
        )

    def _get_decode_profile(self, dev, primary, ae, feats, dtype):
        tuners = getattr(self, "_decode_tuners", None)
        if tuners is None:
            tuners = self._decode_tuners = {}
            self._decode_profiles = {}
        if dev.index not in tuners:
            tuner = DecodeAutoTuner(
                device=dev,
                primary=(dev == primary),
                max_batch=int(getattr(self.opt, "decode_max_batch", 32)),
                max_streams=int(getattr(self.opt, "decode_max_streams", 3)),
                reserve_gib=getattr(self.opt, "decode_reserve_gib", None),
                verbose=True,
                opt=self.opt,
            )
            dim_w = int(getattr(self.opt, "dim_w", 512))

            def bench(b):
                dummy = torch.zeros((b, dim_w), device=dev, dtype=dtype)
                return self._decode_u8(ae, feats, dummy, allow_split=False)

            prof = tuner.tune(bench)
            self._decode_tuners[dev.index] = tuner
            self._decode_profiles[dev.index] = prof
        return self._decode_tuners[dev.index], self._decode_profiles[dev.index]

    def _encoder_args(self):
        if getattr(self, "_has_nvenc", None) is None:
            try:
                txt = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                self._has_nvenc = "h264_nvenc" in txt
            except Exception:
                self._has_nvenc = False
        if self._has_nvenc and getattr(self.opt, "use_nvenc", True):
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-tune",
                "hq",
                "-rc",
                "constqp",
                "-qp",
                "23",
                "-delay",
                "0",
                "-forced-idr",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]

    def _start_ffmpeg(self, video_path, audio_path, H, W, C, fps):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba" if C == 4 else "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if audio_path:
            cmd += ["-i", audio_path]
        cmd += self._encoder_args()
        if audio_path:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-movflags", "+faststart", video_path]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=2 * 1024 * 1024,
        )

    @torch.no_grad()
    def inference_and_save(
        self,
        data: dict,
        video_path: str,
        audio_path: str = None,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
        video_encoder=None,
    ) -> str:
        t_start = time.perf_counter()
        encoder_choice = str(video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC")).strip().lower()
        use_pyav = encoder_choice in ("cpu pyav", "pyav", "pyav_cpu")
        use_torchcodec = encoder_choice in ("torchcodec", "torch codec", "tc")

        # Downgrade BEFORE spawning threads: if torchcodec is missing or its
        # native libs can't load with the installed torch, fall back to PyAV
        # while decoded frames are still in the queue (post-drain fallback
        # would have nothing left to write).
        if use_torchcodec and not _torchcodec_usable():
            logger.warning(
                "[inference_and_save] torchcodec requested but not usable "
                "(missing or binary-incompatible with installed torch); "
                "falling back to PyAV writer."
            )
            use_torchcodec, use_pyav = False, True

        if not use_pyav and not use_torchcodec and shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")

        primary, devs = self._decode_devices()
        torch.cuda.set_device(primary)

        s = data["s"]
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(primary))
        if "s_r" in data:
            r_s = self.encode_identity_into_motion(data["s_r"])
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        if a_cfg_scale is None:
            a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None:
            r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None:
            e_cfg_scale = self.opt.e_cfg_scale
        fps = int(getattr(self.opt, "fps", 25))

        try:
            amp_on = torch.is_autocast_enabled("cuda")
        except TypeError:
            amp_on = torch.is_autocast_enabled()
        try:
            amp_dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            amp_dtype = torch.get_autocast_gpu_dtype()

        def with_amp(fn):
            def run(*a, **k):
                ctx = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if amp_on
                    else contextlib.nullcontext()
                )
                with ctx:
                    return fn(*a, **k)

            return run

        ctx = {}
        for dev in devs:
            ae = self._get_ae(dev, primary)
            if dev.index == primary.index:
                feats = s_r_feats
            else:
                feats = [f.to(dev) for f in s_r_feats]
                if torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                    torch.cuda.synchronize(primary)
            tuner, prof = self._get_decode_profile(dev, primary, ae, feats, s_r.dtype)
            ctx[dev.index] = dict(dev=dev, ae=ae, feats=feats, tuner=tuner, prof=prof)

        override_bs = getattr(self.opt, "decode_batch_size", None)
        if override_bs is not None and int(override_bs) > 0:
            B = int(override_bs)
        else:
            B = max(c["prof"].batch_size for c in ctx.values())
        override = int(getattr(self.opt, "decode_streams", 0))
        for c in ctx.values():
            c["n_streams"] = override or c["tuner"].num_streams(c["prof"], B)
        any_prof = next(iter(ctx.values()))["prof"]
        H, W, C = any_prof.height, any_prof.width, any_prof.channels
        n_workers = sum(c["n_streams"] for c in ctx.values())
        need_host_latents = any(i != primary.index for i in ctx)

        logger.info(
            f"[PLAN] batch={B} frame={W}x{H}x{C} @ {fps}fps, "
            f"{len(ctx)} GPU(s), {n_workers} decode stream(s)"
        )

        stop = threading.Event()
        errors = []
        state = {"total": None, "frames": 0}
        work_q: queue.Queue = queue.Queue()
        done_q: queue.Queue = queue.Queue()
        pool: queue.Queue = queue.Queue()
        _pool_size = max(16, n_workers * 4 + 4)
        # Allocate pinned memory pool with a safe fallback to pageable memory.
        # Pinned alloc can throw c10::Error on some CUDA contexts (e.g. multi-process
        # or after torch.compile), so catch and fall back gracefully.
        _use_pinned = True
        try:
            torch.empty((1, H, W, C), dtype=torch.uint8, pin_memory=True)
        except Exception:
            _use_pinned = False
            logger.warning("[video_saver] pin_memory unavailable, using pageable host buffers")
        for _ in range(_pool_size):
            try:
                buf = torch.empty((B, H, W, C), dtype=torch.uint8, pin_memory=_use_pinned)
            except Exception:
                buf = torch.empty((B, H, W, C), dtype=torch.uint8)
            pool.put(buf)

        def fail(where, e):
            errors.append((where, e, traceback.format_exc()))
            stop.set()

        COALESCE_MS = float(getattr(self.opt, "decode_coalesce_ms", 15.0))
        COALESCE_S = COALESCE_MS / 1_000.0

        def producer():
            try:
                torch.cuda.set_device(primary)
                primary_stream = torch.cuda.current_stream(primary)

                def emit_one(lat_slice, n_valid):
                    if need_host_latents and torch.cuda.is_available():
                        lat_host = torch.empty(lat_slice.shape, dtype=lat_slice.dtype, pin_memory=True)
                        lat_host.copy_(lat_slice, non_blocking=True)
                    else:
                        lat_host = None
                    ev = torch.cuda.Event()
                    ev.record(primary_stream)
                    return ev, lat_host

                def flush_carry(carry, seq):
                    n_valid = len(carry)
                    sl = torch.stack(carry, 0)
                    if n_valid < B:
                        pad = sl[-1:].expand(B - n_valid, -1)
                        sl = torch.cat([sl, pad], 0)
                    ev, lat_host = emit_one(sl, n_valid)
                    work_q.put((seq, sl, n_valid, ev, lat_host))
                    return seq + 1, []

                seq = 0
                carry = []
                carry_t0 = None

                for chunk_idx, latents in enumerate(
                    self.sample(
                        data=data,
                        a_cfg_scale=a_cfg_scale,
                        r_cfg_scale=r_cfg_scale,
                        e_cfg_scale=e_cfg_scale,
                        emo=emo,
                        nfe=nfe,
                        seed=seed,
                        s_r=r_s,
                    )
                ):
                    if stop.is_set():
                        return
                    for frame_latent in latents:
                        carry.append(frame_latent)
                        if len(carry) == 1:
                            carry_t0 = time.perf_counter()

                        if len(carry) == B:
                            seq, carry = flush_carry(carry, seq)
                            carry_t0 = None
                        elif (time.perf_counter() - carry_t0) >= COALESCE_S:
                            seq, carry = flush_carry(carry, seq)
                            carry_t0 = None

                if carry:
                    seq, _ = flush_carry(carry, seq)

                state["total"] = seq
                for _ in range(n_workers):
                    work_q.put(_ABORT)
            except BaseException as e:
                fail("producer", e)

        def worker(c):
            """Decode worker. Creates its own CUDA stream on entry (not in main thread)."""
            dev = c["dev"]
            stream = None
            try:
                torch.cuda.set_device(dev)
                # Create stream inside the worker thread to avoid CUDA context issues
                # when a stream created in the main thread is used in a child thread.
                stream = torch.cuda.Stream(device=dev)
                while not stop.is_set():
                    item = _qget(work_q, stop)
                    if item is _ABORT:
                        break
                    seq, lat, n_valid, ev, lat_host = item

                    buf = _qget(pool, stop)
                    if buf is _ABORT:
                        break

                    with torch.cuda.stream(stream):
                        if dev.index == primary.index:
                            stream.wait_event(ev)
                            u8 = self._decode_u8(c["ae"], c["feats"], lat)
                            buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)
                        else:
                            ev.synchronize()
                            lat_dev = lat_host.to(dev, non_blocking=True)
                            lat_dev.record_stream(stream)
                            u8 = self._decode_u8(c["ae"], c["feats"], lat_dev)
                            buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)

                        done_q.put((seq, buf, n_valid, done))
            except BaseException as e:
                fail(f"decode_worker[{dev}]", e)
            finally:
                # Synchronize and release stream before thread exits to avoid
                # C++ terminate() from stream destructor racing with other threads.
                if stream is not None:
                    try:
                        torch.cuda.synchronize(dev)
                    except Exception:
                        pass

        def writer():
            if use_torchcodec:
                try:
                    from fast_float.utils.pyav_writer import save_video_torchcodec, save_video_pyav
                except ImportError:
                    from pyav_writer import save_video_torchcodec, save_video_pyav

                all_frames = []
                pending, nxt = {}, 0
                try:
                    while not stop.is_set():
                        total = state["total"]
                        if total is not None and nxt >= total:
                            break
                        try:
                            seq, buf, n, done = done_q.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        pending[seq] = (buf, n, done)
                        while nxt in pending:
                            buf, n, done = pending.pop(nxt)
                            if done is not None:
                                done.synchronize()
                            all_frames.append(buf[:n].numpy().copy())
                            state["frames"] += n
                            pool.put(buf)
                            nxt += 1
                    if stop.is_set():
                        return
                    frames_np = np.concatenate(all_frames, axis=0)
                    save_video_torchcodec(frames_np, video_path, audio_path=audio_path, fps=fps)
                except BaseException as e:
                    # Encode failed after draining: write the buffered frames
                    # via PyAV instead of losing them.
                    if all_frames:
                        logger.warning(
                            f"[writer] torchcodec encode failed ({e}); "
                            "writing drained frames via PyAV fallback."
                        )
                        try:
                            save_video_pyav(
                                np.concatenate(all_frames, axis=0),
                                video_path,
                                audio_path=audio_path,
                                fps=fps,
                            )
                        except BaseException as e2:
                            fail("writer_pyav_fallback", e2)
                        return
                    fail("writer_torchcodec", e)
                return

            if use_pyav:
                try:
                    from fast_float.utils.pyav_writer import PyAVVideoWriter
                except ImportError:
                    from pyav_writer import PyAVVideoWriter

                writer_inst = None
                try:
                    writer_inst = PyAVVideoWriter(
                        output_path=video_path,
                        width=W,
                        height=H,
                        fps=fps,
                        channels=C,
                        audio_path=audio_path,
                    )
                    pending, nxt = {}, 0
                    while not stop.is_set():
                        total = state["total"]
                        if total is not None and nxt >= total:
                            break
                        try:
                            seq, buf, n, done = done_q.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        pending[seq] = (buf, n, done)
                        while nxt in pending:
                            buf, n, done = pending.pop(nxt)
                            if done is not None:
                                done.synchronize()
                            writer_inst.write_frames(buf[:n].numpy())
                            state["frames"] += n
                            pool.put(buf)
                            nxt += 1
                    if stop.is_set():
                        return
                    writer_inst.close()
                except BaseException as e:
                    fail("writer_pyav", e)
                finally:
                    if writer_inst is not None:
                        try:
                            writer_inst.close()
                        except Exception:
                            pass
                return

            proc, err_chunks, drain = None, [], None
            try:
                proc = self._start_ffmpeg(video_path, audio_path, H, W, C, fps)
                drain = threading.Thread(
                    target=lambda: err_chunks.append(proc.stderr.read()), daemon=True
                )
                drain.start()
                pending, nxt = {}, 0
                while not stop.is_set():
                    total = state["total"]
                    if total is not None and nxt >= total:
                        break
                    try:
                        seq, buf, n, done = done_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pending[seq] = (buf, n, done)
                    while nxt in pending:
                        buf, n, done = pending.pop(nxt)
                        if done is not None:
                            done.synchronize()
                        proc.stdin.write(buf[:n].numpy().reshape(-1))
                        state["frames"] += n
                        pool.put(buf)
                        nxt += 1
                if stop.is_set():
                    return
                proc.stdin.close()
                rc = proc.wait(timeout=300)
                drain.join(timeout=5)
                if rc != 0:
                    raise RuntimeError(
                        f"ffmpeg exited with {rc}: "
                        f"{b''.join(err_chunks).decode(errors='ignore')[:1000]}"
                    )
            except BrokenPipeError:
                try:
                    proc.wait(timeout=5)
                    drain.join(timeout=5)
                except Exception:
                    pass
                msg = b"".join(err_chunks).decode(errors="ignore")[:1000]
                fail("writer", RuntimeError(f"ffmpeg pipe closed: {msg}"))
            except BaseException as e:
                fail("writer", e)
            finally:
                if proc is not None and proc.poll() is None:
                    proc.kill()

        threads = [threading.Thread(target=with_amp(producer), name="sampler")]
        for i, c in ctx.items():
            for k in range(c["n_streams"]):
                threads.append(
                    threading.Thread(
                        target=with_amp(worker),
                        args=(c,),  # Stream created inside worker thread
                        name=f"decode-cuda{i}-{k}",
                    )
                )
        threads.append(threading.Thread(target=writer, name="writer"))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            where, exc, tb = errors[0]
            raise RuntimeError(f"[{where}] {exc}\n{tb}") from exc

        dt = time.perf_counter() - t_start
        self.last_stats = {"frames": state["frames"], "seconds": dt}
        logger.info(
            f"[DONE] {state['frames']} frames in {dt:.2f}s "
            f"({state['frames'] / dt:.1f} fps end-to-end) -> {video_path}"
        )
        return video_path
