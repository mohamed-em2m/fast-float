"""video_pipeline.py - Auto-tuned, multi-stream, multi-GPU streaming decode + video encoding pipeline for FLOAT."""

import contextlib
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.types import StreamMeta
from fast_float.utils.stream_utils import _ABORT, _qget
from fast_float.utils.timing import timed

try:
    from fast_float.utils.pyav_writer import save_video_torchcodec
except ImportError:
    try:
        from utils.pyav_writer import save_video_torchcodec
    except ImportError:
        from pyav_writer import save_video_torchcodec

logger = logging.getLogger(__name__)


def _drain_ordered(done_q, state, stop, pool, deliver, timeout: float = 0.1) -> None:
    """Re-order decoded batches, hand each to ``deliver``, then recycle its buffer.

    Decode workers finish out of order, but the dispatcher (and every encoder)
    needs generation order. Stops once ``state["total"]`` batches are delivered
    or the shared ``stop`` event is set. Exceptions from ``deliver`` propagate
    to the writer thread, which records them through ``fail``.
    """
    pending, nxt = {}, 0
    while not stop.is_set():
        total = state["total"]
        if total is not None and nxt >= total:
            return
        try:
            seq, buf, n, done = done_q.get(timeout=timeout)
        except queue.Empty:
            continue
        pending[seq] = (buf, n, done)
        while nxt in pending:
            buf, n, done = pending.pop(nxt)
            if done is not None:
                done.synchronize()
            deliver(buf, n)
            state["frames"] += n
            pool.put(buf)
            nxt += 1


class StreamingVideoSaver:
    """Auto-tuned, multi-stream, multi-GPU streaming decode and video save pipeline.

    Mixes into or interacts with FLOAT to decode latents concurrently and stream uint8 frames
    directly into NVENC/FFmpeg/PyAV/torchcodec.
    """

    # ---------------- ffmpeg ---------------- #
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
        if not video_path:
            # subprocess would otherwise fail deep inside Popen with the opaque
            # "expected str, bytes or os.PathLike object, not NoneType".
            raise ValueError(f"_start_ffmpeg requires a non-empty video_path (got {video_path!r})")
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
        )

    # ---------------- main entry ---------------- #
    @torch.no_grad()
    def inference_and_save(
        self,
        data: dict,
        video_path: str | None,
        audio_path: str | None = None,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
        video_encoder=None,
        dispatcher: FrameDispatcher | None = None,
        sampler_state: dict | None = None,
    ) -> str | None:
        t_start = time.perf_counter()
        encoder_choice = str(video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC")).strip().lower()
        use_pyav = encoder_choice in ("cpu pyav", "pyav", "pyav_cpu")
        use_torchcodec = encoder_choice in ("torchcodec", "torch codec", "tc")

        # Live transports (LiveKit, WebSocket taps) consume frames through the
        # dispatcher and intentionally pass no output file (None/empty): there is
        # nothing to mux, and spawning an encoder would either fail on the empty
        # path or waste GPU/CPU time on a file nobody reads.
        stream_only = not video_path
        if stream_only and dispatcher is None:
            raise ValueError("inference_and_save needs a video_path or a dispatcher")
        if not stream_only and not use_pyav and not use_torchcodec and shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")

        primary, devs = self._decode_devices()
        torch.cuda.set_device(primary)

        s = data["s"]
        with timed("encode_image", gpu=True):
            s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(primary))
        if "s_r" in data:
            with timed("encode_identity", gpu=True):
                r_s = self.encode_identity_into_motion(data["s_r"])
        else:
            with timed("encode_identity", gpu=True):
                r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        if a_cfg_scale is None:
            a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None:
            r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None:
            e_cfg_scale = self.opt.e_cfg_scale
        fps = int(getattr(self.opt, "fps", 25) or 25)

        # cuDNN's algorithm search is per *thread*: with benchmark=True every
        # fresh decode worker re-autotunes ~40 conv shapes, which costs ~1.2 s
        # on the first decode call of *every* request (the pipeline spawns new
        # workers per request). Heuristics are ~10% slower per call but remove
        # that stall, which dominates short clips. Set FLOAT_CUDNN_BENCHMARK=1
        # to restore the old behaviour for long renders.
        if os.getenv("FLOAT_CUDNN_BENCHMARK", "0") != "1":
            torch.backends.cudnn.benchmark = False

        # autocast state is thread-local: capture it here, re-enter it in every pipeline thread
        try:
            amp_on = torch.is_autocast_enabled("cuda")
        except TypeError:  # older torch: no device argument
            amp_on = torch.is_autocast_enabled()
        try:
            amp_dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            amp_dtype = torch.get_autocast_gpu_dtype()

        def with_amp(fn):
            def run(*a, **k):
                ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_on else contextlib.nullcontext()
                with ctx:
                    return fn(*a, **k)

            return run

        # ---- per-GPU context: model replica, reference features, tuned profile ----
        # Reference features live on the primary GPU. Copy them to the other GPUs
        # on a dedicated transfer stream each, so the cross-device move is enqueued
        # asynchronously and never drains the primary GPU's stream with a full
        # device synchronize (the old code did synchronize(dev) + synchronize(primary)
        # for every secondary device).
        has_secondary = any(dev.index != primary.index for dev in devs)
        if has_secondary and torch.cuda.is_available():
            # features are final once the encoder stream is done; wait once total
            feats_ready = torch.cuda.Event()
            feats_ready.record(torch.cuda.current_stream(primary))
            feats_ready.synchronize()

        ctx = {}
        for dev in devs:
            ae = self._get_ae(dev, primary)
            if dev.index == primary.index:
                feats = s_r_feats
            else:
                xfer = torch.cuda.Stream(device=dev)
                with torch.cuda.stream(xfer):
                    feats = [f.to(dev, non_blocking=True) for f in s_r_feats]
                xfer.synchronize()  # drain only this device's transfer stream
            tuner, prof = self._get_decode_profile(dev, primary, ae, feats, s_r.dtype)
            ctx[dev.index] = dict(dev=dev, ae=ae, feats=feats, tuner=tuner, prof=prof)

        # Select micro-batch size: user override takes precedence if feasible (checked by tuner),
        # otherwise use maximum tuned size to maximize GPU utilization and tensor core saturation.
        B = max(c["prof"].batch_size for c in ctx.values())
        static_b = self._decoder_static_batch()
        if static_b is not None:
            # Never serve micro-batches the static ONNX graph cannot take.
            B = min(B, static_b)
        override = int(getattr(self.opt, "decode_streams", 0))
        for c in ctx.values():
            c["n_streams"] = override or c["tuner"].num_streams(c["prof"], B)
        any_prof = next(iter(ctx.values()))["prof"]
        H, W, C = any_prof.height, any_prof.width, any_prof.channels
        n_workers = sum(c["n_streams"] for c in ctx.values())
        need_host_latents = any(i != primary.index for i in ctx)

        logger.info(f"[PLAN] batch={B} frame={W}x{H}x{C} @ {fps}fps, {len(ctx)} GPU(s), {n_workers} decode stream(s)")
        for i, c in ctx.items():
            logger.info(f"       cuda:{i} {torch.cuda.get_device_name(i)}: {c['n_streams']} stream(s)")

        # ---- shared state ----
        stop = threading.Event()
        errors = []
        state = {"total": None, "frames": 0}

        if dispatcher is not None:
            dispatcher.bind(
                StreamMeta(
                    fps=float(fps),
                    width=int(W),
                    height=int(H),
                    channels=int(C),
                    request_id=str(getattr(self, "request_id", "") or ""),
                    audio_path=audio_path,
                    started_at_s=time.monotonic(),
                ),
                stop_event=stop,
            )
        work_q: queue.Queue = queue.Queue()  # latents only: tiny, unbounded is fine
        done_q: queue.Queue = queue.Queue()
        pool: queue.Queue = queue.Queue()  # pinned host buffers = backpressure
        for _ in range(max(16, n_workers * 4 + 4)):
            pool.put(torch.empty((B, H, W, C), dtype=torch.uint8, pin_memory=True))

        def fail(where, e):
            errors.append((where, e, traceback.format_exc()))
            stop.set()

        # ---- producer: ODE sampling on primary GPU, dynamic-batch dispatch ----
        COALESCE_MS = float(getattr(self.opt, "decode_coalesce_ms", 15.0))
        COALESCE_S = COALESCE_MS / 1_000.0

        def producer():
            try:
                torch.cuda.set_device(primary)
                primary_stream = torch.cuda.current_stream(primary)

                def emit_one(lat_slice, n_valid):
                    """Record a per-item event and (if needed) copy latents to host non-blocking."""
                    if need_host_latents and torch.cuda.is_available():
                        lat_host = torch.empty(lat_slice.shape, dtype=lat_slice.dtype, pin_memory=True)
                        lat_host.copy_(lat_slice, non_blocking=True)
                    else:
                        lat_host = None
                    ev = torch.cuda.Event()
                    ev.record(primary_stream)
                    return ev, lat_host

                def flush_carry(carry, seq):
                    """Pad carry to B, emit one work item, return (seq+1, [])."""
                    n_valid = len(carry)
                    sl = torch.stack(carry, 0)  # [n_valid, dim]
                    if n_valid < B:
                        pad = sl[-1:].expand(B - n_valid, -1)
                        sl = torch.cat([sl, pad], 0)  # [B, dim]
                    ev, lat_host = emit_one(sl, n_valid)
                    work_q.put((seq, sl, lat_host, n_valid, ev))
                    return seq + 1, []

                seq = 0
                carry: list = []  # accumulate individual latent rows
                deadline: float = 0.0  # wall-clock deadline for current carry window

                for chunk in self.sample(
                    data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                    s_r=s_r,
                    state=sampler_state,
                ):
                    if stop.is_set():
                        return
                    lat = chunk.reshape(-1, chunk.shape[-1])  # [n, dim]
                    for row in lat.unbind(0):  # iterate frame-by-frame
                        if not carry:
                            deadline = time.perf_counter() + COALESCE_S
                        carry.append(row)
                        # Flush when batch is full OR coalesce window has expired
                        if len(carry) >= B or time.perf_counter() >= deadline:
                            seq, carry = flush_carry(carry, seq)

                # Flush whatever remains at end-of-video
                if carry:
                    seq, carry = flush_carry(carry, seq)

                state["total"] = seq
            except BaseException as e:
                fail("producer", e)
            finally:
                for _ in range(n_workers):
                    work_q.put(None)

        # ---- decode workers: one CUDA stream each, spread over all GPUs ----
        def worker(c, stream):
            dev = c["dev"]
            is_primary = dev.index == primary.index
            try:
                torch.cuda.set_device(dev)
                with torch.no_grad():  # grad mode is thread-local
                    while True:
                        item = _qget(work_q, stop)
                        if item is _ABORT or item is None:
                            return
                        seq, lat_gpu, lat_host, n_valid, ev = item

                        buf = _qget(pool, stop)
                        if buf is _ABORT:
                            return

                        with torch.cuda.stream(stream):
                            if is_primary:
                                stream.wait_event(ev)
                                lat_gpu.record_stream(stream)
                                lat = lat_gpu
                            else:
                                ev.synchronize()
                                lat = lat_host.to(dev, non_blocking=True)
                                lat.record_stream(stream)
                            with timed("decode", gpu=True, device=str(dev), batch=B):
                                u8 = self._decode_u8(c["ae"], c["feats"], lat)
                            buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)
                        done_q.put((seq, buf, n_valid, done))
            except BaseException as e:
                fail(f"decode_worker[{dev}]", e)

        # ---- writer: restore order, feed video encoder (torchcodec / PyAV / FFmpeg) ----
        def writer():
            if stream_only:
                # No file: deliver frames to the dispatcher in order and recycle
                # their buffers. The dispatcher copies on submit, so the pool
                # buffer can be reused immediately.

                def deliver(buf, n):
                    with timed("video_write"):
                        dispatcher.submit(buf[:n].numpy())

                try:
                    _drain_ordered(done_q, state, stop, pool, deliver)
                except BaseException as e:
                    fail("writer", e)
                return

            if use_torchcodec:
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
                            with timed("frame_gather"):
                                all_frames.append(buf[:n].numpy().copy())
                            if dispatcher is not None:
                                dispatcher.submit(buf[:n].numpy())
                            state["frames"] += n
                            pool.put(buf)
                            nxt += 1
                    if stop.is_set():
                        return
                    frames_np = np.concatenate(all_frames, axis=0)
                    with timed("video_encode"):
                        save_video_torchcodec(
                            frames_np,
                            video_path,
                            audio_path=audio_path,
                            fps=fps,
                        )
                except BaseException as e:
                    fail("writer_torchcodec", e)
                return

            if use_pyav:
                try:
                    from fast_float.utils.pyav_writer import PyAVVideoWriter
                except ImportError:
                    try:
                        from utils.pyav_writer import PyAVVideoWriter
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
                            with timed("video_write"):
                                writer_inst.write_frames(buf[:n].numpy())
                            if dispatcher is not None:
                                dispatcher.submit(buf[:n].numpy())
                            state["frames"] += n
                            pool.put(buf)
                            nxt += 1
                    if stop.is_set():
                        return
                    with timed("video_finalize"):
                        writer_inst.close()
                except BaseException as e:
                    fail("writer_pyav", e)
                finally:
                    if writer_inst is not None:
                        with contextlib.suppress(Exception):
                            writer_inst.close()
                return

            proc, err_chunks, drain = None, [], None
            try:
                proc = self._start_ffmpeg(video_path, audio_path, H, W, C, fps)
                drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
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
                        with timed("video_write"):
                            proc.stdin.write(buf[:n].numpy().reshape(-1))
                        if dispatcher is not None:
                            dispatcher.submit(buf[:n].numpy())
                        state["frames"] += n
                        pool.put(buf)
                        nxt += 1
                if stop.is_set():
                    return
                with timed("video_finalize"):
                    proc.stdin.close()
                    rc = proc.wait(timeout=300)
                    drain.join(timeout=5)
                if rc != 0:
                    raise RuntimeError(
                        f"ffmpeg exited with {rc}: {b''.join(err_chunks).decode(errors='ignore')[:1000]}"
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
            threads.extend(
                threading.Thread(
                    target=with_amp(worker),
                    args=(c, torch.cuda.Stream(device=c["dev"])),
                    name=f"decode-cuda{i}-{k}",
                )
                for k in range(c["n_streams"])
            )
        threads.append(threading.Thread(target=writer, name="writer"))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            where, exc, tb = errors[0]
            if dispatcher is not None:
                dispatcher.error(exc)
            raise RuntimeError(f"[{where}] {exc}\n{tb}") from exc

        dt = time.perf_counter() - t_start
        self.last_stats = {"frames": state["frames"], "seconds": dt}
        dest = video_path or "dispatcher stream"
        logger.info(
            f"[DONE] {state['frames']} frames in {dt:.2f}s ({state['frames'] / dt:.1f} fps end-to-end) -> {dest}"
        )
        return video_path


class SaveVideoPipeline(StreamingVideoSaver):
    """Standalone video save pipeline orchestrator."""

    def __init__(self, opt=None):
        self.opt = opt
        self._has_nvenc = None
        self.last_stats = {}


__all__ = ["SaveVideoPipeline", "StreamingVideoSaver"]
