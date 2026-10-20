"""decoder.py - Decoder, batch decoder, profile manager, and multi-GPU decode replica handling for FLOAT."""

import contextlib
import copy
import logging
import os
import threading

import torch

from fast_float.utils.decode_tuner import DecodeAutoTuner
from fast_float.utils.timing import timed

logger = logging.getLogger(__name__)

_CHANNELS_LAST_ENV = "FLOAT_DECODE_CHANNELS_LAST"
_AUTOCAST_ENV = "FLOAT_DECODE_AUTOCAST"
_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
_ON_VALUES = frozenset({"1", "true", "yes", "on"})
# Pre-expanding broadcast style vectors is only cheaper than per-chunk cuDNN
# copies while they stay small; guard against spatial feature maps.
_FEAT_EXPAND_BUDGET_BYTES = 64 * 1024 * 1024


class DecoderMixin:
    """Mixin containing latent-to-image decoding logic, batching, multi-GPU decode distribution,

    and tuner profiles for FLOAT.
    """

    # ---------------- devices / replicas ---------------- #
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
            logger.warning(f"[MULTI-GPU] ngpus={want} requested but only {visible} visible; using {n}")
        else:
            logger.info(f"[MULTI-GPU] using {n}/{visible} GPU(s) (primary=cuda:{p_idx})")
        others = [torch.device("cuda", i) for i in range(visible) if i != primary.index][: max(0, n - 1)]
        devs = [primary, *others]
        if n > 1 and not getattr(self.opt, "decode_on_primary", True):
            devs = others
        return primary, devs

    def _get_ae(self, dev, primary):
        """Motion autoencoder on `dev`: the original on the primary GPU, cached replicas elsewhere.

        If you reload weights into self.motion_autoencoder, call self._ae_replicas.clear().
        """
        if dev == primary or (dev.index is not None and primary.index is not None and dev.index == primary.index):
            return self.motion_autoencoder
        with self._decode_lock:
            reps = getattr(self, "_ae_replicas", None)
            if reps is None:
                reps = self._ae_replicas = {}
            idx = dev.index if dev.index is not None else str(dev)
            if idx not in reps:
                # Runners implement __deepcopy__ (clean session rebuild);
                # plain modules copy normally.
                ae = copy.deepcopy(self.motion_autoencoder).to(dev).eval()
                ae.requires_grad_(False)
                reps[idx] = ae
            return reps[idx]

    def _decoder_static_batch(self):
        """Static ONNX decode batch when the decoder backend is ONNX, else None."""
        backend = getattr(self, "_decoder_onnx", None)
        b = getattr(backend, "batch", None) if backend is not None else None
        return int(b) if isinstance(b, int) and b > 0 else None

    # ---------------- decode ---------------- #
    def _decoder_for(self, device):
        """ONNX decoder session bound to ``device`` (primary proto, replicas elsewhere).

        The ONNX decoder lives on the model, not on the motion autoencoder, so
        ``_get_ae`` replicas do not carry one. Without this, every decode worker
        -- including secondary GPUs -- would call the single primary-pinned
        session and all decode work would run on the primary GPU.
        """
        proto = getattr(self, "_decoder_onnx", None)
        if proto is None:
            return None
        if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
            return proto
        idx = device.index if device.index is not None else torch.cuda.current_device()
        primary = self._cuda_device()
        if primary.index is None or idx == primary.index:
            return proto
        with self._decode_lock:
            reps = getattr(self, "_decoder_onnx_replicas", None)
            if reps is None:
                reps = self._decoder_onnx_replicas = {}
            if idx not in reps:
                reps[idx] = proto._recreate(device_id=idx)
            return reps[idx]

    def _channels_last_decode(self) -> bool:
        """Whether the eager/compiled decoder should run in channels_last.

        Off by default (``FLOAT_DECODE_CHANNELS_LAST=1`` or
        ``opt.decode_channels_last``): worth A/B-testing per device, since it
        only pays off if the GPU kernels prefer NHWC.
        """
        flag = getattr(self.opt, "decode_channels_last", None)
        if flag is not None:
            return bool(flag)
        return os.environ.get(_CHANNELS_LAST_ENV, "").strip().lower() in _ON_VALUES

    def _decode_autocast_dtype(self):
        """Autocast dtype for the eager/compiled decoder, or ``None`` (default).

        Off by default: the decoder's AdaIN/SPADE normalizations and the fp32
        ONNX reference path make reduced precision a per-device A/B decision,
        not a safe global default. Enable with ``FLOAT_DECODE_AUTOCAST=fp16``
        (or ``bf16``, or ``opt.decode_autocast``).
        """
        value = getattr(self.opt, "decode_autocast", None)
        if value is None:
            value = os.environ.get(_AUTOCAST_ENV, "")
        key = str(value).strip().lower()
        if key in {"", "0", "off", "false", "no"}:
            return None
        return _AUTOCAST_DTYPES.get(key)

    def _prepare_decoder(self, ae) -> None:
        """One-time per-decoder setup: channels_last weights + per-replica compile.

        ``dispatch`` compiles only the primary model's ``dec``, so a multi-GPU
        replica reusing it would run the primary device's weights.
        """
        dec = ae.dec
        if self._channels_last_decode() and not getattr(dec, "_ff_channels_last", False):
            with self._decode_lock:
                if not getattr(dec, "_ff_channels_last", False):
                    dec.to(memory_format=torch.channels_last)
                    dec._ff_channels_last = True
        if getattr(self, "_decoder_compiled", None) is None:
            return
        primary = getattr(self, "motion_autoencoder", None)
        if primary is not None and ae is primary:
            return
        if getattr(ae, "_ff_decoder_compiled", None) is None:
            with self._decode_lock:
                if getattr(ae, "_ff_decoder_compiled", None) is None:
                    ae._ff_decoder_compiled = torch.compile(dec)

    def _bind_decoder(self, ae, device):
        """Retrieve the decoder runner once per request/worker, avoiding per-chunk locks."""
        backend = self._decoder_for(device)
        if backend is not None:
            return backend
        compiled = getattr(self, "_decoder_compiled", None)
        if compiled is not None or self._channels_last_decode():
            self._prepare_decoder(ae)
        if compiled is None:
            return ae.dec
        primary = getattr(self, "motion_autoencoder", None)
        return compiled if (primary is None or ae is primary) else (getattr(ae, "_ff_decoder_compiled", None) or ae.dec)

    def _decode_call(self, ae, latent: torch.Tensor, feats: list, bound_dec=None) -> torch.Tensor:
        """Route the decode call through the configured backend.

        ``_decoder_onnx`` (static-batch ONNX, one session per GPU) and
        ``_decoder_compiled`` are set by ``fast_float.onnx.dispatch`` when the
        config selects them; otherwise this is the plain
        ``ae.dec(latent, alpha=None, feats=...)`` path on the per-device replica.

        ``bound_dec`` is resolved once per request by the caller so decode
        workers do not re-enter the ONNX replica lock on every chunk.
        """
        dec = bound_dec if bound_dec is not None else self._bind_decoder(ae, latent.device)
        is_onnx = hasattr(dec, "providers")
        amp_dtype = None if is_onnx else self._decode_autocast_dtype()
        device_type = getattr(getattr(latent, "device", None), "type", None)
        ctx = contextlib.nullcontext()
        if amp_dtype is not None and device_type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        with ctx:
            out = dec(latent, feats) if is_onnx else dec(latent, alpha=None, feats=feats)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    @staticmethod
    def _quantize_u8_hwc(out: torch.Tensor) -> torch.Tensor:
        """[-1, 1] float NCHW -> uint8 HWC.

        When the decoder is channels_last the final ``permute(...).contiguous()``
        returns the input unchanged instead of copying a full batch.
        """
        out = out.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
        return out.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _slice_feats(feats: list, i: int, m: int) -> list:
        out = []
        for f in feats:
            if f.shape[0] == 1:
                out.append(f.expand(m, *f.shape[1:]).contiguous() if m > 1 else f)
            else:
                out.append(f[i : i + m])
        return out

    def _note_decode_oom(self) -> None:
        """Evict the allocator cache only after repeated OOMs.

        ``empty_cache`` frees every cached block and synchronizes the device, so
        paying it on each split stalls every decode worker.
        """
        streak = getattr(self, "_decode_oom_streak", 0) + 1
        limit = max(1, int(getattr(self.opt, "decode_oom_evict_after", 2)))
        if streak < limit:
            self._decode_oom_streak = streak
            return
        self._decode_oom_streak = 0
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    def _decode_tiled_u8(self, ae, feats: list, lat, b: int, static_b: int, bound_dec=None) -> torch.Tensor:
        """Tile a batch the static ONNX graph cannot take, writing uint8 HWC once."""
        first = self._quantize_u8_hwc(
            self._decode_call(ae, lat[:static_b], self._slice_feats(feats, 0, static_b), bound_dec=bound_dec)
        )
        out = first.new_empty((b, *first.shape[1:]))
        out[: first.shape[0]].copy_(first)
        done = first.shape[0]
        while done < b:
            m = min(static_b, b - done)
            chunk = self._quantize_u8_hwc(
                self._decode_call(ae, lat[done : done + m], self._slice_feats(feats, done, m), bound_dec=bound_dec)
            )
            out[done : done + chunk.shape[0]].copy_(chunk)
            done += chunk.shape[0]
        return out

    @torch.inference_mode()
    def _decode_u8(self, ae, feats_in, lat, allow_split=True, bound_dec=None) -> torch.Tensor:
        """lat [b, dim] -> uint8 [b, H, W, C] on lat's device/current stream. Halves batch on OOM."""
        b = lat.shape[0]
        # One explicit contiguous copy of broadcast feats: the decoder touches
        # them in every layer, so this beats N implicit cuDNN copies of a
        # stride-0 expanded view.
        feats = [f.expand(b, *f.shape[1:]).contiguous() if (f.shape[0] == 1 and b > 1) else f for f in feats_in]
        static_b = self._decoder_static_batch()
        if bound_dec is None:
            bound_dec = self._bind_decoder(ae, lat.device)
        try:
            if static_b is not None and b > static_b:
                # Static-batch ONNX graph: tile to it (pad/trim inside).
                out = self._decode_tiled_u8(ae, feats, lat, b, static_b, bound_dec=bound_dec)
            else:
                out = self._quantize_u8_hwc(self._decode_call(ae, lat, feats, bound_dec=bound_dec))
            self._decode_oom_streak = 0
            return out
        except torch.cuda.OutOfMemoryError:
            if not allow_split or b <= 1:
                raise
        # (outside the except block so the failed tensors are released first)
        self._note_decode_oom()
        h = b // 2
        return torch.cat(
            [
                self._decode_u8(ae, feats_in, lat[:h], allow_split=allow_split, bound_dec=bound_dec),
                self._decode_u8(ae, feats_in, lat[h:], allow_split=allow_split, bound_dec=bound_dec),
            ],
            0,
        )

    def _get_decode_profile(self, dev, primary, ae, feats, dtype):
        with self._decode_lock:
            if getattr(self, "_decode_profiles", None) is None:
                self._decode_profiles, self._decode_tuners = {}, {}
            if dev.index in self._decode_profiles:
                return self._decode_tuners[dev.index], self._decode_profiles[dev.index]
            # Cap the tuner at the ONNX static batch when that backend is on:
            # larger micro-batches would only be tiled back down (B1/B3).
            max_batch = int(getattr(self.opt, "decode_max_batch", 16))
            static_b = self._decoder_static_batch()
            if static_b is not None:
                max_batch = min(max_batch, static_b)
            tuner = DecodeAutoTuner(
                dev,
                primary=(dev.index == primary.index),
                max_batch=max_batch,
                max_streams=int(getattr(self.opt, "decode_max_streams", 1)),
                reserve_gib=getattr(self.opt, "decode_reserve_gib", None),
                opt=self.opt,
            )
            dim = self.opt.dim_w

            def fn(b):
                lat = torch.randn(b, dim, device=dev, dtype=dtype)
                return self._decode_u8(ae, feats, lat, allow_split=False)

            # The tuning sweep is GPU seconds spent before the first request, and
            # its result only depends on the device / library versions / frame
            # shape / tuner settings. Key on all of those so a driver, torch or
            # resolution change re-measures instead of reusing a stale profile.
            cache_key = "|".join(
                [
                    "v1",
                    str(torch.__version__),
                    str(torch.version.cuda),
                    str(torch.cuda.get_device_name(dev)),
                    ".".join(str(x) for x in torch.cuda.get_device_capability(dev)),
                    f"{int(getattr(self.opt, 'input_size', 0))}x{dim}",
                    str(dtype),
                    f"mb{max_batch}",
                    f"pl{tuner.plateau}",
                    f"sf{tuner.safety}",
                    f"rg{tuner.reserve_gib}",
                    f"ob{getattr(self.opt, 'decode_batch_size', 0)}",
                    f"cl{int(self._channels_last_decode())}",
                    f"ac{self._decode_autocast_dtype()}",
                ]
            )
            with timed("decode_tune", gpu=True, device=str(dev)):
                self._decode_profiles[dev.index] = tuner.tune(fn, cache_key=cache_key)
            self._decode_tuners[dev.index] = tuner
            return self._decode_tuners[dev.index], self._decode_profiles[dev.index]

    def clear_decode_cache(self):
        """Clears memoized batch size and tuner profiles so GPU VRAM is re-evaluated."""
        if hasattr(self, "_decode_bs_cache") and self._decode_bs_cache:
            self._decode_bs_cache.clear()
        if hasattr(self, "_decode_profiles") and self._decode_profiles:
            self._decode_profiles.clear()
        if hasattr(self, "_decode_tuners") and self._decode_tuners:
            self._decode_tuners.clear()

    def get_optimal_decode_batch_size(self, device=None) -> int:
        """Automatically select optimal frame decode batch size based on GPU memory.

        Maximized for high GPU utilization and throughput.

        The result is memoized per device index in ``self._decode_bs_cache`` so that
        ``torch.cuda.mem_get_info`` is only called once per device, not once per chunk.
        """
        if hasattr(self, "opt") and getattr(self.opt, "decode_batch_size", None) is not None:
            val = int(self.opt.decode_batch_size)
            if val > 0:
                return val

        if not torch.cuda.is_available():
            return 4

        try:
            if device is None:
                rank = getattr(self.opt, "rank", 0) if hasattr(self, "opt") else 0
                device = getattr(self, "device", None) or torch.device(f"cuda:{rank}")
            if isinstance(device, str):
                device = torch.device(device)
            if device.type != "cuda":
                return 4

            idx = device.index if device.index is not None else torch.cuda.current_device()

            # Return memoized result to avoid repeated cudaMemGetInfo calls
            cache = getattr(self, "_decode_bs_cache", None)
            if cache is None:
                self._decode_bs_cache = {}
                cache = self._decode_bs_cache
            if idx in cache:
                return cache[idx]

            free_b, total_b = torch.cuda.mem_get_info(idx)
            gib = 1024**3
            free_gib = free_b / gib
            total_gib = total_b / gib

            # StyleDecoder decoding takes ~20-40 MB per frame (e.g. ~300-500 MB for batch 16).
            # When ~5 GiB free out of 15 GiB is available, batch 16 is easily accommodated!
            if free_gib >= 6.0 or (total_gib >= 14.0 and free_gib >= 3.8):
                bs = 16
            elif free_gib >= 3.0 or (total_gib >= 10.0 and free_gib >= 1.8):
                bs = 8
            elif free_gib >= 1.2 or total_gib >= 6.0:
                bs = 4
            elif free_gib >= 0.7:
                bs = 2
            else:
                bs = 1

            cache[idx] = bs
            return bs
        except Exception:
            return 4

    def _default_decode_batch(self, device) -> int:
        """Tuned micro-batch from the decode profile, else the VRAM heuristic.

        ``video_pipeline`` already plans with ``prof.batch_size``; without this
        the sampler path picked a different (heuristic) batch for the same GPU.
        """
        if (
            not getattr(self.opt, "decode_batch_size", None)
            and device is not None
            and getattr(device, "type", None) == "cuda"
        ):
            idx = device.index if device.index is not None else torch.cuda.current_device()
            prof = (getattr(self, "_decode_profiles", None) or {}).get(idx)
            if prof is not None and int(prof.batch_size) > 0:
                return int(prof.batch_size)
        return self.get_optimal_decode_batch_size(device)

    def _expand_feats(self, feats: list, n: int) -> list:
        """Expand per-frame style vectors once per request so chunks only slice."""
        out = []
        for f in feats:
            if f.shape[0] == 1 and n > 1 and f.numel() * f.element_size() * n <= _FEAT_EXPAND_BUDGET_BYTES:
                out.append(f.expand(n, *f.shape[1:]).contiguous())
            else:
                out.append(f)
        return out

    def _align_feats_3d(self, feats: list, b_dim: int, t_dim: int, n_total: int) -> list:
        aligned = []
        for f in feats:
            if f.shape[0] == 1:
                aligned.append(f)
            elif f.shape[0] == b_dim:
                aligned.append(f.unsqueeze(1).expand(-1, t_dim, *f.shape[1:]).reshape(n_total, *f.shape[1:]))
            elif f.shape[0] == n_total:
                aligned.append(f)
            else:
                aligned.append(f[:1])
        return self._expand_feats(aligned, n_total)

    def _align_feats_2d(self, feats: list, n_total: int) -> list:
        aligned = []
        for f in feats:
            if f.shape[0] in (1, n_total):
                aligned.append(f)
            elif n_total % f.shape[0] == 0:
                t = n_total // f.shape[0]
                aligned.append(f.unsqueeze(1).expand(-1, t, *f.shape[1:]).reshape(n_total, *f.shape[1:]))
            else:
                aligned.append(f[:1])
        return self._expand_feats(aligned, n_total)

    def _decode_chunks(self, ae, lat: torch.Tensor, feats: list, bs: int, bound_dec=None) -> torch.Tensor:
        """Decode ``lat`` in mini-batches into one preallocated output tensor."""
        n = lat.shape[0]
        first_m = min(bs, n)
        first = self._decode_call(ae, lat[:first_m], self._slice_feats(feats, 0, first_m), bound_dec=bound_dec)

        out = first.new_empty((n, *first.shape[1:]))
        step = first.shape[0]
        out[:step].copy_(first)

        while step < n:
            m = min(bs, n - step)
            chunk = self._decode_call(ae, lat[step : step + m], self._slice_feats(feats, step, m), bound_dec=bound_dec)
            got = chunk.shape[0]
            if got <= 0:
                raise RuntimeError("decoder returned an empty batch")
            out[step : step + got].copy_(chunk)
            step += got
        return out

    @torch.inference_mode()
    def _decode_latent_batched(
        self,
        ae,
        r_d: torch.Tensor,
        s_r_feats: list,
        batch_size: int | None = None,
        device=None,
    ) -> torch.Tensor:
        """Decode motion latents into images in mini-batches (e.g., 4 frames gathered together)

        to prevent OOM and maximize GPU throughput. Supports 2D [N, D] and 3D [B, T, D] latents.
        """
        if device is None:
            device = r_d.device if (hasattr(r_d, "device") and r_d.device.type == "cuda") else None

        bs = batch_size if batch_size is not None else self._default_decode_batch(device)
        bs = max(1, int(bs))
        static_b = self._decoder_static_batch()
        if static_b is not None:
            bs = min(bs, static_b)

        bound_dec = self._bind_decoder(ae, device or getattr(r_d, "device", None) or self._cuda_device())

        if r_d.ndim == 3:
            b_dim, t_dim, d_dim = r_d.shape
            n_total = b_dim * t_dim
            flat = r_d.reshape(n_total, d_dim)
            out = self._decode_chunks(
                ae, flat, self._align_feats_3d(s_r_feats, b_dim, t_dim, n_total), bs, bound_dec=bound_dec
            )
            return out.reshape(b_dim, t_dim, *out.shape[1:])

        is_1d = r_d.ndim == 1
        if is_1d:
            r_d = r_d.unsqueeze(0)
        n_total = r_d.shape[0]
        out = self._decode_chunks(ae, r_d, self._align_feats_2d(s_r_feats, n_total), bs, bound_dec=bound_dec)
        return out.squeeze(0) if is_1d else out

    @torch.inference_mode()
    def decode_latent_into_image(
        self,
        s_r: torch.Tensor,
        s_r_feats: list,
        r_d: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Decode latent into image with broadcasted features (supports 2D [N, D] and 3D [B, T, D])

        Processes frames in mini-batches (e.g. 4 frames at a time, auto-selected based on GPU VRAM)
        to prevent OOM and maximize decoding throughput.
        """
        return self._decode_latent_batched(
            self.motion_autoencoder,
            r_d,
            s_r_feats,
            batch_size=batch_size,
        )

    ######## Motion Encoder - Helpers ########
    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        if x_r.ndim == 1:
            x_r = x_r.unsqueeze(0)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x


class FloatDecoder(DecoderMixin):
    """Standalone decoder component that can wrap a motion autoencoder for decoding latents."""

    def __init__(self, opt, motion_autoencoder=None):
        self.opt = opt
        self.motion_autoencoder = motion_autoencoder
        self._decode_lock = threading.Lock()
        self._decode_bs_cache = {}
        self._decode_profiles = {}
        self._decode_tuners = {}
        self._ae_replicas = {}
        self._decode_oom_streak = 0


__all__ = ["DecoderMixin", "FloatDecoder"]
