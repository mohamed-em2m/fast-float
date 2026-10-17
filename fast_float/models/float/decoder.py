"""decoder.py - Decoder, batch decoder, profile manager, and multi-GPU decode replica handling for FLOAT."""

import copy
import logging
import threading

import torch

from fast_float.utils.decode_tuner import DecodeAutoTuner
from fast_float.utils.timing import timed

logger = logging.getLogger(__name__)


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
    def _decode_call(self, ae, latent: torch.Tensor, feats: list) -> torch.Tensor:
        """Route the decode call through the configured backend.

        ``_decoder_onnx`` (static-batch ONNX) and ``_decoder_compiled`` are set
        by ``fast_float.onnx.dispatch`` when the config selects them; otherwise
        this is the plain ``ae.dec(latent, alpha=None, feats=...)`` path.
        """
        backend = getattr(self, "_decoder_onnx", None)
        if backend is not None:
            return backend(latent, feats)
        dec = getattr(self, "_decoder_compiled", None) or ae.dec
        out = dec(latent, alpha=None, feats=feats)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _decode_u8(self, ae, feats_in, lat, allow_split=True) -> torch.Tensor:
        """lat [b, dim] -> uint8 [b, H, W, C] on lat's device/current stream. Halves batch on OOM."""
        b = lat.shape[0]
        # One explicit contiguous copy of broadcast feats: the decoder touches
        # them in every layer, so this beats N implicit cuDNN copies of a
        # stride-0 expanded view.
        feats = [f.expand(b, *f.shape[1:]).contiguous() if (f.shape[0] == 1 and b > 1) else f for f in feats_in]
        static_b = self._decoder_static_batch()
        try:
            if static_b is not None and b > static_b:
                # Static-batch ONNX graph: tile to it (pad/trim inside).
                parts = []
                for i in range(0, b, static_b):
                    m = min(static_b, b - i)
                    chunk_feats = [f[i : i + m] if f.shape[0] == b else f for f in feats]
                    parts.append(self._decode_call(ae, lat[i : i + m], chunk_feats))
                out = torch.cat(parts, dim=0)
            else:
                out = self._decode_call(ae, lat, feats)
            out = out.float()
            out = out.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
            return out.permute(0, 2, 3, 1).contiguous()
        except torch.cuda.OutOfMemoryError:
            if not allow_split or b <= 1:
                raise
        # (outside the except block so the failed tensors are released first)
        torch.cuda.empty_cache()
        h = b // 2
        return torch.cat(
            [
                self._decode_u8(ae, feats_in, lat[:h]),
                self._decode_u8(ae, feats_in, lat[h:]),
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

            with timed("decode_tune", gpu=True, device=str(dev)):
                self._decode_profiles[dev.index] = tuner.tune(fn)
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

    @torch.no_grad()
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

        bs = batch_size if batch_size is not None else self.get_optimal_decode_batch_size(device)
        bs = max(1, int(bs))
        static_b = self._decoder_static_batch()
        if static_b is not None:
            bs = min(bs, static_b)

        if r_d.ndim == 3:
            b_dim, t_dim, d_dim = r_d.shape
            n_total = b_dim * t_dim
            r_d_flat = r_d.reshape(n_total, d_dim)

            aligned_feats = []
            for f in s_r_feats:
                if f.shape[0] == 1:
                    aligned_feats.append(("broadcast", f))
                elif f.shape[0] == b_dim:
                    full_f = f.unsqueeze(1).expand(-1, t_dim, *f.shape[1:]).reshape(n_total, *f.shape[1:])
                    aligned_feats.append(("slice", full_f))
                elif f.shape[0] == n_total:
                    aligned_feats.append(("slice", f))
                else:
                    aligned_feats.append(("broadcast", f[:1]))

            if n_total <= bs:
                chunk_feats = [
                    f.expand(n_total, *f.shape[1:]).contiguous() if mode == "broadcast" else f
                    for mode, f in aligned_feats
                ]
                out = self._decode_call(ae, r_d_flat, chunk_feats)
                return out.view(b_dim, t_dim, *out.shape[1:])
            else:
                out_chunks = []
                for i in range(0, n_total, bs):
                    chunk_r_d = r_d_flat[i : i + bs]
                    m = chunk_r_d.shape[0]
                    chunk_feats = [
                        (f.expand(m, *f.shape[1:]).contiguous() if mode == "broadcast" else f[i : i + m])
                        for mode, f in aligned_feats
                    ]
                    chunk_out = self._decode_call(ae, chunk_r_d, chunk_feats)
                    out_chunks.append(chunk_out)
                out = torch.cat(out_chunks, dim=0)
                return out.view(b_dim, t_dim, *out.shape[1:])
        else:
            is_1d = r_d.ndim == 1
            if is_1d:
                r_d = r_d.unsqueeze(0)
            n_total = r_d.shape[0]

            aligned_feats = []
            for f in s_r_feats:
                if f.shape[0] == 1:
                    aligned_feats.append(("broadcast", f))
                elif f.shape[0] == n_total:
                    aligned_feats.append(("slice", f))
                elif n_total % f.shape[0] == 0:
                    t = n_total // f.shape[0]
                    full_f = f.unsqueeze(1).expand(-1, t, *f.shape[1:]).reshape(n_total, *f.shape[1:])
                    aligned_feats.append(("slice", full_f))
                else:
                    aligned_feats.append(("broadcast", f[:1]))

            if n_total <= bs:
                chunk_feats = [
                    f.expand(n_total, *f.shape[1:]).contiguous() if mode == "broadcast" else f
                    for mode, f in aligned_feats
                ]
                out = self._decode_call(ae, r_d, chunk_feats)
                return out.squeeze(0) if is_1d else out
            else:
                out_chunks = []
                for i in range(0, n_total, bs):
                    chunk_r_d = r_d[i : i + bs]
                    m = chunk_r_d.shape[0]
                    chunk_feats = [
                        (f.expand(m, *f.shape[1:]).contiguous() if mode == "broadcast" else f[i : i + m])
                        for mode, f in aligned_feats
                    ]
                    chunk_out = self._decode_call(ae, chunk_r_d, chunk_feats)
                    out_chunks.append(chunk_out)
                out = torch.cat(out_chunks, dim=0)
                return out.squeeze(0) if is_1d else out

    @torch.no_grad()
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


__all__ = ["DecoderMixin", "FloatDecoder"]
