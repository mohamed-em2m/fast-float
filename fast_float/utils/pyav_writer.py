"""
PyAV-based in-process video encoder and audio muxer.
Provides fast, zero-subprocess CPU alternatives to FFmpeg CLI / NVENC:
  - CPU PyAV  : streams frames via libx264 with audio muxed inline.
  - Torchcodec: encodes all frames via torchcodec VideoEncoder (CPU or CUDA),
                then muxes audio with PyAV (or ffmpeg fallback).
"""

import os
from typing import Union

import numpy as np

try:
    import av

    _PYAV_AVAILABLE = True
except ImportError:
    av = None
    _PYAV_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None

import contextlib

from fast_float.utils.logger import logger


def is_pyav_available() -> bool:
    """Check if PyAV is installed and has H.264 encoding support."""
    if not _PYAV_AVAILABLE or av is None:
        return False
    try:
        return "libx264" in av.codec.codecs_available or "h264" in av.codec.codecs_available
    except Exception:
        return False


def is_torchcodec_available() -> bool:
    """Check if torchcodec is installed with VideoEncoder support."""
    try:
        from torchcodec.encoders import VideoEncoder  # noqa: F401

        return True
    except ImportError:
        return False


class PyAVVideoWriter:
    """
    High-performance in-process video encoder using PyAV (CPU libx264).
    Streams frames directly without spawning an external ffmpeg process.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: int = 25,
        channels: int = 3,
        audio_path: str | None = None,
        preset: str = "ultrafast",
        crf: int = 23,
    ):
        if not _PYAV_AVAILABLE or av is None:
            raise RuntimeError("PyAV is not installed. Install with `uv add av` or `pip install av`.")

        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.channels = int(channels)
        self.audio_path = audio_path
        self.frames_written = 0
        self.pix_fmt_in = "rgba" if self.channels == 4 else "rgb24"
        self._closed = False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Open container with faststart so the MP4 can stream immediately
        self.container = av.open(
            output_path,
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )

        # Video stream
        codec_name = "libx264" if "libx264" in av.codec.codecs_available else "h264"
        self.vstream = self.container.add_stream(codec_name, rate=self.fps)
        self.vstream.width = self.width
        self.vstream.height = self.height
        self.vstream.pix_fmt = "yuv420p"
        self.vstream.options = {
            "preset": preset,
            "crf": str(crf),
            "tune": "zerolatency",
        }

        # Audio stream setup (MUST be declared before any packets are muxed)
        self.audio_in = None
        self.in_astream = None
        self.out_astream = None
        self.resampler = None

        if self.audio_path and os.path.exists(self.audio_path):
            try:
                self.audio_in = av.open(self.audio_path)
                if self.audio_in.streams.audio:
                    self.in_astream = self.audio_in.streams.audio[0]
                    out_rate = 44100
                    self.out_astream = self.container.add_stream("aac", rate=out_rate)
                    self.out_astream.format = "fltp"
                    channels_count = getattr(self.in_astream, "channels", 1) or 1
                    self.out_astream.layout = "stereo" if channels_count > 1 else "mono"
                    self.resampler = av.AudioResampler(
                        format="fltp",
                        layout=self.out_astream.layout,
                        rate=out_rate,
                    )
            except Exception as e:
                logger.warning(f"⚠️ PyAV audio track init failed for {self.audio_path}: {e}")
                if self.audio_in is not None:
                    with contextlib.suppress(Exception):
                        self.audio_in.close()
                self.audio_in = None
                self.out_astream = None
                self.resampler = None

    def write_frames(self, frames_np: np.ndarray) -> None:
        """Write a batch of uint8 frames (N, H, W, C) or a single frame (H, W, C)."""
        if self._closed or self.container is None:
            raise RuntimeError("Cannot write frames to a closed PyAVVideoWriter.")

        if not isinstance(frames_np, np.ndarray):
            frames_np = np.asarray(frames_np)

        if frames_np.ndim == 3:
            frames_np = np.expand_dims(frames_np, axis=0)

        n = len(frames_np)
        for i in range(n):
            frame = frames_np[i]
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            vframe = av.VideoFrame.from_ndarray(frame, format=self.pix_fmt_in)
            for packet in self.vstream.encode(vframe):
                self.container.mux(packet)

        self.frames_written += n

    def close(self) -> None:
        """Flush remaining frames, mux audio, and finalize the MP4 container."""
        if self._closed or self.container is None:
            return

        try:
            # 1. Flush video stream
            for packet in self.vstream.encode():
                self.container.mux(packet)

            # 2. Process and mux audio up to video duration (-shortest behavior)
            if (
                self.audio_in is not None
                and self.in_astream is not None
                and self.out_astream is not None
                and self.resampler is not None
            ):
                try:
                    max_duration = self.frames_written / float(self.fps) if self.fps > 0 else None
                    for aframe in self.audio_in.decode(self.in_astream):
                        if max_duration is not None and aframe.time is not None and aframe.time >= max_duration:
                            break
                        resampled = self.resampler.resample(aframe)
                        for rf in resampled:
                            rf.pts = None
                            for apacket in self.out_astream.encode(rf):
                                self.container.mux(apacket)

                    # Flush resampler
                    for rf in self.resampler.resample(None):
                        rf.pts = None
                        for apacket in self.out_astream.encode(rf):
                            self.container.mux(apacket)

                    # Flush audio stream
                    for apacket in self.out_astream.encode():
                        self.container.mux(apacket)
                except Exception as e:
                    logger.warning(f"⚠️ PyAV audio muxing warning: {e}")
                finally:
                    with contextlib.suppress(Exception):
                        self.audio_in.close()
                    self.audio_in = None

        finally:
            try:
                self.container.close()
            except Exception as e:
                logger.warning(f"⚠️ PyAV container close warning: {e}")
            self.container = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_video_pyav(
    vid: Union["torch.Tensor", np.ndarray],
    video_path: str,
    audio_path: str | None = None,
    fps: int = 25,
    preset: str = "ultrafast",
    crf: int = 23,
) -> str:
    """
    Encode an entire video tensor/array to MP4 with PyAV CPU encoder.
    vid shape: (T, H, W, C) or (T, C, H, W).
    """
    if torch is not None and isinstance(vid, torch.Tensor):
        if vid.is_cuda:
            torch.cuda.synchronize()
            if vid.dim() == 4 and vid.shape[1] in (3, 4) and vid.shape[-1] not in (3, 4):
                vid = vid.permute(0, 2, 3, 1).contiguous()
            if torch.is_floating_point(vid):
                vid = (vid.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
            else:
                vid = vid.to(torch.uint8)
            torch.cuda.synchronize()
            vid_np = vid.cpu().numpy()
        else:
            if vid.dim() == 4 and vid.shape[1] in (3, 4) and vid.shape[-1] not in (3, 4):
                vid = vid.permute(0, 2, 3, 1).contiguous()
            if torch.is_floating_point(vid):
                vid = (vid.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
            else:
                vid = vid.to(torch.uint8)
            vid_np = vid.numpy()
    else:
        vid_np = np.asarray(vid)
        if vid_np.ndim == 4 and vid_np.shape[1] in (3, 4) and vid_np.shape[-1] not in (3, 4):
            vid_np = np.transpose(vid_np, (0, 2, 3, 1))
        if np.issubdtype(vid_np.dtype, np.floating):
            vid_np = np.clip(vid_np * 127.5 + 127.5, 0, 255).astype(np.uint8)
        elif vid_np.dtype != np.uint8:
            vid_np = vid_np.astype(np.uint8)

    _T, H, W, C = vid_np.shape
    with PyAVVideoWriter(
        output_path=video_path,
        width=W,
        height=H,
        fps=fps,
        channels=C,
        audio_path=audio_path,
        preset=preset,
        crf=crf,
    ) as writer:
        writer.write_frames(vid_np)

    return video_path


# ---------------------------------------------------------------------------
# Torchcodec encoder helpers
# ---------------------------------------------------------------------------


def _mux_audio_onto_video(
    video_only_path: str,
    audio_path: str,
    output_path: str,
    n_frames: int = 0,
    fps: int = 25,
) -> None:
    """
    Mux audio from *audio_path* onto the video-only *video_only_path* using PyAV.
    The video stream is remuxed (no re-encode); audio is transcoded to AAC.
    Falls back to an ffmpeg copy-mux subprocess when PyAV is unavailable.
    """
    if not _PYAV_AVAILABLE or av is None:
        # Fast fallback: copy-mux with ffmpeg (no video re-encode)
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_only_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return

    max_duration = n_frames / float(fps) if n_frames > 0 and fps > 0 else None

    in_video = av.open(video_only_path)
    in_audio = av.open(audio_path)
    out = av.open(output_path, mode="w", format="mp4", options={"movflags": "+faststart"})

    try:
        # ---- video: remux without re-encoding ----
        in_vstream = in_video.streams.video[0]
        out_vstream = out.add_stream_from_template(in_vstream)

        # ---- audio: transcode to AAC ----
        in_astreams = in_audio.streams.audio
        out_astream = None
        resampler = None
        if in_astreams:
            in_astream = in_astreams[0]
            out_rate = 44100
            out_astream = out.add_stream("aac", rate=out_rate)
            out_astream.format = "fltp"
            channels_count = getattr(in_astream, "channels", 1) or 1
            out_astream.layout = "stereo" if channels_count > 1 else "mono"
            resampler = av.AudioResampler(
                format="fltp",
                layout=out_astream.layout,
                rate=out_rate,
            )
        else:
            in_astream = None

        # Remux video packets
        for packet in in_video.demux(in_vstream):
            if packet.dts is None:
                continue
            packet.stream = out_vstream
            out.mux(packet)

        # Encode + mux audio
        if in_astream and out_astream and resampler:
            try:
                for aframe in in_audio.decode(in_astream):
                    if max_duration is not None and aframe.time is not None and aframe.time >= max_duration:
                        break
                    for rf in resampler.resample(aframe):
                        rf.pts = None
                        for pkt in out_astream.encode(rf):
                            out.mux(pkt)
                for rf in resampler.resample(None):
                    rf.pts = None
                    for pkt in out_astream.encode(rf):
                        out.mux(pkt)
                for pkt in out_astream.encode():
                    out.mux(pkt)
            except Exception as e:
                logger.warning(f"⚠️ torchcodec audio mux warning: {e}")
    finally:
        in_video.close()
        in_audio.close()
        out.close()


def save_video_torchcodec(
    vid: Union["torch.Tensor", np.ndarray],
    video_path: str,
    audio_path: str | None = None,
    fps: int = 25,
    codec: str = "libx264",
    crf: int = 23,
) -> str:
    """
    Encode a video tensor/array with **torchcodec** ``VideoEncoder``.

    torchcodec handles only video; audio is muxed afterwards via PyAV
    (or a fallback ffmpeg copy-mux that does not re-encode the video).

    Args:
        vid: ``(T, H, W, C)`` or ``(T, C, H, W)`` uint8/float array or tensor.
        video_path: Destination ``.mp4`` path.
        audio_path: Optional WAV/MP3/… to mux in.
        fps: Frame-rate of the output video.
        codec: libav codec name (default ``"libx264"``).
        crf: Constant-rate factor for quality (lower = better).
    """
    try:
        from torchcodec.encoders import VideoEncoder as _TCEncoder
    except ImportError:
        raise RuntimeError("torchcodec is not installed. Install with: pip install torchcodec") from None

    # ------ normalise input to (T, C, H, W) uint8 CPU tensor ------
    if torch is not None and isinstance(vid, torch.Tensor):
        if vid.is_cuda:
            torch.cuda.synchronize()
        t = vid
        if t.dim() == 4 and t.shape[-1] in (3, 4):  # (T, H, W, C) -> (T, C, H, W)
            t = t.permute(0, 3, 1, 2).contiguous()
        t = (t.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8) if torch.is_floating_point(t) else t.to(torch.uint8)
        frames_tensor = t.cpu()
    else:
        vid_np = np.asarray(vid)
        if vid_np.ndim == 4 and vid_np.shape[1] in (3, 4) and vid_np.shape[-1] not in (3, 4):
            pass  # already (T, C, H, W)
        elif vid_np.ndim == 4 and vid_np.shape[-1] in (3, 4):
            vid_np = np.transpose(vid_np, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)
        if np.issubdtype(vid_np.dtype, np.floating):
            vid_np = np.clip(vid_np * 127.5 + 127.5, 0, 255).astype(np.uint8)
        elif vid_np.dtype != np.uint8:
            vid_np = vid_np.astype(np.uint8)
        if torch is not None:
            frames_tensor = torch.from_numpy(vid_np)
        else:
            raise RuntimeError("torch is required for torchcodec; install PyTorch.")

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    need_audio = bool(audio_path and os.path.exists(audio_path))
    tmp_video = (video_path + "._tc_tmp.mp4") if need_audio else video_path

    try:
        logger.info(
            f"[torchcodec] encoding {len(frames_tensor)} frames @ {fps}fps codec={codec} crf={crf} -> {tmp_video}"
        )
        encoder = _TCEncoder(frames=frames_tensor, frame_rate=float(fps))
        encoder.to_file(
            dest=tmp_video,
            codec=codec,
            crf=crf,
            pixel_format="yuv420p",
        )

        if need_audio:
            logger.info(f"[torchcodec] muxing audio {audio_path} -> {video_path}")
            _mux_audio_onto_video(
                tmp_video,
                audio_path,
                video_path,
                n_frames=len(frames_tensor),
                fps=fps,
            )
    finally:
        if need_audio:
            with contextlib.suppress(Exception):
                os.remove(tmp_video)

    return video_path
