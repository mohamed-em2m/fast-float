"""
FLOAT - Gradio Interface for Audio-Driven Talking Face Generation
Includes Batch Generation and Low-Latency Near-Real-Time Live Streaming.

Architecture Highlights
------------------------
1. Batch Generation:
   One-shot full-sequence talking face synthesis from uploaded reference image
   and audio files, supporting multiple hardware encoders (FFmpeg NVENC, CPU PyAV, Torchcodec).

2. Seamless Live Streaming (Near-Real-Time):
   - Pre-Encodes Reference Face ONCE: Face detection, alignment, cropping, and
     motion latent identity features (x_r_feats, r_s) are computed once at session start.
   - Temporal Continuity Across Chunks: Carries forward motion latents (prev_x, prev_wa)
     across consecutive chunks, completely eliminating visual seams and mouth jumping between clips.
   - Decoupled Pipeline & Zero-Copy Audio: Direct audio feature extraction and ODE
     sampling without repetitive disk roundtrips.
   - Multi-Backend Video Encoding: Supports FFmpeg NVENC (hardware), CPU PyAV, or Torchcodec.

Requires Gradio >= 4.40 (for gr.Timer and streaming gr.Audio input).
"""

import os
import sys
import time
import math
import shutil
import logging
import tempfile
import datetime
import threading
import traceback
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Union

# Suppress Albumentations auto-update check and noisy deprecation warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Suppress verbose PyTorch Dynamo compiler bytecode tracing
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("torch.fx").setLevel(logging.WARNING)

# ---- logging setup -------------------------------------------------------- #
try:
    from fast_float.utils.logger import setup_logger
except ImportError:
    try:
        from utils.logger import setup_logger
    except ImportError:
        def setup_logger(name="float_fast", **_):
            _log = logging.getLogger(name)
            _log.setLevel(logging.INFO)
            _log.handlers.clear()
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _log.addHandler(_h)
            _log.propagate = False
            return _log

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

try:
    import torch._dynamo
    torch._dynamo.config.verbose = False
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import gradio as gr

logger = setup_logger("float_live_app")

try:
    from fast_float.core.inference_optimized import (
        OptimizedInferenceAgent as InferenceAgent,
        InferenceOptions,
    )
except ImportError:
    try:
        from core.inference_optimized import (
            OptimizedInferenceAgent as InferenceAgent,
            InferenceOptions,
        )
    except ImportError:
        try:
            from inference_optimized import (
                OptimizedInferenceAgent as InferenceAgent,
                InferenceOptions,
            )
        except ImportError:
            from fast_float.core.generate import (
                InferenceAgent,
                InferenceOptions,
            )

try:
    from fast_float.sched.types import FloatOmniRequest
except ImportError:
    FloatOmniRequest = None


def _coerce_int(value: Any, default: int) -> int:
    """Gradio sliders return floats — coerce safely to plain int."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    """Coerce value safely to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_torchcodec_ready() -> bool:
    try:
        from fast_float.utils.pyav_writer import is_torchcodec_available
        return is_torchcodec_available()
    except Exception:
        try:
            from utils.pyav_writer import is_torchcodec_available
            return is_torchcodec_available()
        except Exception:
            return False


def get_encoder_choices() -> list[str]:
    """Return available encoder choices (torchcodec only if actually usable)."""
    choices = ["FFmpeg NVENC", "CPU PyAV"]
    if is_torchcodec_ready():
        choices.append("Torchcodec")
    return choices


def detect_default_encoder() -> str:
    """Auto-detect whether to default to FFmpeg NVENC, CPU PyAV, or Torchcodec."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "h264_nvenc" in res.stdout:
            return "FFmpeg NVENC"
    except Exception:
        pass
    if is_torchcodec_ready():
        return "Torchcodec"
    return "CPU PyAV"


# --------------------------------------------------------------------------- #
# Per-session live state (held in gr.State per connected user session)
# --------------------------------------------------------------------------- #
@dataclass
class LiveSession:
    ref_image_path: Optional[str] = None
    sample_rate: Optional[int] = None
    buffer: list = field(default_factory=list)  # list of 1-D numpy float chunks
    buffered_seconds: float = 0.0
    chunk_idx: int = 0
    running: bool = False
    last_latency: float = 0.0
    session_dir: Optional[str] = None

    # Precomputed reference identity features (cached once per session)
    s_tensor: Optional[torch.Tensor] = None
    r_s: Optional[torch.Tensor] = None
    ref_feats: Optional[list[torch.Tensor]] = None
    cropped_preview: Optional[np.ndarray] = None
    is_preprocessed: bool = False

    # Temporal continuity across chunks (eliminates seams)
    prev_x: Optional[torch.Tensor] = None
    prev_wa: Optional[torch.Tensor] = None
    total_generated_frames: int = 0

    def reset_audio(self):
        """Clears the audio accumulator buffer."""
        self.buffer = []
        self.buffered_seconds = 0.0

    def clear_gpu_cache(self):
        """Cleans up GPU tensor allocations on session stop."""
        self.s_tensor = None
        self.r_s = None
        self.ref_feats = None
        self.prev_x = None
        self.prev_wa = None
        self.cropped_preview = None
        self.is_preprocessed = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


CHUNK_SECONDS_DEFAULT = 1.5  # default duration of audio accumulated per chunk
MAX_LIVE_CLIPS_KEPT = 20     # rotate old temp files so disk space remains bounded


# --------------------------------------------------------------------------- #
# Shared Model Wrapper
# --------------------------------------------------------------------------- #
class FloatModel:
    def __init__(
        self,
        ckpt_path="./checkpoints/float.pth",
        res_dir="./results",
        wav2vec_model_path=None,
        ngpus=1,
    ):
        overrides = ["--ckpt_path", ckpt_path, "--res_dir", res_dir]
        if wav2vec_model_path:
            overrides.extend(["--wav2vec_model_path", wav2vec_model_path])
        original_argv = sys.argv.copy()
        try:
            sys.argv = [sys.argv[0]] + overrides
            self.opt = InferenceOptions().parse()
        finally:
            sys.argv = original_argv
        self.opt.ngpus = ngpus
        self.opt.rank = 0
        os.makedirs(self.opt.res_dir, exist_ok=True)

        self.live_dir = os.path.join(self.opt.res_dir, "live_chunks")
        os.makedirs(self.live_dir, exist_ok=True)

        logger.info("Loading FLOAT model...")
        self.agent = InferenceAgent(self.opt)
        self.infer_lock = threading.Lock()
        logger.info("Model loaded successfully!")

    # ---- 1. Precompute Reference Identity Once per Live Session ----------- #
    def prepare_live_session(self, ref_image_path: str, no_crop: bool = False) -> dict[str, Any]:
        """Aligns, crops, and pre-encodes the reference face into latent identity features.

        Executes once at the start of a live session to eliminate 200-400ms per-chunk
        overhead during continuous streaming.
        """
        device = f"cuda:{self.opt.rank}" if torch.cuda.is_available() else "cpu"
        dp = self.agent.data_processor

        img_rgb = dp.default_img_loader(ref_image_path)
        if not no_crop:
            cropped_img = dp.process_img(img_rgb)
        else:
            cropped_img = img_rgb

        s_tensor = dp.transform(image=cropped_img)["image"].unsqueeze(0).to(device)

        with self.infer_lock:
            with torch.no_grad():
                x_r, x_r_lambda, x_r_feats = self.agent.G.encode_image_into_latent(s_tensor)
                r_s = self.agent.G.motion_autoencoder.dec.direction(x_r_lambda)

        return {
            "s": s_tensor,
            "r_s": r_s,
            "ref_feats": x_r_feats,
            "cropped_preview": cropped_img,
        }

    # ---- 2. Run Single Live Chunk with Temporal Continuity ---------------- #
    def run_live_chunk(
        self,
        session: LiveSession,
        audio_np: np.ndarray,
        sr: int,
        out_path: str,
        chunk_wav_path: str,
        emotion: str = "S2E",
        a_cfg: float = 2.0,
        r_cfg: float = 1.0,
        e_cfg: float = 1.0,
        nfe: int = 6,
        seed: int = 25,
        decode_batch_size: int = 0,
        video_encoder: str = "FFmpeg NVENC",
        temporal_continuity: bool = True,
    ) -> tuple[str, dict[str, float]]:
        """Generates a talking head video clip for one audio chunk.

        Maintains temporal continuity across chunks using session.prev_x and session.prev_wa,
        ensuring smooth natural motion without boundary seams.
        """
        device = f"cuda:{self.opt.rank}" if torch.cuda.is_available() else "cpu"
        fps = int(getattr(self.opt, "fps", 25))
        nfe = _coerce_int(nfe, 6)
        seed = _coerce_int(seed, 25)
        a_cfg = _coerce_float(a_cfg, 2.0)
        r_cfg = _coerce_float(r_cfg, 1.0)
        e_cfg = _coerce_float(e_cfg, 1.0)
        dbs = _coerce_int(decode_batch_size, 0)

        # 1. Save audio chunk for muxing
        sf.write(chunk_wav_path, audio_np, sr)

        # 2. Extract Wav2Vec2 input features
        t0 = time.perf_counter()
        dp = self.agent.data_processor
        audio_tensor = dp.wav2vec_preprocessor(
            audio_np, sampling_rate=sr, return_tensors="pt"
        ).input_values[0].unsqueeze(0).to(device)

        T_frames = int(math.floor(audio_tensor.shape[1] / self.opt.sampling_rate * fps))
        if T_frames <= 0:
            T_frames = 1

        with torch.no_grad():
            wa = self.agent.G.audio_encoder.inference(audio_tensor, seq_len=T_frames)

            # Emotion feature
            if emotion is not None and emotion not in ("S2E", None):
                emo_idx = self.agent.G.emotion_encoder.label2id.get(str(emotion).lower(), None)
                if emo_idx is not None:
                    we = (
                        F.one_hot(
                            torch.tensor(emo_idx, device=device),
                            num_classes=self.opt.dim_e,
                        )
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                else:
                    we = torch.zeros(1, 1, self.opt.dim_e, device=device)
            else:
                we = self.agent.G.emotion_encoder.predict_emotion(audio_tensor)
                if we.ndim == 2:
                    we = we.unsqueeze(1)

        t_audio = time.perf_counter() - t0

        # Temporal continuity handling
        prev_x = session.prev_x if temporal_continuity else None
        prev_wa = session.prev_wa if temporal_continuity else None

        req = FloatOmniRequest(
            request_id=f"live_{session.chunk_idx}",
            ref_image=session.s_tensor,
            audio=audio_tensor,
            ref_latent=session.r_s,
            ref_feats=session.ref_feats,
            wa=wa,
            we=we,
            total_chunks=max(1, int(math.ceil(T_frames / self.agent.G.num_frames_for_clip))),
            total_frames=T_frames,
            current_chunk=0,
            prev_x=prev_x,
            prev_wa=prev_wa,
            a_cfg_scale=a_cfg,
            r_cfg_scale=r_cfg,
            e_cfg_scale=e_cfg,
            nfe=nfe,
            output_format="tensor",
        )

        torch.manual_seed(seed + session.chunk_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + session.chunk_idx)

        # Apply decode batch size override
        if dbs > 0:
            self.opt.decode_batch_size = dbs
        else:
            self.opt.decode_batch_size = None

        with self.infer_lock:
            # 3. ODE latent sampling
            t_samp0 = time.perf_counter()
            task = self.agent.G.sample_omni_chunk(req)
            t_sample = time.perf_counter() - t_samp0

            # Store updated temporal state for next chunk
            if temporal_continuity:
                session.prev_x = req.prev_x
                session.prev_wa = req.prev_wa
            else:
                session.prev_x = None
                session.prev_wa = None

            # 4. StyleDecoder frame synthesis
            t_dec0 = time.perf_counter()
            frames_u8 = self.agent.G.decode_omni_chunk(task)  # [T, H, W, 3] uint8
            t_decode = time.perf_counter() - t_dec0

            # 5. Video encoding & audio muxing
            t_enc0 = time.perf_counter()
            self._save_video_clip(
                frames=frames_u8,
                video_path=out_path,
                audio_path=chunk_wav_path,
                fps=fps,
                video_encoder=video_encoder,
            )
            t_encode = time.perf_counter() - t_enc0

        frames_count = int(frames_u8.shape[0]) if hasattr(frames_u8, "shape") else T_frames
        session.total_generated_frames += frames_count

        timings = {
            "audio": t_audio,
            "sample": t_sample,
            "decode": t_decode,
            "encode": t_encode,
            "total": t_audio + t_sample + t_decode + t_encode,
            "frames": frames_count,
            "fps": frames_count / max(0.001, (t_sample + t_decode + t_encode)),
        }
        return out_path, timings

    def _save_video_clip(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        video_path: str,
        audio_path: str,
        fps: int = 25,
        video_encoder: str = "FFmpeg NVENC",
    ) -> str:
        """Encodes frames and muxes audio using the chosen video encoder backend."""
        encoder_choice = str(video_encoder).strip().lower()
        use_pyav = encoder_choice in ("cpu pyav", "pyav", "pyav_cpu")
        use_torchcodec = encoder_choice in ("torchcodec", "torch codec", "tc")

        if use_torchcodec:
            try:
                from fast_float.utils.pyav_writer import save_video_torchcodec
                return save_video_torchcodec(frames, video_path, audio_path=audio_path, fps=fps)
            except Exception as e:
                logger.warning(f"Torchcodec encode fallback: {e}")

        if use_pyav:
            try:
                from fast_float.utils.pyav_writer import save_video_pyav
                return save_video_pyav(frames, video_path, audio_path=audio_path, fps=fps)
            except Exception as e:
                logger.warning(f"PyAV encode fallback: {e}")

        # FFmpeg / NVENC in-process stream
        frames_np = frames.numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
        if frames_np.ndim == 4:
            T, H, W, C = frames_np.shape
        else:
            raise ValueError(f"Unexpected video frame shape {frames_np.shape}")

        proc = self.agent.G._start_ffmpeg(video_path, audio_path, H, W, C, fps)
        try:
            assert proc.stdin is not None
            proc.stdin.write(frames_np.tobytes())
            proc.stdin.close()
            rc = proc.wait(timeout=60)
            if rc != 0:
                raise RuntimeError(f"FFmpeg exited with error code {rc}")
        except Exception:
            if proc.poll() is None:
                proc.kill()
            raise
        return video_path

    # ---- 3. Full-Sequence Batch Generation -------------------------------- #
    def run(
        self,
        ref_path,
        audio_path,
        out_path,
        emotion,
        a_cfg,
        r_cfg,
        e_cfg,
        nfe,
        seed,
        no_crop,
        decode_batch_size=0,
        video_encoder="FFmpeg NVENC",
    ):
        nfe = _coerce_int(nfe, 10)
        seed = _coerce_int(seed, 25)
        a_cfg = _coerce_float(a_cfg, 2.0)
        r_cfg = _coerce_float(r_cfg, 1.0)
        e_cfg = _coerce_float(e_cfg, 1.0)
        dbs = _coerce_int(decode_batch_size, 0)

        target_model = getattr(self.agent, "G_opt", getattr(self.agent, "model", None))
        if dbs > 0:
            self.opt.decode_batch_size = dbs
        else:
            self.opt.decode_batch_size = None

        if target_model is not None:
            if hasattr(target_model, "clear_decode_cache"):
                target_model.clear_decode_cache()
            elif hasattr(target_model, "_decode_bs_cache"):
                target_model._decode_bs_cache.clear()

        with self.infer_lock:
            return self.agent.run_inference(
                res_video_path=out_path,
                ref_path=ref_path,
                audio_path=audio_path,
                a_cfg_scale=a_cfg,
                r_cfg_scale=r_cfg,
                e_cfg_scale=e_cfg,
                emo=emotion,
                nfe=nfe,
                no_crop=no_crop,
                seed=seed,
                verbose=False,
                video_encoder=video_encoder,
            )

    def generate_video(
        self,
        ref_image,
        audio_file,
        emotion,
        a_cfg_scale,
        r_cfg_scale,
        e_cfg_scale,
        nfe,
        seed,
        no_crop,
        decode_batch_size=0,
        video_encoder="FFmpeg NVENC",
        progress=gr.Progress(),
    ):
        try:
            progress(0, desc="Preparing inputs...")
            if ref_image is None:
                return None, "❌ Please upload a reference image"
            if audio_file is None:
                return None, "❌ Please upload an audio file"

            nfe = _coerce_int(nfe, 10)
            seed = _coerce_int(seed, 25)
            a_cfg_scale = _coerce_float(a_cfg_scale, 2.0)
            r_cfg_scale = _coerce_float(r_cfg_scale, 1.0)
            e_cfg_scale = _coerce_float(e_cfg_scale, 1.0)

            video_name = Path(ref_image).stem
            audio_name = Path(audio_file).stem
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            res_video_path = os.path.join(
                self.opt.res_dir,
                f"{call_time}-{video_name}-{audio_name}-nfe{nfe}-seed{seed}"
                f"-acfg{a_cfg_scale}-ecfg{e_cfg_scale}-{emotion}.mp4",
            )

            progress(0.3, desc="Running optimized inference...")
            output_path = self.run(
                ref_image,
                audio_file,
                res_video_path,
                emotion,
                a_cfg_scale,
                r_cfg_scale,
                e_cfg_scale,
                nfe,
                seed,
                no_crop,
                decode_batch_size=decode_batch_size,
                video_encoder=video_encoder,
            )
            progress(1.0, desc="Complete!")
            return output_path, f"✅ Video generated: {Path(output_path).name}"
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ Error during generation: {e}\n{tb}")
            return None, f"❌ Error during generation: {e}\n\n--- Traceback ---\n{tb}"


# --------------------------------------------------------------------------- #
# Audio Ingestion Helper
# --------------------------------------------------------------------------- #
def _to_mono_float(chunk_sr_data):
    """Gradio streaming mic audio arrives as (sample_rate, np.ndarray)."""
    sr, data = chunk_sr_data
    sr = int(sr)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return sr, data


# --------------------------------------------------------------------------- #
# Gradio Interface Builder
# --------------------------------------------------------------------------- #
def build_interface(
    ckpt_path="./checkpoints/float.pth",
    res_dir="./results",
    wav2vec_model_path=None,
    ngpus=1,
    default_video_encoder: Optional[str] = None,
):
    if default_video_encoder is None:
        default_video_encoder = detect_default_encoder()

    model = FloatModel(
        ckpt_path=ckpt_path,
        res_dir=res_dir,
        wav2vec_model_path=wav2vec_model_path,
        ngpus=ngpus,
    )

    custom_css = """
    .gradio-container { font-family: 'Inter', -apple-system, sans-serif; }
    .live-status { font-weight: 600; font-family: monospace; font-size: 0.95rem; }
    .stat-badge { background: #f0f4f8; padding: 2px 6px; border-radius: 4px; font-weight: 500; }
    """

    with gr.Blocks(css=custom_css, title="FLOAT-Fast Talking Face Generation") as demo:
        gr.Markdown(
            """
            # 🎭 FLOAT-Fast: Audio-Driven Talking Face Generation
            ### Ultra-Optimized Streaming Inference with Continuous Temporal Continuity
            Generate talking face videos from a reference portrait and speech audio —
            as a complete batch video or as a low-latency live streaming session.
            """
        )

        with gr.Tabs():
            # =========================================================== #
            # TAB 1 — Batch Generation (Upload Audio File)
            # =========================================================== #
            with gr.Tab("📁 Batch Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Reference Inputs")
                        b_ref_image = gr.Image(
                            label="Reference Portrait Image",
                            type="filepath",
                            sources=["upload"],
                            height=300,
                        )
                        b_audio_file = gr.Audio(
                            label="Speech Audio File (WAV/MP3)",
                            type="filepath",
                            sources=["upload"],
                        )

                        with gr.Accordion("⚙️ Advanced Optimization Settings", open=False):
                            b_emotion = gr.Dropdown(
                                choices=[
                                    "S2E",
                                    "angry",
                                    "disgust",
                                    "fear",
                                    "happy",
                                    "neutral",
                                    "sad",
                                    "surprise",
                                ],
                                value="S2E",
                                label="Emotion Conditioning",
                            )
                            b_no_crop = gr.Checkbox(
                                label="Skip Automatic Face Cropping", value=False
                            )
                            b_seed = gr.Slider(
                                0, 10000, value=25, step=1, label="Random Seed"
                            )
                            b_decode_batch = gr.Radio(
                                choices=["Auto", "1", "2", "4", "8", "16", "32", "64"],
                                value="Auto",
                                label="🗂️ Decode Batch Size",
                                info="Frames decoded per GPU micro-batch ('Auto' tunes to available VRAM).",
                            )
                            b_video_encoder = gr.Radio(
                                choices=get_encoder_choices(),
                                value=default_video_encoder,
                                label="🎬 Video Encoder Backend",
                                info="FFmpeg NVENC (NVIDIA Hardware), CPU PyAV (In-process libx264), Torchcodec.",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Flow Matching Controls")
                        with gr.Group():
                            b_nfe = gr.Slider(
                                1, 50, value=10, step=1, label="NFE (Denoising Steps)"
                            )
                            b_a_cfg = gr.Slider(
                                0.0, 5.0, value=2.0, step=0.1, label="Audio CFG Scale"
                            )
                            b_r_cfg = gr.Slider(
                                0.0, 3.0, value=1.0, step=0.1, label="Reference CFG Scale"
                            )
                            b_e_cfg = gr.Slider(
                                0.0, 3.0, value=1.0, step=0.1, label="Emotion CFG Scale"
                            )

                        b_generate_btn = gr.Button(
                            "🚀 Generate Video", variant="primary", size="lg"
                        )
                        b_status = gr.Textbox(label="Status & Progress", lines=3)
                        b_video_out = gr.Video(label="Generated Video Output", height=400)

                b_generate_btn.click(
                    fn=model.generate_video,
                    inputs=[
                        b_ref_image,
                        b_audio_file,
                        b_emotion,
                        b_a_cfg,
                        b_r_cfg,
                        b_e_cfg,
                        b_nfe,
                        b_seed,
                        b_no_crop,
                        b_decode_batch,
                        b_video_encoder,
                    ],
                    outputs=[b_video_out, b_status],
                )

            # =========================================================== #
            # TAB 2 — Low-Latency Live Stream (Microphone)
            # =========================================================== #
            with gr.Tab("🔴 Live Stream (Near-Real-Time)"):
                gr.Markdown(
                    """
                    **How Live Streaming Works:**
                    1. Upload a portrait image once. Face alignment and motion latent features are **precomputed**.
                    2. Hit **▶️ Start Live**, then talk into your microphone.
                    3. Each short speech chunk is denoised and rendered back-to-back with **Temporal Continuity**,
                       carrying motion memory forward so video clips flow seamlessly without facial jumps.
                    """
                )

                session_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1):
                        l_ref_image = gr.Image(
                            label="Reference Portrait (Precomputed once)",
                            type="filepath",
                            sources=["upload"],
                            height=260,
                        )
                        l_cropped_preview = gr.Image(
                            label="Aligned Face Preview",
                            interactive=False,
                            height=180,
                            visible=False,
                        )

                        with gr.Accordion("⚙️ Live Streaming Optimization Controls", open=True):
                            with gr.Row():
                                l_chunk_len = gr.Slider(
                                    1.0,
                                    3.0,
                                    value=CHUNK_SECONDS_DEFAULT,
                                    step=0.25,
                                    label="Chunk Duration (seconds)",
                                    info="Audio accumulated before generating. 1.0-1.5s is optimal for near-real-time.",
                                )
                                l_nfe = gr.Slider(
                                    2,
                                    15,
                                    value=6,
                                    step=1,
                                    label="NFE (lower = faster)",
                                    info="4-6 steps deliver <350ms generation latency.",
                                )

                            l_temporal_continuity = gr.Checkbox(
                                label="🔄 Seamless Temporal Continuity",
                                value=True,
                                info="Preserves ODE motion latents across chunks to eliminate boundary seams.",
                            )

                            with gr.Row():
                                l_emotion = gr.Dropdown(
                                    choices=[
                                        "S2E",
                                        "angry",
                                        "disgust",
                                        "fear",
                                        "happy",
                                        "neutral",
                                        "sad",
                                        "surprise",
                                    ],
                                    value="S2E",
                                    label="Emotion Control",
                                )
                                l_no_crop = gr.Checkbox(
                                    label="Skip Face Cropping", value=False
                                )

                            with gr.Row():
                                l_a_cfg = gr.Slider(
                                    0.0, 5.0, value=2.0, step=0.1, label="Audio CFG"
                                )
                                l_r_cfg = gr.Slider(
                                    0.0, 3.0, value=1.0, step=0.1, label="Ref CFG"
                                )
                                l_e_cfg = gr.Slider(
                                    0.0, 3.0, value=1.0, step=0.1, label="Emotion CFG"
                                )

                            with gr.Row():
                                l_seed = gr.Slider(
                                    0, 10000, value=25, step=1, label="Random Seed"
                                )
                                l_decode_batch = gr.Radio(
                                    choices=["Auto", "1", "2", "4", "8", "16"],
                                    value="Auto",
                                    label="Decode Batch Size",
                                )

                            l_video_encoder = gr.Radio(
                                choices=get_encoder_choices(),
                                value=default_video_encoder,
                                label="Video Encoder Backend",
                                info="FFmpeg NVENC provides zero-overhead GPU encoding.",
                            )

                        l_mic = gr.Audio(
                            label="Microphone Audio Input",
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                        )

                        with gr.Row():
                            l_start_btn = gr.Button("▶️ Start Live", variant="primary", size="lg")
                            l_stop_btn = gr.Button("⏹️ Stop Live", size="lg")

                    with gr.Column(scale=1):
                        l_video_out = gr.Video(
                            label="Live Stream Playback",
                            height=400,
                            autoplay=True,
                            loop=False,
                        )
                        l_status = gr.Textbox(
                            label="Live Engine Telemetry",
                            lines=4,
                            elem_classes=["live-status"],
                        )

                l_timer = gr.Timer(0.4, active=False)

                # ---- Session Lifecycle -------------------------------------- #
                def start_live(ref_image, no_crop, session: Optional[LiveSession]):
                    if ref_image is None:
                        return (
                            session,
                            gr.update(active=False),
                            "❌ Upload a reference portrait image first.",
                            gr.update(visible=False),
                        )
                    try:
                        session = LiveSession()
                        session.ref_image_path = ref_image
                        session.running = True
                        session.session_dir = tempfile.mkdtemp(
                            prefix="float_live_", dir=model.live_dir
                        )

                        # Precompute image identity ONCE
                        t_prep0 = time.perf_counter()
                        prep = model.prepare_live_session(ref_image, no_crop=no_crop)
                        session.s_tensor = prep["s"]
                        session.r_s = prep["r_s"]
                        session.ref_feats = prep["ref_feats"]
                        session.cropped_preview = prep["cropped_preview"]
                        session.is_preprocessed = True
                        t_prep = time.perf_counter() - t_prep0

                        status_msg = (
                            f"🟢 Session Initialized ({t_prep:.2f}s)\n"
                            f"Identity pre-encoded & temporal cache primed. Start speaking into your mic!"
                        )
                        return (
                            session,
                            gr.update(active=True),
                            status_msg,
                            gr.update(value=session.cropped_preview, visible=True),
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"❌ Failed to initialize live session: {e}\n{tb}")
                        return (
                            None,
                            gr.update(active=False),
                            f"❌ Initialization failed: {e}",
                            gr.update(visible=False),
                        )

                def stop_live(session: Optional[LiveSession]):
                    if session is not None:
                        session.running = False
                        session.clear_gpu_cache()
                        if session.session_dir and os.path.isdir(session.session_dir):
                            shutil.rmtree(session.session_dir, ignore_errors=True)
                    return None, gr.update(active=False), "⏹️ Live stream stopped.", gr.update(visible=False)

                l_start_btn.click(
                    fn=start_live,
                    inputs=[l_ref_image, l_no_crop, session_state],
                    outputs=[session_state, l_timer, l_status, l_cropped_preview],
                )
                l_stop_btn.click(
                    fn=stop_live,
                    inputs=[session_state],
                    outputs=[session_state, l_timer, l_status, l_cropped_preview],
                )

                # ---- Continuous Mic Ingestion ------------------------------ #
                def on_audio_stream(chunk, session: Optional[LiveSession]):
                    if session is None or chunk is None or not session.running:
                        return session
                    sr, data = _to_mono_float(chunk)
                    if data.size == 0:
                        return session
                    session.sample_rate = sr
                    session.buffer.append(data)
                    session.buffered_seconds += len(data) / float(sr)
                    return session

                l_mic.stream(
                    fn=on_audio_stream,
                    inputs=[l_mic, session_state],
                    outputs=[session_state],
                )

                # ---- Periodic Inference Dispatch --------------------------- #
                def on_tick(
                    session: Optional[LiveSession],
                    chunk_len,
                    emotion,
                    a_cfg,
                    r_cfg,
                    e_cfg,
                    nfe,
                    seed,
                    decode_batch_size,
                    video_encoder,
                    temporal_continuity,
                ):
                    if session is None or not session.running:
                        return gr.update(), session, gr.update()

                    target_len = _coerce_float(chunk_len, CHUNK_SECONDS_DEFAULT)

                    if session.buffered_seconds < target_len or not session.buffer:
                        return (
                            gr.update(),
                            session,
                            gr.update(
                                value=f"🟢 Listening... buffering speech "
                                f"({session.buffered_seconds:.2f}s / {target_len:.2f}s)"
                            ),
                        )

                    if not session.session_dir or not os.path.isdir(session.session_dir):
                        os.makedirs(session.session_dir or model.live_dir, exist_ok=True)

                    # Flush accumulated audio for this chunk
                    audio = np.concatenate(session.buffer)
                    sr = session.sample_rate
                    session.reset_audio()
                    session.chunk_idx += 1
                    idx = session.chunk_idx

                    chunk_wav = os.path.join(session.session_dir, f"chunk_{idx:05d}.wav")
                    chunk_mp4 = os.path.join(session.session_dir, f"chunk_{idx:05d}.mp4")

                    try:
                        out_path, timings = model.run_live_chunk(
                            session=session,
                            audio_np=audio,
                            sr=sr,
                            out_path=chunk_mp4,
                            chunk_wav_path=chunk_wav,
                            emotion=emotion,
                            a_cfg=a_cfg,
                            r_cfg=r_cfg,
                            e_cfg=e_cfg,
                            nfe=nfe,
                            seed=seed,
                            decode_batch_size=decode_batch_size,
                            video_encoder=video_encoder,
                            temporal_continuity=temporal_continuity,
                        )
                        session.last_latency = timings["total"]
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"❌ Chunk {idx} failed: {e}\n{tb}")
                        return (
                            gr.update(),
                            session,
                            gr.update(value=f"❌ Chunk {idx} failed: {e}\n\n{tb}"),
                        )

                    # Rotate old chunk files to prevent disk usage growth
                    if idx > MAX_LIVE_CLIPS_KEPT:
                        old_idx = idx - MAX_LIVE_CLIPS_KEPT
                        for ext in (".wav", ".mp4"):
                            old_path = os.path.join(session.session_dir, f"chunk_{old_idx:05d}{ext}")
                            if os.path.exists(old_path):
                                try:
                                    os.remove(old_path)
                                except Exception:
                                    pass

                    rtf = session.last_latency / max(0.001, target_len)
                    rtf_str = f"{rtf:.2f}x Real-Time ({'Faster' if rtf < 1.0 else 'Slower'} than real-time)"
                    status = (
                        f"🟢 Live Chunk #{idx} [{target_len:.1f}s speech -> {timings['frames']} frames]\n"
                        f"⏱ Latency: {session.last_latency * 1000:.0f}ms | {timings['fps']:.1f} FPS | {rtf_str}\n"
                        f"📊 Breakdown: Sampler {timings['sample'] * 1000:.0f}ms | Decoder {timings['decode'] * 1000:.0f}ms | Encoder {timings['encode'] * 1000:.0f}ms\n"
                        f"🔄 Temporal Memory: {'Active (Seamless)' if temporal_continuity else 'Off'} | Encoder: {video_encoder}"
                    )
                    return gr.update(value=out_path), session, gr.update(value=status)

                l_timer.tick(
                    fn=on_tick,
                    inputs=[
                        session_state,
                        l_chunk_len,
                        l_emotion,
                        l_a_cfg,
                        l_r_cfg,
                        l_e_cfg,
                        l_nfe,
                        l_seed,
                        l_decode_batch,
                        l_video_encoder,
                        l_temporal_continuity,
                    ],
                    outputs=[l_video_out, session_state, l_status],
                )

                try:
                    demo.unload(
                        fn=stop_live,
                        inputs=[session_state],
                        outputs=[session_state, l_timer, l_status, l_cropped_preview],
                    )
                except Exception:
                    pass

        # =============================================================== #
        # Help & Technical Guide
        # =============================================================== #
        with gr.Accordion("ℹ️ Architecture Guide & Performance Tuning", open=False):
            gr.Markdown(
                """
                ### Performance Tuning Recommendations
                - **NFE (Number of Function Evaluations)**:
                  Setting NFE to **4–6** provides the single largest latency reduction (<350ms per chunk) while preserving crisp lip-sync.
                - **Chunk Duration**:
                  Setting Chunk Duration to **1.25s – 1.5s** strikes the optimal balance between natural speech context and low latency.
                - **Seamless Temporal Continuity**:
                  When enabled, FLOAT caches the last motion frames (`prev_x`, `prev_wa`) and feeds them into the next chunk.
                  This eliminates facial jumping or boundary seams between successive clips.
                - **Video Encoder**:
                  - **FFmpeg NVENC**: Direct hardware H.264 encoding via NVIDIA GPU (lowest CPU latency).
                  - **Torchcodec**: Direct PyTorch CUDA video encoding.
                  - **CPU PyAV**: Zero-subprocess in-memory libx264 software encoder.
                """
            )

    return demo


def parse_launch_args():
    import argparse
    import argparse as _argparse

    parser = argparse.ArgumentParser(description="Launch FLOAT Gradio Interface")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auth", type=str, default=None)
    parser.add_argument(
        "--auth_message", type=str, default="Please login to access FLOAT"
    )
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/float.pth")
    parser.add_argument("--res_dir", type=str, default="./results")
    parser.add_argument(
        "--wav2vec_model_path", type=str, default="facebook/wav2vec2-base-960h"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--queue", action=_argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_threads", type=int, default=4)
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--prevent_thread_lock", action="store_true")
    parser.add_argument("--ssl_keyfile", type=str, default=None)
    parser.add_argument("--ssl_certfile", type=str, default=None)
    parser.add_argument("--ssl_keyfile_password", type=str, default=None)
    parser.add_argument("--favicon_path", type=str, default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument(
        "--video_encoder",
        type=str,
        default=None,
        choices=["FFmpeg NVENC", "CPU PyAV", "Torchcodec"],
        help="Default video encoder ('FFmpeg NVENC', 'CPU PyAV', or 'Torchcodec'). Auto-detected if omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_launch_args()

    demo = build_interface(
        ckpt_path=args.ckpt_path,
        res_dir=args.res_dir,
        wav2vec_model_path=args.wav2vec_model_path,
        ngpus=args.ngpus,
        default_video_encoder=args.video_encoder,
    )

    auth_tuple = None
    if args.auth:
        try:
            username, password = args.auth.split(":", 1)
            auth_tuple = (username, password)
        except ValueError:
            logger.warning("⚠️ Invalid auth format. Use 'username:password'")

    logger.info("=" * 60)
    logger.info("🚀 Launching FLOAT Gradio Interface (with Live tab)")
    logger.info("=" * 60)
    logger.info(f"📍 Server: {args.server_name}:{args.port}")
    logger.info(f"💾 Results directory: {args.res_dir}")
    logger.info(f"🤖 Model checkpoint: {args.ckpt_path}")
    logger.info("=" * 60)

    blocked_paths = []
    ckpt_dir = os.path.dirname(args.ckpt_path)
    if ckpt_dir:
        blocked_paths.append(os.path.abspath(ckpt_dir))
    for d in ("models", ".venv", ".git", "config"):
        blocked_paths.append(os.path.abspath(d))

    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
        "inbrowser": args.inbrowser,
        "prevent_thread_lock": args.prevent_thread_lock,
        "allowed_paths": [os.path.abspath(args.res_dir)],
        "blocked_paths": blocked_paths,
    }
    if auth_tuple:
        launch_kwargs["auth"] = auth_tuple
        launch_kwargs["auth_message"] = args.auth_message
    if args.ssl_keyfile and args.ssl_certfile:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
        if args.ssl_keyfile_password:
            launch_kwargs["ssl_keyfile_password"] = args.ssl_keyfile_password
    if args.favicon_path:
        launch_kwargs["favicon_path"] = args.favicon_path
    if args.debug:
        launch_kwargs["debug"] = True
        logger.setLevel(logging.DEBUG)
        logging.getLogger("fast_float").setLevel(logging.DEBUG)
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        logging.getLogger("torch.fx").setLevel(logging.WARNING)

    try:
        if args.queue:
            demo.queue(max_size=args.max_threads)
        demo.launch(**launch_kwargs)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Error launching interface: {e}")
        if args.debug:
            raise
