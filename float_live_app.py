"""
FLOAT - Gradio Interface for Audio-Driven Talking Face Generation
Now with a "Live" tab: microphone audio -> chunked, near-real-time video.

IMPORTANT — read this before you run it
----------------------------------------
FLOAT is a diffusion-based model: it generates each clip through NFE denoising
steps. That means it cannot produce a continuous, frame-perfect live video
stream the way a webcam filter can — there is no way around running inference
on some window of audio before you get pixels back.

What this file gives you instead is a "chunked near-real-time" pipeline:

  1. Your microphone audio is captured continuously in small pieces.
  2. Those pieces are buffered until you have ~CHUNK_SECONDS of audio.
  3. FLOAT runs inference on that chunk (keep NFE low, e.g. 4-8, for speed).
  4. The resulting short clip is pushed to the video player and autoplays.
  5. The next chunk starts buffering immediately, so clips arrive back-to-back.

End-to-end latency ≈ CHUNK_SECONDS + (inference time for that chunk).
On a fast GPU with NFE=6-8 you can get this down to roughly 1-3 seconds of
lag. On CPU or with high NFE it will feel more like "semi-live" batches
than a live call. There will also be a small visual "seam" between clips
since each chunk is generated independently (no temporal state is carried
across chunks) — increasing CHUNK_SECONDS reduces how often you see a seam,
at the cost of more latency.

Requires Gradio >= 4.40 (for gr.Timer and streaming gr.Audio input).
    pip install --upgrade gradio soundfile numpy
"""

import os
import time
import queue
import shutil
import tempfile
import datetime
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import gradio as gr

from generate import InferenceAgent, InferenceOptions


# --------------------------------------------------------------------------- #
# Shared model wrapper (one instance, shared across all sessions/users)
# --------------------------------------------------------------------------- #
class FloatModel:
    def __init__(self):
        self.opt = InferenceOptions().parse()
        self.opt.rank, self.opt.ngpus = 0, 1
        os.makedirs(self.opt.res_dir, exist_ok=True)

        self.live_dir = os.path.join(self.opt.res_dir, "live_chunks")
        os.makedirs(self.live_dir, exist_ok=True)

        print("Loading FLOAT model...")
        self.agent = InferenceAgent(self.opt)
        # A lock around inference is the safe default: most diffusion
        # pipelines aren't written to be called concurrently from multiple
        # threads on one GPU. If your agent IS safe for concurrent calls
        # (e.g. you've wrapped it with its own batching), you can remove
        # this lock to let multiple users' chunks process in parallel.
        self.infer_lock = threading.Lock()
        print("Model loaded successfully!")

    def run(self, ref_path, audio_path, out_path, emotion, a_cfg, r_cfg,
            e_cfg, nfe, seed, no_crop):
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
            )

    # ---- batch (original) path, unchanged in spirit from the original app ---
    def generate_video(self, ref_image, audio_file, emotion, a_cfg_scale,
                        r_cfg_scale, e_cfg_scale, nfe, seed, no_crop,
                        progress=gr.Progress()):
        try:
            progress(0, desc="Preparing inputs...")
            if ref_image is None:
                return None, "❌ Please upload a reference image"
            if audio_file is None:
                return None, "❌ Please upload an audio file"

            video_name = Path(ref_image).stem
            audio_name = Path(audio_file).stem
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            res_video_path = os.path.join(
                self.opt.res_dir,
                f"{call_time}-{video_name}-{audio_name}-nfe{nfe}-seed{seed}"
                f"-acfg{a_cfg_scale}-ecfg{e_cfg_scale}-{emotion}.mp4",
            )

            progress(0.3, desc="Running inference...")
            output_path = self.run(ref_image, audio_file, res_video_path,
                                    emotion, a_cfg_scale, r_cfg_scale,
                                    e_cfg_scale, nfe, seed, no_crop)
            progress(1.0, desc="Complete!")
            return output_path, f"✅ Saved to: {output_path}"
        except Exception as e:
            return None, f"❌ Error during generation: {e}"


# --------------------------------------------------------------------------- #
# Per-session live state (NOT shared between users — held in a gr.State)
# --------------------------------------------------------------------------- #
@dataclass
class LiveSession:
    ref_image_path: Optional[str] = None
    sample_rate: Optional[int] = None
    buffer: list = field(default_factory=list)   # list of 1-D numpy float chunks
    buffered_seconds: float = 0.0
    chunk_idx: int = 0
    running: bool = False
    last_latency: float = 0.0
    session_dir: Optional[str] = None

    def reset_audio(self):
        self.buffer = []
        self.buffered_seconds = 0.0


CHUNK_SECONDS = 1.5      # how much audio to accumulate before generating a clip
MAX_LIVE_CLIPS_KEPT = 20  # rotate old temp files so disk doesn't fill up


def _to_mono_float(chunk_sr_data):
    """Gradio streaming mic audio arrives as (sample_rate, np.ndarray)."""
    sr, data = chunk_sr_data
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        # int16/int32 PCM -> float32 [-1, 1]
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return sr, data


def build_interface():
    model = FloatModel()

    # ---------- custom CSS ----------
    custom_css = """
    .gradio-container { font-family: 'Arial', sans-serif; }
    .live-status { font-weight: 600; }
    """

    with gr.Blocks(css=custom_css, title="FLOAT - Talking Face Generation") as demo:
        gr.Markdown(
            """
            # 🎭 FLOAT: Audio-Driven Talking Face Generation
            Generate talking face videos from a reference image and audio —
            as a one-shot batch job, or as a near-real-time live stream.
            """
        )

        with gr.Tabs():
            # =========================================================== #
            # TAB 1 — Original batch generation (upload full audio file)
            # =========================================================== #
            with gr.Tab("📁 Batch (upload audio)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Input Files")
                        b_ref_image = gr.Image(label="Reference Image", type="filepath",
                                                sources=["upload"], height=300)
                        b_audio_file = gr.Audio(label="Audio File", type="filepath",
                                                 sources=["upload"])

                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            b_emotion = gr.Dropdown(
                                choices=['S2E', 'angry', 'disgust', 'fear', 'happy',
                                         'neutral', 'sad', 'surprise'],
                                value='S2E', label="Emotion Control")
                            b_no_crop = gr.Checkbox(label="Skip Face Cropping", value=False)
                            b_seed = gr.Slider(0, 10000, value=25, step=1, label="Random Seed")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Generation Parameters")
                        with gr.Group():
                            b_nfe = gr.Slider(1, 50, value=10, step=1, label="NFE")
                            b_a_cfg = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Audio CFG Scale")
                            b_r_cfg = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Reference CFG Scale")
                            b_e_cfg = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Emotion CFG Scale")

                        b_generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                        b_status = gr.Textbox(label="Status", lines=3)
                        b_video_out = gr.Video(label="Generated Video", height=400)

                b_generate_btn.click(
                    fn=model.generate_video,
                    inputs=[b_ref_image, b_audio_file, b_emotion, b_a_cfg, b_r_cfg,
                            b_e_cfg, b_nfe, b_seed, b_no_crop],
                    outputs=[b_video_out, b_status],
                )

            # =========================================================== #
            # TAB 2 — Live (microphone -> chunked near-real-time video)
            # =========================================================== #
            with gr.Tab("🔴 Live (microphone)"):
                gr.Markdown(
                    f"""
                    **How it works:** upload a reference face once, hit **Start**,
                    then talk into your mic. Every ~{CHUNK_SECONDS:.1f}s of speech is
                    turned into a short clip and played back automatically.
                    There is inherent latency (buffering + inference time) — this is
                    *near*-real-time, not frame-perfect live video. See the header
                    comment in this file for why.
                    """
                )

                session_state = gr.State(LiveSession())

                with gr.Row():
                    with gr.Column(scale=1):
                        l_ref_image = gr.Image(label="Reference Image (set once)",
                                                type="filepath", sources=["upload"], height=280)

                        with gr.Accordion("⚙️ Live Generation Settings", open=True):
                            gr.Markdown("*Keep NFE low for lower latency.*")
                            l_emotion = gr.Dropdown(
                                choices=['S2E', 'angry', 'disgust', 'fear', 'happy',
                                         'neutral', 'sad', 'surprise'],
                                value='S2E', label="Emotion Control")
                            l_no_crop = gr.Checkbox(label="Skip Face Cropping", value=False)
                            l_seed = gr.Slider(0, 10000, value=25, step=1, label="Random Seed")
                            l_nfe = gr.Slider(1, 20, value=6, step=1,
                                               label="NFE (lower = faster/less latency)")
                            l_a_cfg = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Audio CFG Scale")
                            l_r_cfg = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Reference CFG Scale")
                            l_e_cfg = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Emotion CFG Scale")

                        l_mic = gr.Audio(
                            label="Microphone",
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                        )

                        with gr.Row():
                            l_start_btn = gr.Button("▶️ Start Live", variant="primary")
                            l_stop_btn = gr.Button("⏹️ Stop")

                    with gr.Column(scale=1):
                        l_video_out = gr.Video(label="Live Output", height=400,
                                                autoplay=True, loop=False)
                        l_status = gr.Textbox(label="Live Status", lines=3,
                                               elem_classes=["live-status"])

                # Ticks periodically; each tick checks whether enough audio has
                # been buffered, and if so, runs one inference call.
                l_timer = gr.Timer(0.5, active=False)

                # ---- session lifecycle -------------------------------------------------
                def start_live(ref_image, session: LiveSession):
                    if ref_image is None:
                        return session, gr.update(active=False), "❌ Upload a reference image first."
                    session = LiveSession()
                    session.ref_image_path = ref_image
                    session.running = True
                    session.session_dir = tempfile.mkdtemp(
                        prefix="float_live_", dir=model.live_dir)
                    return session, gr.update(active=True), "🟢 Listening... start talking."

                def stop_live(session: LiveSession):
                    session.running = False
                    if session.session_dir and os.path.isdir(session.session_dir):
                        shutil.rmtree(session.session_dir, ignore_errors=True)
                    return session, gr.update(active=False), "⏹️ Stopped."

                l_start_btn.click(
                    fn=start_live,
                    inputs=[l_ref_image, session_state],
                    outputs=[session_state, l_timer, l_status],
                )
                l_stop_btn.click(
                    fn=stop_live,
                    inputs=[session_state],
                    outputs=[session_state, l_timer, l_status],
                )

                # ---- audio ingestion ----------------------------------------------------
                def on_audio_stream(chunk, session: LiveSession):
                    if chunk is None or not session.running:
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

                # ---- periodic inference tick --------------------------------------------
                def on_tick(session: LiveSession, emotion, a_cfg, r_cfg, e_cfg,
                            nfe, seed, no_crop):
                    if not session.running:
                        return gr.update(), session, gr.update()

                    if session.buffered_seconds < CHUNK_SECONDS or not session.buffer:
                        return gr.update(), session, gr.update(
                            value=f"🟢 Listening... buffering audio "
                                  f"({session.buffered_seconds:.1f}s / {CHUNK_SECONDS:.1f}s)")

                    # Pull everything currently buffered and reset for the next chunk.
                    audio = np.concatenate(session.buffer)
                    sr = session.sample_rate
                    session.reset_audio()
                    session.chunk_idx += 1
                    idx = session.chunk_idx

                    chunk_wav = os.path.join(session.session_dir, f"chunk_{idx:05d}.wav")
                    chunk_mp4 = os.path.join(session.session_dir, f"chunk_{idx:05d}.mp4")
                    sf.write(chunk_wav, audio, sr)

                    t0 = time.time()
                    try:
                        out_path = model.run(
                            ref_path=session.ref_image_path,
                            audio_path=chunk_wav,
                            out_path=chunk_mp4,
                            emotion=emotion, a_cfg=a_cfg, r_cfg=r_cfg,
                            e_cfg=e_cfg, nfe=nfe, seed=seed, no_crop=no_crop,
                        )
                    except Exception as e:
                        return gr.update(), session, gr.update(value=f"❌ Chunk {idx} failed: {e}")

                    session.last_latency = time.time() - t0

                    # Rotate old chunk files so a long session doesn't fill disk.
                    if idx > MAX_LIVE_CLIPS_KEPT:
                        old_idx = idx - MAX_LIVE_CLIPS_KEPT
                        for ext in (".wav", ".mp4"):
                            old_path = os.path.join(
                                session.session_dir, f"chunk_{old_idx:05d}{ext}")
                            if os.path.exists(old_path):
                                os.remove(old_path)

                    status = (f"🟢 Live — chunk {idx} generated in "
                              f"{session.last_latency:.1f}s "
                              f"(clip length {CHUNK_SECONDS:.1f}s)")
                    return gr.update(value=out_path), session, gr.update(value=status)

                l_timer.tick(
                    fn=on_tick,
                    inputs=[session_state, l_emotion, l_a_cfg, l_r_cfg, l_e_cfg,
                            l_nfe, l_seed, l_no_crop],
                    outputs=[l_video_out, session_state, l_status],
                )

        with gr.Accordion("ℹ️ Help & Information", open=False):
            gr.Markdown(
                """
                ## Batch tab
                Same as the original app: upload a full audio file, get one video back.

                ## Live tab
                1. Upload a reference image and click **Start Live**.
                2. Allow microphone access when your browser prompts you.
                3. Speak — every ~1.5s of audio becomes a short clip that autoplays.
                4. Click **Stop** to end the session (this also clears temp files).

                **Tuning for lower latency:**
                - Lower `NFE` (try 4-8) — this is the single biggest latency lever.
                - Shorten `CHUNK_SECONDS` at the top of this file for less buffering
                  delay per clip (but more per-clip inference overhead and more
                  visible seams between clips).
                - Run on a GPU; CPU inference will not keep up with speech in real time.

                **Why not truly frame-by-frame live?** FLOAT denoises each clip over
                multiple steps (NFE); there's no streaming/causal inference path in
                the base model, so the smallest unit of output is "one clip," not
                "one frame." Getting below ~1s of latency reliably would require a
                distilled/one-step variant of the model or an architecture built for
                causal streaming generation — that's a model-level change, not
                something this Gradio wrapper can add on its own.
                """
            )

    return demo


def parse_launch_args():
    import argparse
    parser = argparse.ArgumentParser(description='Launch FLOAT Gradio Interface')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--server_name', type=str, default="0.0.0.0")
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--auth', type=str, default=None)
    parser.add_argument('--auth_message', type=str, default="Please login to access FLOAT")
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/float.pth")
    parser.add_argument('--res_dir', type=str, default="./results")
    parser.add_argument('--wav2vec_model_path', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--queue', action='store_true', default=True)
    parser.add_argument('--max_threads', type=int, default=4)
    parser.add_argument('--inbrowser', action='store_true')
    parser.add_argument('--prevent_thread_lock', action='store_true')
    parser.add_argument('--ssl_keyfile', type=str, default=None)
    parser.add_argument('--ssl_certfile', type=str, default=None)
    parser.add_argument('--ssl_keyfile_password', type=str, default=None)
    parser.add_argument('--favicon_path', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_launch_args()

    import sys
    inference_args = []
    if args.ckpt_path:
        inference_args.extend(['--ckpt_path', args.ckpt_path])
    if args.res_dir:
        inference_args.extend(['--res_dir', args.res_dir])
    if args.wav2vec_model_path:
        inference_args.extend(['--wav2vec_model_path', args.wav2vec_model_path])

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + inference_args
    demo = build_interface()
    sys.argv = original_argv

    auth_tuple = None
    if args.auth:
        try:
            username, password = args.auth.split(':', 1)
            auth_tuple = (username, password)
        except ValueError:
            print("⚠️ Invalid auth format. Use 'username:password'")

    print("\n" + "=" * 60)
    print("🚀 Launching FLOAT Gradio Interface (with Live tab)")
    print("=" * 60)
    print(f"📍 Server: {args.server_name}:{args.port}")
    print(f"💾 Results directory: {args.res_dir}")
    print(f"🤖 Model checkpoint: {args.ckpt_path}")
    print("=" * 60 + "\n")

    launch_kwargs = {
        'server_name': args.server_name,
        'server_port': args.port,
        'share': args.share,
        'show_error': True,
        'inbrowser': args.inbrowser,
        'prevent_thread_lock': args.prevent_thread_lock,
    }
    if auth_tuple:
        launch_kwargs['auth'] = auth_tuple
        launch_kwargs['auth_message'] = args.auth_message
    if args.ssl_keyfile and args.ssl_certfile:
        launch_kwargs['ssl_keyfile'] = args.ssl_keyfile
        launch_kwargs['ssl_certfile'] = args.ssl_certfile
        if args.ssl_keyfile_password:
            launch_kwargs['ssl_keyfile_password'] = args.ssl_keyfile_password
    if args.favicon_path:
        launch_kwargs['favicon_path'] = args.favicon_path
    if args.debug:
        launch_kwargs['debug'] = True

    try:
        if args.queue:
            demo.queue(max_size=args.max_threads)
        demo.launch(**launch_kwargs)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error launching interface: {e}")
        if args.debug:
            raise
