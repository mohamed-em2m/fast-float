import asyncio
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure current directory and repository root are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for path in (str(CURRENT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fast_float.initialize_env import initialize_environment, initialize_onnx_lib_paths

initialize_environment()
initialize_onnx_lib_paths()

from fast_float.core.inference_optimized import (
    InferenceOptions,
    OptimizedInferenceAgent,
)
from fast_float.core.service import (
    run_inference_batched_blocking,
    run_inference_blocking,
)
from fast_float.core.state import app_state
from fast_float.routers import api_router
from fast_float.utils.gpu import print_gpu_status
from fast_float.utils.logger import logger
from fast_float.utils.media import (
    DEFAULT_IMAGE_PATH,
    TARGET_IMAGE_SIZE,
    ensure_default_image,
)
from fast_float.utils.video_optimizer import VideoSaveOptimizer

# Keep strong references to fire-and-forget tasks so they are not garbage
# collected mid-run (RUF006).
_BACKGROUND_TASKS: set[asyncio.Task] = set()

# Configuration
PORT = int(os.getenv("PORT", "8000"))

# Watchdog for the background boot warmup. The warmup runs in the thread pool,
# so a wedged inference (dead CUDA context, driver reset) would otherwise keep
# warming_up=True forever and every request would wait out its timeout. The
# thread cannot be killed; we only guarantee that the server stops reporting
# "warming up" and says what happened.
WARMUP_TIMEOUT_S = float(os.getenv("WARMUP_TIMEOUT_S", "900"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
OPTIMIZATION_MODE = os.getenv("OPTIMIZATION_MODE", "all")
USE_FP16 = os.getenv("USE_FP16", "true").lower() in ("true", "1", "yes")
# Comma-separated list of allowed browser origins, e.g.
# CORS_ORIGINS=http://localhost:3000,https://studio.example.com
# localhost and 127.0.0.1 are different origins to a browser, so allow both.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()
]


async def _run_boot_warmups():
    """Boot warmup (single + B=2 batched) in a background task.

    Runs after startup completes so /health answers immediately. Failure is
    isolated (sequential inference is unaffected). Clears warming_up always.
    """
    loop = asyncio.get_running_loop()
    errors: list[str] = []

    async def _run(fn, *args):
        # Bounded: a wedged inference must not keep warming_up set forever.
        return await asyncio.wait_for(
            loop.run_in_executor(app_state.executor, fn, *args),
            timeout=WARMUP_TIMEOUT_S,
        )

    try:
        logger.info("🔥 Warming up model with minimal inference...")
        app_state.set_warmup_stage("single-inference")
        warmup_start = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as _tmp_wa:
                warmup_audio = _tmp_wa.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _tmp_wv:
                warmup_video = _tmp_wv.name
            silent_audio = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz
            sf.write(warmup_audio, silent_audio, 16000)

            await _run(
                run_inference_blocking,
                app_state.agent,
                warmup_video,
                DEFAULT_IMAGE_PATH,
                warmup_audio,
                {
                    "a_cfg_scale": 2.0,
                    "r_cfg_scale": 1.0,
                    "e_cfg_scale": 1.0,
                    "emotion": "S2E",
                    "nfe": 10,
                    "no_crop": False,
                    "seed": 25,
                },
            )
            for f in (warmup_audio, warmup_video):
                if os.path.exists(f):
                    os.remove(f)
            logger.info(f"✅ Model warmup complete ({time.time() - warmup_start:.2f}s)")
        except Exception as e:
            errors.append(f"warmup: {e}")
            logger.warning(f"⚠️ Model warmup skipped/failed: {e}")

        # Batched-shape warmup (B=2 silent clips) so the first real batched
        # request doesn't pay cuDNN autotuning on unseen batch shapes.
        app_state.set_warmup_stage("batched-inference")
        try:
            batch_audios, batch_videos = [], []
            for _ in range(2):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as _tmp_wa:
                    wa = _tmp_wa.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _tmp_wv:
                    wv = _tmp_wv.name
                sf.write(wa, np.zeros(32000, dtype=np.float32), 16000)  # 2s silence
                batch_audios.append(wa)
                batch_videos.append(wv)
            await _run(
                run_inference_batched_blocking,
                app_state.agent,
                [
                    {
                        "audio_path": wa,
                        "padded_path": wa,
                        "duration": 2.0,
                        "out_path": wv,
                    }
                    for wa, wv in zip(batch_audios, batch_videos, strict=False)
                ],
                DEFAULT_IMAGE_PATH,
                {
                    "a_cfg_scale": 2.0,
                    "r_cfg_scale": 1.0,
                    "e_cfg_scale": 1.0,
                    "emotion": "S2E",
                    "nfe": 2,
                    "no_crop": False,
                    "seed": 25,
                },
            )
            for f in batch_audios + batch_videos:
                if os.path.exists(f):
                    os.remove(f)
            logger.info("✅ Batched warmup complete (B=2)")
        except Exception as e:
            errors.append(f"batched warmup: {e}")
            logger.warning(f"⚠️ Batched warmup skipped: {e}")
    except asyncio.CancelledError:
        app_state.finish_warmup("cancelled")
        raise
    finally:
        # Always releases requests waiting in /generate (see
        # AppState.wait_for_warmup) — including after a timeout or failure.
        app_state.finish_warmup("; ".join(errors) or None)
        logger.info(f"🏁 Boot warmup done ({app_state.warmup_elapsed():.1f}s), /generate open")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model loading, warmup, and resource cleanup."""
    app_state.initialize_runtime()
    app_state.max_concurrent_requests = MAX_CONCURRENT_REQUESTS

    logger.info("=" * 70)
    logger.info("🚀 Starting FLOAT Fast API - NVENC Optimized v5.1.0")
    logger.info(f"⚡ Optimization: {OPTIMIZATION_MODE} | FP16: {USE_FP16}")
    logger.info(f"📐 Target resolution: {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}")

    # 1. Initialize video optimizer
    app_state.video_optimizer = VideoSaveOptimizer()
    logger.info(f"🎬 Video Encoder: {app_state.video_optimizer.encoder_type}")
    print_gpu_status()
    logger.info("=" * 70)

    # 2. Ensure default image exists
    ensure_default_image(DEFAULT_IMAGE_PATH, TARGET_IMAGE_SIZE)

    # 3. Initialize inference options
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]

    opt = InferenceOptions().parse()
    opt.rank = 0
    # run_all.sh / run_api.sh export NGPUS (decode GPUs); sys.argv belongs to
    # hypercorn so the flag can't be passed on the command line.
    opt.ngpus = int(os.getenv("NGPUS", str(getattr(opt, "ngpus", 1))))
    opt.res_dir = RESULTS_DIR
    opt.no_crop = False
    opt.optimization_mode = OPTIMIZATION_MODE
    opt.use_fp16 = USE_FP16
    opt.n_diff_steps = 100
    opt.diff_schedule = "cosine"
    opt.diffusion_mode = "sample"
    sys.argv = orig_argv

    os.makedirs(opt.res_dir, exist_ok=True)
    app_state.opt = opt

    # 4. Load FLOAT model
    logger.info(f"🤖 Loading FLOAT model with {OPTIMIZATION_MODE} optimization...")
    app_state.agent = OptimizedInferenceAgent(opt)
    logger.info("✅ Model loaded successfully")

    # 5. Boot warmup runs AFTER startup (background task) so /health answers
    # immediately. Set SKIP_WARMUP=1 to disable. /generate returns 503
    # while app_state.warming_up is True.
    if os.getenv("SKIP_WARMUP", "0") == "1":
        logger.info("⏭️ Boot warmup skipped (SKIP_WARMUP=1)")
    else:
        app_state.begin_warmup()
        _task = asyncio.create_task(_run_boot_warmups())
        _BACKGROUND_TASKS.add(_task)
        _task.add_done_callback(_BACKGROUND_TASKS.discard)

    # 6. Public URLs setup
    app_state.public_url = os.getenv("AVATER_PUBLIC_URL", f"http://localhost:{PORT}")
    app_state.public_backup_url = os.getenv("BACKUP_AVATER_PUBLIC_URL")

    logger.info("=" * 70)
    logger.info("📺 FLOAT Fast API Ready!")
    logger.info(f"🌐 Public URL: {app_state.public_url}")
    if app_state.public_backup_url:
        logger.info(f"🌐 Backup URL: {app_state.public_backup_url}")
    logger.info(f"🎬 Encoder: {app_state.video_optimizer.encoder_type}")
    logger.info("=" * 70)

    yield

    # Shutdown logic
    logger.info("🛑 Shutting down server...")
    from fast_float.routers.webrtc import _cleanup_peer

    for pc in list(app_state.pcs):
        try:
            await pc.close()
        except Exception:
            pass
        finally:
            _cleanup_peer(pc)

    app_state.shutdown()
    logger.info("👋 Shutdown complete.")


# FastAPI Application
app = FastAPI(
    title="FLOAT WebRTC API - NVENC Optimized",
    version="5.1.0",
    lifespan=lifespan,
)

# NOTE: No GZipMiddleware on purpose. Starlette's GZip compresses every
# streaming response (including video/mp4, only text/event-stream is
# excluded), which wastes CPU on already-compressed video and corrupts
# byte-range streaming. JSON API responses are a few hundred bytes and
# compress to nothing anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular routers
app.include_router(api_router)


@app.get("/", tags=["system"])
async def root():
    """Service status and quick links."""
    return {
        "service": "FLOAT Talking-Face Video Generation API",
        "version": "5.1.0",
        "status": "online",
        "docs_url": "/docs",
        "public_url": app_state.public_url,
    }


if __name__ == "__main__":
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    config.loglevel = "info"
    config.startup_timeout = 600

    asyncio.run(serve(app, config))
