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

# Only the repository root belongs on sys.path. Adding fast_float/ itself makes
# its subpackages importable as top-level modules, which shadows real pip
# packages: `fast_float/livekit` would be imported as `livekit` and break the
# LiveKit RTC SDK (`from livekit import rtc`) with a circular import.
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.initialize_env import initialize_environment, initialize_onnx_lib_paths

initialize_environment()
initialize_onnx_lib_paths()

from fast_float.logging_utils import (
    RequestContextMiddleware,
    configure_logging,
    current_log_file,
    install_asyncio_exception_handler,
    reroute_server_loggers,
)

configure_logging(role="api")

from fast_float.config.float_settings import load_float_options
from fast_float.core.inference_optimized import OptimizedInferenceAgent
from fast_float.core.service import (
    run_inference_batched_blocking,
    run_inference_blocking,
)
from fast_float.core.state import app_state
from fast_float.livekit import selftest
from fast_float.livekit.settings import LiveKitSettings
from fast_float.routers import api_router
from fast_float.utils.gpu import print_gpu_status
from fast_float.utils.logger import logger
from fast_float.utils.media import (
    DEFAULT_IMAGE_PATH,
    ensure_default_image,
)
from fast_float.utils.stun import StunServer
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
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
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
    # Unify hypercorn/uvicorn loggers with the app sinks (the servers create
    # their log config lazily, after the app import).
    reroute_server_loggers()
    install_asyncio_exception_handler(asyncio.get_running_loop())
    app_state.initialize_runtime()
    app_state.max_concurrent_requests = MAX_CONCURRENT_REQUESTS

    # Model/inference options come from settings.toml [float] (env values are
    # only fallbacks for boxes without the shared file). Load them before the
    # model is constructed so architecture, ODE and decode knobs agree.
    opt = load_float_options()
    opt.rank = 0
    app_state.opt = opt

    # Built-in STUN responder for ICE gathering on the raw WebRTC transport.
    # The API host itself is the STUN endpoint (instead of requiring LiveKit's
    # coturn or a public Google server); an explicit [webrtc].stun_urls list
    # disables it. Best effort: a busy UDP port never blocks startup.
    stun_server: StunServer | None = None
    ice_settings = LiveKitSettings.from_env()
    if ice_settings.configured:
        logger.info(f"🔑 LiveKit: {ice_settings.url} (key {ice_settings.api_key[:6]}…, config: {ice_settings.source})")
    else:
        logger.info("🔑 LiveKit: not configured — the SFU live transport is disabled")
    if ice_settings.stun_enabled and not ice_settings.stun_urls:
        stun_server = StunServer(port=ice_settings.stun_port)
        if await stun_server.start():
            app_state.stun_port = ice_settings.stun_port

    # Probe the SFU in the background so /health and /livekit/status answer
    # immediately; the result populates the Config checklist's connectivity
    # row. Best effort — a failure only shows as a warning, never blocks boot.
    selftest.start_probe(ice_settings)

    logger.info("=" * 70)
    logger.info("🚀 Starting FLOAT Fast API - NVENC Optimized v5.1.0")
    logger.info(f"🪵 Logs: role=api file={current_log_file() or 'console only'}")
    logger.info(f"⚡ Optimization: {opt.optimization_mode} | FP16: {opt.use_fp16}")
    logger.info(f"📐 Target resolution: {opt.input_size}x{opt.input_size}")

    # 1. Initialize video optimizer
    app_state.video_optimizer = VideoSaveOptimizer()
    logger.info(f"🎬 Video Encoder: {app_state.video_optimizer.encoder_type}")
    print_gpu_status()
    logger.info("=" * 70)

    # 2. Ensure default image exists (at the model's own resolution, so the
    # placeholder is not upscaled by the preprocess transform)
    ensure_default_image(DEFAULT_IMAGE_PATH, int(opt.input_size))

    # 3. Prepare runtime directories from the configured options
    os.makedirs(opt.res_dir, exist_ok=True)

    # 4. Load FLOAT model
    logger.info(f"🤖 Loading FLOAT model with {opt.optimization_mode} optimization...")
    app_state.agent = OptimizedInferenceAgent(opt)
    logger.info("✅ Model loaded successfully")

    # 4b. Dual-process health check. Live conversations render through the
    # worker's shared-memory ring; warn at boot instead of failing the first
    # conversation with an opaque FileNotFoundError inside the request.
    try:
        from fast_float.config.dynamic import load_settings
        from fast_float.ipc import ipc_enabled, segment_exists

        ipc_cfg = load_settings().get("ipc", {})
        ipc_cfg = ipc_cfg if isinstance(ipc_cfg, dict) else {}
        if ipc_enabled(ipc_cfg):
            shm_name = str(ipc_cfg.get("shm_name", "float_frame_ring"))
            if segment_exists(shm_name):
                logger.info(f"🧵 IPC worker detected (shm={shm_name}) — live rendering runs in the worker")
            else:
                logger.warning(
                    "⚠️  [ipc] enabled but shared memory %r is not published: the inference "
                    "worker is not running, so live conversations will return 503. Start the "
                    "full stack with ./scripts/run_dual_process.sh, or set FLOAT_IPC_ENABLED=false "
                    "to render in-process (./scripts/run_all.sh does that automatically).",
                    shm_name,
                )
        else:
            logger.info("🧵 IPC worker not used (FLOAT_IPC_ENABLED / [ipc] enabled) — live rendering is in-process")
    except Exception as exc:  # pragma: no cover - diagnostics must never block boot
        logger.debug(f"IPC boot check skipped: {exc}")

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

    if stun_server is not None:
        await stun_server.stop()
        app_state.stun_port = None

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
# Outermost: every request (including CORS preflights) gets a request id and
# a bound logging context, and the id is echoed back in X-Request-ID.
app.add_middleware(RequestContextMiddleware)

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

    from fast_float.logging_utils.config import HEALTH_PATHS, access_log_enabled
    from fast_float.logging_utils.uvicorn_logging import create_hypercorn_log_config

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    config.loglevel = "info"
    # Access logs are opt-in (``FLOAT_ACCESS_LOG=1``). Hypercorn disables them
    # when ``accesslog`` is None (its default); "-" makes it create the
    # ``hypercorn.access`` logger, which the log config below routes through our
    # formatters/handlers (health checks filtered). Leave it off unless asked -
    # one line per request, including every /progress poll.
    config.accesslog = "-" if access_log_enabled() else None
    config.startup_timeout = 600

    # Hypercorn builds its Logger lazily (after this import), which would
    # overwrite the handlers installed by configure_logging(). Apply the
    # unified config explicitly; server records get their own rotating file so
    # they never race the app's logs/api.log.
    server_log = current_log_file()
    config.logconfig_dict = create_hypercorn_log_config(
        excluded_paths=list(HEALTH_PATHS),
        log_file=str(server_log.with_name("server.log")) if server_log else None,
    )

    asyncio.run(serve(app, config))
