import os
import time
import base64
import tempfile
import datetime
import asyncio
from typing import Dict, Tuple, Any

import cv2

from fast_float.utils.logger import logger
from fast_float.utils.media import (
    DEFAULT_IMAGE_PATH,
    ensure_default_image,
    resize_image_to_256,
    process_audio_format,
)
from fast_float.utils.chunking import (
    split_wav_into_chunks,
    merge_mp4_chunks,
    get_audio_duration_seconds,
)
from fast_float.core.state import app_state

ALLOWED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "webp"}


def _probe_frame_count(video_path: str) -> int:
    """Best-effort frame count probe for fps reporting. Returns 0 on failure."""
    try:
        cap = cv2.VideoCapture(video_path)
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return max(count, 0)
    except Exception:
        return 0


def run_inference_blocking(
    agent: Any,
    video_path_raw: str,
    image_path: str,
    audio_path: str,
    request_data: Dict[str, Any],
) -> str:
    """Synchronous inference runner invoked within ThreadPoolExecutor."""
    return agent.run_inference(
        res_video_path=video_path_raw,
        ref_path=image_path,
        audio_path=audio_path,
        a_cfg_scale=request_data.get("a_cfg_scale", 2.0),
        r_cfg_scale=request_data.get("r_cfg_scale", 1.0),
        e_cfg_scale=request_data.get("e_cfg_scale", 1.0),
        emo=request_data.get("emotion", "S2E"),
        nfe=request_data.get("nfe", 10),
        no_crop=request_data.get("no_crop", False),
        seed=request_data.get("seed", 25),
        verbose=True,
    )


def _prepare_image(request_data: Dict[str, Any], temp_files: list) -> str:
    """Decode + resize the reference image (or fall back to the default)."""
    if request_data.get("image_base64"):
        image_bytes = base64.b64decode(request_data["image_base64"])
        image_format = str(request_data.get("image_format", "jpg")).lower().lstrip(".")
        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image_format '{image_format}'")
        image_path_temp = tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{image_format}"
        ).name
        with open(image_path_temp, "wb") as f:
            f.write(image_bytes)
        temp_files.append(image_path_temp)

        image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        resize_image_to_256(image_path_temp, image_path)
        temp_files.append(image_path)
        return image_path

    return ensure_default_image(DEFAULT_IMAGE_PATH)


async def generate_video(
    request_data: Dict[str, Any],
    request_id: str,
    state=app_state,
) -> Tuple[str, Dict[str, float]]:
    """Complete pipeline to process inputs, run model inference, and return final video.

    When ``chunk_seconds`` > 0 and the audio is longer than one window, the audio
    is sliced into overlapping chunks, each rendered independently (with progress
    reported to ``state.progress``), then merged back into a single MP4.
    """
    temp_files: list = []
    total_start_time = time.time()
    timings: Dict[str, float] = {}

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    try:
        logger.info(f"[{request_id}] 🚀 Starting video generation")
        state.set_progress(request_id, status="processing", stage="preprocessing", percent=1.0)

        # 1. Audio processing
        preprocess_start = time.time()
        wav_bytes = process_audio_format(
            request_data["audio_base64"],
            request_data.get("audio_format", "pcm"),
            request_data.get("sample_rate", 16000),
            request_data.get("channels", 1),
            request_data.get("sample_width", 2),
        )

        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        temp_files.append(audio_path)

        # 2. Image processing
        image_path = _prepare_image(request_data, temp_files)
        timings["preprocess"] = time.time() - preprocess_start

        # 3. Output paths
        res_dir = (
            getattr(state.opt, "res_dir", "./results") if state.opt else "./results"
        )
        os.makedirs(res_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path_final = os.path.join(res_dir, f"{stamp}_{request_id}.mp4")

        fps = int(getattr(state.opt, "fps", 25)) if state.opt else 25
        encoder = (
            state.video_optimizer.selected_encoder
            if state.video_optimizer and state.video_optimizer.selected_encoder
            else "libx264"
        )

        if state.inference_lock is None:
            state.inference_lock = asyncio.Lock()

        # 4. Decide single-pass vs chunked based on audio length
        chunk_seconds = float(request_data.get("chunk_seconds", 0.0) or 0.0)
        overlap_seconds = float(request_data.get("chunk_overlap_seconds", 0.5) or 0.0)
        duration = get_audio_duration_seconds(audio_path)

        inference_start = time.time()
        num_chunks = 1

        async with state.inference_lock:
            if chunk_seconds > 0 and duration > chunk_seconds:
                num_chunks, output_path = await _generate_chunked(
                    loop, state, request_data, request_id, audio_path, image_path,
                    res_dir, stamp, chunk_seconds, overlap_seconds, encoder, fps, temp_files,
                )
            else:
                state.set_progress(
                    request_id, status="processing", stage="generating",
                    total_chunks=1, completed_chunks=0, percent=5.0,
                )
                output_path = await loop.run_in_executor(
                    state.executor,
                    run_inference_blocking,
                    state.agent,
                    video_path_final,
                    image_path,
                    audio_path,
                    request_data,
                )

        timings["inference"] = time.time() - inference_start
        timings["total"] = time.time() - total_start_time
        timings["chunks"] = num_chunks

        # Real video fps: encoded frames per second of inference wall time.
        frame_count = _probe_frame_count(output_path)
        timings["fps"] = (
            frame_count / timings["inference"]
            if frame_count > 0 and timings["inference"] > 0
            else 0.0
        )

        # 5. Update performance stats
        stats = state.performance_stats
        stats["total_requests"] += 1
        stats["total_inference_time"] += timings["inference"]
        stats["total_preprocess_time"] += timings["preprocess"]
        stats["total_frames"] = stats.get("total_frames", 0) + frame_count
        if stats["total_inference_time"] > 0:
            stats["average_fps"] = stats["total_frames"] / stats["total_inference_time"]

        state.set_progress(request_id, status="done", stage="done", percent=100.0,
                           total_chunks=num_chunks, completed_chunks=num_chunks)

        # 6. Clean temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                logger.debug(f"[{request_id}] Cleanup warning: {cleanup_error}")

        # 7. Verify final video
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Final video not found at {output_path}")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"[{request_id}] ✅ Complete: {timings['total']:.2f}s "
            f"(prep: {timings['preprocess']:.2f}s, inf: {timings['inference']:.2f}s, "
            f"chunks: {num_chunks}) | {file_size:.2f}MB"
        )

        state.set_progress(request_id, status="done", stage="done", percent=100.0,
                           message=f"{num_chunks} chunk(s)")

        return output_path, timings

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Generation failed: {str(e)}", exc_info=True)
        state.set_progress(request_id, status="error", stage="error", message=str(e))
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        raise


async def _generate_chunked(
    loop,
    state,
    request_data: Dict[str, Any],
    request_id: str,
    audio_path: str,
    image_path: str,
    res_dir: str,
    stamp: str,
    chunk_seconds: float,
    overlap_seconds: float,
    encoder: str,
    fps: int,
    temp_files: list,
) -> Tuple[int, str]:
    """Render each audio chunk, then merge. Returns (chunk_count, merged_path)."""
    chunks = split_wav_into_chunks(
        audio_path, chunk_seconds, overlap_seconds, out_dir=res_dir
    )
    # The original file is only reused when no split happened (single element).
    for ch in chunks:
        if ch.get("synthetic"):
            temp_files.append(ch["path"])

    num_chunks = len(chunks)
    state.set_progress(
        request_id, status="processing", stage="generating",
        total_chunks=num_chunks, completed_chunks=0, percent=0.0,
        message=f"0/{num_chunks} chunks",
    )
    logger.info(f"[{request_id}] 🧩 Chunked generation: {num_chunks} chunks")

    chunk_videos: list = []
    for i, ch in enumerate(chunks):
        state.set_progress(
            request_id, stage="generating", completed_chunks=i,
            percent=round(i / num_chunks * 90, 1),
            message=f"{i}/{num_chunks} chunks",
        )
        chunk_out = os.path.join(res_dir, f"{stamp}_{request_id}_chunk{i:03d}.mp4")
        await loop.run_in_executor(
            state.executor,
            run_inference_blocking,
            state.agent,
            chunk_out,
            image_path,
            ch["path"],
            request_data,
        )
        chunk_videos.append(chunk_out)
        temp_files.append(chunk_out)

    state.set_progress(
        request_id, status="merging", stage="merging",
        completed_chunks=num_chunks, percent=92.0, message="Merging chunks",
    )

    merged_path = os.path.join(res_dir, f"{stamp}_{request_id}.mp4")
    await loop.run_in_executor(
        state.executor,
        merge_mp4_chunks,
        chunk_videos,
        merged_path,
        overlap_seconds,
        encoder,
        fps,
    )
    return num_chunks, merged_path
