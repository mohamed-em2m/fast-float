"""Standalone video encoding utilities including SimpleVideoEncoder (MJPEG AVI without FFmpeg)."""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleVideoEncoder:
    """A pure-Python video encoder creating AVI files with Motion JPEG encoding (OpenCV)."""

    def __init__(self, width: int, height: int, fps: int, output_path: str) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.frame_count = 0
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the video. Frame should be RGB uint8 numpy array (H, W, 3)."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match video size ({self.height}, {self.width}, 3)"
            )

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr_frame)
        self.frame_count += 1

    def write_avi_with_audio(self, audio_path: Optional[str] = None) -> str:
        """Write frames to AVI file using OpenCV VideoWriter."""
        if self.frame_count == 0:
            raise ValueError("No frames to write!")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        temp_video = self.output_path.replace(".mp4", "_temp.avi").replace(
            ".avi", "_temp.avi"
        )

        out = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")

        logger.info(f"Writing {self.frame_count} frames to {temp_video}...")
        for frame in tqdm(self.frames, desc="Encoding video"):
            out.write(frame)

        out.release()
        logger.info(f"✅ Video encoding complete: {temp_video}")

        if audio_path and os.path.exists(audio_path):
            logger.warning(
                "Audio merging without ffmpeg is limited. Saving video-only version."
            )
            logger.info(
                f"To add audio manually, use: ffmpeg -i {temp_video} -i {audio_path} -c:v copy -c:a aac {self.output_path}"
            )

        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rename(temp_video, self.output_path)

        return self.output_path

    def write(self) -> str:
        """Write frames without audio."""
        return self.write_avi_with_audio(audio_path=None)
