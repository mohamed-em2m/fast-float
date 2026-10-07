import os
import io
import base64
import tempfile
from typing import Optional
import numpy as np
import cv2
from PIL import Image
from pydub import AudioSegment
import soundfile as sf

from fast_float.utils.logger import logger

DEFAULT_IMAGE_PATH = "./images/img.jpg"
TARGET_IMAGE_SIZE = 256


def ensure_default_image(image_path: str = DEFAULT_IMAGE_PATH, size: int = TARGET_IMAGE_SIZE) -> str:
    """Create default placeholder image if it does not already exist."""
    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
        logger.warning(f"Default image not found at {image_path}, creating placeholder")
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Default",
            (size // 4, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(image_path, img)
        logger.info(f"Created default {size}x{size} image at {image_path}")
    return image_path


def resize_image_to_256(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Load image, convert alpha/RGBA to RGB, and save cleanly as JPEG.
    """
    try:
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

        img = Image.open(image_path)

        # Fix alpha transparency by pasting on white background
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        img.save(output_path, "JPEG", quality=95)
        return output_path

    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        raise


def convert_pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM audio bytes to valid WAV bytes with validation."""
    try:
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
            offset = 128
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
            offset = 0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
            offset = 0
        else:
            dtype = np.int16
            max_val = 32768.0
            offset = 0

        audio_array = np.frombuffer(pcm_data, dtype=dtype)
        if len(audio_array) == 0:
            raise ValueError("Empty audio data")

        if channels > 1:
            remainder = len(audio_array) % channels
            if remainder != 0:
                audio_array = audio_array[: len(audio_array) - remainder]
            audio_array = audio_array.reshape(-1, channels)

        if dtype == np.uint8:
            audio_float = (audio_array.astype(np.float32) - offset) / max_val
        else:
            audio_float = audio_array.astype(np.float32) / max_val

        audio_float = np.clip(audio_float, -1.0, 1.0)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        if len(wav_bytes) < 44:
            raise ValueError(f"Generated WAV too small: {len(wav_bytes)} bytes")

        return wav_bytes

    except Exception as e:
        logger.error(f"PCM to WAV conversion failed: {e}")
        raise


def process_audio_format(
    audio_content: str,
    ext: str,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Decode base64 audio and ensure output is a valid 16kHz mono WAV byte buffer."""
    try:
        audio_bytes = base64.b64decode(audio_content)
        ext = ext.lower().lstrip(".")

        logger.debug(
            f"Processing audio: format={ext}, size={len(audio_bytes)} bytes, sr={sample_rate}, ch={channels}"
        )

        if ext in ["pcm", "raw", "wav"]:
            wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
        else:
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                if audio_segment.frame_rate != sample_rate:
                    audio_segment = audio_segment.set_frame_rate(sample_rate)

                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
            except Exception as e:
                logger.warning(f"Audio container conversion failed: {e}, attempting raw PCM fallback")
                wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)

        if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            raise ValueError("Invalid WAV format after conversion")

        return wav_bytes

    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise
