"""CLI inference options parser for FLOAT optimized inference."""

from __future__ import annotations

from typing import Any

try:
    from fast_float.schemas.base_options import BaseOptions
except ImportError:
    try:
        from schemas.base_options import BaseOptions
    except ImportError:
        from options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
    """Command-line options configuration for FLOAT optimized inference."""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self, parser: Any) -> Any:
        super().initialize(parser)
        parser.add_argument("--ref_path", default=None, type=str, help="reference image")
        parser.add_argument("--aud_path", default=None, type=str, help="audio file")
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="skip face cropping")
        parser.add_argument("--res_video_path", default=None, type=str, help="output video path")
        parser.add_argument(
            "--ckpt_path",
            default="./checkpoints/float.pth",
            type=str,
            help="checkpoint path",
        )
        parser.add_argument("--res_dir", default="./results", type=str, help="result directory")

        parser.add_argument(
            "--optimization_mode",
            default="all",
            type=str,
            choices=[
                "none",
                "torch_compile",
                "channels_last",
                "amp",
                "cudnn_benchmark",
                "all",
            ],
            help="Optimization strategy to use",
        )
        parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")
        parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
        parser.add_argument(
            "--video_encoder",
            default="FFmpeg NVENC",
            type=str,
            choices=["FFmpeg NVENC", "CPU PyAV", "Torchcodec"],
            help="Video encoding backend ('FFmpeg NVENC', 'CPU PyAV', or 'Torchcodec')",
        )

        return parser
