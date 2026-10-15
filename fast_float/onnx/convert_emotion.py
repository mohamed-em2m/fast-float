"""Unit 1: convert the S2E emotion classifier to ONNX.

Builds ``Wav2Vec2ForSpeechClassification`` from the local emotion dir,
overlays the deployed ``float.pth`` weights (``emotion_encoder.*`` prefix,
mirroring the app's loader), exports waveform -> 7-dim probabilities, then
proves torch-vs-ORT parity on two probe inputs.

Usage (from repo root):
    python -m fast_float.onnx.convert_emotion --out ./onnx/emotion_v1.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Repo-root bootstrap so this works via `python -m` and as a plain script.
# NOTE: only REPO_ROOT may be added here. Never add fast_float/ itself:
# it contains an `onnx/` subpackage that would shadow the pip `onnx`
# dependency (breaking `import onnx.external_data_helper` via onnxscript).
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.models.wav2vec2_ser import (  # noqa: E402
    Wav2Vec2ForSpeechClassification,
)
from fast_float.onnx.export import export_module, make_ort_session  # noqa: E402
from fast_float.onnx.parity import check_parity  # noqa: E402

EMOTION_PREFIX = "emotion_encoder.wav2vec2_for_emotion."
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EmotionExportWrapper(torch.nn.Module):
    """Waveform -> emotion probabilities. ``return_dict=False`` keeps the graph
    tracer-friendly (plain tuple out of the HF module)."""

    def __init__(self, core: Wav2Vec2ForSpeechClassification):
        super().__init__()
        self.core = core

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        logits = self.core(input_values, return_dict=False)[0]
        return torch.softmax(logits, dim=-1)


def load_core_manual(emo_dir: str) -> Wav2Vec2ForSpeechClassification:
    """Build the classifier without ``from_pretrained``.

    New transformers versions run a finalization step
    (``_move_missing_keys_from_meta_to_device``) that assumes loader
    internals our locally-forked SER class predates, crashing the load even
    though the weights are fine. Manual config + state-dict load sidesteps it.
    """
    from transformers import Wav2Vec2Config

    # New transformers also call tie_weights() inside __init__/init_weights via
    # an attribute the fork predates. Every weight is overwritten from disk
    # right below (plus the float.pth overlay after), so init-time tying is a
    # safe no-op — and skipping it keeps old transformers behavior identical.
    Wav2Vec2ForSpeechClassification.tie_weights = lambda self, *a, **k: None

    cfg = Wav2Vec2Config.from_pretrained(emo_dir, local_files_only=True)
    core = Wav2Vec2ForSpeechClassification(cfg)
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(emo_dir, name)
        if os.path.exists(path):
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file

                sd = load_file(path, device="cpu")
            else:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            break
    else:
        raise FileNotFoundError(f"no weight file in {emo_dir}")
    missing, unexpected = core.load_state_dict(sd, strict=False)
    print(
        f"[weights] manual load: {len(sd)} tensors "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    try:
        core.tie_weights()
    except Exception:
        pass
    return core


def build_core(emo_dir: str) -> Wav2Vec2ForSpeechClassification:
    try:
        return Wav2Vec2ForSpeechClassification.from_pretrained(
            emo_dir, local_files_only=True
        )
    except AttributeError as e:
        print(f"[weights] from_pretrained incompatible ({e}); manual load")
        return load_core_manual(emo_dir)


def build_emotion_module(emo_dir: str, ckpt_path: str) -> EmotionExportWrapper:
    if not os.path.isdir(emo_dir):
        raise FileNotFoundError(
            f"emotion dir not found: {emo_dir} "
            "(run scripts/download_checkpoints.sh first)"
        )
    core = build_core(emo_dir)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        from collections import Counter

        tops = Counter(k.split(".")[0] for k in sd.keys())
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        sub = {
            k[len(EMOTION_PREFIX):]: v
            for k, v in sd.items()
            if k.startswith(EMOTION_PREFIX)
        }
        if not sub:
            print(
                "[weights] WARN: no emotion_encoder.* keys in ckpt — exporting "
                "base emo-dir weights (same fallback the app uses)"
            )
        missing, unexpected = core.load_state_dict(sub, strict=False)
        print(
            f"[weights] overlaid {len(sub)} tensors from "
            f"{ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"[weights] WARN: {ckpt_path} not found — exporting base "
            "emotion weights (parity still valid, weights differ from deploy)"
        )
    return EmotionExportWrapper(core).eval()


def probe_inputs(seconds: float, sr: int = 16000, seed: int = 0):
    n = int(sr * seconds)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, n, generator=g).clamp(-1, 1)
    t = torch.arange(n, dtype=torch.float32) / sr
    # speech-like: harmonic stack with vibrato, normalized like a waveform
    sine = (
        0.5 * torch.sin(2 * np.pi * 130 * t)
        + 0.3 * torch.sin(2 * np.pi * 260 * t)
        + 0.2 * torch.sin(2 * np.pi * 390 * t)
    ).unsqueeze(0)
    return {"noise": noise, "sine": sine}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert S2E emotion classifier to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument(
        "--emo-dir",
        default="./checkpoints/wav2vec-english-speech-emotion-recognition",
    )
    ap.add_argument("--out", default="./onnx/emotion_v1.onnx")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=30,
                    help="warmed benchmark iters per probe (0 = skip)")
    ap.add_argument("--cpu-ort", action="store_true",
                    help="allow CPU ORT session on a CUDA box (dev only)")
    args = ap.parse_args(argv)

    module = build_emotion_module(args.emo_dir, args.ckpt)
    sample = torch.randn(1, int(16000 * args.seconds)).clamp(-1, 1)

    info = export_module(
        module,
        sample,
        args.out,
        input_names=["input_values"],
        output_names=["probs"],
        dynamic_axes={"input_values": {0: "batch", 1: "length"}, "probs": {0: "batch"}},
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")

    if not args.skip_parity:
        sess = make_ort_session(
            info["path"],
            use_gpu=torch.cuda.is_available() and not args.cpu_ort,
            require_gpu=torch.cuda.is_available() and not args.cpu_ort,
        )
        print(f"[ort] providers={sess.get_providers()}")
        with torch.no_grad():
            for name, wav in probe_inputs(args.seconds).items():
                ref = module(wav)
                ort_out = sess.run(None, {"input_values": wav.numpy()})[0]
                check_parity(ref, ort_out, name=name, atol=args.atol, rtol=args.rtol)
                top = int(np.argmax(ort_out[0]))
                print(f"[probe:{name}] torch_top={LABELS[int(np.argmax(ref.numpy()[0]))]} "
                      f"onnx_top={LABELS[top]}")
                if args.bench_iters > 0:
                    from fast_float.onnx.parity import benchmark
                    wav_np = wav.numpy()
                    benchmark(
                        lambda: module(wav),
                        lambda: sess.run(None, {"input_values": wav_np}),
                        name=name, iters=args.bench_iters,
                    )
    print("[done] emotion converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
