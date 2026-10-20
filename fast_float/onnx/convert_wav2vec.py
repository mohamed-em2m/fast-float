"""Unit 2: convert the FLOAT wav2vec2 audio encoder to ONNX.

Wraps :class:`fast_float.models.wav2vec2.Wav2VecModel` (base-960h)
feature-projection + 12-layer transformer + the trained
``audio_encoder.audio_projection`` head from ``float.pth``::

    conv_feats [B, T, 512] -> audio feats [B, T, dim_w]

The conv front-end + ``F.interpolate`` stay in torch (milliseconds;
interpolate carries a dynamic output size that does not trace cleanly),
while all heavy compute runs in ONNX. The app feeds the graph the output
of ``Wav2VecModel.feature_extract``; batch and frames are dynamic, so one
file covers every clip length.

Usage (from repo root):
    python -m fast_float.onnx.convert_wav2vec --out ./onnx/wav2vec_v1.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Repo-root bootstrap (same rule as convert_emotion: only REPO_ROOT may be
# added — never fast_float/ itself, whose onnx/ subpackage would shadow
# the pip `onnx` dependency).
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.models.wav2vec2 import (
    Wav2VecModel,
    load_pretrained_with_weight_norm,
    restore_legacy_weight_norm,
)
from fast_float.onnx.export import (
    FP16_MAX_MEAN,
    FP16_MIN_COSINE,
    add_ort_args,
    export_module,
    fp16_tol,
    make_session_from_args,
)
from fast_float.onnx.parity import check_parity

PROJ_PREFIX = "audio_encoder.audio_projection."
N_LAYERS_STACKED = 12  # only_last_features=False: hidden_states[1:] of base model


class Wav2vecExportWrapper(torch.nn.Module):
    """Interpolated conv feats -> projected audio feats.

    Takes the *post-interpolation* conv features ``[B, T, 512]`` (produced
    in torch by ``Wav2VecModel.feature_extract``) through feature-projection
    + the 12-layer transformer + the trained ``audio_projection`` head::

        conv_feats [B, T, 512] -> audio feats [B, T, dim_w]

    Batch and frames are both dynamic and no ``Resize``/``seq_len`` input
    is needed, so one file covers every clip length. The conv front-end +
    interpolation stay in torch (milliseconds); all heavy compute
    (transformer + 9216-wide projection matmul) is in ONNX.
    ``return_dict=False`` keeps the graph tracer-friendly.
    """

    def __init__(self, core: Wav2VecModel, projection: torch.nn.Module):
        super().__init__()
        self.core = core
        self.projection = projection

    def forward(self, conv_feats: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.core.feature_projection(conv_feats)
        hidden = self.core._mask_hidden_states(hidden)
        _, all_hidden_states = self.core._encode_with_hidden_states(hidden)
        a = torch.stack(all_hidden_states[1:], dim=1).permute(0, 2, 1, 3)
        b, f, _, h = a.shape
        a = a.reshape(b, f, N_LAYERS_STACKED * h)
        return self.projection(a)


def stack_and_project(out, projection: torch.nn.Module) -> torch.Tensor:
    """Tail of ``AudioEncoder.get_wav2vec2_feature``: 12-layer stack ->
    reshape -> trained projection head. Shared by the parity reference so
    the converter checks the exact app computation."""
    a = torch.stack(out.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
    b, f, _, h = a.shape
    return projection(a.reshape(b, f, N_LAYERS_STACKED * h))


def load_core(w2v_dir: str) -> Wav2VecModel:
    """from_pretrained first (stock-ish subclass usually loads fine);
    manual config + state-dict load on the transformers>=4.4x finalizer
    incompatibility (same fallback as the emotion converter).

    Both branches run ``restore_legacy_weight_norm``: on torch>=2.1 the loader
    silently drops the checkpoint's ``pos_conv_embed.conv.weight_g/weight_v``
    and leaves the parametrized weight at its random init."""
    try:
        return load_pretrained_with_weight_norm(Wav2VecModel, w2v_dir, local_files_only=True)
    except AttributeError as e:
        print(f"[weights] from_pretrained incompatible ({e}); manual load")
    from transformers import Wav2Vec2Config

    cfg = Wav2Vec2Config.from_pretrained(w2v_dir, local_files_only=True)
    core = Wav2VecModel(cfg)
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(w2v_dir, name)
        if os.path.exists(path):
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file

                sd = load_file(path, device="cpu")
            else:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            break
    else:
        raise FileNotFoundError(f"no weight file in {w2v_dir}")
    missing, unexpected = core.load_state_dict(sd, strict=False)
    restored = restore_legacy_weight_norm(core, sd)
    print(
        f"[weights] manual load: {len(sd)} tensors (missing={len(missing)}, "
        f"unexpected={len(unexpected)}, weight_norm restored={restored})"
    )
    return core


def build_projection(ckpt_path: str) -> torch.nn.Module:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sub = {k[len(PROJ_PREFIX) :]: v for k, v in sd.items() if k.startswith(PROJ_PREFIX)}
    if not sub:
        raise KeyError(f"no {PROJ_PREFIX}* keys in {ckpt_path}")
    w = sub["0.weight"]
    in_dim, dim_w = w.shape[1], w.shape[0]
    print(f"[weights] projection: Linear({in_dim}->{dim_w}) + LayerNorm + SiLU")
    if in_dim != N_LAYERS_STACKED * 768:
        raise ValueError(
            f"ckpt projection expects {in_dim} inputs — this exporter covers only_last_features=False (12x768=9216)"
        )
    proj = torch.nn.Sequential(
        torch.nn.Linear(in_dim, dim_w),
        torch.nn.LayerNorm(dim_w),
        torch.nn.SiLU(),
    )
    proj.load_state_dict(sub, strict=True)
    print(f"[weights] loaded {len(sub)} projection tensors from {ckpt_path}")
    return proj


def build_export_module(w2v_dir: str, ckpt_path: str) -> Wav2vecExportWrapper:
    if not os.path.isdir(w2v_dir):
        raise FileNotFoundError(f"wav2vec dir not found: {w2v_dir}")
    core = load_core(w2v_dir)
    proj = build_projection(ckpt_path)
    return Wav2vecExportWrapper(core, proj).eval()


def frames_for(seconds: float, sr: int = 16000, fps: int = 25) -> int:
    import math

    return math.ceil(int(sr * seconds) * fps / sr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert FLOAT wav2vec2 encoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--w2v-dir", default="./checkpoints/wav2vec2-base-960h")
    ap.add_argument("--out", default="./onnx/wav2vec_v1.onnx")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument(
        "--seconds-alt",
        type=float,
        default=4.0,
        help="second clip length proving dynamic seq_len",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--skip-parity", action="store_true")
    add_ort_args(ap)
    args = ap.parse_args(argv)

    module = build_export_module(args.w2v_dir, args.ckpt)
    core = module.core
    # Trace sample: interpolated conv feats [B, T, 512] (batch 2 keeps the
    # batch dim dynamic under dynamo).
    sample_feats = torch.randn(2, frames_for(args.seconds), 512).clamp(-1, 1)

    info = export_module(
        module,
        sample_feats,
        args.out,
        input_names=["conv_feats"],
        output_names=["audio_feats"],
        dynamic_axes={
            "conv_feats": {0: "batch", 1: "frames"},
            "audio_feats": {0: "batch", 1: "frames"},
        },
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")

    if not args.skip_parity:
        sess, fp16 = make_session_from_args(
            info["path"],
            args,
            # conv_feats is dynamic in both batch (0) and frames (1); give TRT a
            # profile spanning the deployed clip range so an engine can build.
            opt_shapes={"conv_feats": [2, frames_for(args.seconds)]},
            extra_dynamic={"conv_feats": {1: (10, 2000)}},
        )
        print(f"[ort] providers={sess.get_providers()}")
        atol, rtol = fp16_tol(args.atol, fp16), fp16_tol(args.rtol, fp16)
        # Audio feats are bounded/small-magnitude, so under TRT FP16 the absolute
        # drift is not covered by the fp16_tol factor alone; add the functional
        # gate the other bounded units (emotion/refenc) use -- the same gate
        # ``select_backends`` re-applies when it measures the artifact.
        func = {"min_cosine": FP16_MIN_COSINE, "max_mean": FP16_MAX_MEAN} if fp16 else {}

        def run_case(name: str, wav: torch.Tensor, frames: int):
            # Reference: the full torch path (convs + interpolate + head).
            with torch.no_grad():
                ref = core(wav, seq_len=frames, return_dict=True)
                ref = stack_and_project(ref, module.projection)
                feats = core.feature_extract(wav, frames)
            ort_out = sess.run(None, {"conv_feats": feats.numpy()})[0]
            check_parity(ref, ort_out, name=name, atol=atol, rtol=rtol, **func)
            print(f"[probe:{name}] shape torch={tuple(ref.shape)} onnx={tuple(ort_out.shape)}")

        import math

        sr = 16000
        n = int(sr * args.seconds)
        t = frames_for(args.seconds)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            tt = torch.arange(n, dtype=torch.float32) / sr
            probes = {
                "noise": (torch.randn(1, n, generator=g).clamp(-1, 1), t),
                "sine": (
                    (0.5 * torch.sin(2 * math.pi * 130 * tt) + 0.3 * torch.sin(2 * math.pi * 260 * tt)).unsqueeze(0),
                    t,
                ),
                "batch2": (
                    torch.cat(
                        [
                            torch.randn(1, n, generator=torch.Generator().manual_seed(1)).clamp(-1, 1),
                            torch.randn(1, n, generator=torch.Generator().manual_seed(2)).clamp(-1, 1),
                        ],
                        dim=0,
                    ),
                    t,
                ),
                "altlen": (
                    torch.randn(
                        1,
                        int(sr * args.seconds_alt),
                        generator=torch.Generator().manual_seed(3),
                    ).clamp(-1, 1),
                    frames_for(args.seconds_alt),
                ),
                "short1s": (
                    torch.randn(1, sr, generator=torch.Generator().manual_seed(4)).clamp(-1, 1),
                    frames_for(1.0),
                ),
            }
            for name, (wav, frames) in probes.items():
                run_case(name, wav, frames)
    print("[done] wav2vec converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
