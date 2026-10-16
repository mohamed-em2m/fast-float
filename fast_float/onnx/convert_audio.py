"""Convert the audio condition encoder (unit 4) to ONNX.

Exports ``AudioEncoder``'s deployed path (wav2vec2-base core + hidden-state
stack + projection) at a STATIC frame count T (--frames) and exact waveform
length (T * sr / fps). The app calls inference() once per request with
T = ceil(audio_len * fps / sr), which varies per request — so the runtime
pads/trims to the exported T exactly like the decoder's static batch
(see convert_decoder.py). Dynamic batch only.

Bakes the ``only_last_features`` flag at build time (deployed default False
= stack 12 layers). Reuses the emotion converter's manual-load fallback for
new-transformers incompatibility.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn

from fast_float.onnx.export import export_module, make_ort_session
from fast_float.onnx.parity import benchmark, check_parity

AUDIO_PREFIX = "audio_encoder."
DIM_W = 512


def _patch_tie_weights(cls):
    # New transformers call tie_weights() inside __init__/init_weights via an
    # attribute older subclasses predate. All weights load from disk right
    # after, so init-time tying is a safe no-op.
    cls.tie_weights = lambda self, *a, **k: None


def load_w2v_core(w2v_dir: str):
    """Wav2Vec2 core with from_pretrained, else manual load. Returns (core, n_layers)."""
    from fast_float.models.wav2vec2 import Wav2VecModel

    if not os.path.isdir(w2v_dir):
        print(f"[weights] WARN: {w2v_dir} not found — random-init core (parity only)")
        from transformers import Wav2Vec2Config

        cfg = Wav2Vec2Config()
        _patch_tie_weights(Wav2VecModel)
        return Wav2VecModel(cfg), cfg.num_hidden_layers, False
    try:
        core = Wav2VecModel.from_pretrained(w2v_dir, local_files_only=True)
        print("[weights] w2v core via from_pretrained")
        return core, core.config.num_hidden_layers, True
    except Exception as e:
        # New transformers (>=4.4x) can raise TypeError, RuntimeError, or
        # AttributeError. Always fall back to manual load.
        print(f"[weights] from_pretrained incompatible ({type(e).__name__}: {e}); manual load")
        from transformers import Wav2Vec2Config

        cfg = Wav2Vec2Config.from_pretrained(w2v_dir, local_files_only=True)
        _patch_tie_weights(Wav2VecModel)
        core = Wav2VecModel(cfg)
        from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

        for name in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME, "model.safetensors",
                     "pytorch_model.bin", "rust_model.ot"):
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
        print(f"[weights] manual load: {len(sd)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        try:
            core.tie_weights()
        except Exception:
            pass
        return core, cfg.num_hidden_layers, True


class AudioExport(nn.Module):
    """Deployed inference() path, exact-length input, baked T and flag."""

    def __init__(self, w2v, projection: nn.Module, only_last: bool, seq_len: int):
        super().__init__()
        self.w2v = w2v
        self.projection = projection
        self.only_last = only_last
        self.seq_len = seq_len

    def forward(self, a):
        out = self.w2v(a, seq_len=self.seq_len, output_hidden_states=True)
        if self.only_last:
            h = out.last_hidden_state
        else:
            h = torch.stack(out.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            h = h.reshape(h.shape[0], h.shape[1], -1)
        return self.projection(h)


def reference_inference(w2v, projection, only_last: bool, a, seq_len: int,
                        sampling_rate: int, fps: float):
    """Mirror of AudioEncoder.inference (pad/trim + same calls)."""
    expected_len = int(seq_len * sampling_rate / fps)
    if a.shape[1] < expected_len:
        a = torch.nn.functional.pad(a, (0, expected_len - a.shape[1]), mode="replicate")
    elif a.shape[1] > expected_len:
        a = a[:, :expected_len]
    out = w2v(a, seq_len=seq_len, output_hidden_states=not only_last)
    if only_last:
        h = out.last_hidden_state
    else:
        h = torch.stack(out.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
        h = h.reshape(h.shape[0], h.shape[1], -1)
    return projection(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert audio encoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--w2v-dir", default="./checkpoints/wav2vec2-base-960h")
    ap.add_argument("--out", default="./onnx/audio_enc_v1.onnx")
    ap.add_argument("--frames", type=int, default=300,
                    help="static output frame count T (runtime pads/trims to it)")
    ap.add_argument("--sampling-rate", type=int, default=16000)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--dim-w", type=int, default=512)
    ap.add_argument("--only-last-features", action="store_true")
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument("--cpu-ort", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    w2v, n_layers, _ = load_w2v_core(args.w2v_dir)
    hid = w2v.config.hidden_size
    in_dim = hid if args.only_last_features else n_layers * hid
    print(f"[model] w2v layers={n_layers} hid={hid} only_last={args.only_last_features} "
          f"proj_in={in_dim} T={args.frames}")

    projection = nn.Sequential(
        nn.Linear(in_dim, args.dim_w), nn.LayerNorm(args.dim_w), nn.SiLU()
    )
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        tops = Counter(k.split(".")[0] for k in sd.keys())
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        sub = {k[len(AUDIO_PREFIX):]: v for k, v in sd.items()
               if k.startswith(AUDIO_PREFIX)}
        # ckpt carries only the projection (frozen base lives in the HF dir).
        proj_sub = {k: v for k, v in sub.items() if k.startswith("audio_projection.")}
        if proj_sub:
            mapped = {k[len("audio_projection."):]: v for k, v in proj_sub.items()}
            missing, unexpected = projection.load_state_dict(mapped, strict=False)
            print(f"[weights] overlaid projection {len(mapped)} tensors "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("[weights] WARN: no audio_encoder.audio_projection.* keys — "
                  "random projection (parity still valid)")
    else:
        print(f"[weights] WARN: {args.ckpt} not found — random projection")

    w2v.to(device).eval()
    projection.to(device).eval()
    module = AudioExport(w2v, projection, args.only_last_features,
                         args.frames).eval()

    wav_len = int(args.frames * args.sampling_rate / args.fps)
    print(f"[shapes] waveform_len={wav_len} frames={args.frames}")
    batches = [int(s) for s in args.batches.split(",") if s.strip()]
    sample = torch.randn(1, wav_len, device=device).clamp(-1, 1)

    # Export on CPU (tracer constant-folder trips on mixed devices).
    module.to("cpu")
    info = export_module(
        module,
        sample.detach().cpu(),
        args.out,
        input_names=["waveform"],
        output_names=["wa"],
        dynamic_axes={"waveform": {0: "batch"}, "wa": {0: "batch"}},
        opset=args.opset,
        mark_dims=(0,),  # length baked via interpolate size; batch only
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")
    module.to(device)

    if not args.skip_parity:
        sess = make_ort_session(
            info["path"],
            use_gpu=torch.cuda.is_available() and not args.cpu_ort,
            require_gpu=torch.cuda.is_available() and not args.cpu_ort,
        )
        print(f"[ort] providers={sess.get_providers()}")
        with torch.no_grad():
            for bi, b in enumerate(batches):
                g = torch.Generator(device="cpu").manual_seed(500 + bi)
                a = torch.randn(b, wav_len, generator=g).clamp(-1, 1).to(device)
                ref = reference_inference(w2v, projection, args.only_last_features,
                                          a, args.frames, args.sampling_rate, args.fps)
                ort_out = sess.run(None, {"waveform": a.detach().cpu().numpy()})[0]
                check_parity(ref, ort_out, name=f"audio_b{b}",
                             atol=args.atol, rtol=args.rtol)
                if args.bench_iters > 0:
                    a_np = a.detach().cpu().numpy()
                    benchmark(
                        lambda: reference_inference(
                            w2v, projection, args.only_last_features, a,
                            args.frames, args.sampling_rate, args.fps),
                        lambda: sess.run(None, {"waveform": a_np}),
                        name=f"audio_b{b}",
                        iters=args.bench_iters,
                    )
    print("[done] audio converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
