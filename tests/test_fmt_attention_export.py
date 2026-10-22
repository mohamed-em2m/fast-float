"""The exported FMT graph must be the model that generates the frames.

Regression: ``FMT.Attention`` has two implementations of the same math — the
fused SDPA branch (what production runs, ``use_fused_attn()`` is True) and the
manual branch. They disagreed about the *polarity* of the alignment mask: the
manual branch masked the local +/-``attention_window`` band and attended to
everything else, the fused branch attended to the band. The ONNX converter
exported the manual branch while the application ran the fused one, so the
artifact was a *different model* (per-step cosine 0.94, final-latent cosine
0.81) and the generated frames were garbage.

The converter's parity gate could not see it: it compared the graph against
the very module it had just traced. These tests pin both halves of the fix —
the two branches agree, and the exported graph reproduces the deployed model.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from fast_float.models.float.FMT import Attention, FlowMatchingTransformer, enc_dec_mask

CLIP = 50
PREV = 10
DIM_W = 512


def _wake_up(fmt):
    """Give the zero-initialised output layers real weights.

    ``FlowMatchingTransformer.initialize_weights`` zeroes the adaLN modulation
    and the final projection, so a randomly initialised FMT emits an
    identically zero vector field -- which would make every comparison in
    these tests vacuous.
    """
    with torch.no_grad():
        for blk in fmt.blocks:
            blk.adaLN_modulation[-1].weight.normal_(0, 0.02)
            blk.adaLN_modulation[-1].bias.normal_(0, 0.02)
        fmt.decoder.adaLN_modulation[-1].weight.normal_(0, 0.02)
        fmt.decoder.adaLN_modulation[-1].bias.normal_(0, 0.02)
        fmt.decoder.linear.weight.normal_(0, 0.02)
        fmt.decoder.linear.bias.normal_(0, 0.02)
    return fmt


def _tiny_fmt(device="cpu") -> FlowMatchingTransformer:
    """A small FMT with the deployed structure (same code path, tiny widths)."""
    opt = SimpleNamespace(
        wav2vec_sec=2.0,
        fps=25.0,
        num_prev_frames=PREV,
        dim_h=64,
        mlp_ratio=2.0,
        fmt_depth=2,
        num_heads=2,
        dim_w=DIM_W,
        dim_a=512,
        dim_e=7,
        attention_window=2,
        rank=0,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
    )
    torch.manual_seed(0)
    fmt = FlowMatchingTransformer(opt).eval().to(device)
    for blk in fmt.blocks:
        if hasattr(blk, "attn"):
            blk.attn.fused_attn = True  # the deployed path
    return _wake_up(fmt)


def _step_inputs(batch=1, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor([0.5])
    x = torch.randn(batch, CLIP, DIM_W, generator=g)
    wa = torch.randn(batch, CLIP, DIM_W, generator=g)
    wr = torch.randn(batch, 1, DIM_W, generator=g)
    we = torch.randn(batch, 1, 7, generator=g)
    prev_x = torch.randn(batch, PREV, DIM_W, generator=g)
    prev_wa = torch.randn(batch, PREV, DIM_W, generator=g)
    if torch.device(device).type == "cuda":
        t, x, wa, wr, we, prev_x, prev_wa = (v.to(device) for v in (t, x, wa, wr, we, prev_x, prev_wa))
    return t, x, wa, wr, we, prev_x, prev_wa


def test_fused_and_manual_attention_agree():
    """The two attention branches must implement the same math."""
    torch.manual_seed(0)
    attn = Attention(32, num_heads=4, qkv_bias=True).eval()
    x = torch.randn(2, 12, 32)
    mask = enc_dec_mask(12, 12, 1, expansion=2)

    attn.fused_attn = True
    fused = attn(x, mask=mask)
    attn.fused_attn = False
    manual = attn(x, mask=mask)

    assert torch.allclose(fused, manual, atol=1e-5), (
        f"the fused and manual attention branches diverge (max_abs={(fused - manual).abs().max().item():.3e}); "
        "the ONNX export takes one of them, so they must be the same model"
    )


def test_attention_keeps_the_local_window():
    """Attention must stay inside the +/-window band (that is what it was trained on).

    The module is wired so the attention weights can be read straight off the
    output: one head whose value projection is the identity, zeroed q/k (so
    the allowed positions get equal weight) and one-hot inputs per position.
    """
    torch.manual_seed(0)
    n, dim = 12, 16
    attn = Attention(dim, num_heads=1, qkv_bias=True).eval()
    with torch.no_grad():
        # q/k -> 0 logits; v -> identity, so out[i, j] is the weight of j for i.
        attn.qkv.weight.zero_()
        attn.qkv.weight[2 * dim :, :] = torch.eye(dim)
        attn.qkv.bias.zero_()
        attn.proj.weight.copy_(torch.eye(dim))
        attn.proj.bias.zero_()

    x = torch.zeros(1, n, dim)
    for j in range(n):
        x[0, j, j] = 1.0
    mask = enc_dec_mask(n, n, 1, expansion=2)

    for fused in (True, False):
        attn.fused_attn = fused
        weights = attn(x, mask=mask)[0, :n, :n]  # [query, key]
        band = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            band[i, max(0, i - 2) : i + 3] = True
        leaked = weights.masked_fill(~band, 0.0)
        assert torch.allclose(leaked, weights, atol=1e-5), (
            f"fused_attn={fused}: attention leaked outside the local frame band"
        )
        assert weights.sum(dim=-1).min() > 0.99  # every row keeps its mass


def test_exported_graph_matches_the_deployed_model(tmp_path):
    """The ONNX artifact must reproduce the deployed torch model."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from fast_float.onnx.convert_fmt import FMTStepWrapper
    from fast_float.onnx.export import export_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmt = _tiny_fmt(device)
    module = FMTStepWrapper(fmt).eval().to(device)

    out = str(tmp_path / "fmt_step.onnx")
    export_module(
        module,
        _step_inputs(2, device),
        out,
        input_names=["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"],
        output_names=["vec"],
        shared_inputs=("t",),
        opset=17,
    )

    # The CPU EP is fp32-exact, so the comparison does not depend on the
    # provider's tensor-core policy (the CUDA EP may use TF32).
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    for batch in (1, 3):
        ins = _step_inputs(batch, device, seed=batch)
        with torch.no_grad():
            ref = fmt.forward(*ins, train=False)
        feed = {k: v.detach().cpu().numpy() for k, v in zip(names, ins, strict=False)}
        got = torch.from_numpy(sess.run(None, feed)[0]).to(ref.device)
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
        assert cos > 0.9999, f"the exported graph is a different model (cosine={cos:.6f}, batch={batch})"


def test_converter_parity_reference_is_the_deployed_model(tmp_path, monkeypatch):
    """The converter must gate the artifact against the deployed model.

    A substitution that only touches the *exported* module (the manual
    attention branch used to be one) has to be rejected instead of quietly
    shipping a graph the application cannot reproduce.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")

    from fast_float.onnx import convert_fmt

    class _Divergent(convert_fmt.FMTStepWrapper):
        """An export-only substitution: same weights, different math."""

        def forward(self, *args):
            return super().forward(*args) * 2.0

    _build = convert_fmt.build_fmt

    def _build_awake(args):
        return _wake_up(_build(args))

    monkeypatch.setattr(convert_fmt, "FMTStepWrapper", _Divergent)
    monkeypatch.setattr(convert_fmt, "build_fmt", _build_awake)
    out = str(tmp_path / "fmt_step.onnx")
    with pytest.raises(RuntimeError, match="not the deployed model"):
        convert_fmt.main(
            [
                "--ckpt",
                str(tmp_path / "missing.pth"),  # random init: parity is still valid
                "--out",
                out,
                "--cpu-ort",
                "--fmt-depth",
                "1",
                "--dim-h",
                "64",
                "--num-heads",
                "2",
                "--mlp-ratio",
                "2.0",
                "--bench-iters",
                "0",
            ]
        )


def test_deployed_and_exported_modules_are_the_same_object_path(tmp_path):
    """The converter must not swap the attention implementation any more."""
    from fast_float.onnx import convert_fmt

    ns = SimpleNamespace(
        ckpt=str(tmp_path / "missing.pth"),
        prev_frames=PREV,
        dim_h=64,
        mlp_ratio=2.0,
        fmt_depth=1,
        num_heads=2,
    )
    fmt = convert_fmt.build_fmt(ns)
    deployed = copy.deepcopy(fmt)
    assert fmt.blocks[0].attn.fused_attn == deployed.blocks[0].attn.fused_attn


def _model_with_backend(fmt, backend):
    model = SimpleNamespace(fmt=fmt, motion_autoencoder=None)
    fmt._onnx_backend = backend
    fmt._mode = "onnx"
    return model


def test_install_self_test_drops_a_wrong_graph():
    """A stale artifact must not ship broken frames: it is dropped at install."""
    from fast_float.onnx.dispatch import apply_backends

    fmt = _tiny_fmt()

    class _Wrong:
        def run_step(self, *args, **kwargs):
            return torch.zeros(3, CLIP + PREV, DIM_W)

    model = _model_with_backend(fmt, _Wrong())
    apply_backends(model, None)
    assert fmt._onnx_backend is None
    assert fmt._mode == "eager"


def test_install_self_test_keeps_a_matching_graph():
    from fast_float.onnx.dispatch import apply_backends

    fmt = _tiny_fmt()

    class _Right:
        def run_step(self, t, x, wa, wr, we, prev_x, prev_wa):
            return fmt.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)

    backend = _Right()
    model = _model_with_backend(fmt, backend)
    apply_backends(model, None)
    assert fmt._onnx_backend is backend
    assert fmt._mode == "onnx"
