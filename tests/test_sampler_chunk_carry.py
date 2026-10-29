"""Live chunks must continue the previous chunk's motion, not restart it.

LiveKit renders every audio chunk with its own ``sample`` call. Without the
``state`` carry the first window of the next chunk starts from zero
``prev_x``/``prev_wa`` and is never blended, so the avatar jumps at every chunk
boundary. These tests pin the carry: the same dict threaded through two calls
must make the second call's head continue the first call's tail, while a
missing/invalid state must keep the historical behaviour.
"""

import itertools
import types
from typing import ClassVar

import pytest
import torch

from fast_float.models.float import sampler as sampler_mod
from fast_float.models.float.sampler import SamplerMixin

DIM = 8
PREV = 2
CLIP = 4  # frames per 2 s window
VEL = 0.01  # vector field: one normal frame step in the fake model
FRAMES = 8  # 8 frames of audio at 25 fps -> 2 windows


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    """Keep the test hermetic: no CUDA context, no GPU state."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class _AudioEncoder:
    def __init__(self, dim_w):
        self.dim_w = dim_w

    def inference(self, a, seq_len):
        # Deterministic conditioning: one distinct constant per frame.
        base = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
        return base.expand(a.shape[0], -1, self.dim_w).clone()


class _EmotionEncoder:
    label2id: ClassVar[dict[str, int]] = {"s2e": 0}

    def predict_emotion(self, a):
        return torch.zeros(a.shape[0], 3)


class _FMT:
    """dx/dt = VEL + 0.1*mean(prev_x): the motion continues from the carried tail."""

    def forward_with_cfv(self, t, x, wa, wr, we, prev_x, prev_wa, **kwargs):
        out = torch.full_like(x, VEL) + 0.1 * prev_x.mean(dim=(1, 2)).view(-1, 1, 1)
        return torch.cat([torch.zeros_like(prev_x), out], dim=1)


class _Sampler(SamplerMixin):
    def __init__(self):
        self.opt = types.SimpleNamespace(
            rank=0,
            dim_w=DIM,
            fps=25,
            sampling_rate=16000,
            nfe=2,
            dim_e=3,
            num_prev_frames=PREV,
        )
        self.num_frames_for_clip = CLIP
        self.num_prev_frames = PREV
        self.stream_encode = None
        self.odeint_kwargs = {"atol": 1e-6, "rtol": 1e-6, "method": "euler"}
        self.audio_encoder = _AudioEncoder(DIM)
        self.emotion_encoder = _EmotionEncoder()
        self.fmt = _FMT()


def _render(model, monkeypatch, fill, state=None):
    """One ``sample`` call whose initial noise is a flat ``fill`` per window.

    A constant fill keeps the fake trajectory smooth (consecutive frames differ
    by the field alone) while a different ``fill`` per call is exactly the live
    situation: the next chunk re-synthesises from unrelated noise, so an
    un-carried head lands in a different pose.
    """
    calls = {"n": 0}

    def fake_noise(gens, batch, frames, dim, device):
        value = fill + calls["n"]
        calls["n"] += 1
        return torch.full((batch, frames, dim), float(value), device=device)

    monkeypatch.setattr(sampler_mod, "_initial_noise", fake_noise)
    a = torch.zeros(1, int(FRAMES * model.opt.sampling_rate / model.opt.fps))
    return torch.cat(list(model.sample({"r_s": None, "a": a}, nfe=2, seed=7, state=state)), dim=0)


def test_no_state_is_unchanged_and_deterministic(monkeypatch):
    a = _render(_Sampler(), monkeypatch, fill=0.0)
    b = _render(_Sampler(), monkeypatch, fill=0.0)
    assert a.shape == (FRAMES, DIM)
    assert torch.equal(a, b)


def test_state_carries_motion_tail_across_calls(monkeypatch):
    model = _Sampler()
    state = {}

    first = _render(model, monkeypatch, fill=0.0, state=state)
    # The write-back is the tail the next call is conditioned on.
    assert set(state) == {"prev_x", "prev_wa", "seam_tail"}
    assert state["prev_x"].shape == (1, PREV, DIM)
    assert state["prev_wa"].shape == (1, PREV, DIM)
    assert state["seam_tail"].shape == (1, 2, DIM)
    # Only the last window is carried, not the first one.
    assert torch.equal(state["prev_x"][0], first[-PREV:])
    assert torch.equal(state["seam_tail"][0], first[-2:])

    # Same noise on both sides: the only difference is the carried state.
    plain = _render(_Sampler(), monkeypatch, fill=10.0)
    carried = _render(model, monkeypatch, fill=10.0, state=state)

    # A fresh chunk restarts ~10 units away from the previous pose; the carried
    # one lands within the first blend weight of it (w = k/(k+1) = 0.8 here).
    seam_plain = (plain[0] - first[-1]).abs().mean()
    seam_carried = (carried[0] - first[-1]).abs().mean()
    assert seam_plain > 5.0, seam_plain
    assert seam_carried < 0.25 * seam_plain, (seam_carried, seam_plain)

    # The ramp relaxes: each frame is pulled less toward the extrapolation than
    # the previous one, so the head rejoins the model's own motion.
    raw = plain[0]  # constant fill -> every frame of a call has the same value
    residuals = [(carried[j] - raw).abs().mean().item() for j in range(CLIP)]
    assert all(a > b for a, b in itertools.pairwise(residuals)), residuals


def test_chunk_seam_blend_is_wider_than_interior_blend():
    """The cross-call seam uses its own (wider) ramp."""
    assert sampler_mod.CHUNK_SEAM_BLEND_FRAMES >= sampler_mod.SEAM_BLEND_FRAMES


def test_invalid_state_is_dropped_instead_of_crashing(monkeypatch):
    model = _Sampler()
    reference = _render(model, monkeypatch, fill=10.0)

    for bad in (
        {"prev_x": torch.zeros(1, PREV, DIM + 1), "prev_wa": torch.zeros(1, PREV, DIM)},
        {"prev_x": torch.zeros(2, PREV, DIM), "prev_wa": torch.zeros(2, PREV, DIM)},
        {"prev_x": "nope", "prev_wa": None},
        {"prev_x": torch.zeros(1, PREV, DIM), "prev_wa": torch.zeros(1, PREV, DIM), "seam_tail": "nope"},
    ):
        assert torch.equal(_render(model, monkeypatch, fill=10.0, state=dict(bad)), reference), bad


def test_missing_seam_tail_still_conditions_the_next_window(monkeypatch):
    """prev_x/prev_wa are what the model sees; the blend is only cosmetic."""
    model = _Sampler()
    state = {}
    first = _render(model, monkeypatch, fill=0.0, state=state)
    state.pop("seam_tail")
    carried = _render(model, monkeypatch, fill=10.0, state=state)
    assert carried.shape == first.shape
    assert not torch.equal(carried, _render(_Sampler(), monkeypatch, fill=10.0))
