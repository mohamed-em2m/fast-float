"""Emotion prediction must survive very short streaming chunks.

``flush()`` dispatches a TTS segment's tail even when it is only a few
milliseconds. wav2vec2's conv front-end requires >= 400 samples, and the
emotion encoder used to pass the raw tail through, crashing inference with
"Kernel size can't be greater than actual input size".
"""

import torch

from fast_float.models.float.condition_encoders import (
    MIN_WAV2VEC_SAMPLES,
    Audio2Emotion,
    _pad_short_waveform,
)


class _FakeCore:
    """Records the waveform length the SER model would receive."""

    def __init__(self):
        self.seen_lengths = []

    def __call__(self, a):
        self.seen_lengths.append(a.shape[-1])
        return type("Out", (), {"logits": torch.zeros(a.shape[0], 7)})()


def _emotion_encoder(core):
    enc = Audio2Emotion.__new__(Audio2Emotion)
    enc._onnx_backend = None
    enc._mode = "eager"
    enc._compiled_body = None
    enc.wav2vec2_for_emotion = core
    return enc


def test_pad_short_waveform_reaches_the_conv_minimum():
    padded = _pad_short_waveform(torch.randn(1, 10))
    assert padded.shape == (1, MIN_WAV2VEC_SAMPLES)


def test_pad_short_waveform_leaves_normal_clips_alone():
    clip = torch.randn(1, 16_000)
    assert _pad_short_waveform(clip) is clip


def test_pad_short_waveform_handles_empty_audio():
    assert _pad_short_waveform(torch.zeros(2, 0)).shape == (2, MIN_WAV2VEC_SAMPLES)


def test_predict_emotion_pads_a_millisecond_tail():
    core = _FakeCore()
    enc = _emotion_encoder(core)

    probs = enc.predict_emotion(torch.zeros(1, 8))

    assert probs.shape == (1, 7)
    assert core.seen_lengths == [MIN_WAV2VEC_SAMPLES]


def test_predict_emotion_normal_clip_is_not_padded():
    core = _FakeCore()
    enc = _emotion_encoder(core)

    enc.predict_emotion(torch.zeros(1, 12_345))

    assert core.seen_lengths == [12_345]
