"""Parity floors and selection significance.

Regression: ``compare_outputs`` ORed the functional branch with the elementwise
gate, so a loose ``atol`` (40x for TRT FP16) skipped ``min_cosine`` entirely. A
decoder graph off by 4.375 with cosine 0.983 was reported as PASS.
"""

import numpy as np

from fast_float.onnx.parity import compare_outputs, is_significant


def _broken_but_within_atol():
    """Outputs that are structurally wrong yet pass ``np.allclose(atol=40)``."""
    rng = np.random.default_rng(0)
    ref = rng.normal(scale=0.5, size=(64, 64, 3)).astype(np.float32)
    out = (ref + 5.0).astype(np.float32)  # |diff| = 5 << atol=40
    return ref, out


def test_loose_atol_cannot_bypass_the_cosine_floor():
    ref, out = _broken_but_within_atol()
    rep = compare_outputs(ref, out, atol=40.0, rtol=0.4, min_cosine=0.995, max_mean=0.05)

    assert rep["elementwise_ok"] is True  # the old code stopped here
    assert rep["cosine"] < 0.995
    assert rep["cosine_ok"] is False
    assert rep["ok"] is False


def test_cosine_floor_is_enforced_even_without_a_mean_floor():
    ref, out = _broken_but_within_atol()
    rep = compare_outputs(ref, out, atol=40.0, rtol=0.4, min_cosine=0.995)
    assert rep["ok"] is False


def test_functional_branch_still_rescues_fp16_rounding():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(32, 32)).astype(np.float32)
    out = (ref + 1e-3).astype(np.float32)  # fp16-style absolute drift

    strict = compare_outputs(ref, out, atol=1e-6, rtol=1e-6)
    functional = compare_outputs(ref, out, atol=1e-6, rtol=1e-6, min_cosine=0.995, max_mean=0.02)

    assert strict["ok"] is False
    assert functional["elementwise_ok"] is False
    assert functional["ok"] is True


def test_mean_floor_rejects_a_large_average_error():
    rng = np.random.default_rng(2)
    ref = rng.normal(scale=10.0, size=(32, 32)).astype(np.float32)
    out = (ref + 0.5).astype(np.float32)  # cosine ~1, mean_abs = 0.5

    rep = compare_outputs(ref, out, atol=40.0, rtol=0.4, min_cosine=0.995, max_mean=0.05)

    assert rep["cosine_ok"] is True
    assert rep["mean_ok"] is False
    assert rep["ok"] is False


def test_no_floors_keeps_the_strict_elementwise_gate():
    ref, out = _broken_but_within_atol()
    assert compare_outputs(ref, out, atol=40.0, rtol=0.4)["ok"] is True
    assert compare_outputs(ref, out, atol=0.1, rtol=0.1)["ok"] is False


def test_significance_rejects_a_win_inside_the_noise_band():
    # 100 -> 97 is 1.03x with +-1 ms jitter: the old min_gain=1.03 picked it.
    assert is_significant(97.0, 1.0, 100.0, 1.0, min_gain=1.03) is False


def test_significance_accepts_a_real_win():
    assert is_significant(50.0, 1.0, 100.0, 1.0) is True
    # 12% faster with very low jitter still clears 1.10x and 3*MAD
    assert is_significant(88.0, 0.5, 100.0, 0.5) is True


def test_significance_still_requires_min_gain():
    assert is_significant(95.0, 0.1, 100.0, 0.1, min_gain=1.10) is False


def test_significance_rejects_degenerate_measurements():
    assert is_significant(0.0, 0.0, 100.0, 0.0) is False
    assert is_significant(10.0, 0.0, 0.0, 0.0) is False
