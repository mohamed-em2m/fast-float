"""Generation must be reproducible from the request seed, regardless of batching."""

import torch

from fast_float.core.service import _chunk_seed
from fast_float.models.float.sampler import _initial_noise, _noise_generator


def test_same_seed_same_noise():
    a = _initial_noise([_noise_generator(25, "cpu")], 1, 4, 8, "cpu")
    b = _initial_noise([_noise_generator(25, "cpu")], 1, 4, 8, "cpu")
    assert torch.equal(a, b)


def test_different_seed_different_noise():
    a = _initial_noise([_noise_generator(25, "cpu")], 1, 4, 8, "cpu")
    b = _initial_noise([_noise_generator(26, "cpu")], 1, 4, 8, "cpu")
    assert not torch.equal(a, b)


def test_row_noise_is_independent_of_batch_composition():
    """The whole point: a chunk's noise must not change when it is batched."""
    solo = _initial_noise([_noise_generator(100, "cpu")], 1, 4, 8, "cpu")
    batched = _initial_noise([_noise_generator(100, "cpu"), _noise_generator(101, "cpu")], 2, 4, 8, "cpu")
    assert torch.equal(solo[0], batched[0])
    solo2 = _initial_noise([_noise_generator(101, "cpu")], 1, 4, 8, "cpu")
    assert torch.equal(solo2[0], batched[1])


def test_legacy_global_rng_path_still_works():
    x = _initial_noise(None, 2, 3, 4, "cpu")
    assert x.shape == (2, 3, 4)


def test_chunk_seed_shifts_by_position():
    req = {"seed": 25}
    assert [_chunk_seed(req, i) for i in range(3)] == [25, 26, 27]
    assert _chunk_seed({}, 4) == 29  # default seed 25
    assert _chunk_seed({"seed": None}, 1) == 26
    assert _chunk_seed({"seed": "7"}, 2) == 9
