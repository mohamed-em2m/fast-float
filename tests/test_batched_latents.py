"""Sampler -> batched-decode latent join.

Regression: ``sampler.sample`` squeezes the batch dim when B == 1, so a
single-chunk request produced ``[T, dim]`` pieces. ``torch.cat(parts, dim=1)``
then made ``[T, 2*dim]`` and ``lat_all[i, trim:t_i, :]`` raised
"too many indices for tensor of dimension 2".
"""

import pytest
import torch

from fast_float.models.float.sampler import stack_latent_parts


def test_single_chunk_is_rebatched():
    piece = torch.randn(30, 128)  # sample() with B == 1 yields [T, dim]

    lat_all = stack_latent_parts([piece], 1)

    assert tuple(lat_all.shape) == (1, 30, 128)
    assert torch.equal(lat_all[0], piece)
    # The per-item slice the batched path performs must now work.
    assert tuple(lat_all[0, 2:9, :].shape) == (7, 128)


def test_chunks_are_joined_on_time():
    a = torch.randn(2, 10, 128)
    b = torch.randn(2, 15, 128)

    lat_all = stack_latent_parts([a, b], 2)

    assert tuple(lat_all.shape) == (2, 25, 128)
    assert torch.equal(lat_all[:, :10], a)
    assert torch.equal(lat_all[:, 10:], b)


def test_items_must_match_the_sampled_batch():
    with pytest.raises(RuntimeError, match="expected \\[N, T, dim\\]"):
        stack_latent_parts([torch.randn(4, 128)], 3)


def test_three_dimensional_parts_are_not_rebatched():
    lat_all = stack_latent_parts([torch.randn(1, 30, 128)], 1)
    assert tuple(lat_all.shape) == (1, 30, 128)


def test_empty_sampler_output_is_an_error():
    with pytest.raises(RuntimeError, match="no latent chunks"):
        stack_latent_parts([], 1)
