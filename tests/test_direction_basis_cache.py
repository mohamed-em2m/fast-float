"""Direction's cached orthonormal basis must not go stale across weight loads."""

import torch

from fast_float.models.float.styledecoder import Direction


def test_basis_is_cached_between_calls():
    d = Direction(20)
    first = d._orthonormal_basis()
    assert d._orthonormal_basis() is first


def test_basis_is_recomputed_when_weights_are_overlaid_in_place():
    d = Direction(20)
    first = d._orthonormal_basis()

    with torch.no_grad():
        d.weight.copy_(torch.randn_like(d.weight))

    second = d._orthonormal_basis()
    assert second is not first
    assert d._orthonormal_basis() is second


def test_basis_is_recomputed_when_the_parameter_is_replaced():
    d = Direction(20)
    first = d._orthonormal_basis()

    d.weight = torch.nn.Parameter(torch.randn_like(d.weight))

    assert d._orthonormal_basis() is not first


def test_fingerprint_does_not_call_item():
    d = Direction(20)
    fp = d._fingerprint(d.weight)
    assert fp[0] == d.weight._version
    assert fp[1] == d.weight.data_ptr()
    assert all(not torch.is_tensor(v) for v in fp)
