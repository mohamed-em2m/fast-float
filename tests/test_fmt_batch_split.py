"""Batched CFG sampling sends more samples than the TensorRT profile allows."""

import torch

from fast_float.runtimes.onnx.backends import FMTStepBackend


def _backend(cap):
    """A backend with the TRT cap set and no session (we stub _run_once)."""
    be = FMTStepBackend.__new__(FMTStepBackend)
    be._batch_cap = cap
    be._split_warned = True
    be.calls = []
    return be


def _stub(be):
    def _run_once(t, x, wa, wr, we, prev_x, prev_wa):
        be.calls.append(int(x.shape[0]))
        # Identity on the batch dim; prev_x is concatenated like the real graph.
        return torch.cat([x, prev_x], dim=1)

    be._run_once = _run_once


def _inputs(batch):
    return (
        torch.tensor([0.5]),
        torch.zeros(batch, 50, 512),
        torch.zeros(batch, 50, 512),
        torch.zeros(batch, 1, 512),
        torch.zeros(batch, 1, 7),
        torch.zeros(batch, 10, 512),
        torch.zeros(batch, 10, 512),
    )


def test_batch_within_profile_is_one_call():
    be = _backend(16)
    _stub(be)
    out = be.run_step(*_inputs(16))
    assert be.calls == [16]
    assert out.shape == (16, 60, 512)


def test_batch_above_profile_is_split_and_reassembled():
    be = _backend(16)
    _stub(be)
    out = be.run_step(*_inputs(24))  # 8 chunks x CFG(3)
    assert be.calls == [16, 8]
    assert out.shape == (24, 60, 512)


def test_split_preserves_order():
    be = _backend(4)
    _stub(be)
    t, x, wa, wr, we, prev_x, prev_wa = _inputs(10)
    x = torch.arange(10).view(10, 1, 1).expand(10, 50, 512).float()
    out = be.run_step(t, x, wa, wr, we, prev_x, prev_wa)
    assert be.calls == [4, 4, 2]
    # First column of each row carries the sample index, so a misordered
    # concat would be visible immediately.
    assert torch.equal(out[:, 0, 0], torch.arange(10).float())


def test_shared_scalar_t_is_not_sliced():
    be = _backend(2)
    seen = []

    def _run_once(t, x, wa, wr, we, prev_x, prev_wa):
        seen.append(tuple(t.shape))
        return x

    be._run_once = _run_once
    be.run_step(*_inputs(5))
    assert seen == [(1,), (1,), (1,)]


def test_batch_sized_t_is_sliced_with_its_chunk():
    """A batch-sized ``t`` must follow the split, or each part sees the wrong time."""
    be = _backend(2)
    seen = []

    def _run_once(t, x, wa, wr, we, prev_x, prev_wa):
        seen.append(t.detach().clone())
        return x

    be._run_once = _run_once
    t, x, wa, wr, we, prev_x, prev_wa = _inputs(5)
    t = torch.arange(5).float()
    be.run_step(t, x, wa, wr, we, prev_x, prev_wa)
    assert [tuple(s.shape) for s in seen] == [(2,), (2,), (1,)]
    assert torch.equal(torch.cat(seen), torch.arange(5).float())


def test_no_split_without_tensorrt():
    be = _backend(None)
    _stub(be)
    out = be.run_step(*_inputs(24))
    assert be.calls == [24]
    assert out.shape == (24, 60, 512)
