"""RuntimeSession: per-call counters and reporting into utils/timing.py."""

import pytest

from fast_float.runtimes.base_runtime import RuntimeSession
from fast_float.utils.timing import timing_run


class _Sess(RuntimeSession):
    component = "test"
    backend = "onnx"

    def forward(self, x):
        if x == "boom":
            raise ValueError("boom")
        return x


def test_counters_update_without_a_timing_run():
    sess = _Sess()
    assert sess.forward(1) == 1
    with pytest.raises(ValueError, match="boom"):
        sess.forward("boom")
    assert sess.stats.calls == 2
    assert sess.stats.errors == 1


def test_forward_reports_a_runtime_stage():
    sess = _Sess()
    with timing_run("t") as collector:
        sess.forward(1)
        sess.forward(2)
        stage = collector._stages.get("runtime:test")
        assert stage is not None
        assert len(stage.walls) == 2
