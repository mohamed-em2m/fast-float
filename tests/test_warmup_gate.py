"""Requests arriving during boot warmup wait instead of failing with a 503."""

import asyncio

from fast_float.core.state import AppState


def _run(coro):
    return asyncio.run(coro)


def test_wait_returns_immediately_when_not_warming():
    state = AppState()

    async def main():
        return await state.wait_for_warmup(timeout=0.01)

    assert _run(main()) is True


def test_wait_is_released_by_finish_warmup():
    state = AppState()

    async def main():
        state.begin_warmup()
        assert state.warming_up is True
        waiter = asyncio.create_task(state.wait_for_warmup(timeout=5))
        await asyncio.sleep(0.05)
        assert not waiter.done()  # held, not rejected
        state.finish_warmup()
        return await waiter

    assert _run(main()) is True
    assert state.warming_up is False
    assert state.warmup_stage == "done"
    assert state.warmup_error is None


def test_wait_times_out_when_warmup_never_finishes():
    state = AppState()

    async def main():
        state.begin_warmup()
        return await state.wait_for_warmup(timeout=0.05)

    assert _run(main()) is False
    assert state.warming_up is True  # caller decides (503); flag untouched


def test_release_survives_failed_warmup():
    state = AppState()

    async def main():
        state.begin_warmup()
        waiter = asyncio.create_task(state.wait_for_warmup(timeout=5))
        await asyncio.sleep(0.01)
        state.finish_warmup("boom")
        return await waiter

    assert _run(main()) is True
    assert state.warming_up is False
    assert state.warmup_stage == "failed"
    assert state.warmup_error == "boom"


def test_stage_and_elapsed_are_reported():
    state = AppState()

    async def main():
        state.begin_warmup()
        state.set_warmup_stage("single-inference")
        await asyncio.sleep(0.02)
        elapsed = state.warmup_elapsed()
        state.finish_warmup()
        return elapsed

    assert _run(main()) > 0
    assert state.warmup_stage == "done"
