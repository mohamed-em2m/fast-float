"""Idle-handling regressions: prompt ``playback_finished`` and session refresh.

A long silence used to cost latency in two places:

* the avatar's idle loop (pre-rendered silence) kept feeding
  ``AVSynchronizer``, so ``wait_for_playout()`` could only ever time out and
  ``lk.playback_finished`` arrived 30 s late;
* Gemini dropped the idle Live session (~2 min), so the next utterance paid
  for a reconnect and the audio queued for the dead socket was discarded.
"""

from __future__ import annotations

import asyncio
import contextlib

from fast_float.livekit.agent_runner import AgentAvatarRunner
from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

# ---------------------------------------------------------------------------
# Idle-loop pause (both generators)
# ---------------------------------------------------------------------------


def test_float_generator_pause_idle_stops_silence_frames():
    async def _test():
        gen = FloatVideoGenerator()
        silence = ("silence",)
        gen._silence_frames = [silence]

        gen.pause_idle()
        agen = gen.__aiter__()
        try:
            await asyncio.wait_for(agen.__anext__(), 0.05)
            raise AssertionError("paused generator still emitted a silence frame")
        except asyncio.TimeoutError:
            pass
        await agen.aclose()

        gen.resume_idle()
        resumed = gen.__aiter__()
        assert await asyncio.wait_for(resumed.__anext__(), 1.0) == silence
        await resumed.aclose()

    asyncio.run(_test())


def test_zmq_generator_pause_idle_stops_silence_frames():
    async def _test():
        gen = ZmqFloatVideoGenerator(shm_name="test_idle_pause")

        async def _connected():
            return None

        gen._ensure_connected = _connected  # type: ignore[method-assign]
        silence = ("silence",)
        gen._silence_frames = [silence]

        gen.pause_idle()
        agen = gen._iter()
        try:
            await asyncio.wait_for(agen.__anext__(), 0.05)
            raise AssertionError("paused generator still emitted a silence frame")
        except asyncio.TimeoutError:
            pass
        await agen.aclose()

        gen.resume_idle()
        resumed = gen._iter()
        assert await asyncio.wait_for(resumed.__anext__(), 1.0) == silence
        await resumed.aclose()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Segment watcher drains with the idle loop paused
# ---------------------------------------------------------------------------


def test_segment_loop_pauses_idle_while_draining_playout():
    async def _test():
        events: list[object] = []

        class Gen:
            paused = False

            def pause_idle(self) -> None:
                self.paused = True
                events.append("pause")

            def resume_idle(self) -> None:
                self.paused = False
                events.append("resume")

            async def wait_until_idle(self, timeout=None) -> bool:
                return True

        class Sync:
            async def wait_for_playout(self) -> None:
                # The idle loop must already be paused, otherwise silence frames
                # keep refilling the queue and this only ends on timeout.
                events.append(("drain", gen.paused))

        class Receiver:
            agent_identity = "agent-1"

            async def notify_playback_finished(
                self, *, playback_position: float, interrupted: bool
            ) -> None:
                events.append(("finished", interrupted))

        gen = Gen()
        runner = AgentAvatarRunner(None, video_gen=gen, receiver=Receiver())
        runner._av_sync = Sync()
        runner._last_agent_identity = "agent-1"

        task = asyncio.create_task(runner._segment_loop())
        try:
            await runner._segment_queue.put(runner._epoch)
            for _ in range(200):
                if ("finished", False) in events:
                    break
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert ("drain", True) in events
        assert events.index("pause") < events.index(("drain", True))
        assert events.index(("drain", True)) < events.index("resume")
        # A paused idle loop must never mask a real drain timeout.
        assert ("finished", False) in events

    asyncio.run(_test())


def test_segment_loop_resumes_idle_after_a_drain_timeout(monkeypatch):
    monkeypatch.setattr("fast_float.livekit.agent_runner.SEGMENT_PLAYOUT_TIMEOUT_S", 0.01)

    async def _test():
        events: list[str] = []

        class Gen:
            def pause_idle(self) -> None:
                events.append("pause")

            def resume_idle(self) -> None:
                events.append("resume")

            async def wait_until_idle(self, timeout=None) -> bool:
                return True

        class Sync:
            async def wait_for_playout(self) -> None:
                await asyncio.sleep(10)

        class Receiver:
            agent_identity = "agent-1"

            async def notify_playback_finished(self, **kwargs) -> None:
                events.append("finished")

        runner = AgentAvatarRunner(None, video_gen=Gen(), receiver=Receiver())
        runner._av_sync = Sync()
        runner._last_agent_identity = "agent-1"

        task = asyncio.create_task(runner._segment_loop())
        try:
            await runner._segment_queue.put(runner._epoch)
            for _ in range(200):
                if "finished" in events:
                    break
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert events == ["pause", "resume", "finished"]

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Gemini idle refresh
# ---------------------------------------------------------------------------


def _fake_session_class():
    from fast_float.livekit.gemini import agent as gemini_agent

    class _Base:
        def __init__(self) -> None:
            self._session_should_close = asyncio.Event()
            self._current_generation = None

        def _start_new_generation(self) -> None:
            pass

        def _mark_current_generation_done(self) -> None:
            pass

        def _handle_input_speech_started(self) -> None:
            pass

        def _handle_input_speech_stopped(self) -> None:
            pass

    return gemini_agent._resilient_session_class(_Base)


def test_idle_gemini_session_is_restarted_before_google_drops_it(monkeypatch):
    from fast_float.livekit.gemini import agent as gemini_agent

    monkeypatch.setattr(gemini_agent, "_IDLE_REFRESH_SEC", 0.05)
    monkeypatch.setattr(gemini_agent, "_IDLE_REFRESH_TICK_SEC", 0.01)

    async def _test():
        sess = _fake_session_class()()
        task = asyncio.create_task(sess._idle_refresh_watchdog())
        try:
            await asyncio.wait_for(sess._session_should_close.wait(), 1.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # A clean restart keeps the resumption handle (set by the plugin when it
        # reconnects); the watchdog must only ask for the restart.
        assert sess._session_should_close.is_set()

    asyncio.run(_test())


def test_gemini_idle_refresh_waits_for_activity_to_settle(monkeypatch):
    from fast_float.livekit.gemini import agent as gemini_agent

    monkeypatch.setattr(gemini_agent, "_IDLE_REFRESH_SEC", 0.05)
    monkeypatch.setattr(gemini_agent, "_IDLE_REFRESH_TICK_SEC", 0.01)

    async def _test():
        # Continuous user speech / turns keep the session alive.
        sess = _fake_session_class()()
        task = asyncio.create_task(sess._idle_refresh_watchdog())
        try:
            for _ in range(5):
                await asyncio.sleep(0.02)
                sess._handle_input_speech_started()
            assert not sess._session_should_close.is_set()

            # An in-flight response is activity too, even without new events.
            sess._current_generation = type("Gen", (), {"_done": False})()
            await asyncio.sleep(0.15)
            assert not sess._session_should_close.is_set()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(_test())
