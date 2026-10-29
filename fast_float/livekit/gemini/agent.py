"""LiveKit Agents worker: Gemini Live speech-to-speech + FLOAT voice tools.

Run from the project root::

    ./scripts/run_gemini_agent.sh
    # or: uv run python -m fast_float.livekit.gemini.agent start

The worker is part of this project (one venv/lockfile), but runs as a separate
process so LiveKit can dispatch jobs to it. The LiveKit + Gemini values come
from the shared ``fast_float/config/secerts.toml`` / ``settings.toml`` (the
single source of truth, also read by the API); the repo-root ``.env`` is only
a fallback for boxes without those files (``cp .env.example .env``).

Use ``start`` rather than the deprecated ``dev`` command: livekit-agents
removed in-process auto-reload, so ``dev`` only prints deprecation warnings
now.

The worker registers as ``float-gemini``. When the FLOAT API starts a Gemini
conversation it dispatches this agent into the room; the agent listens to the
browser microphone and streams its synthesized voice to the avatar worker
(``float-avatar``) over LiveKit DataStream. FLOAT renders the lip-synced video
and publishes both tracks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import ssl
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:  # certifi ships with httpx/google-genai; the system store is a fallback.
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

from dotenv import load_dotenv

# Load the repo-root .env by its real path (not CWD-relative) so the worker
# picks it up even when launched from another directory; a CWD .env can still
# override afterwards (python-dotenv never overwrites existing vars).
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
load_dotenv()

from google.genai import types as genai_types
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput

# RoomOptions is not (yet) re-exported at livekit.agents top level in 1.8.x —
# only the legacy RoomInput/OutputOptions pair is (hence the SDK deprecation
# warning we silence here by switching).
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import google

try:
    # Not re-exported by the plugin package; guarded so a plugin layout change
    # degrades to the stock session instead of breaking the worker import.
    from livekit.plugins.google.realtime.realtime_api import (
        RealtimeSession as _GoogleRealtimeSession,
    )
except Exception:  # pragma: no cover - depends on the installed plugin layout
    _GoogleRealtimeSession = None

from fast_float.livekit.gemini.config import AgentConfig, dispatch_config, sanitize_gemini_model
from fast_float.livekit.gemini.tools import FloatApi
from fast_float.logging_utils import configure_logging, current_log_file

logger = logging.getLogger("float_gemini")

config = AgentConfig.from_env()
# AgentServer reads LIVEKIT_* from os.environ when it registers with the SFU
# (config.from_env also exports the winners there), but pass them explicitly
# as well: registration is the one thing that must never depend on import
# order, and a missing value here is an opaque "ws_url is required" exit.
server = AgentServer(
    ws_url=config.livekit_url or None,
    api_key=config.livekit_api_key or None,
    api_secret=config.livekit_api_secret or None,
    # Bind the health HTTP server to an ephemeral port (0) instead of the SDK's
    # production default 8081: a second worker — or a leftover one from an
    # earlier run — otherwise dies on startup with
    # "[Errno 98] ... ('0.0.0.0', 8081): address already in use" and then sits
    # draining for up to an hour while looking alive. Nothing probes this
    # endpoint (no K8s here) and SFU registration doesn't need a fixed port.
    port=0,
)


class FloatAssistant(Agent):
    """Gemini Live agent with tools that inspect/stop the FLOAT system."""

    def __init__(self) -> None:
        super().__init__(instructions=config.instructions)
        self.api = FloatApi(config.float_api_url, timeout_s=config.float_api_timeout_s)
        # Strong refs to fire-and-forget tasks: the event loop only holds weak
        # ones, so a bare create_task() can be garbage-collected mid-flight.
        self._background: set[asyncio.Task[None]] = set()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _room_name(self) -> str:
        job = get_job_context()
        return job.room.name or ""

    # ---------------------------------------------------------------- status

    @function_tool()
    async def float_health(self) -> str:
        """Report the FLOAT API health, model state and warmup stage."""
        try:
            return await self.api.health()
        except RuntimeError as exc:
            return f"FLOAT health is unavailable: {exc}"

    @function_tool()
    async def float_gpu(self) -> str:
        """Report GPU name, VRAM usage and CUDA availability."""
        try:
            return await self.api.gpu()
        except RuntimeError as exc:
            return f"GPU status is unavailable: {exc}"

    @function_tool()
    async def generation_progress(self, request_id: str) -> str:
        """Look up a FLOAT video-generation request by its request id."""
        try:
            return await self.api.progress(request_id)
        except RuntimeError as exc:
            return f"Could not read generation progress: {exc}"

    @function_tool()
    async def conversation_status(self) -> str:
        """Report whether this room has a live FLOAT avatar session."""
        try:
            return await self.api.conversation_status(await self._room_name())
        except RuntimeError as exc:
            return f"Could not read conversation status: {exc}"

    # ----------------------------------------------------------- room control

    @function_tool()
    async def end_conversation(self, context: RunContext) -> str:
        """End the live conversation and stop the FLOAT avatar for this room."""
        room = await self._room_name()
        message = "Conversation ended."
        if room:
            try:
                await self.api.stop_conversation(room)
            except RuntimeError as exc:
                logger.warning("stop_conversation failed: %s", exc)
                message = "I could not reach the avatar service, but I am ending the call."
        # Let the spoken reply finish before the session closes.
        self._spawn(self._shutdown_after_reply(context))
        return message

    async def _shutdown_after_reply(self, context: RunContext) -> None:
        await asyncio.sleep(0.8)
        await context.session.shutdown(drain=False)


_GENAI_SSL_CONTEXT: ssl.SSLContext | None = None


async def _genai_ssl_context() -> ssl.SSLContext:
    """The TLS context google-genai builds for every HTTPX client.

    google-genai 2.x builds `ssl.create_default_context(...)` synchronously
    inside each Client construction — i.e. on the agent loop, right when a
    session starts. On this box that stall measured ~283 ms on the first
    session, which `livekit.agents` flags with
    "event loop blocked ... delays audio and turn handling". The context only
    depends on the trust store, so build it once in a thread and hand the same
    one to every session through `HttpOptions.client_args["verify"]`
    (the documented hook `_ensure_httpx_ssl_ctx` consults before building).
    """
    global _GENAI_SSL_CONTEXT
    if _GENAI_SSL_CONTEXT is None:
        cafile = os.environ.get("SSL_CERT_FILE") or (certifi.where() if certifi else None)
        capath = os.environ.get("SSL_CERT_DIR")

        def _build() -> ssl.SSLContext:
            """Mirror google/genai/_api_client.py's context construction."""
            if cafile or capath:
                return ssl.create_default_context(cafile=cafile, capath=capath)
            return ssl.create_default_context()

        _GENAI_SSL_CONTEXT = await asyncio.to_thread(_build)
    return _GENAI_SSL_CONTEXT


# Gemini Live can drop an established session with 1011 ("internal error") and
# the plugin retries automatically. Its retry reuses the session-resumption
# handle from the session that just died; Google answers that with 1008
# "BidiGenerateContent session not found or expired" during connect, which the
# plugin cannot retry (no session exists yet) and surfaces as fatal — the whole
# AgentSession shuts down even though a fresh session would have worked.
# Retrying with a fresh session is safe: the plugin replays the chat history
# into the new connection, so only the dead handle is lost.
_STALE_SESSION_MARKERS = ("1011", "1008", "session not found or expired")

# Google keeps a session bound to the client that opened it and releases it a
# moment *after* the socket closes (and not at all when the close never reached
# it). A reconnect that carries the resumption handle is then refused with 1008
# "BidiGenerateContent session is already connected to an existing client" —
# and because that happens during connect (`session is None`), the plugin calls
# it fatal: a self-inflicted idle refresh would end the whole conversation.
# The handle itself is still good; the server just needs a moment, or a session
# that does not have to take the old one over. So retry instead of giving up:
# the first attempt keeps the handle (a cleanly closed socket is usually
# released in well under a second), later attempts drop it and let the plugin
# replay the chat context into a brand-new session.
_HELD_SESSION_MARKERS = ("already connected to an existing client",)
_EXPIRED_HANDLE_MARKERS = ("not found", "expired")
_CONNECT_RETRY_DELAYS_SEC = (0.5, 1.0, 2.0, 4.0)

# Google closes an idle Live session after ~2 minutes without turn activity,
# usually without sending a `goAway` first. The next utterance then hits a dead
# socket: the plugin reconnects, dropping any audio already queued for it, and
# the user has to repeat themselves. Restart the session ourselves after a
# shorter idle window so the reconnect happens between turns instead.
_IDLE_REFRESH_SEC = 60.0
_IDLE_REFRESH_TICK_SEC = 5.0


def _error_chain(error: BaseException | None) -> Iterator[BaseException]:
    """Yield ``error`` and every cause/context behind it (cycle-safe)."""
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _message_matches(error: BaseException | None, markers: tuple[str, ...]) -> bool:
    """True when any exception in ``error``'s chain mentions one of ``markers``."""
    for err in _error_chain(error):
        message = str(err).lower()
        if any(marker in message for marker in markers):
            return True
    return False


def _is_stale_session_error(error: BaseException | None) -> bool:
    """True when ``error`` means the resumption handle can no longer be used."""
    if error is None:
        return False
    for err in _error_chain(error):
        if getattr(err, "code", None) in (1008, 1011):
            return True
        message = str(err).lower()
        if any(marker in message for marker in _STALE_SESSION_MARKERS):
            return True
    return False


def _is_session_held_error(error: BaseException | None) -> bool:
    """True when Google still has the previous connection registered."""
    return _message_matches(error, _HELD_SESSION_MARKERS)


def _is_retryable_connect_error(error: BaseException | None) -> bool:
    """True when a *failed connect* is a teardown race, not a bad configuration.

    Both 1008 flavours qualify: the previous client has not been released yet
    ("already connected to an existing client") or its handle has gone stale
    ("session not found or expired"). Neither is caused by the connect config,
    and both are cured by reconnecting — fresh, if the handle is dead.
    """
    return _is_session_held_error(error) or _is_stale_session_error(error)


def _resilient_session_class(base: type) -> type:
    class _FreshRetryRealtimeSession(base):
        # ----------------------------------------------------- handle hygiene

        def _drop_resumption_handle(self, reason: str) -> None:
            """Forget the resumption handle so the next connect starts fresh."""
            had_handle = getattr(self, "_session_resumption_handle", None) is not None
            self._session_resumption_handle = None
            self._resumption_chat_ctx = None
            if had_handle:
                logger.warning(
                    "Gemini Live %s; discarding the resumption handle and reconnecting "
                    "with a fresh session (the chat context is replayed)",
                    reason,
                )

        def _mark_restart_needed(self, on_error: bool = False) -> None:
            if on_error and _is_stale_session_error(getattr(self, "_session_error", None)):
                self._drop_resumption_handle(f"session expired ({self._session_error})")
            super()._mark_restart_needed(on_error)

        def _emit_error(self, error: Exception, recoverable: bool) -> None:
            # The stock loop reports a failed connect as non-recoverable, and
            # livekit-agents closes the AgentSession on the first one — which
            # would defeat the retry in `_main_task_with_connect_retry`. Report
            # it as recoverable while a retry is still available; that method
            # re-emits it as final once the budget is spent.
            if not recoverable and _is_retryable_connect_error(error):
                recoverable = True
            super()._emit_error(error, recoverable)

        # -------------------------------------------------- idle refresh
        def _note_turn_activity(self) -> None:
            """Reset the idle clock; called on every turn/user-speech event."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # no loop (tests / sync construction)
                return
            self._last_turn_at = loop.time()

        def _start_new_generation(self) -> None:
            self._note_turn_activity()
            super()._start_new_generation()

        def _mark_current_generation_done(self) -> None:
            self._note_turn_activity()
            super()._mark_current_generation_done()

        def _handle_input_speech_started(self) -> None:
            self._note_turn_activity()
            super()._handle_input_speech_started()

        def _handle_input_speech_stopped(self) -> None:
            self._note_turn_activity()
            super()._handle_input_speech_stopped()

        async def _main_task(self) -> None:
            # The stock loop reconnects on `_session_should_close`; run the idle
            # watchdog alongside it so a long pause never leaves a dead socket in
            # front of the next utterance.
            self._note_turn_activity()
            watchdog = asyncio.create_task(self._idle_refresh_watchdog(), name="gemini-idle-refresh")
            try:
                await self._main_task_with_connect_retry()
            finally:
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog

        async def _main_task_with_connect_retry(self) -> None:
            """Run the stock loop, surviving a reconnect that races the teardown.

            The stock `_main_task` treats *any* connect failure as fatal (there
            is no session yet, so it cannot retry), and our own idle refresh can
            provoke exactly that: Google still holds the socket we just closed.
            Retrying keeps the conversation alive instead of tearing down the
            AgentSession; after `_CONNECT_RETRY_DELAYS_SEC` attempts the error is
            final and is reported as non-recoverable.
            """
            attempt = 0
            while True:
                try:
                    await super()._main_task()
                    if attempt:
                        logger.info("Gemini Live session reconnected after %d retries", attempt)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if getattr(getattr(self, "_msg_ch", None), "closed", True):
                        raise
                    if not _is_retryable_connect_error(exc):
                        raise
                    if attempt >= len(_CONNECT_RETRY_DELAYS_SEC):
                        logger.error(
                            "Gemini Live could not reconnect after %d retries (%s); ending the session",
                            attempt,
                            exc,
                        )
                        # `_emit_error` downgraded the stock reports so the
                        # retries could run; this one is final, so report it so.
                        super()._emit_error(exc, recoverable=False)
                        raise
                    delay = _CONNECT_RETRY_DELAYS_SEC[attempt]
                    attempt += 1
                    # A dead handle can never be resumed, and a session the
                    # server still holds cannot be taken over: both need a
                    # fresh connection. Only the first retry keeps the handle.
                    if attempt > 1 or _message_matches(exc, _EXPIRED_HANDLE_MARKERS):
                        self._drop_resumption_handle(f"reconnect refused ({exc})")
                    if hasattr(self, "_num_retries"):
                        # The stock retry budget applies per connect attempt;
                        # give the next one a full budget of its own.
                        self._num_retries = 0
                    logger.warning(
                        "Gemini Live connect failed (%s); retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt,
                        len(_CONNECT_RETRY_DELAYS_SEC),
                    )
                    await asyncio.sleep(delay)

        async def _idle_refresh_watchdog(self) -> None:
            loop = asyncio.get_running_loop()
            while True:
                await asyncio.sleep(_IDLE_REFRESH_TICK_SEC)
                should_close = getattr(self, "_session_should_close", None)
                if should_close is None or should_close.is_set():
                    continue
                generation = getattr(self, "_current_generation", None)
                if generation is not None and not getattr(generation, "_done", False):
                    # A response is still streaming; that is activity, not idleness.
                    self._note_turn_activity()
                    continue
                last = getattr(self, "_last_turn_at", None)
                if last is None:
                    last = self._last_turn_at = loop.time()
                    continue
                idle_for = loop.time() - last
                if idle_for < _IDLE_REFRESH_SEC:
                    continue
                logger.info(
                    "Gemini Live session idle for %.0fs; restarting it with the "
                    "resumption handle before Google drops the connection",
                    idle_for,
                )
                self._note_turn_activity()
                # A clean restart (unlike an error) keeps the handle, so the
                # plugin reconnects and the conversation context survives.
                should_close.set()

    return _FreshRetryRealtimeSession


def _resilient_model_class(base: type, session_base: type) -> type:
    """Subclass ``base`` so restarts after a dead session start fresh.

    ``base`` is passed in (rather than subclassed at import time) so the class
    follows whatever ``google.realtime.RealtimeModel`` is active when the model
    is built, including the one tests monkeypatch.
    """
    fresh_session = _resilient_session_class(session_base)

    class _FreshRetryRealtimeModel(base):
        def session(self, *, turn_detection_disabled: bool = False):
            sess = fresh_session(self)
            self._sessions.add(sess)
            return sess

    return _FreshRetryRealtimeModel


def _build_realtime_model(
    api_key: str | None = None, model: str | None = None, ssl_ctx: ssl.SSLContext | None = None
) -> google.realtime.RealtimeModel:
    """Create the Gemini Live model from the environment.

    ``api_key`` / ``model`` win over the environment/shared config: the FLOAT
    API forwards the Studio settings through the dispatch metadata, so one
    Studio can use its own key and pin its own Live model without editing the
    box. No key is needed in Vertex AI mode.
    """
    kwargs: dict[str, object] = {
        "voice": config.voice,
        "input_audio_transcription": genai_types.AudioTranscriptionConfig(),
        "output_audio_transcription": genai_types.AudioTranscriptionConfig(),
        # The avatar's voice is played back through the browser (the mic track
        # still hears some of it even with echoCancellation). Gemini's
        # server-side activity detection used to flag that spill-over as the
        # user barging in and STOPped replies mid-sentence ("counting to 10,
        # cut at 6" — the agent even logs lk.clear_buffer / "playback
        # interrupted"). Trigger-happy defaults: make start-of-speech harder
        # to commit (LOW sensitivity + 200 ms required speech) so a short
        # echo burst cannot truncate an utterance; real barge-ins (actual
        # user speech, >0.5 s in one burst) still get through.
        "realtime_input_config": genai_types.RealtimeInputConfig(
            automatic_activity_detection=genai_types.AutomaticActivityDetection(
                start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=200,
            ),
        ),
    }
    # The dispatch model name is free text from a Studio box; keep a bad value
    # from setting up the whole session, then falling over on the Gemini
    # websocket. Falls back to the box's configured model.
    resolved_model = sanitize_gemini_model((model or "").strip()) or config.model
    if resolved_model:
        kwargs["model"] = resolved_model
    if config.language:
        kwargs["language"] = config.language
    if config.vertexai:
        kwargs["vertexai"] = True
        if config.google_cloud_project:
            kwargs["project"] = config.google_cloud_project
        if config.google_cloud_location:
            kwargs["location"] = config.google_cloud_location
        if api_key:
            logger.info(
                "vertexai mode: the per-session Gemini API key has no effect "
                "(Vertex AI authenticates with project credentials, not API keys)"
            )
    elif api_key or config.google_api_key:
        kwargs["api_key"] = api_key or config.google_api_key
    if ssl_ctx is not None:
        # Keep the plugin's own default here: RealtimeSession builds
        # `types.HttpOptions(timeout=int(DEFAULT_API_CONNECT_OPTIONS.timeout *
        # 1000))` (10.0 s -> 10_000 ms) when no http_options is given. Pre-seed
        # the TLS context in both spellings google-genai consults:
        #   * httpx clients read `verify` (`_ensure_httpx_ssl_ctx`)
        #   * aiohttp requests/websockets read `ssl` (`_ensure_aiohttp_ssl_ctx`
        #     and `_ensure_websocket_ssl_ctx`)
        # Without the `ssl` key the Live session still built a fresh context
        # synchronously on the agent loop, which livekit-agents flagged as
        # "event loop blocked ... in create_default_context".
        kwargs["http_options"] = genai_types.HttpOptions(
            timeout=10_000,
            client_args={"verify": ssl_ctx},
            async_client_args={"verify": ssl_ctx, "ssl": ssl_ctx},
        )
    model_cls = google.realtime.RealtimeModel
    if _GoogleRealtimeSession is not None:
        model_cls = _resilient_model_class(model_cls, _GoogleRealtimeSession)
    return model_cls(**kwargs)


@server.rtc_session(agent_name=config.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    missing = config.missing()
    if missing:
        raise RuntimeError(
            "Missing configuration: " + ", ".join(missing) + ". See .env.example or fast_float/config/secerts.toml."
        )

    await ctx.connect()
    logger.info(
        "starting Gemini Live agent %s in room %s (avatar=%s)",
        config.agent_name,
        ctx.room.name,
        config.avatar_identity,
    )
    logger.info(
        "job for agent %r — LiveKit credentials present: %s, Gemini API key present: %s, model: %s",
        config.agent_name,
        bool(config.livekit_url and config.livekit_api_key and config.livekit_api_secret),
        bool(config.google_api_key),
        config.model or "<plugin default>",
    )

    # Per-session overrides from the Studio (Config panel) rank *above* the
    # deployment config this session — that is their purpose — but they are
    # passed straight to RealtimeModel and never written into the process
    # environment, where a concurrent job could inherit them (os.environ is
    # global to the worker process; only config.from_env, at import time,
    # keeps env in sync with secerts.toml). The model name is still
    # format-validated inside _build_realtime_model before it is used.
    dispatch = dispatch_config(getattr(getattr(ctx, "job", None), "metadata", "") or "")
    session_api_key = dispatch.get("gemini_api_key", "")
    session_model = dispatch.get("gemini_model", "")
    if session_api_key:
        logger.info("using a per-session Gemini API key supplied by the FLOAT API")
    if session_model:
        logger.info("using a per-session Gemini model supplied by the FLOAT API (%s)", session_model)

    session = AgentSession(
        # The shared TLS context is built in a thread before session.start, so
        # the plugin's GenAI client construction (which would otherwise build a
        # fresh SSL context synchronously on this loop) inherits it.
        llm=_build_realtime_model(session_api_key, session_model, ssl_ctx=await _genai_ssl_context()),
        # Avatar audio is streamed to the renderer as it is produced, so it
        # cannot be paused/rewound: the plugin's DataStreamAudioOutput.pause()
        # only logs "pause is not supported". Upstream enables
        # resume_false_interruption by default and warns on every session that
        # the output cannot honour it, so disable it explicitly instead of
        # letting a resumed "false interruption" silently do nothing.
        turn_handling=TurnHandlingOptions(interruption={"resume_false_interruption": False}),
    )

    # The agent's voice is not published directly: it is streamed to FLOAT,
    # which renders the avatar and publishes synchronized A/V (and sends back
    # lk.playback_started / lk.playback_finished).
    session.output.audio = DataStreamAudioOutput(
        ctx.room,
        destination_identity=config.avatar_identity,
        wait_playback_start=True,
    )

    await session.start(
        agent=FloatAssistant(),
        room=ctx.room,
        # room_options= (not the deprecated RoomOutputOptions): the agent's
        # voice is streamed to the renderer via DataStreamAudioOutput, not
        # published as its own track.
        room_options=RoomOptions(audio_output=False),
    )

    # Keep the job alive for the whole conversation. AgentSession emits
    # "close" when the user disconnects or end_conversation runs.
    closed = asyncio.Event()
    session.on("close", lambda *_: closed.set())
    await closed.wait()


def main() -> None:
    missing = config.missing()
    if missing:
        raise SystemExit(
            "Missing configuration: "
            + ", ".join(missing)
            + f". Define them in {config.config_dir}/secerts.toml ([livekit] and "
            "[gemini].api_key) or in the repo-root .env; see .env.example. Start the "
            "worker with ./scripts/run_gemini_agent.sh. The worker and the FLOAT API must "
            "read the same directory (set FLOAT_CONFIG_DIR on both if the checkout lives "
            "elsewhere)."
        )

    # Startup summary, logged from an SDK event because the CLI installs its
    # log handlers inside run_app() — a logger.info() before that is dropped.
    # This is the line to check when the API reports that dispatch found
    # "NO job": it proves which SFU, agent name and config directory this
    # process is running with. worker_started fires even when registration
    # then fails; the SDK's "registered worker" line confirms the handshake.
    def _log_startup(*_args: object) -> None:
        # run_app() installs the SDK's handlers before this event fires; attach
        # ours afterwards so they are not replaced (and so pre-run logs are
        # not dropped).
        configure_logging(role="agent")
        logger.info(
            "worker starting: agent_name=%s url=%s config_dir=%s livekit_credentials=%s gemini_key=%s model=%s log_file=%s",
            config.agent_name,
            config.livekit_url or "<missing>",
            config.config_dir,
            bool(config.livekit_url and config.livekit_api_key and config.livekit_api_secret),
            bool(config.google_api_key),
            config.model or "<plugin default>",
            current_log_file() or "console only",
        )

    server.on("worker_started", _log_startup)
    cli.run_app(server)


if __name__ == "__main__":
    main()
