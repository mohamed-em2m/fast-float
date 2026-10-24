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
import logging
from pathlib import Path
from typing import Any

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
    RoomOutputOptions,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.plugins import google

from fast_float.livekit.gemini.config import AgentConfig, dispatch_config, sanitize_gemini_model
from fast_float.livekit.gemini.tools import FloatApi

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


def _build_realtime_model(api_key: str | None = None, model: str | None = None) -> google.realtime.RealtimeModel:
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
    return google.realtime.RealtimeModel(**kwargs)  # type: ignore[arg-type]


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
        llm=_build_realtime_model(session_api_key, session_model),
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
        room_output_options=RoomOutputOptions(audio_enabled=False),
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
        logger.info(
            "worker starting: agent_name=%s url=%s config_dir=%s livekit_credentials=%s gemini_key=%s model=%s",
            config.agent_name,
            config.livekit_url or "<missing>",
            config.config_dir,
            bool(config.livekit_url and config.livekit_api_key and config.livekit_api_secret),
            bool(config.google_api_key),
            config.model or "<plugin default>",
        )

    server.on("worker_started", _log_startup)
    cli.run_app(server)


if __name__ == "__main__":
    main()
