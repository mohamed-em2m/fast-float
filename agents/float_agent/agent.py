"""LiveKit Agents worker: Gemini Live speech-to-speech + FLOAT voice tools.

Run from ``agents/``::

    uv sync
    cp .env.example .env   # fill in LiveKit + Google keys
    uv run python -m float_agent.agent dev

The worker registers as ``float-gemini``. When the FLOAT API starts a Gemini
conversation it dispatches this agent into the room; the agent listens to the
browser microphone and streams its synthesized voice to the avatar worker
(``float-avatar``) over LiveKit DataStream. FLOAT renders the lip-synced video
and publishes both tracks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load agents/.env by its real path (not CWD-relative) so the worker picks up
# its project env even when launched from another directory; a CWD .env can
# still override afterwards (python-dotenv never overwrites existing vars).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

from google.genai import types as genai_types
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomOutputOptions,
    RunContext,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.plugins import google

from float_agent.config import AgentConfig, dispatch_config
from float_agent.tools import FloatApi

logger = logging.getLogger("float_gemini")

config = AgentConfig.from_env()
server = AgentServer()


class FloatAssistant(Agent):
    """Gemini Live agent with tools that inspect/stop the FLOAT system."""

    def __init__(self) -> None:
        super().__init__(instructions=config.instructions)
        self.api = FloatApi(config.float_api_url, timeout_s=config.float_api_timeout_s)

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
        asyncio.create_task(self._shutdown_after_reply(context))
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
    resolved_model = (model or "").strip() or config.model
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
    elif api_key or config.google_api_key:
        kwargs["api_key"] = api_key or config.google_api_key
    return google.realtime.RealtimeModel(**kwargs)  # type: ignore[arg-type]


@server.rtc_session(agent_name=config.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    missing = config.missing()
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing) + ". See agents/.env.example.")

    await ctx.connect()
    logger.info(
        "starting Gemini Live agent %s in room %s (avatar=%s)",
        config.agent_name,
        ctx.room.name,
        config.avatar_identity,
    )
    logger.info(
        "registered as %r — LiveKit credentials present: %s, Gemini API key present: %s, model: %s",
        config.agent_name,
        bool(config.livekit_url and config.livekit_api_key and config.livekit_api_secret),
        bool(config.google_api_key),
        config.model or "<plugin default>",
    )

    # The FLOAT API forwards Studio-supplied overrides (Gemini key + model)
    # in the dispatch metadata; prefer them over the box config this session.
    dispatch = dispatch_config(getattr(getattr(ctx, "job", None), "metadata", "") or "")
    session_api_key = dispatch.get("gemini_api_key", "")
    session_model = dispatch.get("gemini_model", "")
    if session_api_key:
        logger.info("using a per-session Gemini API key supplied by the FLOAT API")
        # google-genai reads GOOGLE_API_KEY from the environment too.
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = session_api_key
    if session_model:
        logger.info("using a per-session Gemini model supplied by the FLOAT API (%s)", session_model)

    session = AgentSession(llm=_build_realtime_model(session_api_key, session_model))

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
        raise SystemExit("Missing configuration: " + ", ".join(missing) + ". See agents/.env.example.")
    cli.run_app(server)


if __name__ == "__main__":
    main()
