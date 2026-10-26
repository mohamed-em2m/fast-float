"""Conversation session: wires the agent, the FLOAT renderer and the avatar worker.

Two backends are supported:

* ``builtin`` — :class:`ConversationalAgent` (mic PCM posted by the browser) +
  :class:`FloatVideoGenerator` + :class:`AvatarRunner`, all in this process.
* ``gemini`` — the ``livekit-agents`` worker in
  :mod:`fast_float.livekit.gemini` (Gemini Live speech-to-speech) streams TTS
  audio over LiveKit DataStream; :class:`AgentAvatarRunner` renders it and
  publishes synchronized A/V. The worker shares this venv/lockfile but runs
  as its own process, as LiveKit's dispatch model expects.

The browser joins the same room with a viewer token (POST /livekit/token) and
attaches the tracks published by the avatar worker. One session per room, kept
in :data:`_sessions`; manage through the ``/livekit/conversation/*`` endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from typing import Any

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

from fast_float.core.state import app_state
from fast_float.livekit.agent import DEFAULT_AGENT_IDENTITY, ConversationalAgent
from fast_float.livekit.agent_runner import AgentAvatarRunner
from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.receiver import DataStreamAudioReceiver
from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import issue_token
from fast_float.livekit.worker import AvatarOptions, AvatarRunner
from fast_float.utils.gpu import auto_detect_max_streams
from fast_float.utils.media import ensure_default_image

logger = logging.getLogger(__name__)

# GPU concurrency semaphore initialized dynamically or on first access.
_render_semaphore: asyncio.Semaphore | None = None
_current_max_streams: int = 1


def get_render_semaphore(max_streams: int | None = None) -> asyncio.Semaphore:
    """Get or configure the GPU inference concurrency semaphore.

    If max_streams is not specified or 0, automatically determines the optimal
    parallel streams based on GPU VRAM.
    """
    global _render_semaphore, _current_max_streams
    if max_streams is None:
        from fast_float.livekit.settings import LiveKitSettings

        cfg = LiveKitSettings.from_env()
        max_streams = getattr(cfg, "livekit_max_streams", 0)

    target_streams = auto_detect_max_streams(max_streams)

    if _render_semaphore is None or _current_max_streams != target_streams:
        _current_max_streams = target_streams
        _render_semaphore = asyncio.Semaphore(target_streams)
        logger.info(f"[session] GPU parallel streams capacity set to {target_streams}")

    return _render_semaphore


# room -> running session
_sessions: dict[str, ConversationSession] = {}

# Only raster formats the loader understands; anything else falls back to jpg.
_ALLOWED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Google model-id charset, case preserved (the worker lowercases): keeps
# free-text Studio values out of the dispatch metadata.
_MODEL_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,98}", re.IGNORECASE)


def _safe_image_format(value: str | None) -> str:
    """Normalise an upload's container hint to a whitelisted extension."""
    ext = str(value or "jpg").strip().lower().lstrip(".")
    if ext == "jpeg":
        return "jpg"
    return ext if ext in _ALLOWED_IMAGE_FORMATS else "jpg"


def _materialize_image(image_base64: str, image_format: str) -> str:
    """Decode a browser-uploaded portrait into a temp file for the renderer."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not raw:
        raise ValueError("image_base64 is empty")
    fd, path = tempfile.mkstemp(prefix="float-live-", suffix=f".{_safe_image_format(image_format)}")
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    return path


async def dispatch_agent(
    settings: LiveKitSettings,
    room: str,
    *,
    agent_name: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
) -> bool:
    """Ask the SFU to dispatch the Gemini worker into ``room``.

    ``gemini_api_key`` / ``gemini_model`` (when a browser supplied them) are
    forwarded to the worker inside the dispatch ``metadata``; the worker reads
    them from ``job.metadata``. The key is never written to any log.

    Returns the created ``AgentDispatch`` (truthy), or ``None`` (and logs) when
    the LiveKit API is unavailable or the request is rejected; the avatar
    session stays up so an operator can start the worker manually. The caller
    keeps the dispatch so a failed/torn-down session can delete it again.
    """
    name = (agent_name or settings.agent_name or "").strip()
    if not name:
        return None
    try:
        from livekit.api import CreateAgentDispatchRequest, LiveKitAPI

        request = CreateAgentDispatchRequest(**_dispatch_payload(room, name, gemini_api_key, gemini_model))
        async with LiveKitAPI(
            url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
        ) as lk:
            created = await lk.agent_dispatch.create_dispatch(request)
    except Exception as exc:
        logger.warning(f"[session] could not dispatch agent {name!r} into {room}: {exc}")
        return None
    logger.info(f"[session] dispatched agent {name!r} into {room}")
    return created


def _dispatch_payload(
    room: str,
    agent_name: str,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
) -> dict[str, Any]:
    """``CreateAgentDispatchRequest`` kwargs, adding metadata only when needed."""
    payload: dict[str, Any] = {"room": room, "agent_name": agent_name}
    key = (gemini_api_key or "").strip()
    # Only metadata-shaped names are forwarded. Authoritative validation
    # (known Live-API ids, lowercase, plugin list) happens in the worker
    # (fast_float.livekit.gemini.config.sanitize_gemini_model); this is the cheap filter the
    # API can apply before the name ever enters the SFU's dispatch payload.
    model = (gemini_model or "").strip()
    if _MODEL_NAME_RE.fullmatch(model) is None:
        if model:
            logger.warning(f"[session] gemini_model {model[:64]!r} is not a model id; not forwarding it")
        model = ""
    metadata: dict[str, str] = {}
    if key:
        metadata["gemini_api_key"] = key
    if model:
        metadata["gemini_model"] = model
    if metadata:
        payload["metadata"] = json.dumps(metadata)
    return payload


_JOB_STATUS_NAMES = {0: "pending", 1: "running", 2: "success", 3: "failed", 4: "cancelled"}


def _job_status(value: Any) -> str:
    if isinstance(value, int):
        return _JOB_STATUS_NAMES.get(value, str(value))
    return str(value)


async def fetch_room_dispatch(settings: LiveKitSettings, room: str) -> str:
    """Best-effort description of ``room``'s agent dispatch, for error messages.

    Distinguishes the two common failure modes: *no job at all* (no worker is
    registered and available under the dispatched agent name) vs *a failed
    job* (the worker's entrypoint raised after dispatch). Never raises.
    """
    try:
        from livekit.api import LiveKitAPI

        async with LiveKitAPI(
            url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
        ) as lk:
            dispatches = await lk.agent_dispatch.list_dispatch(room)
    except Exception as exc:
        return f"Could not query the SFU for dispatch state: {exc}"
    if not dispatches:
        return f"The SFU reports no dispatch for room {room!r}."

    parts: list[str] = []
    for dispatch in dispatches:
        name = str(getattr(dispatch, "agent_name", "") or "")
        jobs = getattr(getattr(dispatch, "state", None), "jobs", None) or []
        if not jobs:
            parts.append(
                f"dispatch {name!r} has NO job — no worker is registered/available "
                "under that agent name (is the worker process running?)"
            )
            continue
        for job in jobs:
            state = getattr(job, "state", None)
            status = _job_status(getattr(state, "status", ""))
            error = str(getattr(state, "error", "") or "").strip()
            job_id = str(getattr(job, "id", ""))[:8]
            if error:
                parts.append(f"dispatch {name!r} job {job_id} status={status}: {error[:200]}")
            else:
                parts.append(f"dispatch {name!r} job {job_id} status={status}")
    return " ".join(parts)


class ConversationSession:
    """One conversational avatar session in a LiveKit room."""

    def __init__(
        self,
        room_name: str,
        *,
        settings: LiveKitSettings | None = None,
        identity: str | None = None,
        image_path: str | None = None,
        image_base64: str | None = None,
        image_format: str = "jpg",
        tts_fn: Callable[[str], bytes] | None = None,
        agent_backend: str = "builtin",
        agent_name: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
    ) -> None:
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.agent_backend = (agent_backend or "builtin").strip().lower()
        if self.agent_backend not in {"builtin", "gemini"}:
            raise ValueError("agent_backend must be 'builtin' or 'gemini'")
        self.identity = identity or DEFAULT_AGENT_IDENTITY
        self.avatar_identity = (
            self.settings.avatar_identity if self.agent_backend == "gemini" else f"avatar-{room_name[:24]}"
        )
        self.agent_name = (agent_name or self.settings.agent_name or "").strip()
        # Optional per-session Gemini overrides supplied by the Studio UI
        # (API key + model name). Forwarded to the worker via the dispatch
        # metadata; the key is never logged.
        self.gemini_api_key = (gemini_api_key or "").strip()
        self.gemini_model = (gemini_model or "").strip()
        self.image_path = image_path
        self.image_base64 = image_base64
        self.image_format = _safe_image_format(image_format)
        self.tts_fn = tts_fn
        # Written from ``image_base64`` on start and removed on stop, so a
        # browser-uploaded portrait can drive a live session without exposing
        # a server path.
        self._temp_image: str | None = None

        self.agent: ConversationalAgent | None = None
        self.generator: FloatVideoGenerator | None = None
        self.worker_room: Any = None
        self.runner: AvatarRunner | AgentAvatarRunner | None = None
        self.receiver: DataStreamAudioReceiver | None = None
        self.agent_dispatched: bool | None = None
        # The SFU dispatch created for this session, so a torn-down/failed
        # session deletes it again — otherwise every retry leaks another
        # queued dispatch that later start failures list as "NO job".
        self.dispatch_id: str | None = None
        self._stopped = False

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    async def start(self) -> None:
        """Connect the agent, the avatar worker room and start rendering."""
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")
        if not self.settings.configured:
            raise RuntimeError("LiveKit settings not configured (LIVEKIT_URL / API_KEY / API_SECRET)")

        engine = app_state.agent
        if engine is None:
            raise RuntimeError("The FLOAT model is not loaded yet (still warming up)")

        if self.image_path is None and self.image_base64:
            self._temp_image = _materialize_image(self.image_base64, self.image_format)
            self.image_path = self._temp_image
            self.image_base64 = None

        if self.image_path is None:
            self.image_path = ensure_default_image()

        from fast_float.config.dynamic import load_settings

        ipc_cfg = load_settings().get("ipc", {})
        if isinstance(ipc_cfg, dict) and ipc_cfg.get("enabled", False):
            from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

            self.generator = ZmqFloatVideoGenerator(
                ref_image_path=self.image_path,
                nfe=self.settings.livekit_nfe,
                first_chunk_sec=getattr(self.settings, "livekit_first_chunk_sec", 5.0),
                chunk_sec=getattr(self.settings, "livekit_chunk_sec", 5.0),
                control_endpoint=ipc_cfg.get("zmq_control_endpoint", "tcp://127.0.0.1:5555"),
                frames_endpoint=ipc_cfg.get("zmq_frames_endpoint", "tcp://127.0.0.1:5556"),
                shm_name=ipc_cfg.get("shm_name", "float_frame_ring"),
                shm_slots=ipc_cfg.get("shm_ring_size", 64),
                session_id=self.room_name,
            )
        else:
            semaphore = get_render_semaphore(getattr(self.settings, "livekit_max_streams", 0))

            self.generator = FloatVideoGenerator(
                agent=engine,
                image_path=self.image_path,
                semaphore=semaphore,
                nfe=self.settings.livekit_nfe,
                first_chunk_sec=getattr(self.settings, "livekit_first_chunk_sec", 0.4),
                chunk_sec=getattr(self.settings, "livekit_chunk_sec", 2.0),
            )

        # Pre-render the silence idle loop once: FLOAT on silence audio (duration
        # configured via settings.toml [float].livekit_silence_sec, default 2.0s) so
        # the avatar has a natural resting expression between speech segments.
        # Uses the same GPU semaphore as live inference — runs first (before any
        # audio arrives) then frees the slot for real speech chunks.
        silence_sec = getattr(self.settings, "livekit_silence_sec", 2.0)
        try:
            await self.generator.prepare_silence(duration_sec=silence_sec)
        except TypeError:
            await self.generator.prepare_silence()

        if self.agent_backend == "gemini":
            await self._start_external_agent()
            return

        self.agent = ConversationalAgent(
            room_name=self.room_name,
            settings=self.settings,
            identity=self.identity,
            avatar_identity=self.avatar_identity,
            video_gen=self.generator,
            tts_fn=self.tts_fn,
        )
        await self.agent.connect()

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.avatar_identity,
            name="FLOAT Avatar (worker)",
            room=self.room_name,
            can_publish=True,
        )
        self.worker_room = rtc.Room()
        await self.worker_room.connect(self.settings.url, token_result.token)

        self.runner = AvatarRunner(
            self.worker_room,
            self.generator,
            options=AvatarOptions(),
            agent_identity=self.agent.identity,
        )
        await self.runner.start()
        logger.info(f"[session] conversation ready in {self.room_name} ({self.identity} + {self.avatar_identity})")

    async def _start_external_agent(self) -> None:
        """Connect the avatar worker and wait for the Gemini Live agent."""
        if self.generator is None:
            raise RuntimeError("Generator was not created")

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.avatar_identity,
            name="FLOAT Avatar (Gemini Live)",
            room=self.room_name,
            can_publish=True,
            # Kind=AGENT keeps the avatar out of livekit-agents' RoomIO
            # participant selection; its synthesized voice must not loop back.
            agent=True,
        )
        self.worker_room = rtc.Room()
        await self.worker_room.connect(self.settings.url, token_result.token)

        created = await self._dispatch_external_agent()
        self.agent_dispatched = created is not None
        # AgentDispatch.id, empty when the dispatch failed/None -> no cleanup.
        self.dispatch_id = str(getattr(created, "id", "") or "") or None

        # Give a cold-started worker time to boot, register and join. First
        # runs compile/import livekit-agents + google-genai, which can exceed
        # the old 30s window; override with FLOAT_GEMINI_AGENT_WAIT_S.
        wait_s = max(10, int(float(os.getenv("FLOAT_GEMINI_AGENT_WAIT_S", "45") or "45")))
        self.receiver = DataStreamAudioReceiver(self.worker_room, agent_wait_timeout_s=wait_s)
        self.runner = AgentAvatarRunner(
            self.worker_room,
            video_gen=self.generator,
            receiver=self.receiver,
            options=AvatarOptions(),
        )
        try:
            await self.runner.start()
        except TimeoutError as exc:
            dispatch_state = await fetch_room_dispatch(self.settings, self.room_name)
            raise RuntimeError(
                f"{exc} The Gemini worker did not join as {self.agent_name!r} "
                f"(dispatch created: {bool(self.agent_dispatched)}). {dispatch_state} "
                "Start it with `./scripts/run_gemini_agent.sh` (or "
                "`uv run python -m fast_float.livekit.gemini.agent start`). The worker reads the "
                "shared fast_float/config/secerts.toml + settings.toml (the single source of truth "
                "— put the same LiveKit project's LIVEKIT_URL/API_KEY/API_SECRET there, in both "
                "processes' FLOAT_CONFIG_DIR; the repo-root .env is only a fallback and a differing "
                "copy is ignored with a warning). Its agent name must match the API's.",
            ) from exc
        self.identity = self.receiver.agent_identity or self.identity
        logger.info(
            f"[session] Gemini Live conversation ready in {self.room_name} ({self.identity} -> {self.avatar_identity})"
        )

    async def _dispatch_external_agent(self):
        """Ask the SFU to run the Gemini worker, forwarding Studio overrides.

        Returns the created dispatch (or ``None`` when auto-dispatch is
        disabled / the agent name is unset / the API call failed — the
        operator starts the worker manually then).
        """
        if not (self.settings.autodispatch and self.agent_name):
            return None
        return await dispatch_agent(
            self.settings,
            self.room_name,
            agent_name=self.agent_name,
            gemini_api_key=self.gemini_api_key or None,
            gemini_model=self.gemini_model or None,
        )

    @property
    def resolved_agent_identity(self) -> str:
        if self.receiver is not None and self.receiver.agent_identity:
            return self.receiver.agent_identity
        return self.identity

    async def speak(self, text: str) -> None:
        """Turn a user utterance reply into audio and push it to the renderer."""
        if self.agent_backend == "gemini":
            raise RuntimeError("Gemini Live owns the voice; use the room microphone instead")
        if self.agent is None:
            raise RuntimeError("Session not started")
        if self.tts_fn is None:
            raise RuntimeError("No TTS engine attached to this session")
        await self.agent.send_speech(text)

    def push_audio(self, pcm16_mono: bytes) -> None:
        """Feed live microphone PCM (16 kHz, mono, int16) to the avatar.

        The browser posts short chunks; :class:`FloatVideoGenerator` buffers
        them and starts a 2 s render window as soon as enough audio arrives.
        Call this from the event-loop thread. Gemini sessions take the
        microphone through LiveKit instead, so this path is disabled there.
        """
        if self.agent_backend == "gemini":
            raise RuntimeError("Gemini sessions publish the microphone track to the room")
        if self.generator is None:
            raise RuntimeError("Session not started")
        self.generator.push_audio(pcm16_mono)

    def flush_audio(self) -> None:
        """Render the tail of the current utterance immediately."""
        if self.agent_backend == "gemini" or self.generator is None:
            return
        self.generator.flush()

    def interrupt(self) -> None:
        """Barge-in: drop buffered audio and any queued/in-flight render."""
        if self.generator is None:
            return
        if isinstance(self.runner, AgentAvatarRunner):
            self.runner.interrupt()
            return
        self.generator.clear_buffer()

    async def _delete_dispatch(self) -> None:
        """Best-effort removal of this session's SFU dispatch (never raises)."""
        if not self.dispatch_id:
            return
        try:
            from livekit.api import LiveKitAPI

            async with LiveKitAPI(
                url=self.settings.url,
                api_key=self.settings.api_key,
                api_secret=self.settings.api_secret,
            ) as lk:
                await lk.agent_dispatch.delete_dispatch(self.dispatch_id, self.room_name)
            logger.info(f"[session] deleted agent dispatch {self.dispatch_id[:8]} from {self.room_name}")
        except Exception as exc:
            logger.warning(f"[session] could not delete agent dispatch {self.dispatch_id[:8]}: {exc}")

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.generator is not None:
            if hasattr(self.generator, "stop"):
                self.generator.stop()
            elif hasattr(self.generator, "close"):
                with contextlib.suppress(Exception):
                    await self.generator.close()
        if self.runner is not None:
            with contextlib.suppress(Exception):
                await self.runner.stop()
            self.runner = None
        self.receiver = None
        if self.worker_room is not None:
            with contextlib.suppress(Exception):
                await self.worker_room.disconnect()
            self.worker_room = None
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await self.agent.disconnect()
            self.agent = None
        # Delete the SFU dispatch created for this session: a retry would
        # otherwise stack another queued dispatch and every later failure
        # report lists "NO job" for each leak.
        if self.dispatch_id:
            await self._delete_dispatch()
        self.dispatch_id = None
        if self._temp_image:
            with contextlib.suppress(OSError):
                os.unlink(self._temp_image)
            self._temp_image = None
        logger.info(f"[session] conversation stopped in {self.room_name}")


def get_session(room_name: str | None) -> ConversationSession | None:
    """The running session for ``room_name``, if any."""
    if room_name is None:
        return None
    session = _sessions.get(room_name)
    if session is not None and session.is_stopped:
        return None
    return session


async def start_session(
    room_name: str,
    *,
    settings: LiveKitSettings | None = None,
    identity: str | None = None,
    image_path: str | None = None,
    image_base64: str | None = None,
    image_format: str = "jpg",
    tts_fn: Callable[[str], bytes] | None = None,
    agent_backend: str = "builtin",
    agent_name: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
) -> ConversationSession:
    """Start (or reuse) the conversation session for ``room_name``."""
    existing = get_session(room_name)
    if existing is not None:
        logger.info(f"[session] reusing existing conversation session in {room_name}")
        return existing
    session = ConversationSession(
        room_name,
        settings=settings,
        identity=identity,
        image_path=image_path,
        image_base64=image_base64,
        image_format=image_format,
        tts_fn=tts_fn,
        agent_backend=agent_backend,
        agent_name=agent_name,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )
    try:
        await session.start()
    except Exception:
        # A failed start must not leak a temp portrait or a half-open room.
        await session.stop()
        raise
    _sessions[room_name] = session
    return session


async def stop_session(room_name: str) -> None:
    """Stop and forget the conversation session for ``room_name``."""
    session = _sessions.pop(room_name, None)
    if session is not None:
        await session.stop()


__all__ = [
    "ConversationSession",
    "dispatch_agent",
    "get_render_semaphore",
    "get_session",
    "start_session",
    "stop_session",
]
