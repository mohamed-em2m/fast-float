"""Self checks for the LiveKit + Gemini voice pipeline.

The API reports these from ``GET /livekit/status`` so the Studio can warn the
user about missing/incorrect pieces *before* a session fails, instead of only
after a 503:

* cheap checks (presence/coherence from settings) run on every status read;
* the real SFU connectivity probe (mint a token + dial the SFU) is cached and
  runs once at startup plus on demand (``?verify=1``), so the Studio's 6 s
  status poll never hammers the SFU.

Results are intentionally side-effect free: nothing here mints long-lived
tokens or creates sessions — the probe uses a throwaway room/identity.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import issue_token
from fast_float.livekit.worker import RTC_AVAILABLE

logger = logging.getLogger(__name__)

CheckStatus = Literal["ok", "warn", "fail", "skip"]

PROBE_TIMEOUT_S = 12.0

#: Room prefix used by the throwaway connectivity probe.
PROBE_ROOM_PREFIX = "float-selftest"

#: Message shown until the SFU has been probed once.
PROBE_PENDING_MESSAGE = (
    "The SFU has not been probed yet. Start a Gemini session or call "
    "GET /livekit/status?verify=1 to verify the credentials actually connect."
)


@dataclass(frozen=True)
class Check:
    """One named check with a machine status and a human-readable message."""

    name: str
    status: CheckStatus
    message: str


def run_checks(settings: LiveKitSettings) -> list[Check]:
    """Cheap, side-effect free checks derived from the effective settings."""
    checks: list[Check] = []

    if settings.configured:
        checks.append(
            Check(
                "livekit_credentials",
                "ok",
                f"LiveKit project configured (LIVEKIT_URL + API key + secret, from {settings.source}).",
            )
        )
    else:
        checks.append(
            Check(
                "livekit_credentials",
                "fail",
                "LiveKit is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET (env or secerts.toml [livekit]) — the SFU transport stays off.",
            )
        )

    checks.append(
        Check(
            "livekit_connectivity",
            "skip",
            PROBE_PENDING_MESSAGE,
        )
    )

    if settings.gemini_configured:
        checks.append(Check("gemini_api_key", "ok", "Gemini API key is configured on the server."))
    else:
        checks.append(
            Check(
                "gemini_api_key",
                "fail",
                "No Gemini API key on the server. Add [gemini].api_key to "
                "fast_float/config/secerts.toml or export GOOGLE_API_KEY — or paste one "
                "in the Studio's Config panel for per-session use.",
            )
        )

    if settings.agent_name:
        checks.append(Check("agent_name", "ok", f"Gemini dispatches the worker {settings.agent_name!r}."))
    else:
        checks.append(
            Check(
                "agent_name",
                "fail",
                "FLOAT_GEMINI_AGENT_NAME is unset — there is no agent to dispatch for Gemini sessions.",
            )
        )

    if settings.autodispatch:
        checks.append(
            Check("autodispatch", "ok", "Auto-dispatch is on: each Gemini session asks the SFU to run the worker.")
        )
    else:
        checks.append(
            Check(
                "autodispatch",
                "warn",
                "Auto-dispatch is off (FLOAT_GEMINI_AUTODISPATCH=0). The worker must be "
                "dispatched manually for each Gemini session, or it will not join the room.",
            )
        )

    return checks


def warning_messages(checks: list[Check]) -> list[str]:
    """Flat list of the messages a user must act on (warn/fail status)."""
    return [check.message for check in checks if check.status in ("warn", "fail")]


async def probe_livekit(settings: LiveKitSettings) -> Check:
    """Mint a throwaway token and dial the SFU to prove the credentials work.

    This is the only check that touches the network; it uses a short-lived,
    non-publishing token in a throwaway room.
    """
    if not settings.configured:
        return Check(
            "livekit_connectivity",
            "fail",
            "Cannot probe the SFU: LiveKit is not configured (missing URL / key / secret).",
        )
    if not RTC_AVAILABLE:
        return Check(
            "livekit_connectivity",
            "fail",
            "The LiveKit RTC SDK is not installed in this environment — the SFU transport is unavailable.",
        )
    try:
        token = issue_token(
            url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            identity="selftest",
            room=f"{PROBE_ROOM_PREFIX}-{uuid4().hex[:8]}",
            can_publish=False,
        )
    except Exception as exc:  # LiveKitUnavailable and friends
        return Check("livekit_connectivity", "fail", f"Could not mint a probe token: {exc}")
    try:
        from livekit import rtc

        room = rtc.Room()
        try:
            await asyncio.wait_for(room.connect(settings.url, token.token), timeout=PROBE_TIMEOUT_S)
        finally:
            with contextlib.suppress(Exception):
                await room.disconnect()
    except asyncio.TimeoutError:
        return Check(
            "livekit_connectivity",
            "fail",
            f"Timed out connecting to {settings.url} within {PROBE_TIMEOUT_S:g}s — "
            "is the SFU reachable from this server?",
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        return Check(
            "livekit_connectivity",
            "fail",
            f"The SFU rejected the connection: {message[:220]}",
        )
    logger.info("[selftest] LiveKit connectivity probe succeeded")
    return Check("livekit_connectivity", "ok", f"Connected to {settings.url} with the configured keys.")


# ---------------------------------------------------------------------------
# Cached probe orchestration. A single probe result is shared process-wide;
# start_probe() runs once at API startup (background), run_verify() forces a
# fresh one (?verify=1), cached_check() returns the last result or a pending
# placeholder — it never probes by itself, so the Studio's 6 s poll is cheap.
# ---------------------------------------------------------------------------

_connectivity: Check | None = None
_connectivity_lock = asyncio.Lock()
_connectivity_started = False


async def cached_check() -> Check:
    """Last probe result, or the pending placeholder when never probed."""
    if _connectivity is not None:
        return _connectivity
    return Check("livekit_connectivity", "skip", PROBE_PENDING_MESSAGE)


async def run_verify(settings: LiveKitSettings) -> Check:
    """Force a fresh probe against the current settings and cache it."""
    global _connectivity, _connectivity_started
    async with _connectivity_lock:
        _connectivity = await probe_livekit(settings)
        _connectivity_started = True
    return _connectivity


def start_probe(settings: LiveKitSettings) -> None:
    """Best-effort background probe (startup/headless). Never blocks or raises."""

    async def _once() -> None:
        try:
            await run_verify(settings)
        except Exception:
            logger.exception("[selftest] background connectivity probe failed")

    # Keep a reference so the task is never garbage-collected mid-flight.
    with contextlib.suppress(RuntimeError):
        task = asyncio.create_task(_once())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)


_background_tasks: set[asyncio.Task[None]] = set()


__all__ = [
    "PROBE_PENDING_MESSAGE",
    "PROBE_TIMEOUT_S",
    "Check",
    "CheckStatus",
    "cached_check",
    "probe_livekit",
    "run_checks",
    "run_verify",
    "start_probe",
    "warning_messages",
]
