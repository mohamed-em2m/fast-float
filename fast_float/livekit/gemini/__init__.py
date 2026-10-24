"""Gemini Live worker (LiveKit Agents) for FLOAT.

The worker is part of this project and shares its venv/lockfile, but runs as
a separate process so LiveKit can dispatch jobs to it:

    uv run python -m fast_float.livekit.gemini.agent start

Keep this package's ``__init__`` import-free: importing ``config`` must not
pull in the agent/CLI (or torch through the parent ``fast_float.livekit``).
"""

__all__: list[str] = []
