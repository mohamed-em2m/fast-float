"""Dynamic config: settings.toml (operator defaults) + secerts.toml (secrets), env wins.

Files live next to this module:

* ``settings.toml`` — non-secret defaults, committed, edited by the operator.
* ``secerts.toml``  — API keys/secrets (NOTE: not "secrets"; historical name),
  never committed, environment per machine.

Precedence (highest first):

1. environment variables (``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ...)
2. ``secerts.toml``       — per-machine secrets
3. ``settings.toml``      — deployment defaults
4. the caller's built-in fallback

Import ``load_settings``/``load_secrets``/``first_value`` wherever env-less
config is needed; values are re-read when a file's mtime changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomli  # type: ignore[import-not-found]

    _TOMLIB = True
except ImportError:  # pragma: no cover - until the tomli dep is in the lock
    tomli = None  # type: ignore[assignment]
    _TOMLIB = False

CONFIG_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = CONFIG_DIR / "settings.toml"
SECRETS_PATH = CONFIG_DIR / "secerts.toml"


@lru_cache(maxsize=32)
def _read_cached(path: str, mtime_ns: int) -> dict[str, Any]:
    """Parse one TOML file (mtime part of the key keeps the cache honest)."""
    with open(path, "rb") as f:
        return tomli.load(f)  # type: ignore[union-attr]


def _read(path: Path) -> dict[str, Any]:
    if not _TOMLIB:
        return {}
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _read_cached(str(path), stat.st_mtime_ns)


def load_settings() -> dict[str, Any]:
    """The committed operator defaults, as a dict ({} when absent/broken)."""
    try:
        return _read(SETTINGS_PATH)
    except Exception:  # a malformed operator edit must not kill the app
        return {}


def load_secrets() -> dict[str, Any]:
    """The per-machine secret file, as a dict ({} when absent/broken)."""
    try:
        return _read(SECRETS_PATH)
    except Exception:
        return {}


def first_value(*candidates: Any) -> tuple[Any, str]:
    """First candidate that is set.

    Returns ``(value, source_label)``; source is one of ``env``,
    ``secerts.toml``, ``settings.toml`` or ``none`` so callers can report
    where the effective configuration came from.
    """
    for cand in candidates:
        value, source = cand
        if value not in (None, ""):
            return value, source
    return None, "none"


__all__ = [
    "CONFIG_DIR",
    "SECRETS_PATH",
    "SETTINGS_PATH",
    "first_value",
    "load_secrets",
    "load_settings",
]
