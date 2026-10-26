"""Make the flat-layout repo importable in tests (no installable package)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Variables that fast_float.livekit.gemini.config.AgentConfig.from_env() exports
# into the real process environment, because livekit-agents (LIVEKIT_*) and
# google-genai (GOOGLE_API_KEY) read os.environ directly instead of our config
# object.
# Tests must not inherit whichever value ran before them.
_SDK_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _restore_sdk_env():
    """Snapshot/restore the SDK-visible env around every test.

    The export happens after monkeypatch records its undo state, so without
    this a file-first test leaks, say, a LiveKit secret into the next test's
    "credentials are missing" assertions.
    """
    before = {name: os.environ.get(name) for name in _SDK_ENV}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
