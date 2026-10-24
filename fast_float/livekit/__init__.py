"""LiveKit integration: the transport for the live preview and the avatar.

Public names are imported lazily (PEP 562). The Gemini worker imports
``fast_float.livekit.gemini.*`` in its own process; importing this package must
not drag in the inference stack (torch/onnx) just because the worker shares the
same venv. ``from fast_float.livekit import X`` and ``from fast_float.livekit
import session`` keep working exactly as before.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed
    from fast_float.livekit.agent import (
        DEFAULT_AGENT_IDENTITY,
        DEFAULT_AVATAR_IDENTITY,
        ConversationalAgent,
    )
    from fast_float.livekit.agent_runner import AgentAvatarRunner
    from fast_float.livekit.generator import (
        BYTES_PER_FRAME,
        SAMPLES_PER_FRAME,
        FloatVideoGenerator,
        LiveStreamVideoGenerator,
        MediaFileVideoGenerator,
    )
    from fast_float.livekit.receiver import DataStreamAudioReceiver, SegmentEnd
    from fast_float.livekit.settings import LiveKitSettings, valid_room_name
    from fast_float.livekit.tokens import IssuedToken, LiveKitUnavailable, issue_token
    from fast_float.livekit.worker import (
        DEFAULT_AVATAR_NAME,
        AvatarOptions,
        AvatarRunner,
        LiveKitLiveStreamPublisher,
        LiveKitMediaFilePublisher,
        publish_live_stream_task,
        publish_media_file_task,
    )

# name -> (module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "DEFAULT_AGENT_IDENTITY": ("fast_float.livekit.agent", "DEFAULT_AGENT_IDENTITY"),
    "DEFAULT_AVATAR_IDENTITY": ("fast_float.livekit.agent", "DEFAULT_AVATAR_IDENTITY"),
    "ConversationalAgent": ("fast_float.livekit.agent", "ConversationalAgent"),
    "AgentAvatarRunner": ("fast_float.livekit.agent_runner", "AgentAvatarRunner"),
    "BYTES_PER_FRAME": ("fast_float.livekit.generator", "BYTES_PER_FRAME"),
    "SAMPLES_PER_FRAME": ("fast_float.livekit.generator", "SAMPLES_PER_FRAME"),
    "FloatVideoGenerator": ("fast_float.livekit.generator", "FloatVideoGenerator"),
    "LiveStreamVideoGenerator": ("fast_float.livekit.generator", "LiveStreamVideoGenerator"),
    "MediaFileVideoGenerator": ("fast_float.livekit.generator", "MediaFileVideoGenerator"),
    "DataStreamAudioReceiver": ("fast_float.livekit.receiver", "DataStreamAudioReceiver"),
    "SegmentEnd": ("fast_float.livekit.receiver", "SegmentEnd"),
    "LiveKitSettings": ("fast_float.livekit.settings", "LiveKitSettings"),
    "valid_room_name": ("fast_float.livekit.settings", "valid_room_name"),
    "IssuedToken": ("fast_float.livekit.tokens", "IssuedToken"),
    "LiveKitUnavailable": ("fast_float.livekit.tokens", "LiveKitUnavailable"),
    "issue_token": ("fast_float.livekit.tokens", "issue_token"),
    "DEFAULT_AVATAR_NAME": ("fast_float.livekit.worker", "DEFAULT_AVATAR_NAME"),
    "AvatarOptions": ("fast_float.livekit.worker", "AvatarOptions"),
    "AvatarRunner": ("fast_float.livekit.worker", "AvatarRunner"),
    "LiveKitLiveStreamPublisher": ("fast_float.livekit.worker", "LiveKitLiveStreamPublisher"),
    "LiveKitMediaFilePublisher": ("fast_float.livekit.worker", "LiveKitMediaFilePublisher"),
    "publish_live_stream_task": ("fast_float.livekit.worker", "publish_live_stream_task"),
    "publish_media_file_task": ("fast_float.livekit.worker", "publish_media_file_task"),
}

__all__: list[str] = [
    "BYTES_PER_FRAME",
    "DEFAULT_AGENT_IDENTITY",
    "DEFAULT_AVATAR_IDENTITY",
    "DEFAULT_AVATAR_NAME",
    "SAMPLES_PER_FRAME",
    "AgentAvatarRunner",
    "AvatarOptions",
    "AvatarRunner",
    "ConversationalAgent",
    "DataStreamAudioReceiver",
    "FloatVideoGenerator",
    "IssuedToken",
    "LiveKitLiveStreamPublisher",
    "LiveKitMediaFilePublisher",
    "LiveKitSettings",
    "LiveKitUnavailable",
    "LiveStreamVideoGenerator",
    "MediaFileVideoGenerator",
    "SegmentEnd",
    "issue_token",
    "publish_live_stream_task",
    "publish_media_file_task",
    "valid_room_name",
]

# Keep the literal (ruff needs one for PLE0605) in sync with the lazy table:
# `from fast_float.livekit import *` resolves every name through __getattr__,
# so a drift is a real ImportError, not just a cosmetic mismatch.
if set(__all__) != set(_LAZY) or len(__all__) != len(_LAZY):
    raise RuntimeError(
        "fast_float.livekit.__all__ is out of sync with _LAZY: "
        f"missing={sorted(set(_LAZY) - set(__all__))} "
        f"unknown={sorted(set(__all__) - set(_LAZY))}"
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
