"""LiveKit integration: the transport for the live preview and the avatar.

The API process only needs the *token* API (``livekit-api``, numpy-free) to
mint room grants for the browser. The RTC SDK (``livekit``) requires
``numpy>=1.26`` and therefore cannot live in this environment, which pins
``numpy==1.24.4`` for ``onnx<1.17`` — the avatar worker gets its own project
(see ``docs/LIVEKIT_INTEGRATION.md``).
"""

from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import IssuedToken, LiveKitUnavailable, issue_token

__all__ = [
    "IssuedToken",
    "LiveKitSettings",
    "LiveKitUnavailable",
    "issue_token",
]
