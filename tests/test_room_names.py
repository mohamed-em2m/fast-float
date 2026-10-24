"""Room-name audit: every generated/validated name matches LiveKit's rules.

LiveKit room names are `[A-Za-z0-9_-]` up to 64 characters (the server
rejects anything else). These tests pin:

* the validator used by the API endpoints;
* the browser's `newRoomName()` algorithm (LiveAudioStudio.jsx) — mirrored
  here — so every client-generated room is always accepted by the backend;
* the server-side `room_for()` generator, including its fallback when the
  operator sets a broken `room_prefix`.
"""

from __future__ import annotations

import random
import string

from fast_float.livekit.settings import DEFAULT_ROOM_PREFIX, LiveKitSettings, valid_room_name

# Mirrors the Studio's generator (frontend/src/components/LiveAudioStudio.jsx):
#   `float-live-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`
# JS Number.toString(36) emits base-36 digits (0-9a-z), which is exactly the
# charset LiveKit accepts, so this mirror reproduces the same shape.
_BASE36 = string.digits + string.ascii_lowercase


def _base36(value: int, width: int = 0) -> str:
    if value == 0:
        out = "0"
    else:
        digits: list[str] = []
        while value:
            value, remainder = divmod(value, 36)
            digits.append(_BASE36[remainder])
        out = "".join(reversed(digits))
    return out.zfill(width)


def frontend_new_room_name() -> str:
    timestamp = _base36(random.randrange(10**12, 10**13))  # Date.now() scale
    slug = _base36(random.randrange(36**6), width=6)[:6]  # Math.random().slice(2,8)
    return f"float-live-{timestamp}-{slug}"


def test_valid_room_name_enforces_livekit_charset_and_length():
    assert valid_room_name("float-live-abc")
    assert valid_room_name("float_live-xY-9")
    assert valid_room_name("a")  # single char
    assert valid_room_name("f" * 64)
    assert not valid_room_name("")  # empty never valid
    assert not valid_room_name("f" * 65)  # LiveKit cap is 64
    assert not valid_room_name("float room")  # space
    assert not valid_room_name("float/room")  # slash
    assert not valid_room_name("float.room")  # dot
    assert not valid_room_name("float:room")  # colon


def test_frontend_generated_rooms_are_always_valid():
    """The Studio's newRoomName() output must always pass the API validator."""
    for _ in range(500):
        name = frontend_new_room_name()
        assert valid_room_name(name), name
        assert len(name) <= 64, name


def test_example_room_from_production_logs_is_valid():
    """The room name seen in the 'no agent joined' log is valid."""
    assert valid_room_name("float-live-mvmlauno-s3mtxb")


def test_room_for_generates_valid_names():
    settings = LiveKitSettings(room_prefix="float")
    for _ in range(200):
        room = settings.room_for()
        assert valid_room_name(room)
        assert room.startswith("float-")
    # A client-supplied name is passed through untouched.
    assert settings.room_for("custom-room") == "custom-room"


def test_room_for_falls_back_when_room_prefix_is_broken():
    # A prefix full of invalid characters (or one that blows the 64-char cap)
    # must still yield a usable generated room.
    bad = LiveKitSettings(room_prefix="with space/ slash")
    for _ in range(50):
        room = bad.room_for()
        assert valid_room_name(room)
        assert room.startswith(DEFAULT_ROOM_PREFIX)

    huge = LiveKitSettings(room_prefix="x" * 64)  # prefix alone already hits the cap
    for _ in range(50):
        room = huge.room_for()
        assert valid_room_name(room)
        assert len(room) <= 64


def test_generated_room_names_collide_negligibly():
    """Room ids carry enough entropy that concurrent sessions do not collide."""
    settings = LiveKitSettings(room_prefix="float")
    seen = {settings.room_for() for _ in range(5000)}
    assert len(seen) == 5000
