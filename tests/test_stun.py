"""STUN responder + ICE URL tests (main server as the ICE/STUN endpoint)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import struct

from fast_float.livekit.settings import LiveKitSettings, host_only, stun_url
from fast_float.utils.stun import (
    BINDING_RESPONSE,
    MAGIC_COOKIE,
    build_binding_response,
)


def _request(transaction_id: bytes = b"0123456789ab") -> bytes:
    return struct.pack("!HHI", 0x0001, 0, MAGIC_COOKIE) + transaction_id


def _xor_mapped_address(response: bytes, transaction_id: bytes) -> tuple[str, int]:
    # header (20 bytes) then the first attribute
    attr_type, attr_len = struct.unpack_from("!HH", response, 20)
    assert attr_type == 0x0020
    assert attr_len in (8, 20)
    family = response[25]
    port = struct.unpack_from("!H", response, 26)[0] ^ (MAGIC_COOKIE >> 16)
    raw = response[28 : 28 + (4 if family == 1 else 16)]
    mask = struct.pack("!I", MAGIC_COOKIE)
    if family == 2:
        mask += transaction_id
    decoded = bytes(a ^ b for a, b in zip(raw, mask, strict=True))
    return str(ipaddress.ip_address(decoded)), port


def test_binding_response_echoes_the_reflexive_ipv4_address():
    txid = b"0123456789ab"
    response = build_binding_response(_request(txid), ("203.0.113.7", 54321))

    assert response is not None
    message_type, length, cookie = struct.unpack_from("!HHI", response, 0)
    assert message_type == BINDING_RESPONSE
    assert cookie == MAGIC_COOKIE
    assert response[8:20] == txid
    assert length == len(response) - 20
    assert _xor_mapped_address(response, txid) == ("203.0.113.7", 54321)


def test_binding_response_supports_ipv6():
    txid = b"abcdefghijkl"
    response = build_binding_response(_request(txid), ("2001:db8::1", 50000))

    assert response is not None
    assert _xor_mapped_address(response, txid) == ("2001:db8::1", 50000)


def test_non_binding_and_truncated_datagrams_are_ignored():
    assert build_binding_response(b"", ("127.0.0.1", 1)) is None
    assert build_binding_response(b"\x00" * 20, ("127.0.0.1", 1)) is None
    # A binding request with a length larger than the datagram is malformed.
    bad = struct.pack("!HHI", 0x0001, 999, MAGIC_COOKIE) + b"0123456789ab"
    assert build_binding_response(bad, ("127.0.0.1", 1)) is None


def test_stun_url_brackets_ipv6_and_keeps_dns_names():
    assert stun_url("203.0.113.7") == "stun:203.0.113.7:3478"
    assert stun_url("2001:db8::1", 3479) == "stun:[2001:db8::1]:3479"
    assert stun_url("float.example.com") == "stun:float.example.com:3478"
    assert stun_url("") == ""


def test_host_only_handles_ports_proxy_chains_and_ipv6():
    assert host_only("float.example.com:8000") == "float.example.com"
    assert host_only("float.example.com:8000, proxy.internal") == "float.example.com"
    assert host_only("[2001:db8::1]:8000") == "2001:db8::1"
    assert host_only("2001:db8::1") == "2001:db8::1"
    assert host_only("") == ""
    assert host_only(None) == ""


def test_settings_advertise_the_builtin_responder(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"webrtc": {"stun_enabled": True, "stun_port": 3478}},
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    for name in ("WEBRTC_STUN_URLS", "WEBRTC_STUN_HOST", "WEBRTC_STUN_PORT", "WEBRTC_STUN"):
        monkeypatch.delenv(name, raising=False)

    settings = LiveKitSettings.from_env()
    assert settings.resolved_stun_urls("192.0.2.10") == ("stun:192.0.2.10:3478",)


def test_explicit_stun_list_wins_over_the_builtin_responder(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"webrtc": {"stun_urls": ["stun:coturn.example:3478"]}},
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    monkeypatch.delenv("WEBRTC_STUN_URLS", raising=False)

    settings = LiveKitSettings.from_env()
    assert settings.resolved_stun_urls("192.0.2.10") == ("stun:coturn.example:3478",)


def test_ice_servers_include_stun_and_static_turn(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {
            "webrtc": {
                "stun_urls": ["stun:stun.example:3478"],
                "turn_urls": ["turn:turn.example:3478?transport=tcp"],
                "turn_username": "viewer",
                "turn_credential": "secret",
            }
        },
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    for name in ("WEBRTC_STUN_URLS", "WEBRTC_TURN_URLS", "WEBRTC_TURN_USERNAME",
                 "WEBRTC_TURN_CREDENTIAL", "WEBRTC_TURN_SECRET"):
        monkeypatch.delenv(name, raising=False)

    settings = LiveKitSettings.from_env()
    assert settings.resolved_ice_servers() == (
        {"urls": "stun:stun.example:3478"},
        {
            "urls": "turn:turn.example:3478?transport=tcp",
            "username": "viewer",
            "credential": "secret",
        },
    )


def test_turn_rest_credentials_are_derived_from_the_shared_secret(monkeypatch):
    monkeypatch.setenv("WEBRTC_TURN_SECRET", "turn-shared-secret")
    monkeypatch.setenv("WEBRTC_TURN_URLS", "turn:turn.example:3478")
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})

    settings = LiveKitSettings.from_env()
    username, credential = settings.turn_credentials(ttl_s=600)
    expected = base64.b64encode(
        hmac.new(b"turn-shared-secret", username.encode(), hashlib.sha1).digest()
    ).decode("ascii")
    assert username.split(":", 1)[1]
    assert credential == expected

    servers = settings.resolved_ice_servers()
    entry = servers[0]
    assert entry["urls"] == "turn:turn.example:3478"
    assert entry["username"].split(":", 1)[1]
    assert entry["credential"] == base64.b64encode(
        hmac.new(b"turn-shared-secret", entry["username"].encode(), hashlib.sha1).digest()
    ).decode("ascii")
