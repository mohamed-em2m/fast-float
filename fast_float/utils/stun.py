"""Minimal STUN binding server for the raw WebRTC transport.

The browser and aiortc both need a STUN server to discover their public
(reflexive) ICE candidates. Requiring a separate coturn container — or
routing ICE through a LiveKit SFU — is unnecessary for this deployment: the
API host is already the WebRTC signalling endpoint, so it can answer STUN
binding requests itself.

Only the RFC 5389 binding request/response is implemented (no TURN, no auth,
no TCP). That is exactly what ``RTCIceServer("stun:...")`` uses; a binding
server never relays media and cannot be abused as an open relay.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import struct

from fast_float.utils.logger import logger

MAGIC_COOKIE = 0x2112A442
BINDING_REQUEST = 0x0001
BINDING_RESPONSE = 0x0101
ATTR_MAPPED_ADDRESS = 0x0001
ATTR_XOR_MAPPED_ADDRESS = 0x0020

_HEADER = struct.Struct("!HHI")
_ATTR = struct.Struct("!HH")
_COOKIE_BYTES = struct.pack("!I", MAGIC_COOKIE)


def _xor_address(address: str, transaction_id: bytes) -> tuple[int, bytes]:
    """XOR an IP address with the magic cookie (and transaction id for IPv6)."""
    ip = ipaddress.ip_address(address)
    if ip.version == 4:
        family = 0x01
        mask: bytes = _COOKIE_BYTES
    else:
        # STUN address families are 0x01 (IPv4) / 0x02 (IPv6), not the IP
        # version number.
        family = 0x02
        mask = _COOKIE_BYTES + transaction_id
    return family, bytes(a ^ b for a, b in zip(ip.packed, mask, strict=True))


def build_binding_response(data: bytes, address: tuple[str, int]) -> bytes | None:
    """Return an RFC 5389 binding response, or ``None`` for unrelated traffic.

    ``address`` is the source address of the request (the client's reflexive
    address as seen by this server). Non-STUN datagrams, truncated messages and
    other STUN methods are ignored so the UDP socket can share port 3478 with
    nothing else on the host.
    """
    if len(data) < 20:
        return None
    message_type, length, cookie = _HEADER.unpack_from(data, 0)
    if message_type != BINDING_REQUEST or cookie != MAGIC_COOKIE:
        return None
    if length > len(data) - 20:
        return None

    transaction_id = data[8:20]
    family, xored = _xor_address(address[0], transaction_id)
    xored_port = address[1] ^ (MAGIC_COOKIE >> 16)
    value = struct.pack("!BBH", 0, family, xored_port) + xored
    attributes = _ATTR.pack(ATTR_XOR_MAPPED_ADDRESS, len(value)) + value

    return _HEADER.pack(BINDING_RESPONSE, len(attributes), MAGIC_COOKIE) + transaction_id + attributes


class _StunProtocol(asyncio.DatagramProtocol):
    """Answer binding requests; every other datagram is dropped."""

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        response = build_binding_response(data, address)
        if response is None:
            return
        # A vanished socket (shutdown race) must not kill the loop.
        with contextlib.suppress(OSError):
            self.transport.sendto(response, address)  # type: ignore[attr-defined]

    def error_received(self, exc: Exception) -> None:
        logger.debug(f"STUN socket error: {exc}")


class StunServer:
    """A UDP STUN responder bound to ``host:port``."""

    def __init__(self, host: str = "0.0.0.0", port: int = 3478) -> None:
        self.host = host
        self.port = int(port)
        self._transport: asyncio.DatagramTransport | None = None

    @property
    def running(self) -> bool:
        return self._transport is not None and not self._transport.is_closing()

    async def start(self) -> bool:
        """Bind the socket. Returns ``False`` (and logs) when the port is busy."""
        if self.running:
            return True
        loop = asyncio.get_running_loop()
        try:
            self._transport, _ = await loop.create_datagram_endpoint(
                _StunProtocol,
                local_addr=(self.host, self.port),
            )
        except OSError as exc:
            logger.warning(
                f"STUN responder could not bind {self.host}:{self.port} ({exc}); "
                "raw WebRTC keeps host candidates only (set WEBRTC_STUN_PORT or [webrtc].stun_urls)."
            )
            return False
        logger.info(f"🧊 STUN responder listening on udp://{self.host}:{self.port}")
        return True

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            logger.info("STUN responder stopped")


__all__ = [
    "ATTR_MAPPED_ADDRESS",
    "ATTR_XOR_MAPPED_ADDRESS",
    "BINDING_REQUEST",
    "BINDING_RESPONSE",
    "MAGIC_COOKIE",
    "StunServer",
    "build_binding_response",
]
