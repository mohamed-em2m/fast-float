"""Trickle-ICE endpoint tests (no real peer connection or network)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.routers import webrtc as webrtc_router


class FakePeerConnection:
    def __init__(self) -> None:
        self.candidates: list = []

    async def addIceCandidate(self, candidate) -> None:
        self.candidates.append(candidate)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(webrtc_router.router)
    return TestClient(app)


def test_candidate_endpoint_parses_browser_candidate(client, monkeypatch):
    pc = FakePeerConnection()
    monkeypatch.setitem(webrtc_router._pending_peers, "peer-1", pc)

    response = client.post(
        "/offer/candidate",
        json={
            "peer_id": "peer-1",
            "candidate": "candidate:1 1 UDP 2130706431 192.0.2.7 5000 typ host",
            "sdp_mid": "0",
            "sdp_mline_index": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(pc.candidates) == 1
    candidate = pc.candidates[0]
    assert candidate.ip == "192.0.2.7"
    assert candidate.port == 5000
    assert candidate.protocol == "UDP"
    assert candidate.type == "host"
    assert candidate.sdpMid == "0"
    assert candidate.sdpMLineIndex == 0


def test_candidate_endpoint_accepts_end_of_candidates(client, monkeypatch):
    pc = FakePeerConnection()
    monkeypatch.setitem(webrtc_router._pending_peers, "peer-2", pc)

    response = client.post("/offer/candidate", json={"peer_id": "peer-2", "candidate": None})

    assert response.status_code == 200
    assert pc.candidates == [None]


def test_candidate_endpoint_rejects_unknown_peer(client):
    response = client.post(
        "/offer/candidate",
        json={"peer_id": "missing", "candidate": "candidate:1 1 UDP 1 192.0.2.7 5000 typ host"},
    )
    assert response.status_code == 404
