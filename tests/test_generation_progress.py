"""Contract tests for the generation progress payload.

The Studio renders its progress overlay from ``GET /progress/{id}`` and treats
a single-pass render (``percent`` pinned at 5.0) as indeterminate, using
``started_at``/``elapsed_s``/``audio_seconds`` for the live readout. These
fields are optional, so an older payload must still validate.
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.routers.generation import router as generation_router
from fast_float.schemas.generation import GenerationProgress


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(generation_router)
    return TestClient(app)


def test_progress_timing_fields_default_to_unknown():
    progress = GenerationProgress(request_id="r1", status="queued")

    assert progress.audio_seconds == 0.0
    assert progress.started_at is None
    assert progress.elapsed_s == 0.0


def test_progress_endpoint_exposes_timing_and_keeps_legacy_records():
    request_id = "req-progress-contract"
    started = time.time() - 42.0
    app_state.set_progress(
        request_id,
        status="processing",
        stage="generating",
        percent=5.0,
        total_chunks=1,
        completed_chunks=0,
        audio_seconds=12.5,
        started_at=started,
        elapsed_s=42.0,
        # Fields the router must filter out of the response model.
        internal_note="not part of the schema",
    )

    response = _client().get(f"/progress/{request_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert body["audio_seconds"] == 12.5
    assert body["started_at"] == started
    assert body["elapsed_s"] == 42.0
    assert "internal_note" not in body


def test_progress_endpoint_404s_for_unknown_requests():
    response = _client().get("/progress/never-existed")
    assert response.status_code == 404
