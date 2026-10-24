"""Thin HTTP client for the FLOAT API used by the voice tools."""

from __future__ import annotations

import json
from typing import Any

import httpx


def _short(value: Any, limit: int = 1800) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "…"


class FloatApi:
    """Async client for the small subset of FLOAT endpoints the voice agent uses."""

    def __init__(self, base_url: str, *, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.request(method, url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"FLOAT API is unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:600].strip()
            raise RuntimeError(f"FLOAT API returned HTTP {response.status_code}: {detail}")
        try:
            return response.json()
        except ValueError:
            return response.text

    async def health(self) -> str:
        return _short(await self._request("GET", "/health"))

    async def gpu(self) -> str:
        return _short(await self._request("GET", "/gpu"))

    async def progress(self, request_id: str) -> str:
        request_id = str(request_id or "").strip()
        if not request_id:
            raise RuntimeError("request_id is required")
        return _short(await self._request("GET", f"/progress/{request_id}"))

    async def conversation_status(self, room: str) -> str:
        return _short(await self._request("GET", f"/livekit/conversation/status?room={room}"))

    async def stop_conversation(self, room: str) -> str:
        return _short(await self._request("POST", "/livekit/conversation/stop", {"room": room}))


__all__ = ["FloatApi"]
