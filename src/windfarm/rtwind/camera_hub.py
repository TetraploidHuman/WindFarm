"""Latest JPEG frame from live aircraft camera (phone upload)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any


class CameraHub:
    def __init__(self, max_bytes: int = 2_000_000) -> None:
        self._lock = asyncio.Lock()
        self._jpeg: bytes | None = None
        self._updated_at: str | None = None
        self._vehicle_id: str | None = None
        self._max_bytes = max_bytes
        self._seq = 0

    async def put(self, data: bytes, *, vehicle_id: str = "live-1") -> dict[str, Any]:
        if not data:
            return {"ok": False, "error": "empty frame"}
        if len(data) > self._max_bytes:
            return {"ok": False, "error": f"frame too large ({len(data)} bytes)"}
        if data[:3] != b"\xff\xd8\xff":
            return {"ok": False, "error": "expected JPEG data"}
        async with self._lock:
            self._jpeg = data
            self._updated_at = datetime.now(timezone.utc).isoformat()
            self._vehicle_id = vehicle_id
            self._seq += 1
            return {
                "ok": True,
                "bytes": len(data),
                "seq": self._seq,
                "updated_at": self._updated_at,
                "vehicle_id": vehicle_id,
            }

    async def latest_bytes(self) -> bytes | None:
        async with self._lock:
            return self._jpeg

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "has_frame": self._jpeg is not None,
                "bytes": len(self._jpeg) if self._jpeg else 0,
                "seq": self._seq,
                "updated_at": self._updated_at,
                "vehicle_id": self._vehicle_id,
            }

    async def clear(self) -> None:
        async with self._lock:
            self._jpeg = None
            self._updated_at = None
            self._vehicle_id = None
