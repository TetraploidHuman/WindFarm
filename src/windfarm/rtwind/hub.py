from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Callable, Awaitable

from .types import SourceKind, TelemetryFrame, TrackPoint, RtwindConfig


class TelemetryHub:
    """Exclusive active-source hub: only one of live/sim writes frames."""

    def __init__(self, config: RtwindConfig):
        self.config = config
        self.active: SourceKind = config.active_source
        self._latest: TelemetryFrame | None = None
        self._track: deque[TrackPoint] = deque(maxlen=config.track_max)
        self._seq = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()
        self._live_last_rx = 0.0
        self._on_source_change: list[Callable[[SourceKind, SourceKind], Awaitable[None] | None]] = []
        self._on_frame: list[Callable[[TelemetryFrame], Awaitable[None] | None]] = []

    def add_source_listener(self, cb: Callable[[SourceKind, SourceKind], Awaitable[None] | None]) -> None:
        self._on_source_change.append(cb)

    def add_frame_listener(self, cb: Callable[[TelemetryFrame], Awaitable[None] | None]) -> None:
        self._on_frame.append(cb)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    async def publish(self, frame: TelemetryFrame) -> bool:
        """Accept frame only if it matches the active source."""
        async with self._lock:
            if frame.source != self.active:
                return False
            self._seq += 1
            frame.seq = self._seq
            if frame.source == "live":
                self._live_last_rx = time.monotonic()
            self._latest = frame
            self._track.append(
                TrackPoint(
                    t=frame.t,
                    lat=frame.lat,
                    lon=frame.lon,
                    alt_msl=frame.alt_msl,
                    heading=frame.heading,
                    seq=frame.seq,
                )
            )
        await self._broadcast({"type": "telemetry", "frame": frame.to_dict()})
        for cb in self._on_frame:
            result = cb(frame)
            if asyncio.iscoroutine(result):
                await result
        return True

    async def set_active(self, source: SourceKind, *, clear_track: bool = True) -> dict[str, Any]:
        async with self._lock:
            previous = self.active
            if previous == source:
                return self.source_snapshot_unlocked()
            self.active = source
            self._latest = None
            if clear_track:
                self._track.clear()
            self._seq = 0
        for cb in self._on_source_change:
            result = cb(previous, source)
            if asyncio.iscoroutine(result):
                await result
        snap = self.source_snapshot()
        await self._broadcast({"type": "source_changed", **snap})
        return snap

    def source_snapshot_unlocked(self) -> dict[str, Any]:
        live_age = None if self._live_last_rx <= 0 else time.monotonic() - self._live_last_rx
        return {
            "active": self.active,
            "sources": {
                "sim": {"available": self.config.sim_enabled, "running": self.active == "sim"},
                "live": {
                    "available": True,
                    "running": self.active == "live",
                    "last_rx_age_s": live_age,
                    "stale_after_s": self.config.live_stale_seconds,
                },
            },
        }

    def source_snapshot(self) -> dict[str, Any]:
        return self.source_snapshot_unlocked()

    def latest(self) -> TelemetryFrame | None:
        return self._latest

    def track(self, *, since: int = -1, limit: int = 500) -> list[dict[str, Any]]:
        points = [p for p in self._track if p.seq > since]
        if limit > 0:
            points = points[-limit:]
        return [p.to_dict() for p in points]

    def health(self) -> dict[str, Any]:
        latest = self._latest
        return {
            "status": "ok",
            "active": self.active,
            "latest_seq": latest.seq if latest else -1,
            "track_len": len(self._track),
            "subscribers": len(self._subscribers),
            "source": self.source_snapshot(),
        }
