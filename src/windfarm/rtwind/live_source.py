from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from .hub import TelemetryHub
from .mavlink_source import MavlinkState, MavlinkUdpListener
from .types import RtwindConfig, TelemetryFrame, SourceKind


def _parse_ingest(payload: dict[str, Any], *, source: SourceKind = "live") -> TelemetryFrame:
    now = datetime.now(timezone.utc).isoformat()
    lat_raw = payload.get("lat")
    lon_raw = payload.get("lon")
    lat = float(lat_raw) if lat_raw is not None else None
    lon = float(lon_raw) if lon_raw is not None else None
    alt_raw = payload.get("alt_msl", payload.get("alt"))
    alt_msl = float(alt_raw) if alt_raw is not None else None
    return TelemetryFrame(
        source=source,
        vehicle_id=str(payload.get("vehicle_id") or "live-1"),
        t=str(payload.get("t") or now),
        lat=lat,
        lon=lon,
        alt_msl=alt_msl,
        heading=float(payload.get("heading", payload.get("yaw", 0.0))),
        roll=float(payload.get("roll", 0.0)),
        pitch=float(payload.get("pitch", 0.0)),
        yaw=float(payload.get("yaw", payload.get("heading", 0.0))),
        airspeed=float(payload.get("airspeed", 0.0)),
        groundspeed=float(payload.get("groundspeed", payload.get("airspeed", 0.0))),
        climb_rate=float(payload.get("climb_rate", 0.0)),
        battery=float(payload["battery"]) if payload.get("battery") is not None else None,
        link="ok",
        alt_agl=float(payload["alt_agl"]) if payload.get("alt_agl") is not None else None,
        client_ts=float(payload["client_ts"]) if payload.get("client_ts") is not None else None,
    )


class LiveIngestSource:
    """Live telemetry: HTTP ingest + MAVLink UDP listener while Live is active."""

    def __init__(self, hub: TelemetryHub, config: RtwindConfig):
        self.hub = hub
        self.config = config
        self._watchdog_task: asyncio.Task | None = None
        self._mavlink_task: asyncio.Task | None = None
        self._running = False
        self._last_frame: TelemetryFrame | None = None
        self._mavlink_state = MavlinkState()
        self._mavlink = MavlinkUdpListener(config, self._mavlink_state)
        self._last_mavlink_pub = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        last_rx = self._mavlink_state.last_rx()
        age = None if last_rx <= 0 else time.monotonic() - last_rx
        return {
            "running": self._running,
            "ingest_http": True,
            "mavlink": {
                "enabled": self.config.mavlink_enabled,
                "listening": self._mavlink.listening,
                "bind": f"{self.config.mavlink_bind}:{self.config.mavlink_udp_port}",
                "last_rx_age_s": age,
                "error": self._mavlink.error,
            },
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.config.mavlink_enabled:
            self._mavlink.start()
            self._mavlink_task = asyncio.create_task(self._mavlink_publish_loop(), name="rtwind-mavlink-pub")
        self._watchdog_task = asyncio.create_task(self._watchdog(), name="rtwind-live-wd")

    async def stop(self) -> None:
        self._running = False
        self._mavlink.stop()
        for task in (self._watchdog_task, self._mavlink_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._watchdog_task = None
        self._mavlink_task = None

    async def ingest(self, payload: dict[str, Any]) -> TelemetryFrame | None:
        frame = _parse_ingest(payload, source="live")
        self._last_frame = frame
        if self.hub.active != "live":
            return None
        # Never block the ingest socket/HTTP handler on belief/plan work.
        asyncio.create_task(self.hub.publish(frame), name="rtwind-publish")
        return frame

    async def _mavlink_publish_loop(self) -> None:
        interval = 1.0 / max(self.config.mavlink_publish_hz, 1.0)
        while self._running:
            await asyncio.sleep(interval)
            if self.hub.active != "live":
                continue
            last_rx = self._mavlink_state.last_rx()
            if last_rx <= 0:
                continue
            if (time.monotonic() - last_rx) > self.config.live_stale_seconds:
                continue
            frame = self._mavlink_state.to_frame()
            if frame is None:
                continue
            self._last_frame = frame
            await self.hub.publish(frame)

    async def _watchdog(self) -> None:
        while self._running:
            await asyncio.sleep(2.0)
            if self.hub.active != "live":
                continue
            latest = self.hub.latest()
            has_signal = self._last_frame is not None or self._mavlink_state.last_rx() > 0
            if not has_signal and (latest is None or latest.link == "waiting"):
                waiting = TelemetryFrame(
                    source="live",
                    vehicle_id="live-1",
                    t=datetime.now(timezone.utc).isoformat(),
                    lat=self.config.sim_origin_lat,
                    lon=self.config.sim_origin_lon,
                    alt_msl=0.0,
                    heading=0.0,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                    airspeed=0.0,
                    groundspeed=0.0,
                    climb_rate=0.0,
                    link="waiting",
                )
                await self.hub.publish(waiting)
                continue
            if latest is None:
                continue
            age = getattr(self.hub, "_live_last_rx", 0.0)
            if age > 0 and (time.monotonic() - age) > self.config.live_stale_seconds and latest.link not in {"stale", "waiting"}:
                stale = TelemetryFrame(
                    source=latest.source,
                    vehicle_id=latest.vehicle_id,
                    t=datetime.now(timezone.utc).isoformat(),
                    lat=latest.lat,
                    lon=latest.lon,
                    alt_msl=latest.alt_msl,
                    heading=latest.heading,
                    roll=latest.roll,
                    pitch=latest.pitch,
                    yaw=latest.yaw,
                    airspeed=latest.airspeed,
                    groundspeed=latest.groundspeed,
                    climb_rate=latest.climb_rate,
                    battery=latest.battery,
                    link="stale",
                    alt_agl=latest.alt_agl,
                )
                await self.hub.publish(stale)
