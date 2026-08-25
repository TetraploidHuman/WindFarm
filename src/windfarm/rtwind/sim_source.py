from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone

from .hub import TelemetryHub
from .types import RtwindConfig, TelemetryFrame


class SimDroneSource:
    """Synthetic WGS84 orbit around a configurable origin (exclusive when active)."""

    def __init__(self, hub: TelemetryHub, config: RtwindConfig):
        self.hub = hub
        self.config = config
        self._task: asyncio.Task | None = None
        self._running = False
        self._phase = 0.0
        self.origin_lat = config.sim_origin_lat
        self.origin_lon = config.sim_origin_lon
        self.radius_m = 450.0
        self.cruise_alt_msl = 120.0
        self.airspeed = 12.0

    @property
    def running(self) -> bool:
        return self._running

    def reset(self, *, lat: float | None = None, lon: float | None = None) -> None:
        if lat is not None:
            self.origin_lat = lat
        if lon is not None:
            self.origin_lon = lon
        self._phase = 0.0

    def snapshot(self) -> dict[str, float | bool]:
        return {
            "running": self._running,
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "radius_m": self.radius_m,
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="rtwind-sim")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        dt = 1.0 / max(self.config.sim_hz, 0.5)
        while self._running:
            if self.hub.active == "sim":
                frame = self._tick(dt)
                await self.hub.publish(frame)
            await asyncio.sleep(dt)

    def _tick(self, dt: float) -> TelemetryFrame:
        # ~90 s per orbit
        omega = (2.0 * math.pi) / 90.0
        self._phase = (self._phase + omega * dt) % (2.0 * math.pi)
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(math.radians(self.origin_lat))
        east = self.radius_m * math.cos(self._phase)
        north = self.radius_m * math.sin(self._phase)
        lat = self.origin_lat + north / meters_per_deg_lat
        lon = self.origin_lon + east / meters_per_deg_lon
        # Tangent velocity (east, north) → aviation heading (0°=north, clockwise)
        heading = (math.degrees(math.atan2(-math.sin(self._phase), math.cos(self._phase))) + 360.0) % 360.0
        roll = 8.0 * math.sin(self._phase * 2.0) + random.uniform(-0.4, 0.4)
        pitch = 2.0 * math.sin(self._phase * 3.0) + random.uniform(-0.3, 0.3)
        yaw = heading + random.uniform(-1.0, 1.0)
        alt = self.cruise_alt_msl + 8.0 * math.sin(self._phase) + random.uniform(-0.5, 0.5)
        climb = 8.0 * omega * math.cos(self._phase)
        gs = self.airspeed + random.uniform(-0.3, 0.3)
        return TelemetryFrame(
            source="sim",
            vehicle_id="sim-1",
            t=datetime.now(timezone.utc).isoformat(),
            lat=lat,
            lon=lon,
            alt_msl=alt,
            heading=heading,
            roll=roll,
            pitch=pitch,
            yaw=yaw % 360.0,
            airspeed=self.airspeed + random.uniform(-0.2, 0.2),
            groundspeed=gs,
            climb_rate=climb,
            battery=max(0.15, 0.92 - 0.01 * (self._phase / (2.0 * math.pi))),
            link="ok",
        )
