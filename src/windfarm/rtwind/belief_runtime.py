"""Live belief field for rtwind: GPS observations → BeliefMap → geo heatmap."""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

from ..belief import BeliefUpdater, belief_snapshot, create_belief_map
from ..types import Observation
from .geo_grid import GeoGrid
from .types import TelemetryFrame


class BeliefRuntime:
    """Maintains a geo-anchored belief map updated from live/sim telemetry."""

    # Fixed mission patch: 10 km × 10 km at 50 m cells (201×201).
    MISSION_SIZE_KM = 10.0
    MISSION_RESOLUTION_M = 50.0
    MISSION_CELLS = int(MISSION_SIZE_KM * 1000 / MISSION_RESOLUTION_M) + 1

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        resolution_m: float = MISSION_RESOLUTION_M,
        levels: int = 1,
    ):
        self.width = width if width is not None else self.MISSION_CELLS
        self.height = height if height is not None else self.MISSION_CELLS
        self.resolution_m = resolution_m
        self.levels = levels
        self._lock = threading.Lock()
        self._grid: GeoGrid | None = None
        self._belief = create_belief_map(self.width, self.height, levels)
        self._updater = BeliefUpdater()
        self._step = 0
        self._last_frame_seq = -1
        self._obs_count = 0
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def reset(self, lat: float | None = None, lon: float | None = None) -> None:
        with self._lock:
            self._belief = create_belief_map(self.width, self.height, self.levels)
            self._step = 0
            self._obs_count = 0
            self._last_frame_seq = -1
            if lat is not None and lon is not None:
                self._grid = GeoGrid(lat, lon, self.width, self.height, self.resolution_m)
            else:
                self._grid = None

    def ensure_anchored(self, lat: float, lon: float) -> GeoGrid:
        """Anchor once; never recentre/reset during flight (10 km patch is fixed)."""
        if self._grid is None:
            self._grid = GeoGrid(lat, lon, self.width, self.height, self.resolution_m)
        return self._grid

    def observe_frame(self, frame: TelemetryFrame) -> None:
        if not self._enabled:
            return
        with self._lock:
            if frame.seq == self._last_frame_seq:
                return
            grid = self.ensure_anchored(frame.lat, frame.lon)
            cell = grid.latlon_to_cell(frame.lat, frame.lon)
            if cell is None:
                # Outside fixed 10 km mission patch — skip update, keep existing belief.
                return
            ix, iy = cell

            hdg = math.radians(frame.heading)
            gs = float(frame.groundspeed or 0.0)
            # Grid: +x east, +y south
            u_obs = gs * math.sin(hdg)
            v_obs = -gs * math.cos(hdg)
            w_obs = float(frame.climb_rate or 0.0)
            airspeed = float(frame.airspeed or gs)

            obs = Observation(
                timestamp=frame.t or datetime.now(timezone.utc).isoformat(),
                x=ix,
                y=iy,
                u_obs=u_obs,
                v_obs=v_obs,
                z=0,
                w_obs=w_obs,
                airspeed=airspeed,
                ground_speed=gs,
                climb_rate=w_obs,
                acceleration=0.0,
            )

            self._last_frame_seq = frame.seq
            self._step += 1
            self._obs_count += 1
            pu = pv = pw = 0.0
            arrays = self._belief.field_arrays
            if arrays is not None:
                pu = float(arrays["wind_u"][0, iy, ix])
                pv = float(arrays["wind_v"][0, iy, ix])
                pw = float(arrays["wind_w"][0, iy, ix])
            self._updater.update_with_observation(
                self._belief,
                obs,
                pu,
                pv,
                pw,
                self._step,
                observation_radius=3,
                advect=self._step % 5 == 0,
            )

    def field_geo(
        self,
        layer: str = "energy",
        *,
        level: int = 0,
    ) -> dict[str, Any]:
        """Return a DEM-like geo grid for Leaflet ImageOverlay."""
        with self._lock:
            if self._grid is None:
                return {
                    "ok": False,
                    "layer": layer,
                    "bounds": None,
                    "values": [],
                    "min": 0.0,
                    "max": 1.0,
                    "nx": 0,
                    "ny": 0,
                    "resolution_m": self.resolution_m,
                    "obs_count": 0,
                    "hint": "waiting for first telemetry to anchor belief grid",
                }
            snap = belief_snapshot(self._belief, level=level)
            key = {
                "energy": "energy",
                "uncertainty": "uncertainty",
                "confidence": "confidence",
                "entropy": "entropy",
                "safety": "safety",
                "uplift_prob": "uplift_prob",
                "sink_prob": "sink_prob",
                "wind_speed": "wind_speed",
            }.get(layer, "energy")
            values = snap.get(key) or snap["energy"]
            flat = [v for row in values for v in row if v is not None]
            vmin = min(flat) if flat else 0.0
            vmax = max(flat) if flat else 1.0
            if abs(vmax - vmin) < 1e-9:
                vmax = vmin + 1.0
            bounds = self._grid.bounds()
            return {
                "ok": True,
                "layer": key,
                "bounds": bounds,
                "values": values,
                "min": vmin,
                "max": vmax,
                "nx": self.width,
                "ny": self.height,
                "resolution_m": self.resolution_m,
                "center_lat": self._grid.center_lat,
                "center_lon": self._grid.center_lon,
                "obs_count": self._obs_count,
                "step": self._step,
                "t": time.time(),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "anchored": self._grid is not None,
                "center_lat": None if self._grid is None else self._grid.center_lat,
                "center_lon": None if self._grid is None else self._grid.center_lon,
                "width": self.width,
                "height": self.height,
                "resolution_m": self.resolution_m,
                "size_km": self.MISSION_SIZE_KM,
                "obs_count": self._obs_count,
                "step": self._step,
            }
