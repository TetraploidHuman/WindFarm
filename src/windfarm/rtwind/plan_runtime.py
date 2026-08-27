"""Lightweight MPC replanning for rtwind dashboard (iteration A)."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from ..planner import plan_path_details
from ..types import BeliefMap, Mission
from .belief_runtime import BeliefRuntime
from .geo import GeoContext
from .geo_grid import GeoGrid, METERS_PER_DEG_LAT
from .types import TelemetryFrame


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _lerp_latlon(
    a: tuple[float, float],
    b: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def path_progress_lambda(lat: float, lon: float, path: list[list[float]]) -> float:
    """Normalized progress λ ∈ [0, 1] along a lat/lon polyline."""
    if len(path) < 2:
        return 0.0
    pts = [(float(p[0]), float(p[1])) for p in path]
    seg_lens: list[float] = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        d = _haversine_m(pts[i], pts[i + 1])
        seg_lens.append(d)
        cum.append(cum[-1] + d)
    total = cum[-1]
    if total < 1.0:
        return 0.0

    best_along = 0.0
    best_dist = float("inf")
    q = (lat, lon)
    for i, seg_len in enumerate(seg_lens):
        if seg_len < 1e-6:
            t = 0.0
            proj = pts[i]
        else:
            # Project query onto segment in lat/lon (good enough at 10 km scale).
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            dx, dy = bx - ax, by - ay
            t = ((q[0] - ax) * dx + (q[1] - ay) * dy) / (dx * dx + dy * dy + 1e-12)
            t = max(0.0, min(1.0, t))
            proj = _lerp_latlon(pts[i], pts[i + 1], t)
        d = _haversine_m(q, proj)
        along = cum[i] + t * seg_len
        if d < best_dist:
            best_dist = d
            best_along = along
    return max(0.0, min(1.0, best_along / total))


def straight_baseline(
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    n: int = 24,
) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        lat, lon = _lerp_latlon(start, goal, t)
        out.append([lat, lon])
    return out


class PlanRuntime:
    """Background replanner: belief-guided path + straight baseline + λ."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_plan_mono = 0.0
        self._last_obs_count = -1
        self._goal_lat: float | None = None
        self._goal_lon: float | None = None
        self._planned: list[list[float]] = []
        self._baseline: list[list[float]] = []
        self._lambda_val = 0.0
        self._planning_ms: float | None = None
        self._planning_mode: str | None = None
        self._last_error: str | None = None
        self._ready = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._ready,
                "goal_lat": self._goal_lat,
                "goal_lon": self._goal_lon,
                "lambda": round(self._lambda_val, 4),
                "planning_ms": self._planning_ms,
                "planning_mode": self._planning_mode,
                "planned_points": len(self._planned),
                "baseline_points": len(self._baseline),
                "last_error": self._last_error,
            }

    def path_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": self._ready,
                "planned": [p[:] for p in self._planned],
                "baseline": [p[:] for p in self._baseline],
                "lambda": round(self._lambda_val, 4),
                "planning_ms": self._planning_ms,
                "planning_mode": self._planning_mode,
                "goal_lat": self._goal_lat,
                "goal_lon": self._goal_lon,
                "last_error": self._last_error,
            }

    def reset(self) -> None:
        with self._lock:
            self._planned = []
            self._baseline = []
            self._lambda_val = 0.0
            self._planning_ms = None
            self._planning_mode = None
            self._last_error = None
            self._ready = False
            self._last_obs_count = -1
            self._goal_lat = None
            self._goal_lon = None

    def maybe_replan(
        self,
        frame: TelemetryFrame,
        *,
        belief: BeliefRuntime,
        geo: GeoContext,
        origin_lat: float | None,
        origin_lon: float | None,
        min_interval_s: float = 5.0,
    ) -> dict[str, Any] | None:
        if frame.lat is None or frame.lon is None:
            return None
        bst = belief.status()
        if not bst.get("anchored"):
            return None
        obs = int(bst.get("obs_count") or 0)
        now = time.monotonic()
        with self._lock:
            if now - self._last_plan_mono < min_interval_s and obs == self._last_obs_count:
                self._lambda_val = path_progress_lambda(frame.lat, frame.lon, self._planned) if self._planned else 0.0
                return None
        result = self._replan_sync(
            frame,
            belief=belief,
            geo=geo,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            obs_count=obs,
        )
        with self._lock:
            self._last_plan_mono = now
            self._last_obs_count = obs
        return result

    def _replan_sync(
        self,
        frame: TelemetryFrame,
        *,
        belief: BeliefRuntime,
        geo: GeoContext,
        origin_lat: float | None,
        origin_lon: float | None,
        obs_count: int,
    ) -> dict[str, Any] | None:
        t0 = time.perf_counter()
        try:
            grid = belief.grid()
            bmap = belief.belief_map()
            if grid is None or bmap is None:
                return None

            goal_lat, goal_lon = self._mission_goal(
                frame.lat,
                frame.lon,
                grid,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
            )
            start_cell = grid.latlon_to_cell(frame.lat, frame.lon)
            goal_cell = grid.latlon_to_cell(goal_lat, goal_lon)
            if start_cell is None or goal_cell is None:
                raise ValueError("当前位置或目标超出任务区")

            elevation = self._elevation_grid(geo, grid)
            mission = Mission(
                start=(start_cell[0], start_cell[1], 1),
                goal=(goal_cell[0], goal_cell[1], 1),
                max_steps=min(120, max(40, int(math.hypot(goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1]) * 1.5))),
                step_distance_m=grid.resolution_m,
                max_altitude_level=3,
                clearance_agl_level=1.0,
                elevation=elevation,
                corridor_energy_margin=1.02,
            )

            planning = plan_path_details(
                bmap,
                mission,
                risk_weight=1.0,
                safety_weight=1.2,
                anytime_rounds=2,
                search_node_budget=2500,
                horizon_steps=5,
                beam_width=18,
                branch_width=8,
                terminal_progress_weight=24.0,
            )
            raw_path = planning.get("path") or []
            planned = [list(grid.xy_to_latlon(float(p[0]), float(p[1]))) for p in raw_path if len(p) >= 2]
            if len(planned) < 2:
                planned = straight_baseline((frame.lat, frame.lon), (goal_lat, goal_lon))

            baseline = straight_baseline((frame.lat, frame.lon), (goal_lat, goal_lon))
            lam = path_progress_lambda(frame.lat, frame.lon, planned)
            ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._goal_lat = goal_lat
                self._goal_lon = goal_lon
                self._planned = planned
                self._baseline = baseline
                self._lambda_val = lam
                self._planning_ms = round(ms, 1)
                self._planning_mode = str(planning.get("planning_mode") or "mpc")
                self._last_error = None
                self._ready = True

            return {
                "type": "plan_update",
                **self.path_payload(),
                "obs_count": obs_count,
            }
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._planning_ms = round(ms, 1)
                self._last_error = str(exc)
                self._ready = False
                self._lambda_val = 0.0
            return {
                "type": "plan_update",
                **self.path_payload(),
                "obs_count": obs_count,
            }

    def _mission_goal(
        self,
        lat: float,
        lon: float,
        grid: GeoGrid,
        *,
        origin_lat: float | None,
        origin_lon: float | None,
    ) -> tuple[float, float]:
        """Demo mission: 2.5 km east of orbit center (sim origin or belief center)."""
        center_lat = origin_lat if origin_lat is not None else grid.center_lat
        center_lon = origin_lon if origin_lon is not None else grid.center_lon
        mlon = METERS_PER_DEG_LAT * math.cos(math.radians(center_lat))
        goal_lon = center_lon + 2500.0 / mlon
        goal_lat = center_lat
        if not grid.contains_latlon(goal_lat, goal_lon, margin_cells=2.0):
            goal_lat, goal_lon = lat, lon + 1500.0 / mlon
        return goal_lat, goal_lon

    def _elevation_grid(self, geo: GeoContext, grid: GeoGrid) -> list[list[float]] | None:
        bounds = grid.bounds()
        try:
            dem = geo.dem_grid(
                bounds["south"],
                bounds["west"],
                bounds["north"],
                bounds["east"],
                nx=min(grid.width, 48),
                ny=min(grid.height, 48),
            )
            values = dem.get("values") or []
            out: list[list[float]] = []
            for row in values:
                out.append([float(v) if v is not None else 0.0 for v in row])
            return out or None
        except Exception:
            return None

    def update_lambda(self, lat: float | None, lon: float | None) -> float:
        if lat is None or lon is None:
            return 0.0
        with self._lock:
            if not self._planned:
                return self._lambda_val
            self._lambda_val = path_progress_lambda(lat, lon, self._planned)
            return self._lambda_val
