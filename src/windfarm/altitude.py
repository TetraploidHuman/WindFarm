"""AGL / terrain-height helpers.

Mission `z` is an **above-ground-level (AGL) band index**. Geometric MSL height is:

    msl_m = elev(x, y) + z * altitude_step_m

Constant-AGL cruise therefore follows terrain in MSL; climbing hills costs energy
even when the AGL band is unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

from .mathutils import clamp


def sample_elevation(elevation: Sequence[Sequence[float]] | None, x: float, y: float) -> float:
    """Bilinear sample of a terrain elevation grid (meters). Missing → 0."""
    if not elevation:
        return 0.0
    rows = len(elevation)
    cols = len(elevation[0]) if rows else 0
    if rows <= 0 or cols <= 0:
        return 0.0
    x = clamp(float(x), 0.0, float(cols - 1))
    y = clamp(float(y), 0.0, float(rows - 1))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, cols - 1)
    y1 = min(y0 + 1, rows - 1)
    tx = x - x0
    ty = y - y0
    e00 = float(elevation[y0][x0])
    e10 = float(elevation[y0][x1])
    e01 = float(elevation[y1][x0])
    e11 = float(elevation[y1][x1])
    return (e00 * (1.0 - tx) + e10 * tx) * (1.0 - ty) + (e01 * (1.0 - tx) + e11 * tx) * ty


def terrain_delta_m(
    elevation: Sequence[Sequence[float]] | None,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    return sample_elevation(elevation, x1, y1) - sample_elevation(elevation, x0, y0)


def msl_height_m(elevation_m: float, agl_level: float, altitude_step_m: float) -> float:
    return float(elevation_m) + float(agl_level) * float(altitude_step_m)


def effective_dz_levels(agl_dz: float, terrain_dz_m: float, altitude_step_m: float) -> float:
    """Vertical work in altitude-level units: AGL change + terrain follow."""
    step = max(float(altitude_step_m), 1e-6)
    return float(agl_dz) + float(terrain_dz_m) / step


def cruise_agl_floor(
    horizontal_cells: float,
    min_altitude_level: float,
    clearance_agl_level: float,
    approach_cells: float = 5.0,
) -> float:
    """Hard AGL floor: clearance while cruising; allow landing near the goal."""
    base = float(min_altitude_level)
    if horizontal_cells <= approach_cells:
        return base
    return max(base, float(clearance_agl_level))


def format_agl_band_token(z: float) -> str:
    """Stable decimal token for guide/baseline labels (e.g. 1.5 → '1.5')."""
    text = f"{float(z):.3f}".rstrip("0").rstrip(".")
    return text or "0"


def snap_cruise_agl(
    z: float,
    *,
    clearance: float,
    min_level: float,
    max_level: float,
    step: float = 0.025,
) -> float:
    """Snap AGL to the cruise grid relative to clearance (physics stays continuous)."""
    step = max(float(step), 0.01)
    lo = max(float(clearance), float(min_level))
    hi = float(max_level)
    z = clamp(float(z), lo, hi)
    k = round((z - lo) / step)
    return clamp(round(lo + k * step, 10), lo, hi)


def cruise_agl_band_grid(
    clearance: float,
    min_level: float,
    max_level: float,
    *,
    step: float = 0.025,
    span_levels: float = 2.0,
    sticky: float | None = None,
) -> list[float]:
    """Fine AGL cruise candidates from clearance up to clearance+span (capped by max).

    Default step 0.025 level ≈ 1 m when altitude_step_m=40. Not a physics limit —
    only a planning discretization; wind is still sampled continuously via interpolation.
    """
    step = max(float(step), 0.01)
    lo = max(float(clearance), float(min_level))
    hi = min(float(clearance) + float(span_levels), float(max_level))
    if hi < lo - 1e-12:
        return [clamp(float(clearance), float(min_level), float(max_level))]

    n = int(round((hi - lo) / step))
    bands = [round(lo + i * step, 10) for i in range(n + 1)]
    if sticky is not None:
        bands.append(
            snap_cruise_agl(
                float(sticky),
                clearance=clearance,
                min_level=min_level,
                max_level=max_level,
                step=step,
            )
        )

    uniq: dict[int, float] = {}
    for z in bands:
        z = clamp(float(z), float(min_level), float(max_level))
        if z + 1e-9 < lo:
            continue
        key = int(round((z - lo) / step))
        uniq[key] = round(lo + key * step, 10)
    return [clamp(uniq[k], float(min_level), float(max_level)) for k in sorted(uniq)]


def terrain_climb_along_line_m(
    elevation: Sequence[Sequence[float]] | None,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    samples: int = 8,
) -> float:
    """Sum of positive elevation rises along a straight segment (meters)."""
    if not elevation or samples <= 1:
        return max(0.0, terrain_delta_m(elevation, x0, y0, x1, y1))
    total = 0.0
    prev = sample_elevation(elevation, x0, y0)
    for i in range(1, samples + 1):
        t = i / samples
        elev = sample_elevation(elevation, x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        total += max(0.0, elev - prev)
        prev = elev
    return total


def straight_agl_guide_polyline(
    start: tuple[float, float, float] | tuple[float, float],
    goal: tuple[float, float, float] | tuple[float, float],
    agl_cruise: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Constant-AGL straight guide matching eval `straight_agl_baseline` packing.

    Profile: climb in the first ~12% of XY, hold cruise, descend in the last ~12%.
    """
    sx, sy = float(start[0]), float(start[1])
    sz = float(start[2]) if len(start) > 2 else 0.0
    gx, gy = float(goal[0]), float(goal[1])
    gz = float(goal[2]) if len(goal) > 2 else 0.0
    dist = max(math.hypot(gx - sx, gy - sy), 1.0)
    climb_levels = max(0.0, float(agl_cruise) - sz)
    desc_levels = max(0.0, float(agl_cruise) - gz)
    climb_steps = max(1, int(math.ceil(climb_levels))) if climb_levels > 0.05 else 0
    desc_steps = max(1, int(math.ceil(desc_levels))) if desc_levels > 0.05 else 0
    n_cruise = max(1, int(math.ceil(dist)))
    pts: list[tuple[float, float, float]] = [(sx, sy, sz)]
    climb_frac = 0.12 if climb_steps else 0.0
    desc_frac = 0.12 if desc_steps else 0.0
    for i in range(1, climb_steps + 1):
        t = climb_frac * i / climb_steps
        z = sz + (float(agl_cruise) - sz) * i / climb_steps
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, z))
    cruise_start = climb_frac
    cruise_end = 1.0 - desc_frac
    for i in range(1, n_cruise + 1):
        t = cruise_start + (cruise_end - cruise_start) * i / n_cruise
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, float(agl_cruise)))
    for i in range(1, desc_steps + 1):
        t = cruise_end + desc_frac * i / desc_steps
        z = float(agl_cruise) + (gz - float(agl_cruise)) * i / desc_steps
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, z))
    if math.hypot(pts[-1][0] - gx, pts[-1][1] - gy) > 0.01 or abs(pts[-1][2] - gz) > 0.01:
        pts.append((gx, gy, gz))
    return pts


def next_polyline_waypoint(
    path: list[tuple[float, float, float]],
    x: float,
    y: float,
    *,
    ahead_cells: float = 0.05,
) -> tuple[float, float, float] | None:
    """First polyline vertex strictly ahead of (x, y) along the path chord."""
    if len(path) <= 1:
        return None
    sx, sy = path[0][0], path[0][1]
    gx, gy = path[-1][0], path[-1][1]
    total = math.hypot(gx - sx, gy - sy)
    if total < 1e-9:
        return path[-1]
    cur_along = ((float(x) - sx) * (gx - sx) + (float(y) - sy) * (gy - sy)) / total
    cur_along = clamp(cur_along, 0.0, total)
    for p in path[1:]:
        along = ((p[0] - sx) * (gx - sx) + (p[1] - sy) * (gy - sy)) / total
        if along > cur_along + ahead_cells:
            return (float(p[0]), float(p[1]), float(p[2]))
    return (float(path[-1][0]), float(path[-1][1]), float(path[-1][2]))
