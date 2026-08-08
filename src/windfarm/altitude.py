"""AGL / terrain-height helpers.

Mission `z` is an **above-ground-level (AGL) band index**. Geometric MSL height is:

    msl_m = elev(x, y) + z * altitude_step_m

Constant-AGL cruise therefore follows terrain in MSL; climbing hills costs energy
even when the AGL band is unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .mathutils import bilinear_sample, clamp


def sample_elevation(elevation: Sequence[Sequence[float]] | None, x: float, y: float) -> float:
    """Bilinear sample of a terrain elevation grid (meters). Missing → 0."""
    if elevation is None:
        return 0.0
    if isinstance(elevation, np.ndarray):
        if elevation.size == 0:
            return 0.0
        return float(bilinear_sample(elevation, x, y))
    if isinstance(elevation, (list, tuple)) and len(elevation) == 0:
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


_ELEV_NP_CACHE: dict[int, tuple[object, object]] = {}


def _as_elev_array(elevation):
    import numpy as np

    if elevation is None:
        return None
    if isinstance(elevation, np.ndarray):
        return elevation if elevation.dtype == np.float64 else elevation.astype(np.float64)
    key = id(elevation)
    hit = _ELEV_NP_CACHE.get(key)
    if hit is not None and hit[0] is elevation:
        return hit[1]
    elev = np.asarray(elevation, dtype=np.float64)
    if len(_ELEV_NP_CACHE) > 16:
        _ELEV_NP_CACHE.clear()
    _ELEV_NP_CACHE[key] = (elevation, elev)
    return elev


try:
    from numba import njit
    import numpy as np

    @njit(cache=True)
    def _bilinear_scalar(elev, x, y):
        h = elev.shape[0]
        w = elev.shape[1]
        if w <= 0 or h <= 0:
            return 0.0
        if x < 0.0:
            x = 0.0
        elif x > (w - 1):
            x = float(w - 1)
        if y < 0.0:
            y = 0.0
        elif y > (h - 1):
            y = float(h - 1)
        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1 if x0 + 1 < w else w - 1
        y1 = y0 + 1 if y0 + 1 < h else h - 1
        tx = x - x0
        ty = y - y0
        top = elev[y0, x0] + (elev[y0, x1] - elev[y0, x0]) * tx
        bottom = elev[y1, x0] + (elev[y1, x1] - elev[y1, x0]) * tx
        return top + (bottom - top) * ty

    @njit(cache=True)
    def _terrain_delta_batch_jit(elev, xs0, ys0, xs1, ys1):
        n = xs0.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = _bilinear_scalar(elev, xs1[i], ys1[i]) - _bilinear_scalar(elev, xs0[i], ys0[i])
        return out

    @njit(cache=True)
    def _terrain_climb_batch_jit(elev, xs0, ys0, xs1, ys1, n_samples):
        n = xs0.shape[0]
        out = np.empty(n, dtype=np.float64)
        steps = n_samples if n_samples > 1 else 1
        for i in range(n):
            prev = _bilinear_scalar(elev, xs0[i], ys0[i])
            rise = 0.0
            dx = xs1[i] - xs0[i]
            dy = ys1[i] - ys0[i]
            for k in range(1, steps + 1):
                t = k / steps
                val = _bilinear_scalar(elev, xs0[i] + dx * t, ys0[i] + dy * t)
                d = val - prev
                if d > 0.0:
                    rise += d
                prev = val
            out[i] = rise
        return out

except Exception:  # pragma: no cover
    def _terrain_delta_batch_jit(elev, xs0, ys0, xs1, ys1):
        import numpy as np

        n = xs0.shape[0]
        out = np.empty(n, dtype=np.float64)
        h, w = elev.shape
        for i in range(n):
            for label, x, y in ((0, xs0[i], ys0[i]), (1, xs1[i], ys1[i])):
                xx = min(max(float(x), 0.0), float(max(w - 1, 0)))
                yy = min(max(float(y), 0.0), float(max(h - 1, 0)))
                x0 = int(xx)
                y0 = int(yy)
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)
                tx = xx - x0
                ty = yy - y0
                val = (elev[y0, x0] + (elev[y0, x1] - elev[y0, x0]) * tx) * (1.0 - ty) + (
                    elev[y1, x0] + (elev[y1, x1] - elev[y1, x0]) * tx
                ) * ty
                if label == 0:
                    a = val
                else:
                    out[i] = val - a
        return out

    def _terrain_climb_batch_jit(elev, xs0, ys0, xs1, ys1, n_samples):
        import numpy as np

        n = xs0.shape[0]
        out = np.empty(n, dtype=np.float64)
        steps = max(1, int(n_samples))
        h, w = elev.shape
        for i in range(n):
            ts = np.linspace(0.0, 1.0, steps + 1)
            xs = np.clip(xs0[i] + ts * (xs1[i] - xs0[i]), 0.0, max(w - 1, 0))
            ys = np.clip(ys0[i] + ts * (ys1[i] - ys0[i]), 0.0, max(h - 1, 0))
            x0i = np.floor(xs).astype(np.int64)
            y0i = np.floor(ys).astype(np.int64)
            x1i = np.minimum(x0i + 1, w - 1)
            y1i = np.minimum(y0i + 1, h - 1)
            tx = xs - x0i
            ty = ys - y0i
            vals = (
                (elev[y0i, x0i] + (elev[y0i, x1i] - elev[y0i, x0i]) * tx) * (1.0 - ty)
                + (elev[y1i, x0i] + (elev[y1i, x1i] - elev[y1i, x0i]) * tx) * ty
            )
            out[i] = float(np.maximum(0.0, np.diff(vals)).sum())
        return out


def terrain_delta_m_batch(elevation, xs0, ys0, xs1, ys1):
    """Vectorized terrain_delta_m for N segments."""
    import numpy as np

    xs0_a = np.asarray(xs0, dtype=np.float64).reshape(-1)
    n = int(xs0_a.size)
    if elevation is None or n == 0:
        return np.zeros(n, dtype=np.float64)
    elev = _as_elev_array(elevation)
    if elev is None or elev.ndim != 2 or elev.size == 0:
        return np.zeros(n, dtype=np.float64)
    return _terrain_delta_batch_jit(
        elev,
        xs0_a,
        np.asarray(ys0, dtype=np.float64).reshape(-1),
        np.asarray(xs1, dtype=np.float64).reshape(-1),
        np.asarray(ys1, dtype=np.float64).reshape(-1),
    )


def terrain_climb_along_line_batch(elevation, xs0, ys0, xs1, ys1, samples: int = 4):
    """Batch positive elevation rise along N straight segments (meters)."""
    import numpy as np

    xs0_a = np.asarray(xs0, dtype=np.float64).reshape(-1)
    n = int(xs0_a.size)
    if elevation is None or n == 0:
        return np.zeros(n, dtype=np.float64)
    elev = _as_elev_array(elevation)
    if elev is None or elev.ndim != 2 or elev.size == 0:
        return np.zeros(n, dtype=np.float64)
    return _terrain_climb_batch_jit(
        elev,
        xs0_a,
        np.asarray(ys0, dtype=np.float64).reshape(-1),
        np.asarray(xs1, dtype=np.float64).reshape(-1),
        np.asarray(ys1, dtype=np.float64).reshape(-1),
        max(1, int(samples)),
    )

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

    Default step 0.02 level ≈ 1 m when altitude_step_m=50. Not a physics limit —
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
    if elevation is None or samples <= 1:
        return max(0.0, terrain_delta_m(elevation, x0, y0, x1, y1))
    if isinstance(elevation, (list, tuple)) and len(elevation) == 0:
        return 0.0
    import numpy as np

    return float(
        terrain_climb_along_line_batch(
            elevation,
            np.asarray([x0], dtype=np.float64),
            np.asarray([y0], dtype=np.float64),
            np.asarray([x1], dtype=np.float64),
            np.asarray([y1], dtype=np.float64),
            samples=samples,
        )[0]
    )

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
