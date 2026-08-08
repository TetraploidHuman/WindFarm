from __future__ import annotations

import math

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def magnitude(u: float, v: float) -> float:
    return math.hypot(u, v)


def magnitude3(u: float, v: float, w: float) -> float:
    return math.sqrt(u * u + v * v + w * w)


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def meteorological_dir_from_uv(u: float, v: float) -> float:
    return math.atan2(-u, -v)


def uv_from_speed_dir(speed: float, direction_rad: float) -> tuple[float, float]:
    return -speed * math.sin(direction_rad), -speed * math.cos(direction_rad)


def day_fraction(timestamp: str) -> float:
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    return (hour * 60 + minute) / 1440.0


def hour_features(timestamp: str) -> tuple[float, float]:
    phase = 2.0 * math.pi * day_fraction(timestamp)
    return math.sin(phase), math.cos(phase)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bilinear_sample(grid: list[list[float]], x: float, y: float) -> float:
    if isinstance(grid, np.ndarray) or (hasattr(grid, "ndim") and not isinstance(grid, (list, tuple))):
        arr = np.asarray(grid, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return 0.0
        h, w = arr.shape
        x = clamp(float(x), 0.0, float(max(w - 1, 0)))
        y = clamp(float(y), 0.0, float(max(h - 1, 0)))
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        tx = x - x0
        ty = y - y0
        top = lerp(float(arr[y0, x0]), float(arr[y0, x1]), tx)
        bottom = lerp(float(arr[y1, x0]), float(arr[y1, x1]), tx)
        return lerp(top, bottom, ty)
    if grid is None or (isinstance(grid, (list, tuple)) and len(grid) == 0):
        return 0.0
    row0 = grid[0]
    # list-of-ndarray-rows (common when slicing / mixing numpy windows)
    if isinstance(row0, np.ndarray) or (hasattr(row0, "ndim") and not isinstance(row0, (list, tuple))):
        arr = np.asarray(grid, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return 0.0
        return bilinear_sample(arr, x, y)
    if not row0:
        return 0.0
    height = len(grid)
    width = len(grid[0])
    x = clamp(x, 0.0, max(width - 1, 0))
    y = clamp(y, 0.0, max(height - 1, 0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    top = lerp(grid[y0][x0], grid[y0][x1], tx)
    bottom = lerp(grid[y1][x0], grid[y1][x1], tx)
    return lerp(top, bottom, ty)


def _trilinear_sample_batch_numpy(arr: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    z_n, h, w = arr.shape
    x = np.clip(x, 0.0, max(w - 1, 0))
    y = np.clip(y, 0.0, max(h - 1, 0))
    z = np.clip(z, 0.0, max(z_n - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    z1 = np.minimum(z0 + 1, z_n - 1)
    tx = x - x0
    ty = y - y0
    tz = z - z0

    def _plane(zi: np.ndarray) -> np.ndarray:
        v00 = arr[zi, y0, x0]
        v10 = arr[zi, y0, x1]
        v01 = arr[zi, y1, x0]
        v11 = arr[zi, y1, x1]
        top = v00 + (v10 - v00) * tx
        bottom = v01 + (v11 - v01) * tx
        return top + (bottom - top) * ty

    lower = _plane(z0)
    upper = _plane(z1)
    return lower + (upper - lower) * tz


try:
    from numba import njit

    @njit(cache=True)
    def _trilinear_sample_scalar_jit(arr, x, y, z):
        z_n = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        if x < 0.0:
            x = 0.0
        elif x > (w - 1):
            x = float(w - 1)
        if y < 0.0:
            y = 0.0
        elif y > (h - 1):
            y = float(h - 1)
        if z < 0.0:
            z = 0.0
        elif z > (z_n - 1):
            z = float(z_n - 1)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        z0 = int(np.floor(z))
        x1 = x0 + 1 if x0 + 1 < w else w - 1
        y1 = y0 + 1 if y0 + 1 < h else h - 1
        z1 = z0 + 1 if z0 + 1 < z_n else z_n - 1
        tx = x - x0
        ty = y - y0
        tz = z - z0
        v000 = arr[z0, y0, x0]
        v100 = arr[z0, y0, x1]
        v010 = arr[z0, y1, x0]
        v110 = arr[z0, y1, x1]
        v001 = arr[z1, y0, x0]
        v101 = arr[z1, y0, x1]
        v011 = arr[z1, y1, x0]
        v111 = arr[z1, y1, x1]
        top0 = v000 + (v100 - v000) * tx
        bot0 = v010 + (v110 - v010) * tx
        top1 = v001 + (v101 - v001) * tx
        bot1 = v011 + (v111 - v011) * tx
        lower = top0 + (bot0 - top0) * ty
        upper = top1 + (bot1 - top1) * ty
        return lower + (upper - lower) * tz

    @njit(cache=True)
    def _trilinear_sample_batch_jit(arr, xs, ys, zs):
        n = xs.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = _trilinear_sample_scalar_jit(arr, xs[i], ys[i], zs[i])
        return out

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def _trilinear_sample_scalar_jit(arr, x, y, z):
        return float(_trilinear_sample_batch_numpy(arr, np.asarray([x]), np.asarray([y]), np.asarray([z]))[0])

    def _trilinear_sample_batch_jit(arr, xs, ys, zs):
        return _trilinear_sample_batch_numpy(arr, xs, ys, zs)


def trilinear_sample(layers, x: float, y: float, z: float) -> float:
    # Hot path: (Z,H,W) ndarray → scalar Numba kernel (no per-call array alloc).
    if isinstance(layers, np.ndarray) or (hasattr(layers, "ndim") and not isinstance(layers, (list, tuple))):
        arr = np.asarray(layers, dtype=np.float64)
        if arr.ndim == 3 and arr.size > 0:
            return float(_trilinear_sample_scalar_jit(arr, float(x), float(y), float(z)))
        return 0.0
    if layers is None or (isinstance(layers, (list, tuple)) and len(layers) == 0):
        return 0.0
    z = clamp(z, 0.0, max(len(layers) - 1, 0))
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, len(layers) - 1)
    tz = z - z0
    lower = bilinear_sample(layers[z0], x, y)
    upper = bilinear_sample(layers[z1], x, y)
    return lerp(lower, upper, tz)


def trilinear_sample_batch(field, xs, ys, zs):
    """Vectorized trilinear sample on a (Z, H, W) ndarray.

    ``xs``, ``ys``, ``zs`` are 1-D arrays of equal length; returns float64 (N,).
    """
    if isinstance(field, np.ndarray) and field.dtype == np.float64 and field.ndim == 3:
        arr = field
    else:
        arr = np.asarray(field, dtype=np.float64)
    if isinstance(xs, np.ndarray) and xs.dtype == np.float64 and xs.ndim == 1:
        xs_a = xs
    else:
        xs_a = np.asarray(xs, dtype=np.float64).reshape(-1)
    if isinstance(ys, np.ndarray) and ys.dtype == np.float64 and ys.ndim == 1:
        ys_a = ys
    else:
        ys_a = np.asarray(ys, dtype=np.float64).reshape(-1)
    if isinstance(zs, np.ndarray) and zs.dtype == np.float64 and zs.ndim == 1:
        zs_a = zs
    else:
        zs_a = np.asarray(zs, dtype=np.float64).reshape(-1)
    n = int(xs_a.size)
    if arr.ndim != 3 or arr.size == 0 or n == 0:
        return np.zeros(n, dtype=np.float64)
    if _HAS_NUMBA:
        return _trilinear_sample_batch_jit(arr, xs_a, ys_a, zs_a)
    return _trilinear_sample_batch_numpy(arr, xs_a, ys_a, zs_a)
