from __future__ import annotations

import math


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
    if not grid or not grid[0]:
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


def trilinear_sample(layers: list[list[list[float]]], x: float, y: float, z: float) -> float:
    if not layers:
        return 0.0
    z = clamp(z, 0.0, max(len(layers) - 1, 0))
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, len(layers) - 1)
    tz = z - z0
    lower = bilinear_sample(layers[z0], x, y)
    upper = bilinear_sample(layers[z1], x, y)
    return lerp(lower, upper, tz)
