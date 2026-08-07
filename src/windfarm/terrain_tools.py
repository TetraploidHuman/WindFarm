from __future__ import annotations

import math

from .types import TerrainField


ROUGHNESS_BY_CLASS = {
    0: 0.05,
    1: 0.10,
    2: 0.25,
    3: 0.50,
    4: 0.80,
    5: 1.20,
}


def derive_slope_aspect(elevation: list[list[float]], resolution_m: float) -> tuple[list[list[float]], list[list[float]]]:
    height = len(elevation)
    width = len(elevation[0]) if height else 0
    slope = [[0.0 for _ in range(width)] for _ in range(height)]
    aspect = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            left = elevation[y][max(0, x - 1)]
            right = elevation[y][min(width - 1, x + 1)]
            up = elevation[max(0, y - 1)][x]
            down = elevation[min(height - 1, y + 1)][x]
            dzdx = (right - left) / max(2.0 * resolution_m, 1e-6)
            dzdy = (down - up) / max(2.0 * resolution_m, 1e-6)
            slope[y][x] = math.atan(math.hypot(dzdx, dzdy))
            aspect[y][x] = math.atan2(dzdy, dzdx)
    return slope, aspect


def roughness_from_landcover(landcover: list[list[int]]) -> list[list[float]]:
    return [[ROUGHNESS_BY_CLASS.get(value, 0.3) for value in row] for row in landcover]


def prepare_terrain(elevation: list[list[float]], landcover: list[list[int]], resolution_m: float) -> TerrainField:
    slope, aspect = derive_slope_aspect(elevation, resolution_m)
    roughness = roughness_from_landcover(landcover)
    return TerrainField(elevation=elevation, slope=slope, aspect=aspect, roughness=roughness)

