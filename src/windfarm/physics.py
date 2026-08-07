from __future__ import annotations

from dataclasses import dataclass
import math

from .mathutils import clamp, magnitude, meteorological_dir_from_uv, uv_from_speed_dir, wrap_angle
from .types import TerrainField, WindField


@dataclass(slots=True)
class PhysicsDiagnostics:
    f_elev: list[list[float]]
    f_rough: list[list[float]]
    f_slope: list[list[float]]
    delta_elev: list[list[float]]


def mean_grid(grid: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for row in grid:
        for value in row:
            total += value
            count += 1
    return total / max(count, 1)


def downscale_wind(
    u_km: float,
    v_km: float,
    terrain: TerrainField,
    w_km: float = 0.0,
    altitude_levels: int = 5,
) -> tuple[WindField, PhysicsDiagnostics]:
    rows = len(terrain.elevation)
    cols = len(terrain.elevation[0]) if rows else 0
    mean_elev = mean_grid(terrain.elevation)
    base_speed = magnitude(u_km, v_km)
    base_dir = meteorological_dir_from_uv(u_km, v_km)

    levels = max(1, altitude_levels)
    u_field = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(levels)]
    v_field = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(levels)]
    w_field = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(levels)]
    f_elev = [[0.0 for _ in range(cols)] for _ in range(rows)]
    f_rough = [[0.0 for _ in range(cols)] for _ in range(rows)]
    f_slope = [[0.0 for _ in range(cols)] for _ in range(rows)]
    delta_elev = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for y in range(rows):
        for x in range(cols):
            delta = terrain.elevation[y][x] - mean_elev
            factor_elev = clamp(1.0 + 0.0015 * delta, 0.7, 1.3)
            factor_rough = math.exp(-0.4 * terrain.roughness[y][x])
            theta = wrap_angle(terrain.aspect[y][x] - base_dir)
            factor_slope = clamp(1.0 + 0.5 * terrain.slope[y][x] * math.cos(theta), 0.6, 1.4)
            delta_elev[y][x] = delta
            f_elev[y][x] = factor_elev
            f_rough[y][x] = factor_rough
            f_slope[y][x] = factor_slope
            upslope = terrain.slope[y][x] * math.cos(theta)
            ridge_lift = terrain.slope[y][x] * math.sin(theta) * math.sin(theta)
            for level in range(levels):
                level_ratio = 0.0 if levels == 1 else level / (levels - 1)
                shear_gain = 0.08 + 0.18 * level_ratio
                roughness_decay = math.exp(-1.2 * terrain.roughness[y][x] * (1.0 - 0.6 * level_ratio))
                speed = base_speed * factor_elev * factor_slope * roughness_decay * (1.0 + shear_gain)
                direction = wrap_angle(base_dir + (0.1 + 0.06 * level_ratio) * math.sin(theta) + 0.04 * level_ratio)
                u_value, v_value = uv_from_speed_dir(speed, direction)
                thermal_decay = 1.0 - 0.45 * level_ratio
                wave_gain = 0.7 + 0.9 * level_ratio
                w_value = clamp(
                    w_km * (0.9 - 0.2 * level_ratio)
                    + (2.4 * thermal_decay) * upslope
                    - (0.9 - 0.35 * level_ratio) * terrain.roughness[y][x]
                    + wave_gain * 0.6 * ridge_lift
                    + 0.0012 * delta
                    + 0.25 * level_ratio,
                    -3.5,
                    4.5,
                )
                u_field[level][y][x] = u_value
                v_field[level][y][x] = v_value
                w_field[level][y][x] = w_value

    return WindField(u=u_field, v=v_field, w=w_field), PhysicsDiagnostics(
        f_elev=f_elev,
        f_rough=f_rough,
        f_slope=f_slope,
        delta_elev=delta_elev,
    )
