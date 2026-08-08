from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mathutils import magnitude, meteorological_dir_from_uv
from .types import TerrainField, WindField


@dataclass(slots=True)
class PhysicsDiagnostics:
    # Prefer (H, W) ndarrays; list[list[float]] still accepted at boundaries.
    f_elev: object
    f_rough: object
    f_slope: object
    delta_elev: object


def mean_grid(grid: list[list[float]]) -> float:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def downscale_wind(
    u_km: float,
    v_km: float,
    terrain: TerrainField,
    w_km: float = 0.0,
    altitude_levels: int = 5,
    *,
    altitude_step_m: float = 50.0,
    u_100: float | None = None,
    v_100: float | None = None,
) -> tuple[WindField, PhysicsDiagnostics]:
    elev = np.asarray(terrain.elevation, dtype=np.float64)
    if elev.ndim != 2 or elev.size == 0:
        empty = WindField(u=[], v=[], w=[])
        blank: list[list[float]] = []
        return empty, PhysicsDiagnostics(f_elev=blank, f_rough=blank, f_slope=blank, delta_elev=blank)

    slope = np.asarray(terrain.slope, dtype=np.float64)
    aspect = np.asarray(terrain.aspect, dtype=np.float64)
    roughness = np.asarray(terrain.roughness, dtype=np.float64)
    levels = max(1, int(altitude_levels))
    alt_step = max(float(altitude_step_m), 1e-6)

    mean_elev = float(elev.mean())
    speed_10 = magnitude(u_km, v_km)
    dir_10 = meteorological_dir_from_uv(u_km, v_km)
    has_100 = u_100 is not None and v_100 is not None
    if has_100:
        speed_100 = magnitude(float(u_100), float(v_100))
        dir_100 = meteorological_dir_from_uv(float(u_100), float(v_100))
    else:
        speed_100 = speed_10
        dir_100 = dir_10

    delta = elev - mean_elev
    factor_elev = np.clip(1.0 + 0.0015 * delta, 0.7, 1.3)
    factor_rough = np.exp(-0.4 * roughness)
    theta = np.arctan2(np.sin(aspect - dir_10), np.cos(aspect - dir_10))
    factor_slope = np.clip(1.0 + 0.5 * slope * np.cos(theta), 0.6, 1.4)
    upslope = slope * np.cos(theta)
    ridge_lift = slope * np.sin(theta) * np.sin(theta)

    level_idx = np.arange(levels, dtype=np.float64)[:, None, None]
    z_agl = level_idx * alt_step
    log10 = np.log(10.0)
    log100 = np.log(100.0)
    t_shear = np.clip(
        (np.log(np.maximum(z_agl, 10.0)) - log10) / max(log100 - log10, 1e-6),
        0.0,
        1.0,
    )
    if levels == 1:
        t_shear = np.zeros((1, 1, 1), dtype=np.float64)

    base_speed = speed_10 * (1.0 - t_shear) + speed_100 * t_shear
    ddir = (dir_100 - dir_10 + np.pi) % (2.0 * np.pi) - np.pi
    base_dir = dir_10 + t_shear * ddir

    level_ratio = np.linspace(0.0, 1.0, levels, dtype=np.float64)[:, None, None] if levels > 1 else np.zeros((1, 1, 1))
    # Measured 100 m profile replaces most of the old heuristic shear ramp.
    shear_gain = (0.02 + 0.04 * level_ratio) if has_100 else (0.08 + 0.18 * level_ratio)
    roughness_decay = np.exp(-1.2 * roughness[None, :, :] * (1.0 - 0.6 * level_ratio))
    speed = base_speed * factor_elev[None, :, :] * factor_slope[None, :, :] * roughness_decay * (1.0 + shear_gain)
    direction = np.arctan2(
        np.sin(base_dir + (0.1 + 0.06 * level_ratio) * np.sin(theta)[None, :, :] + 0.04 * level_ratio),
        np.cos(base_dir + (0.1 + 0.06 * level_ratio) * np.sin(theta)[None, :, :] + 0.04 * level_ratio),
    )
    u_field = -speed * np.sin(direction)
    v_field = -speed * np.cos(direction)

    thermal_decay = 1.0 - 0.45 * level_ratio
    wave_gain = 0.7 + 0.9 * level_ratio
    w_field = np.clip(
        w_km * (0.9 - 0.2 * level_ratio)
        + (2.4 * thermal_decay) * upslope[None, :, :]
        - (0.9 - 0.35 * level_ratio) * roughness[None, :, :]
        + wave_gain * 0.6 * ridge_lift[None, :, :]
        + 0.0012 * delta[None, :, :]
        + 0.25 * level_ratio,
        -3.5,
        4.5,
    )

    return WindField(u=u_field, v=v_field, w=w_field), PhysicsDiagnostics(
        f_elev=factor_elev,
        f_rough=factor_rough,
        f_slope=factor_slope,
        delta_elev=delta,
    )
