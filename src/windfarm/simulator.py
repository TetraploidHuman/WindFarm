from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
import random

import numpy as np

from .io import write_json
from .physics import downscale_wind
from .terrain_tools import prepare_terrain
from .types import TerrainField


@dataclass(slots=True)
class SimulatedDataset:
    terrain: TerrainField
    coarse_wind: list[dict]
    training_samples: list[dict]
    observations: list[dict]
    truth_fields: list[dict]


class EnvironmentSimulator:
    def __init__(self, seed: int = 7):
        self.random = random.Random(seed)

    def build_terrain(self, width: int, height: int, resolution_m: float) -> TerrainField:
        valley_x = width * 0.34
        ridge_x = width * 0.71
        urban_x0 = int(width * 0.58)
        urban_x1 = int(width * 0.82)
        urban_y0 = int(height * 0.32)
        urban_y1 = int(height * 0.74)
        elevation = [
            [
                162
                + 12 * math.sin(x / 5.2)
                + 17 * math.cos(y / 6.4)
                + 1.6 * x
                - 24 * math.exp(-((x - valley_x) ** 2) / 20.0)
                + 30 * math.exp(-((x - ridge_x) ** 2 + (y - height * 0.50) ** 2) / 52.0)
                + 5 * math.exp(-((x - width * 0.20) ** 2 + (y - height * 0.78) ** 2) / 28.0)
                for x in range(width)
            ]
            for y in range(height)
        ]
        landcover = []
        for y in range(height):
            row = []
            for x in range(width):
                if abs(x - valley_x) < width * 0.05 and y > height * 0.55:
                    row.append(0)
                elif urban_x0 <= x <= urban_x1 and urban_y0 <= y <= urban_y1:
                    row.append(4)
                elif y < height * 0.20 or (x > width * 0.84 and y > height * 0.65):
                    row.append(3)
                elif x < width * 0.18:
                    row.append(1)
                else:
                    row.append(2)
            landcover.append(row)
        return prepare_terrain(elevation, landcover, resolution_m)

    def coarse_wind_at(self, minute_index: int) -> tuple[float, float, float]:
        # Morning boundary-layer evolution with a slow synoptic background.
        day_phase = 2.0 * math.pi * ((minute_index % 1440) / 1440.0)
        synoptic = minute_index / 220.0
        u_km = 2.4 + 0.9 * math.sin(day_phase - 0.7) + 0.35 * math.cos(synoptic)
        v_km = -3.6 + 1.2 * math.cos(day_phase - 0.15) - 0.28 * math.sin(synoptic / 1.7)
        w_km = 0.25 * max(0.0, math.sin(day_phase - 0.8)) + 0.12 * math.cos(synoptic / 2.3)
        return u_km, v_km, w_km

    def truth_field(
        self,
        terrain: TerrainField,
        u_km: float,
        v_km: float,
        minute_index: int,
        w_km: float = 0.0,
        altitude_levels: int = 5,
        *,
        altitude_step_m: float = 50.0,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]:
        height = len(terrain.elevation)
        width = len(terrain.elevation[0]) if height else 0
        baseline, _ = downscale_wind(
            u_km,
            v_km,
            terrain,
            w_km,
            altitude_levels,
            altitude_step_m=altitude_step_m,
            u_100=u_100,
            v_100=v_100,
        )
        u_arr = np.asarray(baseline.u, dtype=np.float64)
        v_arr = np.asarray(baseline.v, dtype=np.float64)
        w_arr = np.asarray(baseline.w, dtype=np.float64)
        truth_u = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(altitude_levels)]
        truth_v = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(altitude_levels)]
        truth_w = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(altitude_levels)]
        for level in range(altitude_levels):
            for y in range(height):
                for x in range(width):
                    du, dv, dw = self._truth_delta(terrain, u_km, v_km, minute_index, x, y, level, altitude_levels)
                    truth_u[level][y][x] = float(u_arr[level, y, x]) + du
                    truth_v[level][y][x] = float(v_arr[level, y, x]) + dv
                    truth_w[level][y][x] = float(w_arr[level, y, x]) + dw
        return truth_u, truth_v, truth_w

    def truth_at_cell(
        self,
        terrain: TerrainField,
        u_km: float,
        v_km: float,
        minute_index: int,
        x: int,
        y: int,
        z: int,
        baseline_u: float,
        baseline_v: float,
        baseline_w: float,
        altitude_levels: int = 5,
    ) -> tuple[float, float, float]:
        du, dv, dw = self._truth_delta(terrain, u_km, v_km, minute_index, x, y, z, altitude_levels)
        return baseline_u + du, baseline_v + dv, baseline_w + dw

    def _truth_delta(
        self,
        terrain: TerrainField,
        u_km: float,
        v_km: float,
        minute_index: int,
        x: int,
        y: int,
        level: int,
        altitude_levels: int,
    ) -> tuple[float, float, float]:
        height = len(terrain.elevation)
        width = len(terrain.elevation[0]) if height else 0
        phase = 2.0 * math.pi * ((minute_index % 1440) / 1440.0)
        phase_sin = math.sin(phase)
        phase_cos = math.cos(phase)
        level_ratio = 0.0 if altitude_levels == 1 else level / (altitude_levels - 1)
        terrain_wave = (x + y) / 7.8
        lee_wave = (x - 0.65 * y) / 6.3
        valley_channel = math.exp(-((x - width * 0.34) ** 2) / 20.0) * (0.35 + 0.18 * math.sin(y / 4.5))
        ridge_lift = math.exp(-((x - width * 0.71) ** 2 + (y - height * 0.50) ** 2) / 58.0)
        aspect_sin = math.sin(terrain.aspect[y][x])
        aspect_cos = math.cos(terrain.aspect[y][x])
        urban_drag = 0.50 * terrain.roughness[y][x]
        thermal = max(0.0, math.sin(phase - 0.8)) * 0.25
        du = (
            (0.70 + 0.18 * level_ratio) * terrain.slope[y][x] * aspect_sin
            + 0.32 * terrain.slope[y][x] * aspect_cos
            - (0.34 - 0.12 * level_ratio) * urban_drag
            + 0.24 * math.sin(terrain_wave + 0.3 * level_ratio)
            - 0.15 * math.cos(lee_wave)
            + 0.20 * phase_sin
            - 0.07 * phase_cos
            + 0.07 * u_km * terrain.slope[y][x]
            + 0.45 * valley_channel
            + (0.24 + 0.14 * level_ratio) * ridge_lift
            + thermal
        )
        dv = (
            -0.36 * terrain.slope[y][x] * aspect_sin
            + (0.58 + 0.08 * level_ratio) * terrain.slope[y][x] * aspect_cos
            + 0.16 * urban_drag
            + 0.20 * math.cos(terrain_wave)
            + 0.18 * math.sin(lee_wave + 0.25 * level_ratio)
            - 0.16 * phase_sin
            + 0.10 * phase_cos
            - 0.05 * v_km * terrain.slope[y][x]
            + (0.30 + 0.12 * level_ratio) * ridge_lift
            - 0.16 * valley_channel
        )
        dw = (
            (0.85 + 0.35 * level_ratio) * ridge_lift
            + (0.55 - 0.18 * level_ratio) * terrain.slope[y][x] * aspect_cos
            - (0.28 - 0.12 * level_ratio) * urban_drag
            + 0.18 * math.sin(terrain_wave + phase + 0.4 * level_ratio)
            + 0.12 * phase_sin
            + (0.35 - 0.08 * level_ratio) * thermal
            + 0.22 * level_ratio
        )
        return du, dv, dw

    def generate_dataset(
        self,
        width: int,
        height: int,
        resolution_m: float,
        time_steps: int,
        start_time: str,
        sample_interval_seconds: int,
        terrain: TerrainField | None = None,
        coarse_wind: list[dict] | None = None,
    ) -> SimulatedDataset:
        terrain = terrain if terrain is not None else self.build_terrain(width, height, resolution_m)
        if terrain.elevation and (len(terrain.elevation) != height or len(terrain.elevation[0]) != width):
            height = len(terrain.elevation)
            width = len(terrain.elevation[0]) if height else width
        coarse_series = list(coarse_wind) if coarse_wind is not None else None
        coarse_wind_out: list[dict] = []
        training_samples: list[dict] = []
        observations: list[dict] = []
        truth_fields: list[dict] = []
        current_time = datetime.fromisoformat(start_time)
        minute_index = current_time.hour * 60 + current_time.minute

        for step in range(time_steps):
            timestamp = current_time.isoformat(timespec="seconds")
            if coarse_series is not None:
                sample = coarse_series[min(step, len(coarse_series) - 1)]
                u_km = float(sample["u_km"])
                v_km = float(sample["v_km"])
                w_km = float(sample.get("w_km", 0.0))
                u_100 = float(sample["u100_km"]) if sample.get("u100_km") is not None else None
                v_100 = float(sample["v100_km"]) if sample.get("v100_km") is not None else None
                timestamp = str(sample.get("timestamp", timestamp))
            else:
                u_km, v_km, w_km = self.coarse_wind_at(minute_index)
                u_100 = v_100 = None
            out_sample = {"timestamp": timestamp, "u_km": u_km, "v_km": v_km, "w_km": w_km}
            if u_100 is not None and v_100 is not None:
                out_sample["u100_km"] = u_100
                out_sample["v100_km"] = v_100
            coarse_wind_out.append(out_sample)
            truth_u, truth_v, truth_w = self.truth_field(
                terrain,
                u_km,
                v_km,
                minute_index,
                w_km,
                altitude_step_m=resolution_m,
                u_100=u_100,
                v_100=v_100,
            )
            altitude_levels = len(truth_u)
            for z in range(altitude_levels):
                for y in range(height):
                    for x in range(width):
                        training_samples.append(
                            {
                                "timestamp": timestamp,
                                "x": x,
                                "y": y,
                                "z": z,
                                "u_km": u_km,
                                "v_km": v_km,
                                "w_km": w_km,
                                "u_obs": truth_u[z][y][x] + self.random.uniform(-0.08, 0.08),
                                "v_obs": truth_v[z][y][x] + self.random.uniform(-0.08, 0.08),
                                "w_obs": truth_w[z][y][x] + self.random.uniform(-0.05, 0.05),
                            }
                        )
            sx = min(width - 4, 5 + minute_index // 90)
            sy = min(height - 4, 7 + minute_index // 120)
            sz = (minute_index // 45) % max(altitude_levels, 1)
            observations.append(
                {
                    "timestamp": timestamp,
                    "x": sx,
                    "y": sy,
                    "z": sz,
                    "u_obs": truth_u[sz][sy][sx] + self.random.uniform(-0.05, 0.05),
                    "v_obs": truth_v[sz][sy][sx] + self.random.uniform(-0.05, 0.05),
                    "w_obs": truth_w[sz][sy][sx] + self.random.uniform(-0.04, 0.04),
                    "airspeed": 13.6 + self.random.uniform(-0.4, 0.4),
                    "ground_speed": 14.1 + self.random.uniform(-0.8, 1.0),
                    "climb_rate": self.random.uniform(-0.35, 0.9),
                    "acceleration": self.random.uniform(-0.2, 0.2),
                }
            )
            truth_fields.append({"timestamp": timestamp, "u": truth_u, "v": truth_v, "w": truth_w})
            current_time += timedelta(seconds=sample_interval_seconds)
            minute_index += max(1, sample_interval_seconds // 60)

        return SimulatedDataset(
            terrain=terrain,
            coarse_wind=coarse_wind_out,
            training_samples=training_samples,
            observations=observations,
            truth_fields=truth_fields,
        )

    def write_dataset(self, dataset: SimulatedDataset, output_dir: str) -> None:
        write_json(f"{output_dir}/terrain.json", {"terrain": asdict(dataset.terrain)})
        write_json(f"{output_dir}/training.json", {"samples": dataset.training_samples})
        write_json(f"{output_dir}/coarse_wind.json", {"coarse_wind": dataset.coarse_wind})
        write_json(f"{output_dir}/observations.json", {"observations": dataset.observations})
        write_json(f"{output_dir}/truth.json", {"truth_fields": dataset.truth_fields})
