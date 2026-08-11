"""Build multi-scenario datasets from real DEM + Open-Meteo wind."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import BeliefConfig, MissionConfig, SimulationConfig, TaskConfig, load_task_config, save_task_config
from .data_ingest import (
    DEFAULT_SCENARIOS,
    ScenarioSpec,
    ensure_srtm_tile,
    fetch_open_meteo_hourly,
    mission_endpoints,
    mission_route_octet,
    resample_coarse_wind,
    terrain_from_srtm,
    write_coarse_wind_json,
    write_terrain_json,
)
from .io import read_json, write_json
from .simulator import EnvironmentSimulator

# Legacy 48×36 @ 50 m geographic span — keep extent, densify cells.
DEFAULT_EXTENT_M = (2350.0, 1750.0)
DEFAULT_RESOLUTION_M = 30.0
REF_ALTITUDE_STEP_M = 50.0
REF_CLIMB_COST_J = 225.0
OBS_RADIUS_M = 200.0


def grid_dims_for_extent(extent_x_m: float, extent_y_m: float, resolution_m: float) -> tuple[int, int]:
    res = max(float(resolution_m), 1e-6)
    width = int(round(float(extent_x_m) / res)) + 1
    height = int(round(float(extent_y_m) / res)) + 1
    return max(width, 2), max(height, 2)


def build_scenario_dataset(
    spec: ScenarioSpec,
    output_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    width: int | None = None,
    height: int | None = None,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    extent_m: tuple[float, float] = DEFAULT_EXTENT_M,
    time_steps: int = 120,
    sample_interval_seconds: int = 15,
    seed: int = 7,
    base_config: str | Path | None = None,
) -> dict:
    """Download DEM/wind if needed, synthesize truth/training on real terrain, write artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    resolution_m = float(resolution_m)
    if width is None or height is None:
        auto_w, auto_h = grid_dims_for_extent(extent_m[0], extent_m[1], resolution_m)
        width = int(width if width is not None else auto_w)
        height = int(height if height is not None else auto_h)

    hgt = ensure_srtm_tile(data_dir, spec.tile_lat, spec.tile_lon)
    terrain = terrain_from_srtm(
        hgt,
        center_lat=spec.lat,
        center_lon=spec.lon,
        width=width,
        height=height,
        resolution_m=resolution_m,
        tile_lat=spec.tile_lat,
        tile_lon=spec.tile_lon,
    )
    write_terrain_json(
        output_dir / "terrain.json",
        terrain,
        meta={
            "scenario": spec.name,
            "description": spec.description,
            "lat": spec.lat,
            "lon": spec.lon,
            "hgt": str(hgt),
            "width": width,
            "height": height,
            "resolution_m": resolution_m,
            "source": "SRTM1/skadi native crop",
        },
    )

    # Align simulation start with the first wind hour so interpolation stays in-range.
    hourly = fetch_open_meteo_hourly(spec.lat, spec.lon, spec.wind_start_date, spec.wind_end_date)
    start_time = hourly[0]["timestamp"]
    for item in hourly:
        if item["timestamp"][11:13] == "08":
            start_time = item["timestamp"]
            break
    coarse = resample_coarse_wind(hourly, time_steps, start_time, sample_interval_seconds)
    write_coarse_wind_json(
        output_dir / "coarse_wind.json",
        coarse,
        meta={
            "scenario": spec.name,
            "source": "Open-Meteo archive (ERA5-backed) 10m+100m",
            "lat": spec.lat,
            "lon": spec.lon,
            "wind_start_date": spec.wind_start_date,
            "wind_end_date": spec.wind_end_date,
            "hourly_count": len(hourly),
        },
    )

    simulator = EnvironmentSimulator(seed=seed)
    dataset = simulator.generate_dataset(
        width=width,
        height=height,
        resolution_m=resolution_m,
        time_steps=time_steps,
        start_time=start_time,
        sample_interval_seconds=sample_interval_seconds,
        terrain=terrain,
        coarse_wind=coarse,
    )
    simulator.write_dataset(dataset, str(output_dir))
    write_terrain_json(
        output_dir / "terrain.json",
        terrain,
        meta={
            "scenario": spec.name,
            "description": spec.description,
            "lat": spec.lat,
            "lon": spec.lon,
            "hgt": str(hgt),
            "source": "SRTM1/skadi native crop",
        },
    )
    write_coarse_wind_json(
        output_dir / "coarse_wind.json",
        coarse,
        meta={
            "scenario": spec.name,
            "source": "Open-Meteo archive (ERA5-backed) 10m+100m",
            "lat": spec.lat,
            "lon": spec.lon,
            "wind_start_date": spec.wind_start_date,
            "wind_end_date": spec.wind_end_date,
        },
    )

    start, goal = mission_endpoints(width, height, spec.start_frac, spec.goal_frac)
    if base_config is not None:
        config = load_task_config(base_config)
    else:
        config = TaskConfig()

    climb_cost = REF_CLIMB_COST_J * (resolution_m / REF_ALTITUDE_STEP_M)
    cruise_band_step = max(0.01, 1.0 / resolution_m)
    obs_radius = max(2, int(round(OBS_RADIUS_M / resolution_m)))

    config.simulation = SimulationConfig(
        width=width,
        height=height,
        resolution_m=resolution_m,
        time_steps=time_steps,
        start_time=start_time,
        sample_interval_seconds=sample_interval_seconds,
        report_keyframe_interval=config.simulation.report_keyframe_interval,
    )
    belief_kwargs = {f: getattr(config.belief, f) for f in BeliefConfig.__dataclass_fields__}
    belief_kwargs["observation_radius"] = obs_radius
    config.belief = BeliefConfig(**belief_kwargs)

    # Default octet; large tune catalogs expand via scripts/build_tune_catalog.py.
    routes = mission_route_octet(start, goal)
    config.mission = MissionConfig(
        start=start,
        goal=goal,
        max_steps=max(40, int(1.8 * (abs(goal[0] - start[0]) + abs(goal[1] - start[1])))),
        step_distance_m=resolution_m,
        battery_capacity_j=config.mission.battery_capacity_j,
        nominal_airspeed=config.mission.nominal_airspeed,
        hover_power_w=config.mission.hover_power_w,
        cruise_power_w=config.mission.cruise_power_w,
        hotel_power_w=config.mission.hotel_power_w,
        headwind_power_per_mps_w=config.mission.headwind_power_per_mps_w,
        climb_power_per_mps_w=config.mission.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=config.mission.descent_power_reduction_per_mps_w,
        reserve_energy_ratio=config.mission.reserve_energy_ratio,
        altitude_step_m=resolution_m,
        min_altitude_level=config.mission.min_altitude_level,
        max_altitude_level=config.mission.max_altitude_level,
        climb_cost_per_level_j=climb_cost,
        clearance_agl_level=config.mission.clearance_agl_level,
        cruise_band_step=cruise_band_step,
        corridor_energy_margin=getattr(config.mission, "corridor_energy_margin", 1.02),
    )
    save_task_config(output_dir / "config.json", config)
    # MissionConfig does not carry routes; stitch them into the on-disk JSON.
    cfg_payload = read_json(output_dir / "config.json")
    cfg_payload["mission"]["routes"] = routes
    write_json(output_dir / "config.json", cfg_payload)

    elev = terrain.elevation
    flat = [v for row in elev for v in row]
    summary = {
        "name": spec.name,
        "description": spec.description,
        "lat": spec.lat,
        "lon": spec.lon,
        "hgt": str(hgt),
        "start_time": start_time,
        "time_steps": time_steps,
        "grid": {"width": width, "height": height, "resolution_m": resolution_m},
        "extent_m": {"x": (width - 1) * resolution_m, "y": (height - 1) * resolution_m},
        "elevation_m": {"min": min(flat), "max": max(flat), "mean": sum(flat) / max(len(flat), 1)},
        "wind": {
            "u_km_mean": sum(s["u_km"] for s in coarse) / len(coarse),
            "v_km_mean": sum(s["v_km"] for s in coarse) / len(coarse),
            "speed_mean": sum((s["u_km"] ** 2 + s["v_km"] ** 2) ** 0.5 for s in coarse) / len(coarse),
            "has_100m_profile": all("u100_km" in s for s in coarse),
        },
        "mission": {"start": list(start), "goal": list(goal), "routes": routes},
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": "SRTM1 native crop + Open-Meteo 10m/100m",
    }
    (output_dir / "scenario_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_default_scenarios(
    scenarios_dir: str | Path = "scenarios",
    data_dir: str | Path = "data",
    base_config: str | Path | None = "config_eval.json",
    only: list[str] | None = None,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    width: int | None = None,
    height: int | None = None,
) -> list[dict]:
    scenarios_dir = Path(scenarios_dir)
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for spec in DEFAULT_SCENARIOS:
        if only and spec.name not in only:
            continue
        print(f"[scenario] building {spec.name}: {spec.description} @ {resolution_m:.0f} m")
        summary = build_scenario_dataset(
            spec,
            scenarios_dir / spec.name,
            data_dir=data_dir,
            base_config=base_config if base_config and Path(base_config).exists() else None,
            resolution_m=resolution_m,
            width=width,
            height=height,
        )
        summaries.append(summary)
        print(
            f"  grid {summary['grid']['width']}×{summary['grid']['height']} @ {summary['grid']['resolution_m']} m, "
            f"elev {summary['elevation_m']['min']:.0f}–{summary['elevation_m']['max']:.0f} m, "
            f"wind_speed≈{summary['wind']['speed_mean']:.2f} m/s, "
            f"mission {summary['mission']['start']}→{summary['mission']['goal']}"
        )
    # Merge into existing index so `--only` rebuilds do not drop other maps.
    by_name: dict[str, dict] = {}
    index_path = scenarios_dir / "index.json"
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text(encoding="utf-8"))
            for row in prev.get("scenarios") or []:
                if isinstance(row, dict) and row.get("name"):
                    by_name[str(row["name"])] = row
        except json.JSONDecodeError:
            pass
    for row in summaries:
        by_name[str(row["name"])] = row
    # Index tracks only the curated DEFAULT_SCENARIOS catalog.
    order = [spec.name for spec in DEFAULT_SCENARIOS]
    merged = [by_name[name] for name in order if name in by_name]
    index = {"scenarios": merged}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return summaries


__all__ = [
    "DEFAULT_EXTENT_M",
    "DEFAULT_RESOLUTION_M",
    "build_default_scenarios",
    "build_scenario_dataset",
    "grid_dims_for_extent",
]
