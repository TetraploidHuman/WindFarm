"""Build multi-scenario datasets from real DEM + Open-Meteo wind."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import MissionConfig, SimulationConfig, TaskConfig, load_task_config, save_task_config
from .data_ingest import (
    DEFAULT_SCENARIOS,
    ScenarioSpec,
    ensure_srtm_tile,
    fetch_open_meteo_hourly,
    mission_endpoints,
    resample_coarse_wind,
    terrain_from_srtm,
    write_coarse_wind_json,
    write_terrain_json,
)
from .simulator import EnvironmentSimulator


def build_scenario_dataset(
    spec: ScenarioSpec,
    output_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    width: int = 24,
    height: int = 18,
    resolution_m: float = 100.0,
    time_steps: int = 120,
    sample_interval_seconds: int = 15,
    seed: int = 7,
    base_config: str | Path | None = None,
) -> dict:
    """Download DEM/wind if needed, synthesize truth/training on real terrain, write artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

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
            "source": "SRTM1/skadi",
        },
    )

    # Align simulation start with the first wind hour so interpolation stays in-range.
    hourly = fetch_open_meteo_hourly(spec.lat, spec.lon, spec.wind_start_date, spec.wind_end_date)
    start_time = hourly[0]["timestamp"]
    # Prefer mid-morning of the first day when available.
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
            "source": "Open-Meteo archive (ERA5-backed)",
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
    # Re-write terrain/coarse with metadata (write_dataset overwrites plain versions).
    write_terrain_json(
        output_dir / "terrain.json",
        terrain,
        meta={
            "scenario": spec.name,
            "description": spec.description,
            "lat": spec.lat,
            "lon": spec.lon,
            "hgt": str(hgt),
            "source": "SRTM1/skadi",
        },
    )
    write_coarse_wind_json(
        output_dir / "coarse_wind.json",
        coarse,
        meta={
            "scenario": spec.name,
            "source": "Open-Meteo archive (ERA5-backed)",
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
    config.simulation = SimulationConfig(
        width=width,
        height=height,
        resolution_m=resolution_m,
        time_steps=time_steps,
        start_time=start_time,
        sample_interval_seconds=sample_interval_seconds,
        report_keyframe_interval=config.simulation.report_keyframe_interval,
    )
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
        altitude_step_m=config.mission.altitude_step_m,
        min_altitude_level=config.mission.min_altitude_level,
        max_altitude_level=config.mission.max_altitude_level,
        climb_cost_per_level_j=config.mission.climb_cost_per_level_j,
        clearance_agl_level=config.mission.clearance_agl_level,
        cruise_band_step=getattr(config.mission, "cruise_band_step", 0.025),
    )
    save_task_config(output_dir / "config.json", config)

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
        "elevation_m": {"min": min(flat), "max": max(flat), "mean": sum(flat) / max(len(flat), 1)},
        "wind": {
            "u_km_mean": sum(s["u_km"] for s in coarse) / len(coarse),
            "v_km_mean": sum(s["v_km"] for s in coarse) / len(coarse),
            "speed_mean": sum((s["u_km"] ** 2 + s["v_km"] ** 2) ** 0.5 for s in coarse) / len(coarse),
        },
        "mission": {"start": list(start), "goal": list(goal)},
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "scenario_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_default_scenarios(
    scenarios_dir: str | Path = "scenarios",
    data_dir: str | Path = "data",
    base_config: str | Path | None = "config_eval.json",
    only: list[str] | None = None,
) -> list[dict]:
    scenarios_dir = Path(scenarios_dir)
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for spec in DEFAULT_SCENARIOS:
        if only and spec.name not in only:
            continue
        print(f"[scenario] building {spec.name}: {spec.description}")
        summary = build_scenario_dataset(
            spec,
            scenarios_dir / spec.name,
            data_dir=data_dir,
            base_config=base_config if base_config and Path(base_config).exists() else None,
        )
        summaries.append(summary)
        print(
            f"  elev {summary['elevation_m']['min']:.0f}–{summary['elevation_m']['max']:.0f} m, "
            f"wind_speed≈{summary['wind']['speed_mean']:.2f} m/s, "
            f"mission {summary['mission']['start']}→{summary['mission']['goal']}"
        )
    index = {"scenarios": summaries}
    (scenarios_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return summaries


def resolve_scenario_spec(name: str) -> ScenarioSpec:
    for spec in DEFAULT_SCENARIOS:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in DEFAULT_SCENARIOS)
    raise KeyError(f"Unknown scenario '{name}'. Known: {known}")
