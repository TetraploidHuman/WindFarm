from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import read_json, write_json


def detect_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def resolve_n_jobs(requested: int | None = 0) -> int:
    """Resolve worker thread count for XGBoost / LightGBM.

    - ``0`` / ``None``: auto-adapt to the host (leave a few cores for OS / planner)
    - ``> 0``: use exactly that many threads (capped by CPU count)
    - ``< 0``: use all logical CPUs
    """
    cpu = detect_cpu_count()
    if requested is None or requested == 0:
        if cpu <= 4:
            return max(1, cpu - 1)
        if cpu <= 16:
            return max(1, cpu - 2)
        # High-core hosts (e.g. 32): keep headroom for planner + system.
        return max(1, cpu - 4)
    if requested < 0:
        return cpu
    return max(1, min(int(requested), cpu))


@dataclass(slots=True)
class ModelConfig:
    split_ratio: float = 0.7
    model_type: str = "xgboost"
    learning_rate: float = 0.1
    epochs: int = 800
    l2: float = 0.01
    num_estimators: int = 96
    max_depth: int = 3
    min_samples_leaf: int = 32
    max_bins: int = 10
    feature_subsample_ratio: float = 0.55
    max_training_samples: int = 4000
    validation_ratio: float = 0.2
    early_stopping_rounds: int = 12
    temporal_window_size: int = 4
    altitude_levels: int = 5
    # 0 = auto-detect host cores; >0 = manual; <0 = use all cores
    n_jobs: int = 0


@dataclass(slots=True)
class BeliefConfig:
    observation_radius: int = 2
    advection_gain: float = 0.25
    decay_per_step: float = 0.04
    process_noise: float = 0.18
    observation_noise: float = 0.12
    advection_noise: float = 0.08


@dataclass(slots=True)
class PlannerConfig:
    risk_weight: float = 1.2
    safety_weight: float = 1.5
    replan_interval_steps: int = 1
    anytime_rounds: int = 4
    heuristic_weight_start: float = 2.4
    heuristic_weight_end: float = 1.0
    search_node_budget: int = 9000
    horizon_steps: int = 7
    beam_width: int = 28
    branch_width: int = 10
    discount_factor: float = 0.93
    terminal_progress_weight: float = 24.0


@dataclass(slots=True)
class SimulationConfig:
    width: int = 32
    height: int = 24
    resolution_m: float = 100.0
    time_steps: int = 360
    start_time: str = "2026-03-21T08:30:00"
    sample_interval_seconds: int = 15
    report_keyframe_interval: int = 3


@dataclass(slots=True)
class MissionConfig:
    start: tuple[int, int] = (4, 18)
    goal: tuple[int, int] = (26, 7)
    max_steps: int = 80
    step_distance_m: float = 100.0
    battery_capacity_j: float = 620000.0
    nominal_airspeed: float = 16.5
    hover_power_w: float = 105.0
    cruise_power_w: float = 150.0
    hotel_power_w: float = 18.0
    headwind_power_per_mps_w: float = 14.0
    climb_power_per_mps_w: float = 125.0
    descent_power_reduction_per_mps_w: float = 58.0
    reserve_energy_ratio: float = 0.22
    altitude_step_m: float = 40.0
    min_altitude_level: int = 0
    max_altitude_level: int = 4
    climb_cost_per_level_j: float = 180.0
    clearance_agl_level: float = 1.0
    cruise_band_step: float = 0.025


@dataclass(slots=True)
class TaskConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_task_config(path: str | Path) -> TaskConfig:
    payload = read_json(path)
    mission_payload = dict(payload.get("mission", {}))
    simulation_payload = dict(payload.get("simulation", {}))
    model_payload = dict(payload.get("model", {}))
    belief_payload = dict(payload.get("belief", {}))
    planner_payload = dict(payload.get("planner", {}))
    if "sample_interval_minutes" in simulation_payload and "sample_interval_seconds" not in simulation_payload:
        simulation_payload["sample_interval_seconds"] = int(simulation_payload.pop("sample_interval_minutes")) * 60
    if "start" in mission_payload:
        mission_payload["start"] = tuple(mission_payload["start"])
    if "goal" in mission_payload:
        mission_payload["goal"] = tuple(mission_payload["goal"])

    def _known(cls, data: dict) -> dict:
        allowed = set(cls.__dataclass_fields__)
        return {key: value for key, value in data.items() if key in allowed}

    return TaskConfig(
        model=ModelConfig(**_known(ModelConfig, model_payload)),
        belief=BeliefConfig(**_known(BeliefConfig, belief_payload)),
        planner=PlannerConfig(**_known(PlannerConfig, planner_payload)),
        simulation=SimulationConfig(**_known(SimulationConfig, simulation_payload)),
        mission=MissionConfig(**_known(MissionConfig, mission_payload)),
    )


def save_task_config(path: str | Path, config: TaskConfig) -> None:
    write_json(path, config.to_dict())
