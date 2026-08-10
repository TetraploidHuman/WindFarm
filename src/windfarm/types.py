from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GridSpec:
    width: int
    height: int
    resolution_m: float


@dataclass(slots=True)
class WindField:
    # Prefer (Z, H, W) ndarrays in hot paths; nested lists still work.
    u: object
    v: object
    w: object


@dataclass(slots=True)
class TerrainField:
    elevation: list[list[float]]
    slope: list[list[float]]
    aspect: list[list[float]]
    roughness: list[list[float]]


@dataclass(slots=True)
class CoarseWindSample:
    timestamp: str
    u_km: float
    v_km: float
    w_km: float = 0.0
    # Optional Open-Meteo 100 m wind (same km units as u_km/v_km).
    u100_km: float | None = None
    v100_km: float | None = None


@dataclass(slots=True)
class Observation:
    timestamp: str
    x: int
    y: int
    u_obs: float
    v_obs: float
    z: int = 0
    w_obs: float = 0.0
    airspeed: float = 0.0
    ground_speed: float = 0.0
    climb_rate: float = 0.0
    acceleration: float = 0.0


@dataclass(slots=True)
class TrainingSample:
    timestamp: str
    x: int
    y: int
    u_km: float
    v_km: float
    u_obs: float
    v_obs: float
    z: int = 0
    w_km: float = 0.0
    w_obs: float = 0.0


@dataclass(slots=True)
class DroneState:
    x: float
    y: float
    z: float
    heading_rad: float
    battery_ratio: float
    airspeed: float


@dataclass(slots=True)
class Mission:
    start: tuple[int, int] | tuple[int, int, int]
    goal: tuple[int, int] | tuple[int, int, int]
    max_steps: int
    step_distance_m: float
    home: tuple[int, int] | tuple[int, int, int] | None = None
    max_return_cost_j: float | None = None
    nominal_airspeed: float = 13.8
    hover_power_w: float = 105.0
    cruise_power_w: float = 150.0
    hotel_power_w: float = 18.0
    headwind_power_per_mps_w: float = 14.0
    climb_power_per_mps_w: float = 125.0
    descent_power_reduction_per_mps_w: float = 58.0
    reserve_energy_ratio: float = 0.22
    altitude_step_m: float = 50.0
    min_altitude_level: int = 0
    max_altitude_level: int = 4
    climb_cost_per_level_j: float = 225.0
    # z is AGL band index; this is the cruise safety floor (not a forced cruise height).
    clearance_agl_level: float = 1.0
    # Planning grid for cruise AGL candidates (levels). 0.02 ≈ 1 m if altitude_step_m=50.
    cruise_band_step: float = 0.02
    # Lateral corridor generation ceiling vs straight (model Joules).
    # None / ≤0 disables corridors; final commit needs ~1.2% win + usable wind/uplift.
    corridor_energy_margin: float | None = 1.02
    # Optional DEM (meters). When set, constant-AGL moves pay terrain climb energy.
    elevation: list[list[float]] | None = None
    # Sticky lateral via (cells) selected by an energy guide; avoids corridor flip-flops.
    guide_via: tuple[float, float] | None = None
    # Distilled 1-via prior (cells) from truth/open-loop — injected as a corridor
    # candidate only; must still pass the same Joules / edge gates. Never sets altitude.
    prior_via: tuple[float, float] | None = None
    # Sticky cruise AGL band from the winning energy guide (may be above clearance floor).
    preferred_cruise_agl: float | None = None
    # True once a climb above clearance was earned with clearance-relative evidence.
    cruise_climb_earned: bool = False


@dataclass(slots=True)
class BeliefCell:
    expected_energy_gain: float = 0.0
    uncertainty: float = 1.0
    confidence: float = 0.0
    mode_prob_sink: float = 0.25
    mode_prob_neutral: float = 0.5
    mode_prob_uplift: float = 0.25
    belief_entropy: float = 1.5
    wind_u: float = 0.0
    wind_v: float = 0.0
    wind_w: float = 0.0
    wind_var_u: float = 1.0
    wind_var_v: float = 1.0
    wind_var_w: float = 1.0
    wind_cov_uv: float = 0.0
    last_update: int = 0
    safety_penalty: float = 0.0
    hazard_prob: float = 0.15


@dataclass(slots=True)
class BeliefMap:
    width: int
    height: int
    levels: int = 1
    cells: list[list[list[BeliefCell]]] = field(default_factory=list)
    # Optional (Z,H,W) float caches for fast trilinear sampling / snapshots.
    # Populated by BeliefUpdater.apply_prediction; patched on observation updates.
    field_arrays: dict[str, object] | None = None
