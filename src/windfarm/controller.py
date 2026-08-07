from __future__ import annotations

import math
from dataclasses import dataclass

from .mathutils import clamp
from .types import DroneState


@dataclass(frozen=True, slots=True)
class FlightEnvelope:
    min_airspeed: float = 8.5
    max_airspeed: float = 22.0
    best_glide_speed: float = 13.8
    best_thermalling_speed: float = 10.5
    max_bank_rad: float = math.radians(42.0)
    max_turn_rate_rad_s: float = math.radians(24.0)
    base_sink_mps: float = 0.58
    polar_quad_coeff: float = 0.014
    induced_drag_coeff: float = 0.33


DEFAULT_ENVELOPE = FlightEnvelope()


def compute_heading(current: tuple[float, float, float], nxt: tuple[float, float, float]) -> float:
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    return math.atan2(dy, dx)


def move_one_step(state: DroneState, nxt: tuple[float, float, float]) -> DroneState:
    return DroneState(
        x=nxt[0],
        y=nxt[1],
        z=nxt[2],
        heading_rad=compute_heading((state.x, state.y, state.z), nxt),
        battery_ratio=max(0.0, state.battery_ratio),
        airspeed=state.airspeed,
    )


def transition_wind_metrics(
    current: tuple[float, float, float],
    nxt: tuple[float, float, float],
    local_u: float,
    local_v: float,
    local_w: float,
) -> tuple[float, float, float]:
    move_dx = nxt[0] - current[0]
    move_dy = nxt[1] - current[1]
    move_dz = nxt[2] - current[2]
    move_norm = max(math.hypot(move_dx, move_dy), 1e-6)
    headwind = -((local_u * move_dx) + (local_v * move_dy)) / move_norm - 0.2 * local_w
    horizontal_distance = math.hypot(move_dx, move_dy)
    return headwind, horizontal_distance, float(move_dz)


def energy_cost_j(airspeed: float, headwind: float, step_distance_m: float) -> float:
    effective_speed = max(4.0, airspeed - headwind)
    travel_time = step_distance_m / effective_speed
    return travel_time * (35.0 + 0.7 * airspeed * airspeed)


def transition_energy_j(
    airspeed: float,
    current: tuple[int, int, int],
    nxt: tuple[int, int, int],
    local_u: float,
    local_v: float,
    local_w: float,
    step_distance_m: float,
    altitude_step_m: float,
    climb_cost_per_level_j: float,
    hover_power_w: float = 105.0,
    cruise_power_w: float = 150.0,
    hotel_power_w: float = 18.0,
    headwind_power_per_mps_w: float = 14.0,
    climb_power_per_mps_w: float = 125.0,
    descent_power_reduction_per_mps_w: float = 58.0,
    terrain_dz_m: float = 0.0,
) -> float:
    headwind, horizontal_distance_cells, move_dz = transition_wind_metrics(current, nxt, local_u, local_v, local_w)
    # AGL band change + terrain follow (constant-AGL over hills still climbs in MSL).
    effective_dz = float(move_dz) + float(terrain_dz_m) / max(altitude_step_m, 1e-6)
    step_distance = math.hypot(horizontal_distance_cells * step_distance_m, abs(effective_dz) * altitude_step_m)
    bank_rad = clamp(abs(math.atan2(move_dy := (nxt[1] - current[1]), max(nxt[0] - current[0], 1e-6))), 0.0, DEFAULT_ENVELOPE.max_bank_rad * 0.45)
    aero = aerodynamic_power_w(
        airspeed=airspeed,
        bank_rad=bank_rad,
        vertical_w=local_w,
        envelope=DEFAULT_ENVELOPE,
    )
    travel_time = step_distance / max(airspeed - headwind, 4.0)
    vertical_rate_mps = (effective_dz * altitude_step_m) / max(travel_time, 1e-6)
    baseline_power = cruise_power_w + hotel_power_w
    wind_power = headwind_power_per_mps_w * max(headwind, 0.0) - 0.45 * headwind_power_per_mps_w * max(-headwind, 0.0)
    maneuver_power = max(0.0, aero - 90.0)
    climb_power = climb_power_per_mps_w * max(vertical_rate_mps, 0.0)
    descent_credit = descent_power_reduction_per_mps_w * max(-vertical_rate_mps, 0.0)
    uplift_credit = 0.30 * descent_power_reduction_per_mps_w * max(local_w, 0.0)
    legacy_vertical_bias = climb_cost_per_level_j * max(effective_dz, 0.0) / max(travel_time, 1e-6)
    total_power = max(
        hover_power_w,
        baseline_power + wind_power + maneuver_power + climb_power + legacy_vertical_bias - descent_credit - uplift_credit,
    )
    return max(0.0, total_power * travel_time)


def advance_continuous_state(
    state: DroneState,
    nxt: tuple[int, int, int],
    local_u: float,
    local_v: float,
    local_w: float,
    safety_penalty: float,
    step_distance_m: float,
    altitude_step_m: float,
    min_altitude_level: int,
    max_altitude_level: int,
) -> DroneState:
    target_x = float(nxt[0])
    target_y = float(nxt[1])
    target_z = float(nxt[2])
    dx = target_x - state.x
    dy = target_y - state.y
    horizontal_distance = math.hypot(dx, dy)
    track_heading = state.heading_rad if horizontal_distance < 1e-6 else math.atan2(dy, dx)
    progress_cells = min(1.0, horizontal_distance)
    thermal_gain = max(local_w, 0.0)
    sink_penalty = max(-local_w, 0.0)
    # Mild loiter in lift only when already near the commanded altitude band.
    loiter_gain = 0.12 * thermal_gain * (1.0 - clamp(abs(target_z - state.z), 0.0, 1.0))
    # Cover ground: baseline was ~0.65 cells/step and stretched short missions.
    forward_scale = clamp(0.80 + 0.16 * thermal_gain - 0.12 * sink_penalty - 0.10 * safety_penalty + loiter_gain, 0.32, 1.12)
    move_fraction = progress_cells * forward_scale
    next_x = state.x + (math.cos(track_heading) * move_fraction if horizontal_distance < 1e-6 else (dx / max(horizontal_distance, 1e-6)) * move_fraction)
    next_y = state.y + (math.sin(track_heading) * move_fraction if horizontal_distance < 1e-6 else (dy / max(horizontal_distance, 1e-6)) * move_fraction)

    climb_command = clamp(target_z - state.z, -0.45, 1.15)
    thermal_climb = local_w / max(altitude_step_m / max(step_distance_m, 1e-6), 1e-6) * 0.12
    # Above the commanded cruise/approach band: dump altitude, do not ride uplift higher.
    if state.z > target_z + 0.35:
        thermal_climb *= 0.15
        climb_command = min(climb_command, 0.85 * (target_z - state.z))
        safety_z = 0.10 * safety_penalty
    elif abs(target_z - state.z) <= 0.35:
        # Level hold: neutralize free lift so we do not ratchet upward while cruising.
        thermal_climb *= 0.2
        climb_command = clamp(target_z - state.z, -0.25, 0.25)
        safety_z = 0.10 * safety_penalty
    elif state.z < target_z - 0.35:
        # Climbing to a commanded band: do not let safety sink cancel the climb
        # (ridge cells were trapping the aircraft below preferred AGL).
        climb_command = max(climb_command, min(1.15, 0.70 * (target_z - state.z)))
        thermal_climb += 0.08 * max(local_w, 0.0)
        safety_z = 0.0
    else:
        safety_z = 0.10 * safety_penalty
    next_z = state.z + climb_command + thermal_climb - safety_z
    next_z = min(next_z, target_z + 0.30)
    if state.z > target_z + 0.2:
        next_z = min(next_z, state.z - 0.05)
    next_z = clamp(next_z, float(min_altitude_level), float(max_altitude_level))
    return DroneState(
        x=next_x,
        y=next_y,
        z=next_z,
        heading_rad=track_heading,
        battery_ratio=max(0.0, state.battery_ratio),
        airspeed=state.airspeed,
    )


def clamp_airspeed(airspeed: float, envelope: FlightEnvelope = DEFAULT_ENVELOPE) -> float:
    return clamp(airspeed, envelope.min_airspeed, envelope.max_airspeed)


def glide_sink_rate_mps(
    airspeed: float,
    bank_rad: float = 0.0,
    envelope: FlightEnvelope = DEFAULT_ENVELOPE,
) -> float:
    v = clamp_airspeed(airspeed, envelope)
    dv = v - envelope.best_glide_speed
    bank_penalty = envelope.induced_drag_coeff * max(0.0, (1.0 / max(math.cos(clamp(bank_rad, 0.0, envelope.max_bank_rad)), 1e-3)) - 1.0)
    return envelope.base_sink_mps + envelope.polar_quad_coeff * dv * dv + bank_penalty


def aerodynamic_power_w(
    airspeed: float,
    bank_rad: float,
    vertical_w: float,
    envelope: FlightEnvelope = DEFAULT_ENVELOPE,
) -> float:
    sink = glide_sink_rate_mps(airspeed, bank_rad, envelope)
    thermal_credit = 55.0 * max(vertical_w - sink, 0.0)
    profile_drag = 42.0 + 0.48 * airspeed * airspeed
    bank_drag = 18.0 * max(0.0, (1.0 / max(math.cos(clamp(bank_rad, 0.0, envelope.max_bank_rad)), 1e-3)) - 1.0)
    return max(12.0, profile_drag + bank_drag - thermal_credit)


def best_thermalling_bank(vertical_w: float, envelope: FlightEnvelope = DEFAULT_ENVELOPE) -> float:
    strength = clamp((vertical_w - 0.3) / 2.0, 0.0, 1.0)
    return clamp((0.22 + 0.38 * strength) * envelope.max_bank_rad, 0.0, envelope.max_bank_rad)


def propagate_control(
    state: DroneState,
    control: dict[str, float | str],
    dt_s: float,
    local_u: float,
    local_v: float,
    local_w: float,
    step_distance_m: float,
    altitude_step_m: float,
    min_altitude_level: int,
    max_altitude_level: int,
    envelope: FlightEnvelope = DEFAULT_ENVELOPE,
) -> tuple[DroneState, float]:
    mode = str(control.get("mode", "cruise"))
    commanded_speed = clamp_airspeed(float(control.get("airspeed", state.airspeed)), envelope)
    climb_bias = float(control.get("climb_bias", 0.0))
    turn_rate = clamp(float(control.get("turn_rate_rad_s", 0.0)), -envelope.max_turn_rate_rad_s, envelope.max_turn_rate_rad_s)
    heading = state.heading_rad + turn_rate * dt_s
    bank_rad = min(envelope.max_bank_rad, abs(turn_rate) / max(envelope.max_turn_rate_rad_s, 1e-6) * envelope.max_bank_rad)
    if mode == "thermal_orbit":
        commanded_speed = min(commanded_speed, envelope.best_thermalling_speed)
        if abs(turn_rate) < 1e-6:
            turn_rate = best_thermalling_bank(local_w, envelope) / max(envelope.max_bank_rad, 1e-6) * envelope.max_turn_rate_rad_s
            heading = state.heading_rad + turn_rate * dt_s
            bank_rad = best_thermalling_bank(local_w, envelope)
        if climb_bias >= 0.0:
            climb_bias = max(climb_bias, 0.65)
    airspeed = state.airspeed + (commanded_speed - state.airspeed) * min(1.0, dt_s / 3.0)
    airspeed = clamp_airspeed(airspeed, envelope)
    body_vx = math.cos(heading) * airspeed
    body_vy = math.sin(heading) * airspeed
    groundspeed_x = body_vx + local_u
    groundspeed_y = body_vy + local_v
    x = state.x + (groundspeed_x * dt_s) / max(step_distance_m, 1e-6)
    y = state.y + (groundspeed_y * dt_s) / max(step_distance_m, 1e-6)
    sink = glide_sink_rate_mps(airspeed, bank_rad, envelope)
    climb_rate = local_w - sink + climb_bias * 0.75
    if mode == "thermal_orbit":
        climb_rate += 0.45 * max(local_w, 0.0)
    if mode == "approach":
        # Final approach overrides residual uplift so the aircraft can close altitude.
        climb_rate = min(climb_rate, climb_bias * 1.15 - max(sink, 0.35))
    z_m = state.z * altitude_step_m + climb_rate * dt_s
    z = clamp(z_m / max(altitude_step_m, 1e-6), float(min_altitude_level), float(max_altitude_level))
    next_state = DroneState(
        x=x,
        y=y,
        z=z,
        heading_rad=heading,
        battery_ratio=state.battery_ratio,
        airspeed=airspeed,
    )
    return next_state, aerodynamic_power_w(airspeed, bank_rad, local_w, envelope) * dt_s
