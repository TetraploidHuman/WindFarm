from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import heapq
import math
import os

import numpy as np

from .altitude import (
    cruise_agl_band_grid,
    cruise_agl_floor,
    format_agl_band_token,
    sample_elevation,
    snap_cruise_agl,
    straight_agl_guide_polyline,
    terrain_climb_along_line_batch,
    terrain_climb_along_line_m,
    terrain_delta_m,
    terrain_delta_m_batch,
)
from .belief import ensure_belief_field_arrays, pack_belief_field
from .controller import (
    DEFAULT_ENVELOPE,
    best_thermalling_bank,
    clamp_airspeed,
    propagate_control,
    transition_energy_batch,
    transition_energy_j,
)
from .mathutils import clamp, trilinear_sample, trilinear_sample_batch
from .types import BeliefMap, DroneState, Mission


State3D = tuple[int, int, int]

_BELIEF_SAMPLE_ATTRS = (
    "wind_u",
    "wind_v",
    "wind_w",
    "expected_energy_gain",
    "uncertainty",
    "safety_penalty",
    "mode_prob_uplift",
    "mode_prob_sink",
)
_WIND_ATTRS = ("wind_u", "wind_v", "wind_w")


@dataclass(slots=True)
class CostBreakdown:
    energy_j: float
    progress_reward_j: float
    uncertainty_cost_j: float
    safety_cost_j: float
    altitude_bias_j: float
    vertical_maneuver_cost_j: float
    total_cost_j: float


def normalize_state(point: tuple[int, int] | tuple[int, int, int]) -> State3D:
    if len(point) == 2:
        return int(point[0]), int(point[1]), 0
    return int(point[0]), int(point[1]), int(point[2])


def neighbors(x: int, y: int, z: int, width: int, height: int, min_z: int, max_z: int) -> list[State3D]:
    result: list[State3D] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if 0 <= nx < width and 0 <= ny < height and min_z <= nz <= max_z:
                    result.append((nx, ny, nz))
    return result


def effective_cost(
    belief_map: BeliefMap,
    current: State3D,
    nxt: State3D,
    mission: Mission,
    risk_weight: float = 1.2,
    safety_weight: float = 1.5,
) -> float:
    return transition_cost_breakdown(
        belief_map,
        current,
        nxt,
        mission,
        risk_weight=risk_weight,
        safety_weight=safety_weight,
    ).total_cost_j


def transition_cost_breakdown(
    belief_map: BeliefMap,
    current: State3D,
    nxt: State3D,
    mission: Mission,
    risk_weight: float = 1.2,
    safety_weight: float = 1.5,
) -> CostBreakdown:
    cx, cy, cz = current
    nx, ny, nz = nxt
    cell = _sample_belief_state(belief_map, float(nx), float(ny), float(nz))
    move_dx = nx - cx
    move_dy = ny - cy
    move_dz = nz - cz
    move_norm = max(math.hypot(move_dx, move_dy), 1e-6)
    tailwind = (cell["wind_u"] * (move_dx / move_norm)) + (cell["wind_v"] * (move_dy / move_norm))
    wind_reward = max(-8.0, min(8.0, tailwind))
    vertical_reward = 2.2 * cell["wind_w"] - 0.6 * max(move_dz, 0)
    horizontal_to_goal = math.hypot(mission.goal[0] - nx, mission.goal[1] - ny)
    altitude_bias_reward = _altitude_bonus(nz, mission, x=nx, y=ny, belief_cell=cell, move_dx=move_dx, move_dy=move_dy)
    # Only push toward goal altitude on final approach; cruise focuses on wind-efficient bands.
    approach_scale = 1.0 if horizontal_to_goal <= 5.0 else 0.15
    altitude_goal_reward = approach_scale * 2.4 * (
        abs(goal_altitude_gap(current, mission) / max(mission.max_altitude_level - mission.min_altitude_level + 1, 1))
        - abs(goal_altitude_gap(nxt, mission) / max(mission.max_altitude_level - mission.min_altitude_level + 1, 1))
    )
    transition_energy = transition_energy_j(
        airspeed=mission.nominal_airspeed,
        current=current,
        nxt=nxt,
        local_u=cell["wind_u"],
        local_v=cell["wind_v"],
        local_w=cell["wind_w"],
        step_distance_m=mission.step_distance_m,
        altitude_step_m=mission.altitude_step_m,
        climb_cost_per_level_j=mission.climb_cost_per_level_j,
        hover_power_w=mission.hover_power_w,
        cruise_power_w=mission.cruise_power_w,
        hotel_power_w=mission.hotel_power_w,
        headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
        climb_power_per_mps_w=mission.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
        terrain_dz_m=_mission_terrain_dz(mission, cx, cy, nx, ny),
    )
    progress_reward = 18.0 * (
        cell["expected_energy_gain"] + 0.35 * wind_reward + vertical_reward + altitude_goal_reward
    )
    uncertainty_cost = 22.0 * risk_weight * max(cell["uncertainty"], 0.0)
    safety_cost = 28.0 * safety_weight * cell["safety_penalty"]
    altitude_bias_cost = -22.0 * altitude_bias_reward
    vertical_maneuver_cost = mission.climb_cost_per_level_j * max(move_dz, 0) + 0.2 * mission.climb_cost_per_level_j * max(-move_dz, 0)
    goal_altitude_direction = goal_altitude_gap(current, mission) - goal_altitude_gap(nxt, mission)
    if horizontal_to_goal <= 5.0:
        if goal_altitude_direction < 0:
            vertical_maneuver_cost += 1.4 * mission.climb_cost_per_level_j * abs(goal_altitude_direction)
        elif goal_altitude_direction > 0:
            vertical_maneuver_cost -= 0.9 * mission.climb_cost_per_level_j * abs(goal_altitude_direction)
    else:
        # Cruise: climbing into helpful wind is cheap; climbing into headwind/sink is expensive.
        if move_dz > 0 and (cell["wind_w"] > 0.15 or wind_reward > 0.5):
            vertical_maneuver_cost *= 0.35
        elif move_dz > 0 and wind_reward < -0.5:
            vertical_maneuver_cost *= 1.4
    total_cost = transition_energy - progress_reward + uncertainty_cost + safety_cost + altitude_bias_cost + vertical_maneuver_cost
    return CostBreakdown(
        energy_j=transition_energy,
        progress_reward_j=progress_reward,
        uncertainty_cost_j=uncertainty_cost,
        safety_cost_j=safety_cost,
        altitude_bias_j=altitude_bias_cost,
        vertical_maneuver_cost_j=vertical_maneuver_cost,
        total_cost_j=total_cost,
    )


def heuristic(point: State3D, goal: State3D, mission: Mission) -> float:
    horizontal = math.hypot(goal[0] - point[0], goal[1] - point[1]) * mission.step_distance_m
    vertical = abs(goal[2] - point[2]) * mission.altitude_step_m
    return math.hypot(horizontal, vertical)


def estimate_energy_to_goal_j(
    point: tuple[float, float, float],
    goal: State3D,
    mission: Mission,
    local_wind: dict[str, float] | None = None,
    belief_map: BeliefMap | None = None,
    probe_corridors: bool = False,
) -> float:
    """Residual energy to goal; samples wind/terrain along the remaining segment.

    When probe_corridors=True and a belief map is available, also scores mild
    lateral offsets and returns the cheapest option.
    """
    energies = estimate_energies_to_goal_batch(
        [point],
        goal,
        mission,
        belief_map=belief_map,
        local_winds=[local_wind] if local_wind is not None else None,
        probe_corridors=probe_corridors,
    )
    return float(energies[0])


def estimate_energies_to_goal_batch(
    points: list[tuple[float, float, float]],
    goal: State3D,
    mission: Mission,
    belief_map: BeliefMap | None = None,
    local_winds: list[dict[str, float] | None] | None = None,
    probe_corridors: bool = False,
) -> list[float]:
    """Batch residual energy-to-goal for many points (same goal)."""
    if not points:
        return []
    gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2])
    n = len(points)
    # Light path: no belief → closed form from local wind / nominal.
    if belief_map is None:
        out: list[float] = []
        for i, point in enumerate(points):
            local = None if local_winds is None else local_winds[i]
            out.append(_segment_energy_closed(point, (gx, gy, gz), mission, local))
        return out

    fracs = (0.2, 0.55, 0.85)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    meta: list[tuple[float, float, float, float, float]] = []  # x0,y0,x1,y1,z
    for point in points:
        x0, y0, z = float(point[0]), float(point[1]), float(point[2])
        meta.append((x0, y0, gx, gy, z))
        for f in fracs:
            xs.append(x0 + f * (gx - x0))
            ys.append(y0 + f * (gy - y0))
            zs.append(z)
    wind = _sample_wind_batch(belief_map, xs, ys, zs)
    wu = np.asarray(wind["wind_u"], dtype=np.float64).reshape(n, 3)
    wv = np.asarray(wind["wind_v"], dtype=np.float64).reshape(n, 3)
    ww = np.asarray(wind["wind_w"], dtype=np.float64).reshape(n, 3)
    xs0 = np.asarray([m[0] for m in meta], dtype=np.float64)
    ys0 = np.asarray([m[1] for m in meta], dtype=np.float64)
    xs1 = np.asarray([m[2] for m in meta], dtype=np.float64)
    ys1 = np.asarray([m[3] for m in meta], dtype=np.float64)
    zs_arr = np.asarray([m[4] for m in meta], dtype=np.float64)
    dx = xs1 - xs0
    dy = ys1 - ys0
    horiz_m = np.hypot(dx, dy) * mission.step_distance_m
    bearings = np.arctan2(dy, dx)
    cos_b = np.cos(bearings)
    sin_b = np.sin(bearings)
    tw = np.mean(wu * cos_b[:, None] + wv * sin_b[:, None], axis=1)
    uplift = np.mean(ww, axis=1)
    airspeed = max(mission.nominal_airspeed, 4.0)
    groundspeed = np.maximum(4.0, airspeed + tw)
    time_s = np.where(horiz_m > 1e-6, horiz_m / groundspeed, 0.0)
    headwind = np.maximum(0.0, -tw)
    nominal_power_w = (
        mission.cruise_power_w
        + mission.hotel_power_w
        + mission.headwind_power_per_mps_w * headwind
        - 18.0 * np.maximum(0.0, uplift)
    )
    base = time_s * np.maximum(40.0, nominal_power_w)
    dz_levels = gz - zs_arr
    climb = np.where(
        dz_levels >= 0.0,
        dz_levels * mission.climb_cost_per_level_j,
        dz_levels * mission.altitude_step_m * mission.descent_power_reduction_per_mps_w,
    )
    base = base + climb
    elev = getattr(mission, "elevation", None)
    if elev is not None:
        rises = terrain_climb_along_line_batch(elev, xs0, ys0, xs1, ys1, samples=3)
        climb_scale = mission.climb_cost_per_level_j / max(mission.altitude_step_m, 1e-6)
        base = base + np.where(horiz_m > 1e-6, rises * climb_scale, 0.0)
    out = base.tolist()

    if probe_corridors and belief_map is not None:
        for i, (x0, y0, x1, y1, z) in enumerate(meta):
            horiz_cells = math.hypot(gx - x0, gy - y0)
            if horiz_cells <= 6.0:
                continue
            px, py = -(gy - y0) / horiz_cells, (gx - x0) / horiz_cells
            for sign in (-1.0, 1.0):
                via = (x0 + 0.5 * (gx - x0) + sign * 1.5 * px, y0 + 0.5 * (gy - y0) + sign * 1.5 * py, z)
                via_e = estimate_energy_to_goal_j(via, goal, mission, belief_map=belief_map, probe_corridors=False)
                out[i] = min(
                    out[i],
                    via_e + 35.0 * 1.5 * mission.step_distance_m / max(mission.nominal_airspeed, 4.0),
                )
    return out


def _segment_energy_closed(
    point: tuple[float, float, float],
    goal: tuple[float, float, float],
    mission: Mission,
    local_wind: dict[str, float] | None,
) -> float:
    x0, y0, z = float(point[0]), float(point[1]), float(point[2])
    gx, gy, gz = goal
    horiz_m = math.hypot(gx - x0, gy - y0) * mission.step_distance_m
    if horiz_m <= 1e-6:
        base = 0.0
    else:
        bearing = math.atan2(gy - y0, gx - x0)
        tw = 0.0
        uplift = 0.0
        if local_wind is not None:
            tw = local_wind.get("wind_u", 0.0) * math.cos(bearing) + local_wind.get("wind_v", 0.0) * math.sin(bearing)
            uplift = float(local_wind.get("wind_w", 0.0))
        airspeed = max(mission.nominal_airspeed, 4.0)
        groundspeed = max(4.0, airspeed + tw)
        time_s = horiz_m / groundspeed
        headwind = max(0.0, -tw)
        nominal_power_w = (
            mission.cruise_power_w
            + mission.hotel_power_w
            + mission.headwind_power_per_mps_w * headwind
            - 18.0 * max(0.0, uplift)
        )
        base = time_s * max(40.0, nominal_power_w)
        terrain_rise_m = terrain_climb_along_line_m(
            getattr(mission, "elevation", None), x0, y0, gx, gy, samples=3
        )
        base += (terrain_rise_m / max(mission.altitude_step_m, 1e-6)) * mission.climb_cost_per_level_j
    dz_levels = gz - z
    if dz_levels >= 0:
        base += dz_levels * mission.climb_cost_per_level_j
    else:
        base += dz_levels * mission.altitude_step_m * mission.descent_power_reduction_per_mps_w
    return base


def plan_path(
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float = 1.2,
    safety_weight: float = 1.5,
    anytime_rounds: int = 4,
    heuristic_weight_start: float = 2.4,
    heuristic_weight_end: float = 1.0,
    search_node_budget: int = 9000,
    horizon_steps: int = 7,
    beam_width: int = 28,
    branch_width: int = 10,
    discount_factor: float = 0.93,
    terminal_progress_weight: float = 24.0,
) -> list[State3D]:
    return plan_path_details(
        belief_map=belief_map,
        mission=mission,
        risk_weight=risk_weight,
        safety_weight=safety_weight,
        anytime_rounds=anytime_rounds,
        heuristic_weight_start=heuristic_weight_start,
        heuristic_weight_end=heuristic_weight_end,
        search_node_budget=search_node_budget,
        horizon_steps=horizon_steps,
        beam_width=beam_width,
        branch_width=branch_width,
        discount_factor=discount_factor,
        terminal_progress_weight=terminal_progress_weight,
    )["path"]


def plan_path_details(
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float = 1.2,
    safety_weight: float = 1.5,
    anytime_rounds: int = 4,
    heuristic_weight_start: float = 2.4,
    heuristic_weight_end: float = 1.0,
    search_node_budget: int = 9000,
    horizon_steps: int = 7,
    beam_width: int = 28,
    branch_width: int = 10,
    discount_factor: float = 0.93,
    terminal_progress_weight: float = 24.0,
    return_cost_map_cache: object | None = None,
) -> dict:
    # Prefer ndarray elevation for vectorized terrain queries during this plan.
    # Mutate in place — do not replace(mission): sticky fields (preferred_cruise_agl,
    # guide_via) must remain on the caller-owned Mission object.
    elev_src = getattr(mission, "elevation", None)
    if elev_src is not None and not isinstance(elev_src, np.ndarray):
        mission.elevation = np.asarray(elev_src, dtype=np.float64)

    start = _continuous_state_tuple(mission.start)
    goal = normalize_state(mission.goal)
    horizontal_to_goal = math.hypot(goal[0] - start[0], goal[1] - start[1])
    guide_horizon_cells = 50.0 / max(float(mission.step_distance_m), 1e-6)
    # Overlap independent subtasks (shared read-only belief): guides ∥ return-map, then MPC.
    # Full Dijkstra is expensive on fine grids — only when the return budget is actually tight.
    use_return_map = (
        mission.home is not None
        and mission.max_return_cost_j is not None
        and horizontal_to_goal > 8.0
        and _return_budget_is_tight(mission, start, goal)
    )
    want_guides = horizontal_to_goal > guide_horizon_cells
    sticky_cruise = getattr(mission, "preferred_cruise_agl", None)
    # After the first guide commit, energy guides dominate open-cruise legs — seed with
    # greedy progress instead of multi-round MPC (guides still override below).
    sticky_open_cruise = (
        sticky_cruise is not None
        and want_guides
        and not use_return_map
        and horizontal_to_goal > max(guide_horizon_cells, 8.0)
    )
    guide_future = None
    return_future = None
    pool: ThreadPoolExecutor | None = None
    return_cost_map = return_cost_map_cache if (use_return_map and return_cost_map_cache is not None) else None
    need_return_compute = use_return_map and return_cost_map is None
    workers = int(want_guides) + int(need_return_compute)
    # Under parallel multi-scenario eval, prefer single-threaded planner helpers
    # (parent already saturates cores with several run-demo processes).
    if os.environ.get("WINDFARM_N_JOBS") and workers > 1:
        workers = 1
    if workers > 0:
        pool = ThreadPoolExecutor(max_workers=workers)
        if want_guides:
            guide_future = pool.submit(_energy_guide_paths, start, goal, belief_map, mission)
        if need_return_compute:
            if workers == 1 and want_guides:
                # Guides already submitted; compute return map on this thread after/with overlap via result order.
                return_future = None
                return_cost_map = compute_return_cost_map(belief_map, mission)
            else:
                return_future = pool.submit(compute_return_cost_map, belief_map, mission)

    if return_future is not None:
        return_cost_map = return_future.result()

    candidates: list[dict] = []
    if sticky_open_cruise:
        best_path = _greedy_progress_path(start, goal, belief_map, mission)
        planning_mode = "greedy_progress"
        candidates.append(
            {
                "mode": planning_mode,
                "path": best_path,
                "path_cost": 0.0,
                "note": "sticky_cruise_skip_mpc",
            }
        )
    else:
        strict_result = _run_anytime_continuous_mpc(
            belief_map=belief_map,
            mission=mission,
            risk_weight=risk_weight,
            safety_weight=safety_weight,
            anytime_rounds=anytime_rounds,
            heuristic_weight_start=heuristic_weight_start,
            heuristic_weight_end=heuristic_weight_end,
            search_node_budget=search_node_budget,
            horizon_steps=horizon_steps,
            beam_width=beam_width,
            branch_width=branch_width,
            discount_factor=discount_factor,
            terminal_progress_weight=terminal_progress_weight,
            return_cost_map=return_cost_map,
            candidate_mode="strict_return",
        )
        best_path = strict_result["path"]
        candidates = strict_result["candidates"]
        planning_mode = "mpc_strict_return"

        needs_relaxed = return_cost_map is not None and (
            not best_path
            or len(best_path) <= 1
            or not _continuous_goal_reached(best_path[-1], goal)
        )
        if needs_relaxed or horizontal_to_goal <= 8.0:
            relaxed_result = _run_anytime_continuous_mpc(
                belief_map=belief_map,
                mission=mission,
                risk_weight=risk_weight,
                safety_weight=safety_weight,
                anytime_rounds=max(2, anytime_rounds),
                heuristic_weight_start=heuristic_weight_start,
                heuristic_weight_end=heuristic_weight_end,
                search_node_budget=max(search_node_budget // 2, 2000),
                horizon_steps=horizon_steps,
                beam_width=max(beam_width // 2, 8),
                branch_width=branch_width,
                discount_factor=discount_factor,
                terminal_progress_weight=terminal_progress_weight * 1.35,
                return_cost_map=None,
                candidate_mode="relaxed_return",
            )
            candidates.extend(relaxed_result["candidates"])
            if relaxed_result["path"] and (
                _continuous_goal_reached(relaxed_result["path"][-1], goal)
                or len(best_path) <= 1
                or heuristic(_continuous_state_tuple(relaxed_result["path"][-1]), goal, mission)
                < heuristic(_continuous_state_tuple(best_path[-1]), goal, mission)
            ):
                best_path = relaxed_result["path"]
                planning_mode = "mpc_relaxed_return"

    best_path = best_path[: mission.max_steps + 1]
    if len(best_path) <= 1:
        best_path = _greedy_progress_path(start, goal, belief_map, mission)
        planning_mode = "greedy_progress"

    # Energy floor: multi-band straight AGL + lateral corridors vs MPC, same transition model.
    guide_paths: list[tuple[str, list[tuple[float, float, float]]]] = []
    try:
        if guide_future is not None:
            guide_paths = guide_future.result()
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
    ranked = []
    mpc_completed_energy = math.inf
    if best_path and len(best_path) > 1:
        ranked.append((planning_mode, best_path))
        mpc_completed_energy = _completed_plan_energy_j(best_path, goal, belief_map, mission)
    for label, path in guide_paths:
        ranked.append((label, path))
    best_energy = math.inf
    best_label = planning_mode
    chosen = best_path
    clearance = float(getattr(mission, "clearance_agl_level", 1.0))
    straight_items: list[tuple[float, str, list[tuple[float, float, float]]]] = []
    for label, path in guide_paths:
        if not label.startswith("guide_straight_agl") or not path or len(path) <= 1:
            continue
        cruise_z = _cruise_z_from_guide_label(label, clearance, mission)
        straight_items.append((cruise_z, label, path))
    straight_energies: dict[float, float] = {}
    straight_paths: dict[float, tuple[str, list[tuple[float, float, float]]]] = {}
    if straight_items:
        energies = _completed_plans_energy_j([p for _, _, p in straight_items], goal, belief_map, mission)
        for (cruise_z, label, path), energy in zip(straight_items, energies):
            straight_energies[cruise_z] = energy
            straight_paths[cruise_z] = (label, path)
    straight_guide_energy = min(straight_energies.values()) if straight_energies else math.inf
    bands = list(straight_energies.keys()) if straight_energies else _cruise_band_candidates(belief_map, mission)
    sticky_for_scores = getattr(mission, "preferred_cruise_agl", None)
    if sticky_for_scores is not None and bands:
        # Keep sticky/clearance fine patches and any coarse-span probes (layer discovery).
        coarse = 0.25
        near = []
        for z in bands:
            zf = float(z)
            on_coarse = abs(zf / coarse - round(zf / coarse)) <= 1e-9
            if (
                abs(zf - float(sticky_for_scores)) <= 0.40 + 1e-9
                or abs(zf - clearance) <= 0.15 + 1e-9
                or on_coarse
            ):
                near.append(z)
        if near:
            bands = near
    step = _cruise_band_step(mission)
    # Primary selector: full-path straight energies (same transition model as eval).
    # Mid-cruise descent thrash is handled by asymmetric sticky margins below — not by
    # switching to cruise-fair (that under-penalizes climb and re-locks ~AGL3).
    if straight_energies:
        select_scores = {z: straight_energies[z] for z in bands if z in straight_energies}
        if not select_scores:
            select_scores = dict(straight_energies)
        cruise_fair: dict[float, float] = {}
    else:
        cruise_fair = _band_selection_scores(start, goal, belief_map, mission, bands)
        select_scores = cruise_fair
    # Tiny climb prior — break exact ties toward clearance (not a hard preference).
    if select_scores and step <= 0.26 + 1e-12:
        floor_z = min(select_scores.keys(), key=lambda z: abs(z - clearance))
        floor_e = max(select_scores[floor_z], 1.0)
        pen = 0.002 * floor_e
        select_scores = {
            z: e + pen * max(0.0, float(z) - clearance) for z, e in select_scores.items()
        }
    sticky_now = getattr(mission, "preferred_cruise_agl", None)
    sz_now = float(start[2])
    # Cancel aero-descent optimism when already established in cruise.
    if sticky_now is not None and sz_now >= clearance + 0.25 and select_scores and horizontal_to_goal > 8.0:
        climb_cost = float(getattr(mission, "climb_cost_per_level_j", 180.0))
        hold = min(sz_now, float(sticky_now))
        select_scores = {
            z: e + climb_cost * max(0.0, hold - float(z))
            for z, e in select_scores.items()
        }
    selected_band = _select_preferred_cruise_band(
        select_scores,
        sticky=getattr(mission, "preferred_cruise_agl", None),
        clearance=clearance,
        remaining_horiz=horizontal_to_goal,
        tiebreak_scores=straight_energies if straight_energies else cruise_fair,
        band_step=step,
    )
    # Soft rate-limit only — evidence gates live in _select_preferred_cruise_band.
    sticky_now = getattr(mission, "preferred_cruise_agl", None)
    if sticky_now is not None and selected_band is not None and horizontal_to_goal > 10.0:
        lo = float(sticky_now) - 0.35
        hi = float(sticky_now) + 0.35
        selected_band = float(min(max(float(selected_band), lo), hi))
    elif sticky_now is not None and selected_band is not None:
        # Approach: allow faster descent than climb.
        selected_band = float(min(float(selected_band), float(sticky_now)))
        selected_band = float(max(float(selected_band), float(sticky_now) - 0.50))
    preferred_straight_energy = (
        straight_energies.get(selected_band, straight_guide_energy)
        if selected_band is not None
        else straight_guide_energy
    )
    # If selected band is missing from path table (shouldn't), rebuild from fair z.
    if selected_band is not None and selected_band not in straight_paths:
        label = _straight_guide_label(selected_band, clearance, step=step)
        path = _agl_guide_polyline(start, goal, belief_map, mission, via_xy=None, cruise_z=selected_band)
        if len(path) > 1:
            straight_paths[selected_band] = (label, path)
            e = _completed_plan_energy_j(path, goal, belief_map, mission)
            straight_energies[selected_band] = e
            preferred_straight_energy = e
    band_eps = max(1e-6, 0.51 * step)
    ranked_candidates: list[tuple[str, list[tuple[float, float, float]]]] = []
    for label, path in ranked:
        if not path or len(path) <= 1:
            continue
        # Must make forward progress on the first hop.
        if _goal_axis_progress(
            DroneState(x=start[0], y=start[1], z=start[2], heading_rad=0.0, battery_ratio=1.0, airspeed=mission.nominal_airspeed),
            path[1],
            goal,
        ) < 0.05 and math.hypot(goal[0] - start[0], goal[1] - start[1]) > 1.0:
            continue
        ranked_candidates.append((label, path))
    ranked_energies = (
        _completed_plans_energy_j([path for _, path in ranked_candidates], goal, belief_map, mission)
        if ranked_candidates
        else []
    )
    for (label, path), energy in zip(ranked_candidates, ranked_energies):
        floor_e = preferred_straight_energy if preferred_straight_energy < math.inf else straight_guide_energy
        # Universal hard floor: never commit below the preferred straight AGL band.
        if not label.startswith("guide_straight_agl") and floor_e < math.inf and energy > floor_e * 1.001:
            continue
        if label.startswith("guide_corridor_"):
            # Same margin knob as generation (default 0.95); no extra 10%+ deadweight.
            corr_m = getattr(mission, "corridor_energy_margin", None)
            corr_m = 0.97 if corr_m is None else float(corr_m)
            if corr_m <= 0.0:
                continue
            if floor_e < math.inf and energy > floor_e * corr_m:
                continue
            if mpc_completed_energy < math.inf and energy > mpc_completed_energy * corr_m:
                continue
            energy += 0.01 * max(floor_e, energy)
        elif label.startswith("guide_straight_agl"):
            band = _cruise_z_from_guide_label(label, clearance, mission)
            if selected_band is not None and abs(band - selected_band) <= band_eps:
                energy *= 0.97
            else:
                energy *= 1.01
        terminal = heuristic(_continuous_state_tuple(path[-1]), goal, mission)
        score = energy + 0.05 * terminal
        if score < best_energy:
            best_energy = score
            chosen = path
            best_label = label
    # Snap MPC / wrong-band straights to the selected cruise band; keep winning corridors.
    if (
        selected_band is not None
        and selected_band in straight_paths
        and horizontal_to_goal > guide_horizon_cells
    ):
        wrong_straight = best_label.startswith("guide_straight_agl") and (
            abs(_cruise_z_from_guide_label(best_label, clearance, mission) - selected_band) > band_eps
        )
        if best_label.startswith("mpc") or wrong_straight:
            best_label, chosen = straight_paths[selected_band]
            best_energy = preferred_straight_energy * 0.97
    if chosen is not None and best_label != planning_mode:
        best_path = chosen
        planning_mode = best_label
        cruise_ref = (
            selected_band
            if selected_band is not None
            else _cruise_z_from_guide_label(best_label, clearance, mission)
        )
        cruise_ref = _commit_preferred_cruise(
            mission,
            cruise_ref,
            clearance,
            band_energies=select_scores if select_scores else straight_energies,
        )
        if best_label.startswith("guide_corridor_"):
            via = None
            for p in best_path[1:4]:
                if abs(p[2] - cruise_ref) <= 0.35 or abs(p[2] - float(getattr(mission, "preferred_cruise_agl", cruise_ref))) <= 0.35:
                    via = (float(p[0]), float(p[1]))
                    break
            if via is None:
                # Fall back to first lateral waypoint even if z differs (profile rewrite).
                for p in best_path[1:4]:
                    via = (float(p[0]), float(p[1]))
                    break
            if via is not None:
                mission.guide_via = via
        elif best_label.startswith("guide_straight_agl"):
            mission.guide_via = None
        else:
            if cruise_ref > clearance + 0.05:
                mission.guide_via = None
        candidates.append(
            {
                "mode": best_label,
                "path": best_path,
                "path_cost": best_energy,
                "note": "energy_guide_override",
            }
        )
    elif selected_band is not None:
        _commit_preferred_cruise(
            mission,
            selected_band,
            clearance,
            band_energies=select_scores if select_scores else straight_energies,
        )
        if not str(best_label).startswith("guide_corridor_"):
            mission.guide_via = None

    best_path = best_path[: mission.max_steps + 1]
    if best_path and _continuous_goal_reached(best_path[-1], goal):
        best_path[-1] = (float(goal[0]), float(goal[1]), float(goal[2]))
    return {
        "path": best_path,
        "path_cost": _path_cost(best_path, belief_map, mission, risk_weight, safety_weight),
        "path_cost_breakdown": _path_cost_breakdown(best_path, belief_map, mission, risk_weight, safety_weight),
        "candidates": candidates,
        "return_cost_map": return_cost_map,
        "return_budget_j": mission.max_return_cost_j,
        "planning_mode": planning_mode,
        "return_constraint_active": planning_mode == "mpc_strict_return",
    }


def _greedy_progress_path(
    start: tuple[float, float, float],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
) -> list[tuple[float, float, float]]:
    """Guarantee a non-empty next waypoint toward the goal when MPC collapses."""
    min_level, max_level = _search_level_bounds(belief_map, mission)
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    horizontal = math.hypot(dx, dy)
    step = min(1.0, horizontal) if horizontal > 1e-6 else 0.0
    if horizontal > 1e-6:
        nx = start[0] + (dx / horizontal) * step
        ny = start[1] + (dy / horizontal) * step
    else:
        nx, ny = start[0], start[1]
    if horizontal > 2.5:
        floor = _effective_agl_floor(mission, horizontal)
        preferred = _preferred_cruise_altitude(
            belief_map,
            start[0],
            start[1],
            goal,
            max(min_level, int(math.floor(floor))),
            max_level,
            current_z=start[2],
            climb_cost_per_level_j=mission.climb_cost_per_level_j,
            clearance_agl_level=getattr(mission, "clearance_agl_level", 1.0),
            preferred_cruise_agl=getattr(mission, "preferred_cruise_agl", None),
        )
        preferred = max(preferred, floor)
        dz = clamp(preferred - start[2], -1.0, 1.0)
    else:
        dz = clamp(goal[2] - start[2], -1.5, 1.5)
        if abs(goal[2] - start[2]) > 0.2:
            nx, ny = start[0], start[1]
    floor_now = _effective_agl_floor(mission, horizontal)
    nz = clamp(start[2] + dz, max(float(min_level), floor_now if horizontal > 2.5 else float(min_level)), float(max_level))
    nxt = (
        clamp(nx, 0.0, belief_map.width - 1),
        clamp(ny, 0.0, belief_map.height - 1),
        nz,
    )
    if math.hypot(nxt[0] - start[0], nxt[1] - start[1]) < 1e-4 and abs(nxt[2] - start[2]) < 1e-4:
        return [start]
    return [start, nxt]


def _polyline_model_energy_j(
    path: list[tuple[float, float, float]],
    belief_map: BeliefMap,
    mission: Mission,
) -> float:
    """Same accounting family as eval baselines: transition_energy_j + uplift discount."""
    energies = _polylines_model_energy_j([path], belief_map, mission)
    return energies[0] if energies else 0.0


def _polylines_model_energy_j(
    paths: list[list[tuple[float, float, float]]],
    belief_map: BeliefMap,
    mission: Mission,
) -> list[float]:
    """Batch-score many polylines in one transition_energy_batch call."""
    if not paths:
        return []
    counts: list[int] = []
    cur_rows: list[tuple[float, float, float]] = []
    nxt_rows: list[tuple[float, float, float]] = []
    for path in paths:
        if len(path) <= 1:
            counts.append(0)
            continue
        n = len(path) - 1
        counts.append(n)
        for a, b in zip(path, path[1:]):
            cur_rows.append((float(a[0]), float(a[1]), float(a[2])))
            nxt_rows.append((float(b[0]), float(b[1]), float(b[2])))
    if not cur_rows:
        return [0.0 for _ in paths]

    currents = np.asarray(cur_rows, dtype=np.float64)
    nexts = np.asarray(nxt_rows, dtype=np.float64)
    samples = _sample_belief_states_batch(belief_map, currents[:, 0], currents[:, 1], currents[:, 2])
    elev = getattr(mission, "elevation", None)
    if elev is not None:
        terrain_dz = terrain_delta_m_batch(
            elev,
            currents[:, 0],
            currents[:, 1],
            nexts[:, 0],
            nexts[:, 1],
        )
    else:
        terrain_dz = np.zeros(currents.shape[0], dtype=np.float64)
    step_e = transition_energy_batch(
        airspeed=mission.nominal_airspeed,
        currents=currents,
        nexts=nexts,
        local_u=samples["wind_u"],
        local_v=samples["wind_v"],
        local_w=samples["wind_w"],
        step_distance_m=mission.step_distance_m,
        altitude_step_m=mission.altitude_step_m,
        climb_cost_per_level_j=mission.climb_cost_per_level_j,
        hover_power_w=mission.hover_power_w,
        cruise_power_w=mission.cruise_power_w,
        hotel_power_w=mission.hotel_power_w,
        headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
        climb_power_per_mps_w=mission.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
        terrain_dz_m=terrain_dz,
    )
    uplift = np.clip(1.0 - 0.20 * np.maximum(samples["wind_w"], 0.0), 0.5, 1.4)
    seg = step_e * uplift
    out: list[float] = []
    offset = 0
    for n in counts:
        if n <= 0:
            out.append(0.0)
        else:
            out.append(float(np.sum(seg[offset : offset + n])))
            offset += n
    return out


def _path_mean_uncertainty(
    path: list[tuple[float, float, float]],
    belief_map: BeliefMap,
) -> float:
    if len(path) <= 1:
        return 1.0
    pts = np.asarray(path[:-1], dtype=np.float64)
    samples = _sample_belief_states_batch(belief_map, pts[:, 0], pts[:, 1], pts[:, 2])
    return float(np.mean(samples["uncertainty"]))


def _polyline_risk_adjusted_energy_j(
    path: list[tuple[float, float, float]],
    belief_map: BeliefMap,
    mission: Mission,
    uncertainty_gain: float = 0.08,
) -> float:
    """Belief energy with an uncertainty surcharge (pessimize untrusted wind wins)."""
    base = _polyline_model_energy_j(path, belief_map, mission)
    if not math.isfinite(base):
        return base
    unc = _path_mean_uncertainty(path, belief_map)
    return base * (1.0 + uncertainty_gain * max(0.0, unc))


def _path_xy_length(path: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _completed_plan_energy_j(
    path: list[tuple[float, float, float]],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
) -> float:
    """Polyline energy of path, finishing to goal with a constant-AGL guide if needed."""
    return _completed_plans_energy_j([path], goal, belief_map, mission)[0]


def _completed_plans_energy_j(
    paths: list[list[tuple[float, float, float]]],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
) -> list[float]:
    """Batch version of _completed_plan_energy_j."""
    if not paths:
        return []
    clearance = float(getattr(mission, "clearance_agl_level", 1.0))
    cruise_z = getattr(mission, "preferred_cruise_agl", None)
    cruise = float(cruise_z) if cruise_z is not None else clearance
    extended: list[list[tuple[float, float, float]]] = []
    for path in paths:
        if not path:
            extended.append([])
            continue
        if _continuous_goal_reached(path[-1], goal):
            extended.append(path)
            continue
        finish = _agl_guide_polyline(
            _continuous_state_tuple(path[-1]),
            goal,
            belief_map,
            mission,
            via_xy=None,
            cruise_z=cruise,
        )
        if len(finish) > 1:
            extended.append(list(path) + list(finish[1:]))
        else:
            extended.append(path)
    return _polylines_model_energy_j(extended, belief_map, mission)


def _agl_guide_polyline(
    start: tuple[float, float, float],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
    via_xy: tuple[float, float] | None = None,
    cruise_z: float | None = None,
) -> list[tuple[float, float, float]]:
    """Constant-AGL guide packed like eval `straight_agl_baseline`.

    Coarse waypoint packing biases band ranking (elevated cruise looks worse than
    it is). Use the same climb/hold/descend discretization as the energy metric.
    """
    clearance = float(getattr(mission, "clearance_agl_level", 1.0))
    min_level, max_level = _search_level_bounds(belief_map, mission)
    if cruise_z is None:
        cruise_z = clearance
    cruise_z = clamp(float(cruise_z), float(min_level), float(max_level))
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2])
    width = max(belief_map.width - 1, 0)
    height = max(belief_map.height - 1, 0)

    def _clamp_path(pts: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for x, y, z in pts:
            p = (
                clamp(float(x), 0.0, float(width)),
                clamp(float(y), 0.0, float(height)),
                clamp(float(z), float(min_level), float(max_level)),
            )
            if not out:
                out.append(p)
                continue
            prev = out[-1]
            if math.hypot(p[0] - prev[0], p[1] - prev[1]) > 0.02 or abs(p[2] - prev[2]) > 0.02:
                out.append(p)
        return out if out else [(sx, sy, sz)]

    if via_xy is None or math.hypot(gx - sx, gy - sy) < 5.0:
        return _clamp_path(straight_agl_guide_polyline((sx, sy, sz), (gx, gy, gz), agl_cruise=cruise_z))

    vx = clamp(float(via_xy[0]), 0.0, float(width))
    vy = clamp(float(via_xy[1]), 0.0, float(height))
    # Two constant-AGL legs: start → via (hold cruise) → goal (descend).
    leg1 = straight_agl_guide_polyline((sx, sy, sz), (vx, vy, cruise_z), agl_cruise=cruise_z)
    leg2 = straight_agl_guide_polyline((vx, vy, cruise_z), (gx, gy, gz), agl_cruise=cruise_z)
    merged = list(leg1) + list(leg2[1:] if leg2 else [])
    return _clamp_path(merged)


def _cruise_band_step(mission: Mission) -> float:
    # Floor at ~0.5 m physical resolution so fine grids stay usable.
    alt = max(float(getattr(mission, "altitude_step_m", 50.0) or 50.0), 1e-6)
    configured = float(getattr(mission, "cruise_band_step", 0.02) or 0.02)
    return max(configured, 0.5 / alt)


def _cruise_band_span(mission: Mission) -> float:
    return float(getattr(mission, "cruise_band_span", 2.0) or 2.0)


def _local_agl_bands(
    center: float,
    *,
    lo: float,
    hi: float,
    step: float,
    radius: float,
) -> list[float]:
    """Fine AGL samples in [center±radius] clipped to [lo, hi]."""
    step = max(float(step), 0.01)
    a = max(float(lo), float(center) - float(radius))
    b = min(float(hi), float(center) + float(radius))
    if b < a - 1e-12:
        return []
    # Align to step grid relative to lo.
    k0 = int(math.ceil((a - lo) / step - 1e-12))
    k1 = int(math.floor((b - lo) / step + 1e-12))
    return [round(lo + k * step, 10) for k in range(k0, k1 + 1)]


def _cruise_band_candidates(belief_map: BeliefMap, mission: Mission) -> list[float]:
    """Cruise AGL candidates for planning.

    For fine steps (<0.2): coarse 0.25 grid over the span, plus a fine neighborhood
    around sticky/clearance (full fine refinement happens after coarse scoring).
    Eval baselines still use the full fine grid via cruise_band_step.
    """
    clearance = float(getattr(mission, "clearance_agl_level", 1.0))
    min_level, max_level = _search_level_bounds(belief_map, mission)
    fine = _cruise_band_step(mission)
    span = _cruise_band_span(mission)
    sticky = getattr(mission, "preferred_cruise_agl", None)
    lo = max(clearance, float(min_level))
    hi = min(clearance + span, float(max_level))
    if fine >= 0.2 - 1e-12:
        return cruise_agl_band_grid(
            clearance, float(min_level), float(max_level), step=fine, span_levels=span, sticky=sticky
        )
    coarse = 0.25
    # Sticky cruise: keep a fine neighborhood for tracking, plus the coarse span so
    # better wind layers remain discoverable (no map-specific height lock-in).
    if sticky is not None and fine < 0.2 - 1e-12:
        bands = [clearance, float(sticky)]
        bands.extend(_local_agl_bands(float(sticky), lo=lo, hi=hi, step=fine, radius=0.35))
        bands.extend(_local_agl_bands(clearance, lo=lo, hi=hi, step=fine, radius=0.15))
        bands.extend(
            cruise_agl_band_grid(
                clearance, float(min_level), float(max_level), step=coarse, span_levels=span, sticky=sticky
            )
        )
        uniq = sorted(set(round(float(z), 10) for z in bands if lo - 1e-12 <= z <= hi + 1e-12))
        return [clamp(z, float(min_level), float(max_level)) for z in uniq]
    bands = cruise_agl_band_grid(
        clearance, float(min_level), float(max_level), step=coarse, span_levels=span, sticky=sticky
    )
    # Always include a fine patch near clearance (often optimal on plains/coast).
    bands.extend(_local_agl_bands(clearance, lo=lo, hi=hi, step=fine, radius=coarse))
    if sticky is not None:
        bands.extend(_local_agl_bands(float(sticky), lo=lo, hi=hi, step=fine, radius=coarse))
    uniq = sorted(set(round(float(z), 10) for z in bands))
    return [clamp(z, float(min_level), float(max_level)) for z in uniq]


def _straight_guide_label(cruise_z: float, clearance: float, step: float = 0.025) -> str:
    eps = max(1e-6, 0.51 * float(step))
    if abs(cruise_z - clearance) < eps:
        return "guide_straight_agl"
    return f"guide_straight_agl_{format_agl_band_token(cruise_z)}"


def _cruise_z_from_guide_label(label: str, clearance: float, mission: Mission | None = None) -> float:
    if label.startswith("guide_straight_agl_"):
        try:
            return float(label.rsplit("_", 1)[-1])
        except ValueError:
            pass
    if label.startswith("guide_straight_agl"):
        return clearance
    preferred = getattr(mission, "preferred_cruise_agl", None) if mission is not None else None
    if preferred is not None:
        return float(preferred)
    return clearance


def _commit_preferred_cruise(
    mission: Mission,
    cruise_z: float,
    clearance: float,
    band_energies: dict[float, float] | None = None,
) -> float:
    """Update preferred cruise AGL.

    Follows the newly selected band when band_energies show a clear improvement
    either up or down. Only ratchet upward when scores are nearly tied.
    """
    step = _cruise_band_step(mission)
    max_level = float(getattr(mission, "max_altitude_level", 4))
    min_level = float(getattr(mission, "min_altitude_level", 0))
    z = snap_cruise_agl(
        float(cruise_z),
        clearance=clearance,
        min_level=min_level,
        max_level=max_level,
        step=step,
    )
    prev = getattr(mission, "preferred_cruise_agl", None)
    if prev is not None and band_energies:
        prev = float(prev)

        def _lookup(level: float) -> float:
            key = min(band_energies.keys(), key=lambda k: abs(k - level))
            return band_energies[key]

        prev_e = _lookup(prev)
        new_e = _lookup(z)
        argmin_z = float(min(band_energies.keys(), key=lambda k: band_energies[k]))
        argmin_e = band_energies[argmin_z]
        # Align with band selector: climb needs ~1.5%, dump needs ~2.5%.
        if z + 1e-9 < prev:
            if new_e > prev_e * 0.975:
                z = prev
            elif z < prev - 0.35 - 1e-9:
                z = prev - 0.35
        elif z > prev + 1e-9 and new_e <= prev_e * 0.985:
            pass  # earned climb
        elif z > prev + 1e-9 and new_e > prev_e * 0.985:
            z = prev
        else:
            if abs(new_e - prev_e) <= prev_e * 0.015:
                z = prev
        if z > argmin_z + 2.0 * step + 1e-9 and _lookup(z) > argmin_e * 1.05 and z >= prev - 1e-9:
            if argmin_z + 1e-9 >= prev and argmin_e <= prev_e * 0.985:
                z = min(argmin_z, prev + 0.35)
        if z > prev + 0.35 + 1e-9:
            z = prev + 0.35
        if z + 1e-9 < prev - 0.35:
            z = prev - 0.35
    elif prev is not None:
        # No scores: allow free update toward newly selected cruise.
        pass
    z = max(z, float(clearance))
    z = snap_cruise_agl(
        z,
        clearance=clearance,
        min_level=min_level,
        max_level=max_level,
        step=step,
    )
    mission.preferred_cruise_agl = z
    return z



def _band_selection_scores(
    start: tuple[float, float, float],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
    bands: list[float],
) -> dict[float, float]:
    """Cruise-fair band scores: wind along-route as if already at cruise + climb fee.

    Comparing full paths from z=0 buries altitude-wind differences under climb cost.
    Scoring cruise-from-altitude + additive climb is terrain-agnostic and makes
    higher-band wind visible to the selector.
    """
    sz = float(start[2])
    climb_cost = float(getattr(mission, "climb_cost_per_level_j", 180.0))
    if not bands:
        return {}
    paths: list[list[tuple[float, float, float]]] = []
    zs: list[float] = []
    for raw_z in bands:
        z = float(raw_z)
        alt_start = (float(start[0]), float(start[1]), z)
        path = _agl_guide_polyline(alt_start, goal, belief_map, mission, via_xy=None, cruise_z=z)
        paths.append(path)
        zs.append(z)
    energies = _polylines_model_energy_j(paths, belief_map, mission)
    scores: dict[float, float] = {}
    for z, path, cruise_e in zip(zs, paths, energies):
        if len(path) <= 1:
            scores[z] = math.inf
        else:
            scores[z] = float(cruise_e) + climb_cost * max(0.0, z - sz)
    return scores


def _select_preferred_cruise_band(
    band_scores: dict[float, float],
    sticky: float | None,
    clearance: float,
    remaining_horiz: float = 0.0,
    tiebreak_scores: dict[float, float] | None = None,
    band_step: float = 0.5,
) -> float | None:
    """Pick cruise AGL from full-path band scores with evidence-gated changes.

    Universal rules (no map-specific thresholds):
    - Leave clearance / climb only if the winner beats the reference by ~1.5%.
    - Dump a sticky layer only with a stronger ~2.5% edge (asymmetric).
    - Rate-limit is applied by the caller; this function picks the evidence target.
    """
    if not band_scores:
        return None
    step = max(float(band_step), 0.01)
    climb_need = 0.985  # ≥1.5% savings to climb
    dump_need = 0.975  # ≥2.5% savings to dump mid-cruise
    best_z = float(min(band_scores.keys(), key=lambda z: band_scores[z]))
    best_e = band_scores[best_z]
    clearance_key = float(min(band_scores.keys(), key=lambda z: abs(float(z) - float(clearance))))
    clearance_e = band_scores[clearance_key]

    if sticky is None or remaining_horiz <= 6.0:
        # Cold start / final approach: stay at clearance unless climb is clearly earned.
        if remaining_horiz <= 6.0:
            if best_z + 1e-9 < clearance_key and best_e <= clearance_e * 0.98:
                return best_z
            if remaining_horiz <= 3.5:
                return float(min(band_scores.keys(), key=lambda z: abs(float(z) - float(clearance_key))))
        if best_e <= clearance_e * climb_need:
            return best_z
        return clearance_key

    sticky_key = float(min(band_scores.keys(), key=lambda z: abs(float(z) - float(sticky))))
    sticky_e = band_scores[sticky_key]

    # Approach: gentle descent only.
    if remaining_horiz <= 10.0:
        if best_z + 1e-9 < sticky_key and best_e <= sticky_e * 0.98:
            return float(min(sticky_key, max(best_z, sticky_key - 0.35)))
        return sticky_key

    # Late cruise: hold unless a strong correction is available (wrong-layer recovery).
    if remaining_horiz <= 16.0:
        if best_e <= sticky_e * dump_need and abs(best_z - sticky_key) > 0.5 * step:
            if best_z > sticky_key:
                return float(min(best_z, sticky_key + 0.35))
            return float(max(best_z, sticky_key - 0.35))
        return sticky_key

    # Early/mid cruise: evidence-gated climb or corrective dump; else hold.
    if best_z >= sticky_key + 0.5 * step and best_e <= sticky_e * climb_need:
        return float(min(best_z, sticky_key + 0.35))
    if best_z + 1e-9 < sticky_key - 0.5 * step and best_e <= sticky_e * dump_need:
        return float(max(best_z, sticky_key - 0.35))
    # Also compare against clearance when sticky is only slightly elevated — blocks
    # noise climbs that never beat clearance by climb_need.
    if sticky_key <= clearance_key + 0.40 + 1e-9:
        if best_z > sticky_key + 1e-9 and best_e > clearance_e * climb_need:
            return sticky_key
        if best_z <= clearance_key + 1e-9 and sticky_e > clearance_e * (1.0 / climb_need):
            # Sticky worse than clearance without having earned the layer → ease down.
            if best_e <= sticky_e * dump_need or best_z <= clearance_key + 1e-9:
                return float(max(clearance_key, sticky_key - 0.35))
    return sticky_key


def _energy_guide_paths(
    start: tuple[float, float, float],
    goal: State3D,
    belief_map: BeliefMap,
    mission: Mission,
) -> list[tuple[str, list[tuple[float, float, float]]]]:
    """Multi-band straight AGL + lateral corridors, all scored later by path energy."""
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    dx, dy = gx - sx, gy - sy
    horiz = math.hypot(dx, dy)
    clearance = float(getattr(mission, "clearance_agl_level", 1.0))
    fine = _cruise_band_step(mission)
    span = _cruise_band_span(mission)
    min_level, max_level = _search_level_bounds(belief_map, mission)
    lo = max(clearance, float(min_level))
    hi = min(clearance + span, float(max_level))
    bands = _cruise_band_candidates(belief_map, mission)
    guides: list[tuple[str, list[tuple[float, float, float]]]] = []
    straight_energy_by_band: dict[float, float] = {}
    pending_zs: list[float] = []
    pending_paths: list[list[tuple[float, float, float]]] = []
    pending_labels: list[str] = []

    def _flush_pending() -> None:
        if not pending_paths:
            return
        energies = _polylines_model_energy_j(pending_paths, belief_map, mission)
        for z, label, path, energy in zip(pending_zs, pending_labels, pending_paths, energies):
            guides.append((label, path))
            if path and len(path) > 1:
                straight_energy_by_band[z] = energy
        pending_zs.clear()
        pending_paths.clear()
        pending_labels.clear()

    def _add_straight(cruise_z: float) -> None:
        z = round(float(cruise_z), 10)
        if z in straight_energy_by_band or z in pending_zs:
            return
        label = _straight_guide_label(z, clearance, step=fine)
        path = _agl_guide_polyline(start, goal, belief_map, mission, via_xy=None, cruise_z=z)
        pending_zs.append(z)
        pending_labels.append(label)
        pending_paths.append(path)

    for cruise_z in bands:
        _add_straight(cruise_z)
    _flush_pending()

    # Hierarchical refine: densify around cheapest full-path bands (not cruise-fair).
    sticky = getattr(mission, "preferred_cruise_agl", None)
    if fine < 0.2 - 1e-12 and straight_energy_by_band:
        path_ranked = sorted(straight_energy_by_band.keys(), key=lambda z: straight_energy_by_band[z])
        if sticky is not None:
            centers = [float(sticky), clearance]
            # Keep coarse-span winners discoverable while sticky (layer search).
            coarse = 0.25
            for z in path_ranked:
                zf = float(z)
                if abs(zf / coarse - round(zf / coarse)) <= 1e-9:
                    centers.append(zf)
                if len([c for c in centers if abs(c / coarse - round(c / coarse)) <= 1e-9]) >= 4:
                    break
        else:
            centers = [float(z) for z in path_ranked[:5]]
            # Seed mid-span so intermediate bands remain reachable from coarse cells.
            for z in (clearance + 0.25, clearance + 0.5, clearance + 0.75, clearance + 1.0):
                if lo - 1e-12 <= z <= hi + 1e-12:
                    centers.append(float(z))
        for center in centers:
            for z in _local_agl_bands(center, lo=lo, hi=hi, step=fine, radius=0.35 if sticky is None else 0.25):
                _add_straight(z)
        _flush_pending()

    if horiz < 6.0 or getattr(mission, "elevation", None) is None:
        return guides

    # Corridor cruise follows the cheapest full-path straight band.
    best_band = min(straight_energy_by_band, key=straight_energy_by_band.get) if straight_energy_by_band else bands[0]
    if sticky is not None and straight_energy_by_band:
        sticky_key = min(straight_energy_by_band.keys(), key=lambda z: abs(z - float(sticky)))
        if straight_energy_by_band[sticky_key] <= straight_energy_by_band[best_band] * 1.012:
            best_band = sticky_key
    straight_floor = straight_energy_by_band.get(best_band, math.inf)
    if not math.isfinite(straight_floor) and straight_energy_by_band:
        straight_floor = min(straight_energy_by_band.values())

    # Sticky via competes with straights; drop it if it no longer beats the floor.
    locked = getattr(mission, "guide_via", None)
    cell_m = max(float(mission.step_distance_m), 1e-6)
    # Risk-adjusted corridor gate (belief noise tax). Off unless explicitly enabled.
    corridor_margin = getattr(mission, "corridor_energy_margin", None)
    if locked is not None:
        if math.hypot(float(locked[0]) - sx, float(locked[1]) - sy) < (125.0 / cell_m):
            mission.guide_via = None
        else:
            locked_path = _agl_guide_polyline(
                start,
                goal,
                belief_map,
                mission,
                via_xy=(float(locked[0]), float(locked[1])),
                cruise_z=best_band,
            )
            if corridor_margin is not None and len(locked_path) > 1:
                locked_e = _polyline_risk_adjusted_energy_j(locked_path, belief_map, mission, uncertainty_gain=0.12)
                if locked_e <= straight_floor * float(corridor_margin):
                    guides.append(("guide_corridor_locked", locked_path))
                else:
                    mission.guide_via = None
            else:
                mission.guide_via = None

    # Lateral corridors: risk-adjusted belief energy must beat straight by a clear margin.
    if corridor_margin is None or float(corridor_margin) <= 0.0:
        return guides
    corridor_margin = float(corridor_margin)
    ux, uy = dx / horiz, dy / horiz
    px, py = -uy, ux
    direct_rise = terrain_climb_along_line_m(mission.elevation, sx, sy, gx, gy, samples=8)
    offsets = tuple(offset_m / cell_m for offset_m in (150.0, 250.0, 350.0))
    detour_max_cells = 250.0 / cell_m
    corridor_candidates: list[tuple[str, list[tuple[float, float, float]]]] = []
    for sign in (-1.0, 1.0):
        for offset in offsets:
            mx = clamp(sx + 0.5 * dx + sign * offset * px, 0.0, belief_map.width - 1)
            my = clamp(sy + 0.5 * dy + sign * offset * py, 0.0, belief_map.height - 1)
            via_rise = terrain_climb_along_line_m(mission.elevation, sx, sy, mx, my, samples=4)
            via_rise += terrain_climb_along_line_m(mission.elevation, mx, my, gx, gy, samples=4)
            if direct_rise > 1.0 and via_rise > direct_rise * 1.05:
                continue
            detour = (math.hypot(mx - sx, my - sy) + math.hypot(gx - mx, gy - my)) - horiz
            if detour > detour_max_cells:
                continue
            path = _agl_guide_polyline(start, goal, belief_map, mission, via_xy=(mx, my), cruise_z=best_band)
            if len(path) <= 1:
                continue
            label = f"guide_corridor_{sign:+.0f}_{offset:.1f}"
            corridor_candidates.append((label, path))
    if corridor_candidates:
        for label, path in corridor_candidates:
            energy = _polyline_risk_adjusted_energy_j(path, belief_map, mission, uncertainty_gain=0.12)
            if straight_floor < math.inf and energy > straight_floor * corridor_margin:
                continue
            # Raw model energy must not exceed straight (blocks pure uncertainty gaming).
            raw = _polyline_model_energy_j(path, belief_map, mission)
            if straight_floor < math.inf and raw > straight_floor * 1.0:
                continue
            guides.append((label, path))
    return guides


def _run_anytime_continuous_mpc(
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float,
    safety_weight: float,
    anytime_rounds: int,
    heuristic_weight_start: float,
    heuristic_weight_end: float,
    search_node_budget: int,
    horizon_steps: int,
    beam_width: int,
    branch_width: int,
    discount_factor: float,
    terminal_progress_weight: float,
    return_cost_map: dict[State3D, float] | None,
    candidate_mode: str,
) -> dict:
    weights = _weight_schedule(anytime_rounds, heuristic_weight_start, heuristic_weight_end)
    start = _continuous_state_tuple(mission.start)
    goal = normalize_state(mission.goal)
    best_path: list[tuple[float, float, float]] = [start]
    best_value = -math.inf
    best_frontier_point = normalize_state(start)
    best_frontier_score = heuristic(best_frontier_point, goal, mission)
    candidates: list[dict] = []
    for round_index, weight in enumerate(weights):
        round_beam_width = max(6, int(beam_width * (0.65 + 0.35 * ((round_index + 1) / max(len(weights), 1)))))
        round_branch_width = max(4, int(branch_width * (0.75 + 0.25 * ((round_index + 1) / max(len(weights), 1)))))
        round_horizon = max(2, min(horizon_steps + round_index, mission.max_steps))
        remaining_budget = max(200, search_node_budget // max(len(weights) - round_index, 1))
        result = _beam_search_continuous_mpc(
            belief_map=belief_map,
            mission=mission,
            risk_weight=risk_weight,
            safety_weight=safety_weight,
            node_budget=remaining_budget,
            horizon_steps=round_horizon,
            beam_width=round_beam_width,
            branch_width=round_branch_width,
            discount_factor=discount_factor,
            terminal_progress_weight=terminal_progress_weight * weight,
            return_cost_map=return_cost_map,
        )
        chosen_path = result["path"] or [_continuous_state_tuple(mission.start)]
        chosen_cost = _path_cost(chosen_path, belief_map, mission, risk_weight, safety_weight)
        candidates.append(
            {
                "round": round_index + 1,
                "mode": candidate_mode,
                "heuristic_weight": weight,
                "path": chosen_path[: mission.max_steps + 1],
                "path_cost": chosen_cost,
                "path_cost_breakdown": _path_cost_breakdown(chosen_path, belief_map, mission, risk_weight, safety_weight),
                "goal_reached": bool(chosen_path and _continuous_goal_reached(chosen_path[-1], goal)),
                "horizon_steps": round_horizon,
                "beam_width": round_beam_width,
            }
        )
        if result["value"] > best_value:
            best_value = result["value"]
            best_path = chosen_path
            best_frontier_point = normalize_state(chosen_path[-1]) if chosen_path else normalize_state(start)
            best_frontier_score = heuristic(best_frontier_point, goal, mission)
        if best_path and _continuous_goal_reached(best_path[-1], goal):
            break
    return {
        "path": best_path[: mission.max_steps + 1],
        "candidates": candidates,
        "best_frontier_point": best_frontier_point,
    }


def _beam_search_continuous_mpc(
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float,
    safety_weight: float,
    node_budget: int,
    horizon_steps: int,
    beam_width: int,
    branch_width: int,
    discount_factor: float,
    terminal_progress_weight: float,
    return_cost_map: dict[State3D, float] | None,
) -> dict:
    start = _continuous_state_tuple(mission.start)
    goal = normalize_state(mission.goal)
    start_heading = math.atan2(goal[1] - start[1], goal[0] - start[0]) if math.hypot(goal[0] - start[0], goal[1] - start[1]) > 1e-6 else 0.0
    min_level, max_level = _search_level_bounds(belief_map, mission)
    dt_s = mission.step_distance_m / max(mission.nominal_airspeed, 1e-6)
    beam = [{
        "state": DroneState(x=start[0], y=start[1], z=start[2], heading_rad=start_heading, battery_ratio=1.0, airspeed=mission.nominal_airspeed),
        "path": [start],
        "value": 0.0,
        "cost": 0.0,
        "goal_reached": _continuous_goal_reached(start, goal),
    }]
    expanded = 0
    best_candidate = {
        "state": start,
        "path": [start],
        "value": -math.inf,
        "cost": 0.0,
        "goal_reached": _continuous_goal_reached(start, goal),
    }

    for depth in range(max(1, horizon_steps)):
        if expanded >= node_budget or not beam:
            break
        next_beam: list[dict] = []
        for candidate in beam:
            current = candidate["state"]
            if candidate["goal_reached"]:
                next_beam.append(candidate)
                continue
            options = _rank_continuous_successors(
                belief_map=belief_map,
                mission=mission,
                current=current,
                path_history=candidate["path"],
                goal=goal,
                risk_weight=risk_weight,
                safety_weight=safety_weight,
                return_cost_map=return_cost_map,
                branch_width=branch_width,
                spent_cost=candidate["cost"],
                dt_s=dt_s,
                min_level=min_level,
                max_level=max_level,
            )
            expanded += len(options)
            for option in options:
                step_cost = option["step_cost"]
                step_reward = option["step_reward"]
                new_path = candidate["path"] + [option["point"]]
                new_value = candidate["value"] + (discount_factor ** depth) * step_reward
                new_cost = candidate["cost"] + step_cost
                goal_reached = _continuous_goal_reached(option["point"], goal)
                terminal_bonus = 0.0
                if goal_reached:
                    terminal_bonus += terminal_progress_weight
                elif depth == horizon_steps - 1:
                    terminal_bonus += _terminal_bonus(normalize_state(option["point"]), goal, mission, terminal_progress_weight)
                next_beam.append(
                    {
                        "state": option["state"],
                        "path": new_path,
                        "value": new_value + terminal_bonus,
                        "cost": new_cost,
                        "goal_reached": goal_reached,
                    }
                )
        if not next_beam:
            break
        next_beam.sort(
            key=lambda item: (
                item["goal_reached"],
                item["value"],
                -heuristic(normalize_state((item["state"].x, item["state"].y, item["state"].z)), goal, mission),
            ),
            reverse=True,
        )
        beam = next_beam[:beam_width]
        if beam:
            frontier_best = max(beam, key=lambda item: (item["goal_reached"], item["value"], len(item["path"])))
            if (
                frontier_best["goal_reached"] and not best_candidate["goal_reached"]
            ) or (
                frontier_best["goal_reached"] == best_candidate["goal_reached"]
                and frontier_best["value"] > best_candidate["value"]
            ):
                best_candidate = frontier_best
        if any(item["goal_reached"] for item in beam):
            best_candidate = max(beam, key=lambda item: (item["goal_reached"], item["value"]))
            break

    return {
        "path": best_candidate["path"][: mission.max_steps + 1],
        "value": best_candidate["value"],
    }


def _reconstruct_path(
    came_from: dict[State3D, State3D | None],
    current: State3D,
    max_steps: int,
) -> list[State3D]:
    if current not in came_from:
        return []
    path = []
    cursor: State3D | None = current
    while cursor is not None:
        path.append(cursor)
        cursor = came_from.get(cursor)
    path.reverse()
    return path[: max_steps + 1]


def _weight_schedule(rounds: int, start: float, end: float) -> list[float]:
    if rounds <= 1:
        return [end]
    return [start + (end - start) * (idx / (rounds - 1)) for idx in range(rounds)]


def _continuous_state_tuple(point: tuple[int, int] | tuple[int, int, int] | tuple[float, float, float]) -> tuple[float, float, float]:
    if len(point) == 2:
        return float(point[0]), float(point[1]), 0.0
    return float(point[0]), float(point[1]), float(point[2])


def _continuous_goal_reached(point: tuple[float, float, float], goal: State3D) -> bool:
    return math.hypot(point[0] - goal[0], point[1] - goal[1]) <= 0.45 and abs(point[2] - goal[2]) <= 0.55


def _terminal_bonus(point: State3D, goal: State3D, mission: Mission, terminal_progress_weight: float) -> float:
    distance = heuristic(point, goal, mission)
    return terminal_progress_weight / (1.0 + distance / max(mission.step_distance_m, 1e-6))


def path_distance_m(
    path: list[tuple[int, int] | tuple[int, int, int] | tuple[float, float, float]],
    step_distance_m: float,
    altitude_step_m: float | None = None,
) -> float:
    distance = 0.0
    # Default: isotropic cube (Z step matches XY) when altitude_step_m omitted.
    alt_m = float(step_distance_m if altitude_step_m is None else altitude_step_m)
    for (x1, y1, z1), (x2, y2, z2) in zip(
        [_continuous_state_tuple(point) for point in path],
        [_continuous_state_tuple(point) for point in path[1:]],
    ):
        horizontal = math.hypot(x2 - x1, y2 - y1) * step_distance_m
        vertical = abs(z2 - z1) * alt_m
        distance += math.hypot(horizontal, vertical)
    return distance


def _path_cost(
    path: list[tuple[float, float, float]],
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float,
    safety_weight: float,
) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for current, nxt in zip(path, path[1:]):
        local = _sample_belief_state(belief_map, nxt[0], nxt[1], nxt[2])
        discrete_current = normalize_state(current)
        discrete_next = normalize_state(nxt)
        total += transition_energy_j(
            airspeed=mission.nominal_airspeed,
            current=current,
            nxt=nxt,
            local_u=local["wind_u"],
            local_v=local["wind_v"],
            local_w=local["wind_w"],
            step_distance_m=mission.step_distance_m,
            altitude_step_m=mission.altitude_step_m,
            climb_cost_per_level_j=mission.climb_cost_per_level_j,
            hover_power_w=mission.hover_power_w,
            cruise_power_w=mission.cruise_power_w,
            hotel_power_w=mission.hotel_power_w,
            headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
            climb_power_per_mps_w=mission.climb_power_per_mps_w,
            descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
            terrain_dz_m=_mission_terrain_dz(mission, current[0], current[1], nxt[0], nxt[1]),
        )
        total += 22.0 * risk_weight * local["uncertainty"] + 28.0 * safety_weight * local["safety_penalty"]
        total -= 18.0 * local["expected_energy_gain"]
        total += 10.0 * abs(discrete_next[2] - discrete_current[2])
    return total


def _path_cost_breakdown(
    path: list[tuple[float, float, float]],
    belief_map: BeliefMap,
    mission: Mission,
    risk_weight: float,
    safety_weight: float,
) -> dict[str, float]:
    totals = {
        "energy_j": 0.0,
        "progress_reward_j": 0.0,
        "uncertainty_cost_j": 0.0,
        "safety_cost_j": 0.0,
        "altitude_bias_j": 0.0,
        "vertical_maneuver_cost_j": 0.0,
        "total_cost_j": 0.0,
    }
    for current, nxt in zip(path, path[1:]):
        local = _sample_belief_state(belief_map, nxt[0], nxt[1], nxt[2])
        energy_j = transition_energy_j(
            airspeed=mission.nominal_airspeed,
            current=current,
            nxt=nxt,
            local_u=local["wind_u"],
            local_v=local["wind_v"],
            local_w=local["wind_w"],
            step_distance_m=mission.step_distance_m,
            altitude_step_m=mission.altitude_step_m,
            climb_cost_per_level_j=mission.climb_cost_per_level_j,
            hover_power_w=mission.hover_power_w,
            cruise_power_w=mission.cruise_power_w,
            hotel_power_w=mission.hotel_power_w,
            headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
            climb_power_per_mps_w=mission.climb_power_per_mps_w,
            descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
            terrain_dz_m=_mission_terrain_dz(mission, current[0], current[1], nxt[0], nxt[1]),
        )
        progress_reward = 18.0 * local["expected_energy_gain"]
        uncertainty_cost = 22.0 * risk_weight * local["uncertainty"]
        safety_cost = 28.0 * safety_weight * local["safety_penalty"]
        altitude_bias = 8.0 * abs(goal_altitude_gap(normalize_state(current), mission)) - 8.0 * abs(goal_altitude_gap(normalize_state(nxt), mission))
        terrain_levels = _mission_terrain_dz(mission, current[0], current[1], nxt[0], nxt[1]) / max(mission.altitude_step_m, 1e-6)
        vertical_cost = 10.0 * abs(nxt[2] - current[2]) + 8.0 * max(0.0, terrain_levels)
        total = energy_j - progress_reward + uncertainty_cost + safety_cost + altitude_bias + vertical_cost
        totals["energy_j"] += energy_j
        totals["progress_reward_j"] += progress_reward
        totals["uncertainty_cost_j"] += uncertainty_cost
        totals["safety_cost_j"] += safety_cost
        totals["altitude_bias_j"] += altitude_bias
        totals["vertical_maneuver_cost_j"] += vertical_cost
        totals["total_cost_j"] += total
    return totals


def _rank_continuous_successors(
    belief_map: BeliefMap,
    mission: Mission,
    current: DroneState,
    path_history: list[tuple[float, float, float]],
    goal: State3D,
    risk_weight: float,
    safety_weight: float,
    return_cost_map: dict[State3D, float] | None,
    branch_width: int,
    spent_cost: float,
    dt_s: float,
    min_level: int,
    max_level: int,
) -> list[dict]:
    local = _sample_belief_state(belief_map, current.x, current.y, current.z)
    floor = _effective_agl_floor(mission, horizontal_to_goal=math.hypot(goal[0] - current.x, goal[1] - current.y))
    effective_min = max(min_level, int(math.floor(floor)))
    preferred_z = _preferred_cruise_altitude(
        belief_map,
        current.x,
        current.y,
        goal,
        effective_min,
        max_level,
        current_z=current.z,
        climb_cost_per_level_j=mission.climb_cost_per_level_j,
        clearance_agl_level=getattr(mission, "clearance_agl_level", 1.0),
        preferred_cruise_agl=getattr(mission, "preferred_cruise_agl", None),
    )
    preferred_z = max(preferred_z, floor)
    controls = _control_library(
        current,
        goal,
        local,
        mission,
        preferred_z=preferred_z,
        belief_map=belief_map,
    )
    bearing = math.atan2(goal[1] - current.y, goal[0] - current.x) if math.hypot(goal[0] - current.x, goal[1] - current.y) > 1e-6 else current.heading_rad
    horizontal = math.hypot(goal[0] - current.x, goal[1] - current.y)
    level_lo = float(effective_min if horizontal > 5.0 else min_level)
    level_hi = float(max_level)

    # Phase 1: propagate controls (control-flow stays Python).
    proposals: list[dict] = []
    for control in controls:
        next_state, aero_energy = propagate_control(
            current,
            control,
            dt_s=dt_s,
            local_u=local["wind_u"],
            local_v=local["wind_v"],
            local_w=local["wind_w"],
            step_distance_m=mission.step_distance_m,
            altitude_step_m=mission.altitude_step_m,
            min_altitude_level=effective_min if horizontal > 5.0 else min_level,
            max_altitude_level=max_level,
        )
        point = (
            clamp(next_state.x, 0.0, belief_map.width - 1),
            clamp(next_state.y, 0.0, belief_map.height - 1),
            clamp(next_state.z, level_lo, level_hi),
        )
        if horizontal > 5.0 and goal[2] <= current.z + 0.25 and point[2] > preferred_z + 0.85:
            continue
        proposals.append(
            {
                "state": next_state,
                "point": point,
                "aero_energy": float(aero_energy),
                "control": control,
            }
        )
    if not proposals:
        return []

    # Phase 2: batch belief sample + terrain-aware transition energy.
    points = np.asarray([item["point"] for item in proposals], dtype=np.float64)
    next_locals = _sample_belief_states_batch(belief_map, points[:, 0], points[:, 1], points[:, 2])
    elev = getattr(mission, "elevation", None)
    if elev is not None:
        terrain_dz = terrain_delta_m_batch(
            elev,
            np.full(len(proposals), current.x),
            np.full(len(proposals), current.y),
            points[:, 0],
            points[:, 1],
        )
    else:
        terrain_dz = np.zeros(len(proposals), dtype=np.float64)
    currents = np.tile(np.asarray([current.x, current.y, current.z], dtype=np.float64), (len(proposals), 1))
    terrain_energies = transition_energy_batch(
        airspeed=current.airspeed,
        currents=currents,
        nexts=points,
        local_u=np.full(len(proposals), local["wind_u"]),
        local_v=np.full(len(proposals), local["wind_v"]),
        local_w=np.full(len(proposals), local["wind_w"]),
        step_distance_m=mission.step_distance_m,
        altitude_step_m=mission.altitude_step_m,
        climb_cost_per_level_j=mission.climb_cost_per_level_j,
        hover_power_w=mission.hover_power_w,
        cruise_power_w=mission.cruise_power_w,
        hotel_power_w=mission.hotel_power_w,
        headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
        climb_power_per_mps_w=mission.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
        terrain_dz_m=terrain_dz,
    )

    # Phase 3: batch residual energies + along-track wind (once for all proposals).
    n_prop = len(proposals)
    current_energy_to_goal = estimate_energy_to_goal_j(
        (current.x, current.y, current.z), goal, mission, local, belief_map=belief_map
    )
    next_pts = [item["point"] for item in proposals]
    next_winds = [{key: float(next_locals[key][i]) for key in next_locals} for i in range(n_prop)]
    next_energies = estimate_energies_to_goal_batch(
        next_pts, goal, mission, belief_map=belief_map, local_winds=next_winds
    )

    segs = np.empty((n_prop * 3, 5), dtype=np.float64)
    look_cells_arr = np.empty(n_prop, dtype=np.float64)
    track_bearing_arr = np.empty(n_prop, dtype=np.float64)
    step_len_arr = np.empty(n_prop, dtype=np.float64)
    goal_look = min(4.0, max(horizontal, 1.0))
    goal_ax = current.x + math.cos(bearing) * goal_look
    goal_ay = current.y + math.sin(bearing) * goal_look
    for i, point in enumerate(next_pts):
        step_dx = point[0] - current.x
        step_dy = point[1] - current.y
        step_len = math.hypot(step_dx, step_dy)
        step_len_arr[i] = step_len
        track_bearing = math.atan2(step_dy, step_dx) if step_len > 1e-6 else bearing
        track_bearing_arr[i] = track_bearing
        look_cells = min(4.0, max(2.0, horizontal * 0.35))
        look_cells_arr[i] = look_cells
        ahead_x = point[0] + math.cos(track_bearing) * look_cells
        ahead_y = point[1] + math.sin(track_bearing) * look_cells
        segs[i * 3 + 0] = (current.x, current.y, point[0], point[1], point[2])
        segs[i * 3 + 1] = (point[0], point[1], ahead_x, ahead_y, point[2])
        segs[i * 3 + 2] = (current.x, current.y, goal_ax, goal_ay, point[2])
    track_tw, track_w = _integrate_tracks_batch(belief_map, segs, samples=2)

    terrain_rises = np.empty(n_prop, dtype=np.float64)
    if elev is not None:
        terrain_rises[:] = terrain_climb_along_line_batch(
            elev,
            np.full(n_prop, current.x),
            np.full(n_prop, current.y),
            points[:, 0],
            points[:, 1],
            samples=2,
        )
    else:
        terrain_rises.fill(0.0)

    scored: list[dict] = []
    near_goal = horizontal <= 5.0
    final_approach = horizontal <= 3.5
    for i, item in enumerate(proposals):
        next_state = item["state"]
        point = item["point"]
        aero_energy = item["aero_energy"]
        control = item["control"]
        next_local = next_winds[i]
        terrain_rise_m = float(terrain_rises[i])
        terrain_energy = float(terrain_energies[i])
        step_energy = 0.55 * aero_energy + 0.45 * terrain_energy

        energy_progress_reward = current_energy_to_goal - float(next_energies[i])

        xy_progress = _goal_axis_progress(current, point, goal)
        altitude_progress = abs(goal[2] - current.z) - abs(goal[2] - point[2])
        alt_weight = 75.0 if final_approach else (20.0 if near_goal else 4.0)
        goal_progress_reward = 82.0 * xy_progress + alt_weight * altitude_progress

        step_len = float(step_len_arr[i])
        seg_tw = float(track_tw[i * 3 + 0])
        seg_w = float(track_w[i * 3 + 0])
        lookahead_tw = float(track_tw[i * 3 + 1])
        lookahead_w = float(track_w[i * 3 + 1])
        goal_tw = float(track_tw[i * 3 + 2])
        goal_w = float(track_w[i * 3 + 2])
        blend_tw = 0.40 * seg_tw + 0.35 * lookahead_tw + 0.25 * goal_tw
        blend_w = 0.40 * seg_w + 0.35 * lookahead_w + 0.25 * goal_w
        wind_assist = (
            70.0 * blend_tw
            - 38.0 * max(0.0, -blend_tw)
            + 48.0 * max(0.0, blend_w)
            - 36.0 * max(0.0, -blend_w)
        )
        if not near_goal:
            band_progress = abs(current.z - preferred_z) - abs(point[2] - preferred_z)
            wind_assist += 40.0 * band_progress - 12.0 * abs(point[2] - preferred_z)
            free_lift = max(0.0, next_local["wind_w"] - 0.1)
            climb_dz = point[2] - current.z
            if abs(current.z - preferred_z) <= 0.3 and abs(point[2] - preferred_z) <= 0.35:
                wind_assist += 18.0 - 40.0 * abs(climb_dz)
            elif climb_dz > 0.05:
                if point[2] <= preferred_z + 0.35 and (free_lift > 0.08 or blend_tw >= -0.2):
                    wind_assist += 28.0 * max(free_lift, 0.05) * climb_dz
                elif point[2] > preferred_z + 0.2 or blend_tw < -0.5:
                    wind_assist -= 90.0 * max(climb_dz, point[2] - preferred_z)
                else:
                    wind_assist -= 50.0 * climb_dz
            if terrain_rise_m > 2.0:
                terrain_cost = (
                    terrain_rise_m / max(mission.altitude_step_m, 1e-6)
                ) * mission.climb_cost_per_level_j
                wind_payoff = max(0.0, 55.0 * blend_tw + 40.0 * max(0.0, blend_w))
                if wind_payoff < 1.25 * terrain_cost:
                    wind_assist -= terrain_cost - 0.4 * wind_payoff
                else:
                    wind_assist -= 0.25 * terrain_cost

        cross_track = _cross_track_cells(mission.start, goal, point[0], point[1])
        if not near_goal and cross_track > 0.75:
            wind_assist -= 14.0 * max(0.0, cross_track - 0.75)
        if step_len > 1e-6 and not near_goal:
            wasted = max(0.0, step_len - max(0.0, xy_progress))
            wind_assist -= 36.0 * wasted
        if not near_goal and xy_progress < 0.20:
            wind_assist -= 55.0 * max(0.0, blend_w)
            energy_progress_reward = min(energy_progress_reward, 10.0)

        risk_cost = 30.0 * risk_weight * next_local["uncertainty"] + 36.0 * safety_weight * next_local["safety_penalty"]
        revisit_penalty = _continuous_revisit_penalty(path_history, point) * 0.25
        speed_penalty = 8.0 * abs(next_state.airspeed - DEFAULT_ENVELOPE.best_glide_speed)
        step_cost = step_energy + risk_cost + speed_penalty + revisit_penalty

        return_budget = mission.max_return_cost_j
        if return_cost_map is not None and return_budget is not None:
            effective_budget = return_budget * (1.45 if near_goal else 1.15)
            return_cost = return_cost_map.get(normalize_state(point), math.inf)
            if return_cost > effective_budget:
                continue
            if spent_cost + step_cost + return_cost > effective_budget:
                continue

        if final_approach and point[2] > current.z + 0.05:
            goal_progress_reward -= 140.0 * (point[2] - current.z)

        uplift_scale = 0.08 if final_approach else (0.45 if near_goal else 1.15)
        belief_energy_bonus = uplift_scale * (
            40.0 * next_local["expected_energy_gain"]
            + 55.0 * next_local["mode_prob_uplift"]
            - 30.0 * next_local["mode_prob_sink"]
        )
        if not near_goal and xy_progress < 0.20:
            belief_energy_bonus *= 0.15
        step_reward = energy_progress_reward + goal_progress_reward + belief_energy_bonus + wind_assist - step_cost
        scored.append(
            {
                "state": next_state,
                "point": point,
                "step_cost": step_cost,
                "step_reward": step_reward,
                "score": step_reward,
                "control": control,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:branch_width]


def _continuous_revisit_penalty(
    path_history: list[tuple[float, float, float]],
    point: tuple[float, float, float],
    lookback: int = 6,
) -> float:
    recent_points = path_history[-lookback:]
    repeats = sum(1 for prev in recent_points if math.hypot(prev[0] - point[0], prev[1] - point[1]) <= 0.75 and abs(prev[2] - point[2]) <= 0.6)
    if repeats <= 0:
        return 0.0
    return 42.0 * repeats


def _goal_axis_progress(
    current: DroneState,
    point: tuple[float, float, float],
    goal: State3D,
) -> float:
    """Euclidean progress toward the goal (cells). Sideways motion gets no credit."""
    dx_goal = goal[0] - current.x
    dy_goal = goal[1] - current.y
    goal_dist = math.hypot(dx_goal, dy_goal)
    if goal_dist <= 1e-6:
        return 0.0
    dx_step = point[0] - current.x
    dy_step = point[1] - current.y
    return max(0.0, (dx_step * dx_goal + dy_step * dy_goal) / goal_dist)


def _cross_track_cells(
    start: tuple[float, float] | tuple[float, float, float] | tuple[int, int] | tuple[int, int, int],
    goal: State3D,
    x: float,
    y: float,
) -> float:
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    dx, dy = gx - sx, gy - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return math.hypot(x - sx, y - sy)
    # Absolute cross product / length = distance to line.
    return abs((x - sx) * dy - (y - sy) * dx) / length


def _lookahead_wind(
    belief_map: BeliefMap,
    x: float,
    y: float,
    z: float,
    bearing: float,
    cells: float = 3.0,
) -> tuple[float, float]:
    """Average along-track wind a few cells ahead of (x, y)."""
    return _integrate_track_wind(
        belief_map,
        x,
        y,
        x + math.cos(bearing) * cells,
        y + math.sin(bearing) * cells,
        z,
        samples=4,
    )


def _integrate_track_wind(
    belief_map: BeliefMap,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z: float,
    samples: int = 5,
) -> tuple[float, float]:
    """Integrate along-track tailwind and vertical wind over a segment."""
    tw, ww = _integrate_tracks_batch(
        belief_map,
        np.asarray([[x0, y0, x1, y1, z]], dtype=np.float64),
        samples=samples,
    )
    return float(tw[0]), float(ww[0])


def _integrate_tracks_batch(
    belief_map: BeliefMap,
    segs: np.ndarray,
    samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch along-track wind integrate.

    ``segs`` shape (N, 5): x0,y0,x1,y1,z → returns (tw[N], w[N]).
    """
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 5)
    n = segs.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    n_samp = max(2, int(samples))
    x0 = segs[:, 0]
    y0 = segs[:, 1]
    x1 = segs[:, 2]
    y1 = segs[:, 3]
    z = segs[:, 4]
    dx = x1 - x0
    dy = y1 - y0
    span = np.hypot(dx, dy)
    bearings = np.arctan2(dy, dx)
    bearings = np.where(span > 1e-9, bearings, 0.0)
    t = np.linspace(0.0, 1.0, n_samp, dtype=np.float64)[None, :]
    xs = np.clip(x0[:, None] + dx[:, None] * t, 0.0, float(belief_map.width - 1)).reshape(-1)
    ys = np.clip(y0[:, None] + dy[:, None] * t, 0.0, float(belief_map.height - 1)).reshape(-1)
    zs = np.repeat(z, n_samp)
    wind = _sample_wind_batch(belief_map, xs, ys, zs)
    wu = wind["wind_u"].reshape(n, n_samp)
    wv = wind["wind_v"].reshape(n, n_samp)
    ww = wind["wind_w"].reshape(n, n_samp)
    cos_b = np.cos(bearings)[:, None]
    sin_b = np.sin(bearings)[:, None]
    tw = np.mean(wu * cos_b + wv * sin_b, axis=1)
    w_mean = np.mean(ww, axis=1)
    return tw, w_mean


def _sample_wind_batch(belief_map: BeliefMap, xs, ys, zs) -> dict[str, np.ndarray]:
    arrays = belief_map.field_arrays
    if arrays is None or "wind_u" not in arrays:
        arrays = ensure_belief_field_arrays(belief_map, _WIND_ATTRS)
    xs_a = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys_a = np.asarray(ys, dtype=np.float64).reshape(-1)
    zs_a = np.asarray(zs, dtype=np.float64).reshape(-1)
    return {attr: trilinear_sample_batch(arrays[attr], xs_a, ys_a, zs_a) for attr in _WIND_ATTRS}


def _heading_turn_rate(current_heading: float, desired_heading: float) -> float:
    err = math.atan2(math.sin(desired_heading - current_heading), math.cos(desired_heading - current_heading))
    return clamp(err, -DEFAULT_ENVELOPE.max_turn_rate_rad_s, DEFAULT_ENVELOPE.max_turn_rate_rad_s)


def _control_library(
    current: DroneState,
    goal: State3D,
    local: dict[str, float],
    mission: Mission,
    preferred_z: float | None = None,
    belief_map: BeliefMap | None = None,
) -> list[dict[str, float | str]]:
    goal_heading = math.atan2(goal[1] - current.y, goal[0] - current.x) if math.hypot(goal[0] - current.x, goal[1] - current.y) > 1e-6 else current.heading_rad
    base_turn = _heading_turn_rate(current.heading_rad, goal_heading)
    vertical_gap = goal[2] - current.z
    horizontal = math.hypot(goal[0] - current.x, goal[1] - current.y)
    # Final approach: only steer toward goal XY and force descent.
    if horizontal <= 3.5 and vertical_gap < -0.15:
        return [
            {"mode": "approach", "turn_rate_rad_s": base_turn, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 2.0, "climb_bias": -1.6},
            {"mode": "approach", "turn_rate_rad_s": base_turn * 0.7, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 3.5, "climb_bias": -2.0},
            {"mode": "approach", "turn_rate_rad_s": base_turn * 1.1, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 1.0, "climb_bias": -1.3},
            {"mode": "cruise", "turn_rate_rad_s": base_turn, "airspeed": DEFAULT_ENVELOPE.best_glide_speed, "climb_bias": -1.0},
        ]

    if horizontal <= 5.0:
        target_z = goal[2]
    elif goal[2] > current.z + 0.25:
        base = preferred_z if preferred_z is not None else current.z
        target_z = max(base, min(goal[2], current.z + 1.25))
    else:
        # Follow energy-optimal preferred band (may be low — no forced clearance).
        target_z = preferred_z if preferred_z is not None else current.z
    band_gap = float(target_z) - current.z
    bearing_tw = local["wind_u"] * math.cos(goal_heading) + local["wind_v"] * math.sin(goal_heading)
    wind_speed = math.hypot(local["wind_u"], local["wind_v"])
    free_lift = max(0.0, local["wind_w"] - 0.12)
    # Ahead terrain rise on the goal ray — if costly, favor lateral probes / hold.
    ahead_rise = 0.0
    if getattr(mission, "elevation", None) is not None and horizontal > 5.0:
        look = min(3.0, horizontal)
        ahead_x = current.x + math.cos(goal_heading) * look
        ahead_y = current.y + math.sin(goal_heading) * look
        ahead_rise = terrain_climb_along_line_m(
            mission.elevation, current.x, current.y, ahead_x, ahead_y, samples=4
        )
    if abs(band_gap) <= 0.25 and horizontal > 5.0:
        # Hold the chosen band; cancel residual uplift so we don't ratchet up for free.
        hold_bias = -(max(0.0, local["wind_w"] - 0.50)) / 0.75
        cruise_bias = clamp(hold_bias, -0.85, 0.1)
    elif band_gap > 0.25:
        cruise_bias = clamp(0.45 + 0.5 * min(band_gap, 1.5), 0.25, 1.15)
        if bearing_tw < -0.6:
            cruise_bias = min(cruise_bias, 0.45)
    else:
        # Preferred is below: descend when the optimizer asked for it.
        cruise_bias = clamp(band_gap * 0.65, -1.05, -0.1)
        if horizontal <= 8.0:
            cruise_bias = min(cruise_bias, -0.35 + 0.45 * band_gap)
    if ahead_rise > 25.0 and bearing_tw < 0.35:
        cruise_bias = min(cruise_bias, 0.15)

    controls: list[dict[str, float | str]] = [
        {"mode": "cruise", "turn_rate_rad_s": base_turn, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 1.5, "climb_bias": cruise_bias},
        {"mode": "cruise", "turn_rate_rad_s": base_turn * 0.65, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 3.2, "climb_bias": cruise_bias},
        {"mode": "cruise", "turn_rate_rad_s": base_turn * 1.1, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 0.4, "climb_bias": cruise_bias},
        {"mode": "energy_save", "turn_rate_rad_s": base_turn * 0.55, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 0.8, "climb_bias": cruise_bias},
    ]

    # Lateral corridor probes + wind-aligned blend (only when the direct ray is costly).
    need_probes = horizontal > 5.0 and (ahead_rise > 20.0 or bearing_tw < 0.25 or (wind_speed > 1.5 and bearing_tw < 0.6))
    if need_probes:
        probe_headings = [goal_heading + math.radians(d) for d in (-30.0, 30.0)]
        if wind_speed > 0.8:
            wind_heading = math.atan2(local["wind_v"], local["wind_u"])
            blend = 0.45 if bearing_tw < 0.1 else 0.30
            blended = math.atan2(
                (1.0 - blend) * math.sin(goal_heading) + blend * math.sin(wind_heading),
                (1.0 - blend) * math.cos(goal_heading) + blend * math.cos(wind_heading),
            )
            probe_headings.append(blended)
            if bearing_tw < -0.2:
                probe_headings.append(
                    math.atan2(
                        0.45 * math.sin(goal_heading) + 0.55 * math.sin(wind_heading),
                        0.45 * math.cos(goal_heading) + 0.55 * math.cos(wind_heading),
                    )
                )
        if ahead_rise > 40.0:
            probe_headings.extend(goal_heading + math.radians(d) for d in (-45.0, 45.0))

        scored_probes: list[tuple[float, float]] = []
        for heading in probe_headings:
            turn = _heading_turn_rate(current.heading_rad, heading)
            score = 0.0
            if belief_map is not None:
                look = min(3.5, horizontal)
                ax = current.x + math.cos(heading) * look
                ay = current.y + math.sin(heading) * look
                tw, ww = _integrate_track_wind(
                    belief_map, current.x, current.y, ax, ay, current.z, samples=4
                )
                score += 60.0 * tw + 35.0 * max(0.0, ww) - 30.0 * max(0.0, -tw)
                rise = terrain_climb_along_line_m(
                    getattr(mission, "elevation", None), current.x, current.y, ax, ay, samples=3
                )
                score -= (rise / max(mission.altitude_step_m, 1e-6)) * mission.climb_cost_per_level_j * 0.55
            else:
                score += local["wind_u"] * math.cos(heading) + local["wind_v"] * math.sin(heading)
            scored_probes.append((score, turn))
        scored_probes.sort(key=lambda item: item[0], reverse=True)
        for score, turn in scored_probes[:3]:
            controls.append(
                {
                    "mode": "cruise",
                    "turn_rate_rad_s": turn,
                    "airspeed": DEFAULT_ENVELOPE.best_glide_speed + (2.4 if score > 0 else 1.0),
                    "climb_bias": cruise_bias,
                }
            )

    if abs(band_gap) > 0.35 and horizontal > 5.0:
        controls.append(
            {
                "mode": "cruise",
                "turn_rate_rad_s": base_turn * 0.8,
                "airspeed": DEFAULT_ENVELOPE.best_glide_speed + (1.6 if band_gap < 0 else 0.2),
                "climb_bias": clamp(1.05 if band_gap > 0 else -1.1, -1.2, 1.1),
            }
        )
    if vertical_gap < -0.15 and horizontal <= 10.0:
        controls.append({"mode": "cruise", "turn_rate_rad_s": base_turn * 0.85, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 2.2, "climb_bias": -1.15})
        controls.append({"mode": "cruise", "turn_rate_rad_s": base_turn, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 3.5, "climb_bias": -0.95})
        if horizontal <= 7.0:
            controls.append({"mode": "approach", "turn_rate_rad_s": base_turn, "airspeed": DEFAULT_ENVELOPE.best_glide_speed + 2.5, "climb_bias": -1.4})

    # Thermal only with usable uplift AND non-adverse along-track wind.
    allow_thermal = (
        horizontal > 6.0
        and preferred_z is not None
        and preferred_z > current.z + 0.3
        and bearing_tw >= -0.35
        and (local["wind_w"] > 0.2 or local["mode_prob_uplift"] > 0.4)
        and ahead_rise < 50.0
    )
    if allow_thermal:
        thermal_turn = max(0.15, best_thermalling_bank(local["wind_w"]) / max(DEFAULT_ENVELOPE.max_bank_rad, 1e-6) * DEFAULT_ENVELOPE.max_turn_rate_rad_s)
        # Keep a forward component via smaller turn rate so we don't pure-orbit.
        controls.append({"mode": "thermal_orbit", "turn_rate_rad_s": thermal_turn * 0.55, "airspeed": DEFAULT_ENVELOPE.best_thermalling_speed, "climb_bias": 0.55})
    return controls


def _sample_belief_states_batch(
    belief_map: BeliefMap,
    xs,
    ys,
    zs,
) -> dict[str, np.ndarray]:
    arrays = ensure_belief_field_arrays(belief_map, _BELIEF_SAMPLE_ATTRS)
    return {attr: trilinear_sample_batch(arrays[attr], xs, ys, zs) for attr in _BELIEF_SAMPLE_ATTRS}


def _sample_belief_state(belief_map: BeliefMap, x: float, y: float, z: float) -> dict[str, float]:
    arrays = belief_map.field_arrays
    if arrays is not None and "wind_u" in arrays:
        # Scalar Numba path — avoid per-call length-1 arrays / ensure overhead.
        return {
            attr: float(trilinear_sample(arrays[attr], x, y, z))
            for attr in _BELIEF_SAMPLE_ATTRS
            if attr in arrays
        }
    arrays = ensure_belief_field_arrays(belief_map, _BELIEF_SAMPLE_ATTRS)
    return {attr: float(trilinear_sample(arrays[attr], x, y, z)) for attr in _BELIEF_SAMPLE_ATTRS}


def _sample_attr(belief_map: BeliefMap, attr: str, x: float, y: float, z: float) -> float:
    arrays = belief_map.field_arrays
    if arrays is not None and attr in arrays:
        return float(
            trilinear_sample_batch(
                arrays[attr],
                np.asarray([x], dtype=np.float64),
                np.asarray([y], dtype=np.float64),
                np.asarray([z], dtype=np.float64),
            )[0]
        )
    z = clamp(z, 0.0, belief_map.levels - 1)
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, belief_map.levels - 1)
    tz = z - z0
    lower = _sample_attr_2d(belief_map, attr, x, y, z0)
    upper = _sample_attr_2d(belief_map, attr, x, y, z1)
    return lower + (upper - lower) * tz


def _sample_attr_2d(belief_map: BeliefMap, attr: str, x: float, y: float, z: int) -> float:
    x = clamp(x, 0.0, belief_map.width - 1)
    y = clamp(y, 0.0, belief_map.height - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, belief_map.width - 1)
    y1 = min(y0 + 1, belief_map.height - 1)
    tx = x - x0
    ty = y - y0
    v00 = getattr(belief_map.cells[z][y0][x0], attr)
    v10 = getattr(belief_map.cells[z][y0][x1], attr)
    v01 = getattr(belief_map.cells[z][y1][x0], attr)
    v11 = getattr(belief_map.cells[z][y1][x1], attr)
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return top + (bottom - top) * ty


def _rank_successors(
    belief_map: BeliefMap,
    mission: Mission,
    current: State3D,
    goal: State3D,
    min_level: int,
    max_level: int,
    risk_weight: float,
    safety_weight: float,
    return_cost_map: dict[State3D, float] | None,
    branch_width: int,
    spent_cost: float,
) -> list[dict]:
    scored: list[dict] = []
    for nxt in neighbors(
        current[0],
        current[1],
        current[2],
        belief_map.width,
        belief_map.height,
        min_level,
        max_level,
    ):
        breakdown = transition_cost_breakdown(
            belief_map,
            current,
            nxt,
            mission,
            risk_weight=risk_weight,
            safety_weight=safety_weight,
        )
        if not math.isfinite(breakdown.total_cost_j):
            continue
        current_altitude_gap = abs(goal[2] - current[2])
        next_altitude_gap = abs(goal[2] - nxt[2])
        if next_altitude_gap > current_altitude_gap:
            continue
        if return_cost_map is not None and mission.max_return_cost_j is not None:
            return_cost = return_cost_map.get(nxt, math.inf)
            if return_cost > mission.max_return_cost_j:
                continue
            if spent_cost + breakdown.total_cost_j + return_cost > mission.max_return_cost_j:
                continue
        current_energy_to_goal = estimate_energy_to_goal_j((float(current[0]), float(current[1]), float(current[2])), goal, mission)
        next_energy_to_goal = estimate_energy_to_goal_j((float(nxt[0]), float(nxt[1]), float(nxt[2])), goal, mission)
        
        energy_progress_reward = current_energy_to_goal - next_energy_to_goal
        
        step_reward = energy_progress_reward - breakdown.total_cost_j
        scored.append(
            {
                "state": nxt,
                "score": step_reward,
                "step_reward": step_reward,
                "step_cost": breakdown.total_cost_j,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:branch_width]


def _return_budget_is_tight(
    mission: Mission,
    start: tuple[float, float, float],
    goal: State3D,
) -> bool:
    """True when remaining return budget may bind — otherwise skip Dijkstra."""
    budget = mission.max_return_cost_j
    home = mission.home
    if budget is None or home is None:
        return False
    home_state = normalize_state(home)
    to_goal = estimate_energy_to_goal_j(start, goal, mission)
    goal_pt = (float(goal[0]), float(goal[1]), float(goal[2]))
    home_from_goal = estimate_energy_to_goal_j(goal_pt, home_state, mission)
    home_from_here = estimate_energy_to_goal_j(start, home_state, mission)
    need = max(to_goal + home_from_goal, home_from_here) * 1.45
    return float(budget) < need


def _coarse_neighbors(
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    min_z: int,
    max_z: int,
    stride: int,
) -> list[State3D]:
    """26-connected neighbors on a stride-aligned XY lattice (full Z)."""
    result: list[State3D] = []
    for dz in (-1, 0, 1):
        for dy in (-stride, 0, stride):
            for dx in (-stride, 0, stride):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if 0 <= nx < width and 0 <= ny < height and min_z <= nz <= max_z:
                    result.append((nx, ny, nz))
    return result


class ReturnCostLookup:
    """Coarse-grid return costs with nearest-cell lookup (dict-compatible .get)."""

    __slots__ = ("costs", "stride")

    def __init__(self, costs: dict[State3D, float], stride: int = 1) -> None:
        self.costs = costs
        self.stride = max(1, int(stride))

    def _key(self, point: State3D | tuple[float, float, float]) -> State3D:
        x, y, z = normalize_state(point)
        stride = self.stride
        if stride > 1:
            x = (x // stride) * stride
            y = (y // stride) * stride
        return x, y, z

    def get(self, point: State3D | tuple[float, float, float], default: float = math.inf) -> float:
        return self.costs.get(self._key(point), default)

    def __getitem__(self, point: State3D | tuple[float, float, float]) -> float:
        key = self._key(point)
        if key not in self.costs:
            raise KeyError(point)
        return self.costs[key]

    def __contains__(self, point: object) -> bool:
        try:
            return self._key(point) in self.costs  # type: ignore[arg-type]
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.costs)


def compute_return_cost_map(
    belief_map: BeliefMap,
    mission: Mission,
    *,
    xy_stride: int | None = None,
) -> ReturnCostLookup:
    """Dijkstra return costs from home.

    On wide fine grids, use XY stride>1 (default 2 when width≥32) so the node
    count drops ~4× while lookups snap to the coarse lattice.
    """
    width = belief_map.width
    height = belief_map.height
    min_level, max_level = _search_level_bounds(belief_map, mission)
    stride = 2 if xy_stride is None and width >= 32 else max(1, int(xy_stride or 1))
    costs: dict[State3D, float] = {}
    if mission.home is None:
        return ReturnCostLookup(costs, stride=stride)
    home = normalize_state(mission.home)
    home = ((home[0] // stride) * stride, (home[1] // stride) * stride, home[2])
    costs[home] = 0.0
    frontier: list[tuple[float, State3D]] = [(0.0, home)]
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current_cost > costs.get(current, math.inf):
            continue
        nbrs = _coarse_neighbors(
            current[0],
            current[1],
            current[2],
            width,
            height,
            min_level,
            max_level,
            stride,
        )
        if not nbrs:
            continue
        nxt_arr = np.asarray(nbrs, dtype=np.float64)
        cur_arr = np.tile(np.asarray(current, dtype=np.float64), (len(nbrs), 1))
        samples = _sample_belief_states_batch(belief_map, nxt_arr[:, 0], nxt_arr[:, 1], nxt_arr[:, 2])
        elev = getattr(mission, "elevation", None)
        if elev is not None:
            terrain_dz = terrain_delta_m_batch(
                elev,
                nxt_arr[:, 0],
                nxt_arr[:, 1],
                cur_arr[:, 0],
                cur_arr[:, 1],
            )
        else:
            terrain_dz = np.zeros(len(nbrs), dtype=np.float64)
        step_costs = transition_energy_batch(
            airspeed=mission.nominal_airspeed,
            currents=nxt_arr,
            nexts=cur_arr,
            local_u=samples["wind_u"],
            local_v=samples["wind_v"],
            local_w=samples["wind_w"],
            step_distance_m=mission.step_distance_m,
            altitude_step_m=mission.altitude_step_m,
            climb_cost_per_level_j=mission.climb_cost_per_level_j,
            hover_power_w=mission.hover_power_w,
            cruise_power_w=mission.cruise_power_w,
            hotel_power_w=mission.hotel_power_w,
            headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
            climb_power_per_mps_w=mission.climb_power_per_mps_w,
            descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
            terrain_dz_m=terrain_dz,
        )
        for nbr, step_cost in zip(nbrs, step_costs):
            new_cost = current_cost + float(step_cost)
            if new_cost < costs.get(nbr, math.inf):
                costs[nbr] = new_cost
                heapq.heappush(frontier, (new_cost, nbr))
    return ReturnCostLookup(costs, stride=stride)


def _return_step_cost(
    belief_map: BeliefMap,
    current: State3D,
    nxt: State3D,
    mission: Mission,
) -> float:
    cx, cy, cz = current
    cell = _sample_belief_state(belief_map, float(cx), float(cy), float(cz))
    return transition_energy_j(
        airspeed=mission.nominal_airspeed,
        current=current,
        nxt=nxt,
        local_u=cell["wind_u"],
        local_v=cell["wind_v"],
        local_w=cell["wind_w"],
        step_distance_m=mission.step_distance_m,
        altitude_step_m=mission.altitude_step_m,
        climb_cost_per_level_j=mission.climb_cost_per_level_j,
        hover_power_w=mission.hover_power_w,
        cruise_power_w=mission.cruise_power_w,
        hotel_power_w=mission.hotel_power_w,
        headwind_power_per_mps_w=mission.headwind_power_per_mps_w,
        climb_power_per_mps_w=mission.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=mission.descent_power_reduction_per_mps_w,
        terrain_dz_m=_mission_terrain_dz(mission, cx, cy, nxt[0], nxt[1]),
    )


def _altitude_bonus(
    z: int,
    mission: Mission,
    x: float | None = None,
    y: float | None = None,
    belief_cell=None,
    move_dx: float = 0.0,
    move_dy: float = 0.0,
) -> float:
    """Cruise: reward wind-efficient altitude. Approach: reward goal altitude."""
    if mission.max_altitude_level <= mission.min_altitude_level:
        return 0.0
    span = max(1, mission.max_altitude_level - mission.min_altitude_level)
    goal = normalize_state(mission.goal)
    if x is None or y is None:
        return 1.0 - abs(z - goal[2]) / span
    horizontal = math.hypot(goal[0] - x, goal[1] - y)
    if horizontal <= 5.0:
        return 1.0 - abs(z - goal[2]) / span
    # Wind-aware cruise score in [-1, 1]-ish via soft mapping.
    move_norm = max(math.hypot(move_dx, move_dy), 1e-6)
    if belief_cell is None:
        return 0.0
    if isinstance(belief_cell, dict):
        wind_u = float(belief_cell["wind_u"])
        wind_v = float(belief_cell["wind_v"])
        wind_w = float(belief_cell["wind_w"])
        energy = float(belief_cell["expected_energy_gain"])
    else:
        wind_u = float(belief_cell.wind_u)
        wind_v = float(belief_cell.wind_v)
        wind_w = float(belief_cell.wind_w)
        energy = float(belief_cell.expected_energy_gain)
    tailwind = (wind_u * (move_dx / move_norm)) + (wind_v * (move_dy / move_norm))
    score = 0.45 * tailwind + 0.55 * wind_w + 0.35 * energy
    return max(-1.0, min(1.0, score / 4.0))


def _preferred_cruise_altitude(
    belief_map: BeliefMap,
    x: float,
    y: float,
    goal: State3D,
    min_level: int,
    max_level: int,
    current_z: float | None = None,
    climb_cost_per_level_j: float = 180.0,
    clearance_agl_level: float = 1.0,
    preferred_cruise_agl: float | None = None,
) -> float:
    """Pick the energy-optimal AGL band above the safety floor — no forced cruise height.

    Climb / hold / descend only when wind+energy scores justify it. Near the goal,
    track goal altitude for landing. Cruise never plans below clearance AGL.
    A sticky preferred_cruise_agl (from multi-band guides) biases the band choice.
    """
    bearing = math.atan2(goal[1] - y, goal[0] - x) if math.hypot(goal[0] - x, goal[1] - y) > 1e-6 else 0.0
    horizontal_cells = max(math.hypot(goal[0] - x, goal[1] - y), 1.0)
    z_now = float(min_level if current_z is None else current_z)
    goal_z = float(goal[2])
    floor = cruise_agl_floor(horizontal_cells, float(min_level), clearance_agl_level)
    search_min = max(min_level, int(math.floor(floor)))

    # Final approach: close altitude to the goal.
    if horizontal_cells <= 5.0:
        if horizontal_cells <= 3.5:
            return clamp(goal_z, float(min_level), float(max_level))
        return clamp(z_now + 0.6 * (goal_z - z_now), float(min_level), float(max_level))

    # Sticky cruise from energy guides: keep probing light (current / sticky / ±1)
    # instead of rescanning every integer band at every MPC node.
    if preferred_cruise_agl is not None and horizontal_cells > 6.0:
        sticky = clamp(float(preferred_cruise_agl), float(search_min), float(max_level))
        candidates = {
            int(round(clamp(z_now, search_min, max_level))),
            int(round(clamp(sticky, search_min, max_level))),
            int(round(clamp(sticky - 1.0, search_min, max_level))),
            int(round(clamp(sticky + 1.0, search_min, max_level))),
            int(round(clamp(floor, search_min, max_level))),
        }
        levels = sorted(candidates)
    else:
        levels = list(range(search_min, max_level + 1))
    if not levels:
        return clamp(z_now, float(min_level), float(max_level))

    # Batch all belief probes: here@z_now, here@each level, and 3 along-track fracs per level.
    fracs = (0.0, 0.3, 0.6)
    track_xy = [(x + f * (goal[0] - x), y + f * (goal[1] - y)) for f in fracs]
    xs: list[float] = [x]
    ys: list[float] = [y]
    zs: list[float] = [z_now]
    # index 0 = current band at (x,y)
    level_here_idx: dict[int, int] = {}
    level_track_idx: dict[int, tuple[int, int, int]] = {}
    for level in levels:
        level_here_idx[level] = len(xs)
        xs.append(x)
        ys.append(y)
        zs.append(float(level))
        track_ids = []
        for sx, sy in track_xy:
            track_ids.append(len(xs))
            xs.append(sx)
            ys.append(sy)
            zs.append(float(level))
        level_track_idx[level] = (track_ids[0], track_ids[1], track_ids[2])

    packed = _sample_belief_states_batch(belief_map, xs, ys, zs)
    cos_b = math.cos(bearing)
    sin_b = math.sin(bearing)
    here = {attr: float(packed[attr][0]) for attr in _BELIEF_SAMPLE_ATTRS}

    def band_tailwind(level: int) -> float:
        i0, i1, i2 = level_track_idx[level]
        tw = 0.0
        for i in (i0, i1, i2):
            tw += float(packed["wind_u"][i]) * cos_b + float(packed["wind_v"][i]) * sin_b
        return tw / 3.0

    def band_score(level: int) -> float:
        i = level_here_idx[level]
        local_u = float(packed["wind_u"][i])
        local_v = float(packed["wind_v"][i])
        local_w = float(packed["wind_w"][i])
        energy = float(packed["expected_energy_gain"][i])
        uplift = float(packed["mode_prob_uplift"][i])
        sink = float(packed["mode_prob_sink"][i])
        unc = float(packed["uncertainty"][i])
        safety = float(packed["safety_penalty"][i])
        tw = band_tailwind(level)
        wind_saving = horizontal_cells * (
            22.0 * tw
            - 20.0 * max(0.0, -tw)
            + 16.0 * max(0.0, local_w)
            - 18.0 * max(0.0, -local_w)
            + 10.0 * energy
            + 5.0 * uplift
            - 12.0 * sink
        )
        free_lift = clamp(2.5 * max(0.0, here["wind_w"] - 0.1), 0.0, 2.0)
        powered_climb = max(0.0, float(level) - z_now - free_lift)
        climb_up = powered_climb * climb_cost_per_level_j
        terminal_align = abs(float(level) - goal_z) * 0.7 * climb_cost_per_level_j
        leave_current = abs(float(level) - z_now) * 0.2 * climb_cost_per_level_j
        return (
            wind_saving
            - climb_up
            - terminal_align
            - leave_current
            - 8.0 * unc
            - 6.0 * safety
        )

    # Default anchor: current band (or clearance floor), never forced higher without wind gain.
    anchor = int(round(clamp(max(z_now, floor) if goal_z <= z_now + 0.25 else max(z_now, goal_z, floor), search_min, max_level)))
    if anchor not in level_here_idx:
        anchor = levels[0]
    best_z = float(anchor)
    best_score = band_score(anchor)
    # Small margin so we only change altitude when the gain is clear.
    margin = 80.0 + 10.0 * max(0.0, 10.0 - horizontal_cells)
    guide_target: float | None = None
    if preferred_cruise_agl is not None and horizontal_cells > 6.0:
        guide_target = clamp(float(preferred_cruise_agl), float(search_min), float(max_level))
        margin = 35.0 + 6.0 * max(0.0, 10.0 - horizontal_cells)
    tw_now = band_tailwind(anchor)
    for level in levels:
        if level == anchor:
            continue
        score = band_score(level)
        if level > z_now + 0.4 and band_tailwind(level) < tw_now - 0.45:
            if guide_target is None or abs(float(level) - guide_target) > 0.6:
                continue
        level_margin = margin
        if guide_target is not None and abs(float(level) - guide_target) <= 0.6:
            level_margin = min(level_margin, 20.0)
            score += 45.0
        if score > best_score + level_margin:
            best_score = score
            best_z = float(level)

    if guide_target is not None:
        target = float(guide_target)
        target_level = int(round(clamp(target, search_min, max_level)))
        if target_level in level_here_idx:
            target_score = band_score(target_level)
            if target_score > best_score - 55.0:
                best_z = target

    # Free-lift ride: allow drifting up one band when uplift is free and not into headwind.
    if here["wind_w"] > 0.35 and best_z <= z_now + 0.2 and max_level >= search_min:
        up = int(min(max_level, math.floor(z_now) + 1))
        if up in level_here_idx and band_tailwind(up) >= tw_now - 0.2:
            if band_score(up) > best_score - 40.0:
                best_z = max(best_z, float(up))

    if here["wind_w"] > 0.18 and horizontal_cells > 7.0 and tw_now >= -0.35:
        soft = min(float(max_level), max(z_now, floor) + 0.9)
        soft_i = int(clamp(round(soft), search_min, max_level))
        if soft_i in level_here_idx and band_tailwind(soft_i) >= tw_now - 0.2 and band_score(soft_i) > best_score - 50.0:
            best_z = max(best_z, min(soft, float(soft_i)))

    # Near goal: start bleeding altitude so we can land without a hover-descent.
    # Hold preferred/clearance cruise until the last ~12% (progress-based in execution).
    if horizontal_cells <= 8.0:
        max_recoverable = goal_z + max(0.4, (horizontal_cells - 0.5) / 2.0)
        if horizontal_cells > 0.5:
            hold = floor
            if preferred_cruise_agl is not None:
                hold = max(hold, float(preferred_cruise_agl))
            max_recoverable = max(max_recoverable, hold)
        best_z = min(best_z, max_recoverable)

    # Rate-limit altitude change per replan; always allow climbing toward an elevated goal.
    best_z = max(best_z, floor)
    climb_cap = 1.35 if guide_target is not None and guide_target > z_now + 0.4 else 1.0
    best_z = clamp(best_z, z_now - 1.0, z_now + climb_cap)
    best_z = clamp(best_z, float(search_min), float(max_level))
    if goal_z > z_now + 0.25:
        best_z = max(best_z, min(goal_z, z_now + 1.1))
    return best_z


def goal_altitude_gap(point: State3D, mission: Mission) -> int:
    return normalize_state(mission.goal)[2] - point[2]


def _search_level_bounds(belief_map: BeliefMap, mission: Mission) -> tuple[int, int]:
    if belief_map.levels <= 0:
        return 0, 0
    min_level = max(0, mission.min_altitude_level)
    max_level = min(belief_map.levels - 1, mission.max_altitude_level)
    if min_level > max_level:
        min_level = max_level
    return min_level, max_level


def _mission_terrain_dz(mission: Mission, x0: float, y0: float, x1: float, y1: float) -> float:
    return terrain_delta_m(getattr(mission, "elevation", None), x0, y0, x1, y1)


def _effective_agl_floor(mission: Mission, horizontal_to_goal: float) -> float:
    return cruise_agl_floor(
        horizontal_to_goal,
        float(mission.min_altitude_level),
        float(getattr(mission, "clearance_agl_level", 1.0)),
    )
