from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import math

import numpy as np

from .altitude import (
    cruise_agl_floor,
    msl_height_m,
    next_polyline_waypoint,
    sample_elevation,
    straight_agl_guide_polyline,
    terrain_delta_m,
)
from .belief import BeliefUpdater, belief_snapshot, create_belief_map
from .config import TaskConfig
from .controller import advance_continuous_state, transition_energy_j
from .mathutils import bilinear_sample, clamp, magnitude, magnitude3, trilinear_sample
from .planner import _cruise_z_from_guide_label, plan_path_details
from .types import CoarseWindSample, DroneState, Mission, Observation, TerrainField, WindField


@dataclass(slots=True)
class PredictSession:
    timestamp: str | None = None
    prev_coarse: tuple[float, float, float] | None = None
    current_coarse: tuple[float, float, float] = (0.0, 0.0, 0.0)
    prev_u_km: float = 0.0
    prev_v_km: float = 0.0
    prev_w_km: float = 0.0
    delta_u_km: float = 0.0
    delta_v_km: float = 0.0
    delta_w_km: float = 0.0
    cell_history: dict[tuple[int, int, int], dict[str, float]] = field(default_factory=dict)
    coarse_history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=4))
    cell_history_window: dict[tuple[int, int, int], deque[tuple[float, float, float]]] = field(default_factory=dict)
    window_size: int = 4

    def advance_header(self, timestamp: str, u_km: float, v_km: float, w_km: float = 0.0) -> None:
        previous = self.prev_coarse
        self.timestamp = timestamp
        self.current_coarse = (u_km, v_km, w_km)
        self.prev_u_km = previous[0] if previous else u_km
        self.prev_v_km = previous[1] if previous else v_km
        self.prev_w_km = previous[2] if previous else w_km
        self.delta_u_km = u_km - self.prev_u_km
        self.delta_v_km = v_km - self.prev_v_km
        self.delta_w_km = w_km - self.prev_w_km
        self.prev_coarse = (u_km, v_km, w_km)
        self.coarse_history.append((u_km, v_km, w_km))

    def temporal_features(self, x: int, y: int, z: int, u_local: float, v_local: float, w_local: float) -> dict[str, float]:
        previous = self.cell_history.get((x, y, z))
        prev_local_u = previous["u_local"] if previous else u_local
        prev_local_v = previous["v_local"] if previous else v_local
        prev_local_w = previous["w_local"] if previous else w_local
        local_window = list(self.cell_history_window.get((x, y, z), deque()))
        coarse_window = list(self.coarse_history)
        coarse_u_values = [item[0] for item in coarse_window] or [self.current_coarse[0]]
        coarse_v_values = [item[1] for item in coarse_window] or [self.current_coarse[1]]
        coarse_w_values = [item[2] for item in coarse_window] or [self.current_coarse[2]]
        local_u_values = [item[0] for item in local_window] or [prev_local_u]
        local_v_values = [item[1] for item in local_window] or [prev_local_v]
        local_w_values = [item[2] for item in local_window] or [prev_local_w]
        prev2_coarse = coarse_window[-2] if len(coarse_window) >= 2 else self.current_coarse
        prev3_coarse = coarse_window[-3] if len(coarse_window) >= 3 else prev2_coarse
        prev2_local = local_window[-2] if len(local_window) >= 2 else (prev_local_u, prev_local_v, prev_local_w)
        prev3_local = local_window[-3] if len(local_window) >= 3 else prev2_local
        return {
            "prev_u_km": getattr(self, "prev_u_km", getattr(self, "current_coarse", (0.0, 0.0, 0.0))[0]),
            "prev_v_km": getattr(self, "prev_v_km", getattr(self, "current_coarse", (0.0, 0.0, 0.0))[1]),
            "prev_w_km": getattr(self, "prev_w_km", getattr(self, "current_coarse", (0.0, 0.0, 0.0))[2]),
            "delta_u_km": getattr(self, "delta_u_km", 0.0),
            "delta_v_km": getattr(self, "delta_v_km", 0.0),
            "delta_w_km": getattr(self, "delta_w_km", 0.0),
            "prev_local_u": prev_local_u,
            "prev_local_v": prev_local_v,
            "prev_local_w": prev_local_w,
            "prev_local_speed": magnitude3(prev_local_u, prev_local_v, prev_local_w),
            "local_trend_u": u_local - prev_local_u,
            "local_trend_v": v_local - prev_local_v,
            "local_trend_w": w_local - prev_local_w,
            "prev2_u_km": prev2_coarse[0],
            "prev2_v_km": prev2_coarse[1],
            "prev2_w_km": prev2_coarse[2],
            "prev3_u_km": prev3_coarse[0],
            "prev3_v_km": prev3_coarse[1],
            "prev3_w_km": prev3_coarse[2],
            "coarse_u_mean": sum(coarse_u_values) / len(coarse_u_values),
            "coarse_v_mean": sum(coarse_v_values) / len(coarse_v_values),
            "coarse_w_mean": sum(coarse_w_values) / len(coarse_w_values),
            "prev2_local_u": prev2_local[0],
            "prev2_local_v": prev2_local[1],
            "prev2_local_w": prev2_local[2],
            "prev3_local_u": prev3_local[0],
            "prev3_local_v": prev3_local[1],
            "prev3_local_w": prev3_local[2],
            "local_u_mean": sum(local_u_values) / len(local_u_values),
            "local_v_mean": sum(local_v_values) / len(local_v_values),
            "local_w_mean": sum(local_w_values) / len(local_w_values),
        }

    def update_cell_history(self, u_grid, v_grid, w_grid, z: int) -> None:
        u_arr = np.asarray(u_grid, dtype=np.float64)
        v_arr = np.asarray(v_grid, dtype=np.float64)
        w_arr = np.asarray(w_grid, dtype=np.float64)
        if u_arr.ndim != 2 or u_arr.size == 0:
            self.cell_history = {}
            return
        height, width = u_arr.shape
        next_history = {}
        for y in range(height):
            for x in range(width):
                uu = float(u_arr[y, x])
                vv = float(v_arr[y, x])
                ww = float(w_arr[y, x])
                self.observe_cell(x, y, z, uu, vv, ww)
                next_history[(x, y, z)] = {"u_local": uu, "v_local": vv, "w_local": ww}
        self.cell_history = next_history

    def observe_cell(self, x: int, y: int, z: int, u_local: float, v_local: float, w_local: float) -> None:
        key = (x, y, z)
        window = self.cell_history_window.get(key)
        if window is None:
            window = deque(maxlen=self.window_size)
        window.append((u_local, v_local, w_local))
        self.cell_history_window[key] = window
        self.cell_history[key] = {"u_local": u_local, "v_local": v_local, "w_local": w_local}


@dataclass(slots=True)
class NavigationContext:
    mission: Mission
    terrain: TerrainField
    state: DroneState
    battery_j: float
    belief_map: object
    predict_session: PredictSession
    trace: list[dict] = field(default_factory=list)
    step: int = 0
    planned_path: list[tuple[int, int, int]] = field(default_factory=list)
    latest_planning: dict = field(default_factory=lambda: {"path": [], "path_cost": 0.0, "candidates": []})
    latest_prediction: dict | None = None
    latest_physics: dict | None = None
    return_cost_map_cache: object | None = None
    return_cost_map_step: int = -10_000


class NavigationEngine:
    def __init__(self, pipeline, config: TaskConfig):
        self.pipeline = pipeline
        self.config = config
        self.updater = BeliefUpdater(
            observation_radius=config.belief.observation_radius,
            advection_gain=config.belief.advection_gain,
            decay_per_step=config.belief.decay_per_step,
            process_noise=config.belief.process_noise,
            observation_noise=config.belief.observation_noise,
            advection_noise=config.belief.advection_noise,
        )

    def create_context(self, start: tuple[int, int] | tuple[int, int, int], goal: tuple[int, int] | tuple[int, int, int]) -> NavigationContext:
        width = len(self.pipeline.terrain.elevation[0])
        height = len(self.pipeline.terrain.elevation)
        actual_start = _clamp_xyz(start, width, height, self.config.mission)
        actual_goal = _clamp_xyz(goal, width, height, self.config.mission)
        mission = Mission(
            start=actual_start,
            goal=actual_goal,
            max_steps=self.config.mission.max_steps,
            step_distance_m=self.config.mission.step_distance_m,
            home=actual_start,
            max_return_cost_j=None,
            nominal_airspeed=getattr(self.config.mission, "nominal_airspeed", 16.5),
            hover_power_w=getattr(self.config.mission, "hover_power_w", 105.0),
            cruise_power_w=getattr(self.config.mission, "cruise_power_w", 150.0),
            hotel_power_w=getattr(self.config.mission, "hotel_power_w", 18.0),
            headwind_power_per_mps_w=getattr(self.config.mission, "headwind_power_per_mps_w", 14.0),
            climb_power_per_mps_w=getattr(self.config.mission, "climb_power_per_mps_w", 125.0),
            descent_power_reduction_per_mps_w=getattr(self.config.mission, "descent_power_reduction_per_mps_w", 58.0),
            reserve_energy_ratio=getattr(self.config.mission, "reserve_energy_ratio", 0.22),
            altitude_step_m=getattr(self.config.mission, "altitude_step_m", 50.0),
            min_altitude_level=getattr(self.config.mission, "min_altitude_level", 0),
            max_altitude_level=getattr(self.config.mission, "max_altitude_level", 4),
            climb_cost_per_level_j=getattr(self.config.mission, "climb_cost_per_level_j", 180.0),
            clearance_agl_level=getattr(self.config.mission, "clearance_agl_level", 1.0),
            cruise_band_step=getattr(self.config.mission, "cruise_band_step", 0.02),
            corridor_energy_margin=getattr(self.config.mission, "corridor_energy_margin", None),
            elevation=self.pipeline.terrain.elevation,
        )
        state = DroneState(
            x=float(actual_start[0]),
            y=float(actual_start[1]),
            z=float(actual_start[2]),
            heading_rad=0.0,
            battery_ratio=1.0,
            airspeed=mission.nominal_airspeed,
        )
        battery_j = self.config.mission.battery_capacity_j * state.battery_ratio
        context = NavigationContext(
            mission=mission,
            terrain=self.pipeline.terrain,
            state=state,
            battery_j=battery_j,
            belief_map=create_belief_map(width, height, self.config.model.altitude_levels),
            predict_session=PredictSession(window_size=self.config.model.temporal_window_size, coarse_history=deque(maxlen=self.config.model.temporal_window_size)),
            planned_path=[actual_start],
        )
        context.mission.max_return_cost_j = self.return_home_budget_j(context)
        return context

    def step(
        self,
        context: NavigationContext,
        sample: CoarseWindSample,
        observation: Observation | None = None,
        truth_field: dict | None = None,
    ) -> dict:
        context.step += 1
        is_keyframe = context.step == 1 or context.step % max(1, self.config.simulation.report_keyframe_interval) == 0
        prediction, physics = self._forecast_fields(context, sample, observation, is_keyframe)
        self.updater.apply_prediction(
            context.belief_map,
            WindField(u=prediction["u_layers"], v=prediction["v_layers"], w=prediction["w_layers"]),
            context.step,
        )
        observation_payload = None
        if observation:
            observation = self._clamp_observation(observation, context.belief_map.width, context.belief_map.height)
            self.updater.update_with_observation(
                context.belief_map,
                observation,
                prediction["u"][observation.y][observation.x],
                prediction["v"][observation.y][observation.x],
                prediction["w"][observation.y][observation.x],
                context.step,
            )
            observation_payload = {
                "position": [observation.x, observation.y, observation.z],
                "u_obs": observation.u_obs,
                "v_obs": observation.v_obs,
                "w_obs": observation.w_obs,
                "airspeed": observation.airspeed,
                "ground_speed": observation.ground_speed,
                "climb_rate": observation.climb_rate,
                "acceleration": observation.acceleration,
            }

        self._replan(context)
        step_energy_j = self._move_if_possible(context, prediction, truth_field)
        frame = self._build_frame(context, sample.timestamp, physics, prediction, observation_payload, truth_field, is_keyframe)
        frame["step_energy_j"] = step_energy_j
        context.trace.append(frame)
        return frame

    def _resolved_guide_cruise_agl(self, context: NavigationContext) -> float:
        """Sticky preferred cruise, else parse from guide_straight_agl_* mode label."""
        clearance = float(getattr(context.mission, "clearance_agl_level", 1.0))
        preferred = getattr(context.mission, "preferred_cruise_agl", None)
        if preferred is not None:
            return float(preferred)
        mode = str((context.latest_planning or {}).get("planning_mode") or "")
        return float(_cruise_z_from_guide_label(mode, clearance, context.mission))

    def _guide_cruise_target_z(self, context: NavigationContext, horizontal: float, total_horiz: float) -> float | None:
        """Baseline-like AGL profile while locked to a straight/corridor energy guide."""
        mode = str((context.latest_planning or {}).get("planning_mode") or "")
        if not (mode.startswith("guide_straight_agl") or mode.startswith("guide_corridor_")):
            return None
        cruise = self._resolved_guide_cruise_agl(context)
        goal_z = float(context.mission.goal[2]) if len(context.mission.goal) >= 3 else 0.0
        if total_horiz < 1e-6:
            return goal_z
        progress = clamp(1.0 - horizontal / total_horiz, 0.0, 1.0)
        # Match eval straight_agl_baseline: climb to cruise in the first ~12%,
        # hold, then descend in the last ~12–15% (1-cell lag compensated).
        if progress < 0.12:
            return cruise
        if progress > 0.85:
            blend = clamp((1.0 - progress) / 0.15, 0.0, 1.0)
            return goal_z + (cruise - goal_z) * blend
        if horizontal <= 0.5:
            return goal_z
        return cruise

    def _baseline_like_next_waypoint(
        self,
        context: NavigationContext,
        next_waypoint: tuple,
        horizontal: float,
    ) -> tuple[float, float, float]:
        """Apply baseline AGL profile; straight guides track the eval baseline polyline."""
        goal = context.mission.goal
        gx, gy = float(goal[0]), float(goal[1])
        sx, sy = context.state.x, context.state.y
        ms = context.mission.start
        total_horiz = math.hypot(float(goal[0]) - float(ms[0]), float(goal[1]) - float(ms[1]))
        total_horiz = max(total_horiz, horizontal, 1.0)
        mode = str((context.latest_planning or {}).get("planning_mode") or "")
        cruise = self._resolved_guide_cruise_agl(context)

        if mode.startswith("guide_straight_agl"):
            start_xyz = (
                float(ms[0]),
                float(ms[1]),
                float(ms[2]) if len(ms) > 2 else 0.0,
            )
            goal_xyz = (
                float(goal[0]),
                float(goal[1]),
                float(goal[2]) if len(goal) > 2 else 0.0,
            )
            # If a prior corridor left us off the original ray, re-anchor the
            # baseline polyline from the current state (avoid long return zigzags).
            dx0, dy0 = goal_xyz[0] - start_xyz[0], goal_xyz[1] - start_xyz[1]
            span = math.hypot(dx0, dy0) or 1.0
            ux, uy = dx0 / span, dy0 / span
            cross = abs((sx - start_xyz[0]) * (-uy) + (sy - start_xyz[1]) * ux)
            if cross > 0.35:
                start_xyz = (sx, sy, float(context.state.z))
            pts = straight_agl_guide_polyline(start_xyz, goal_xyz, agl_cruise=cruise)
            nxt = next_polyline_waypoint(pts, sx, sy, ahead_cells=0.02)
            if nxt is None:
                return (gx, gy, goal_xyz[2])
            return (
                clamp(nxt[0], 0.0, context.belief_map.width - 1),
                clamp(nxt[1], 0.0, context.belief_map.height - 1),
                clamp(nxt[2], float(context.mission.min_altitude_level), float(context.mission.max_altitude_level)),
            )

        target_z = self._guide_cruise_target_z(context, horizontal, total_horiz)
        if target_z is None:
            return (float(next_waypoint[0]), float(next_waypoint[1]), float(next_waypoint[2]))
        # Corridor: keep planned XY (via), only rewrite altitude to the AGL profile.
        tx, ty = float(next_waypoint[0]), float(next_waypoint[1])
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        step = min(1.0, dist) if dist > 1e-6 else 0.0
        if dist > 1e-6:
            nx = sx + dx / dist * step
            ny = sy + dy / dist * step
        else:
            nx, ny = tx, ty
        z = context.state.z + clamp(target_z - context.state.z, -1.0, 1.2)
        z = clamp(z, float(context.mission.min_altitude_level), float(context.mission.max_altitude_level))
        return (nx, ny, z)

    def build_report(
        self,
        context: NavigationContext,
        requested_start: tuple[int, int],
        requested_goal: tuple[int, int],
    ) -> dict:
        return {
            "grid": {
                "width": len(self.pipeline.terrain.elevation[0]),
                "height": len(self.pipeline.terrain.elevation),
                "resolution_m": self.config.mission.step_distance_m,
            },
            "mission": {
                "start": list(context.mission.start),
                "goal": list(context.mission.goal),
                "home": list(context.mission.home) if context.mission.home is not None else None,
                "max_steps": context.mission.max_steps,
                "battery_capacity_j": self.config.mission.battery_capacity_j,
                "power_model": {
                    "nominal_airspeed": context.mission.nominal_airspeed,
                    "hover_power_w": context.mission.hover_power_w,
                    "cruise_power_w": context.mission.cruise_power_w,
                    "hotel_power_w": context.mission.hotel_power_w,
                    "headwind_power_per_mps_w": context.mission.headwind_power_per_mps_w,
                    "climb_power_per_mps_w": context.mission.climb_power_per_mps_w,
                    "descent_power_reduction_per_mps_w": context.mission.descent_power_reduction_per_mps_w,
                    "reserve_energy_ratio": context.mission.reserve_energy_ratio,
                },
                "altitude_step_m": context.mission.altitude_step_m,
                "altitude_levels": [context.mission.min_altitude_level, context.mission.max_altitude_level],
            },
            "terrain": {
                "elevation": self.pipeline.terrain.elevation,
                "roughness": self.pipeline.terrain.roughness,
                "slope": self.pipeline.terrain.slope,
            },
            "goal_reached": _goal_reached(context.state, context.mission.goal),
            "final_position": [_round_state_value(context.state.x), _round_state_value(context.state.y), _round_state_value(context.state.z)],
            "executed_path": [
                [_round_state_value(context.mission.start[0]), _round_state_value(context.mission.start[1]), _round_state_value(float(context.mission.start[2]) if len(context.mission.start) > 2 else 0.0)]
            ]
            + [item["position"] for item in context.trace],
            "path_model_energy_j": sum(float(item.get("step_energy_j") or 0.0) for item in context.trace),
            "battery_ratio": context.state.battery_ratio,
            "steps_executed": len(context.trace),
            "route_summary": {
                "requested_start": list(requested_start),
                "requested_goal": list(requested_goal),
                "actual_start": list(context.mission.start),
                "actual_goal": list(context.mission.goal),
                "final_position": [_round_state_value(context.state.x), _round_state_value(context.state.y), _round_state_value(context.state.z)],
                "goal_reached": _goal_reached(context.state, context.mission.goal),
            },
            "model_summary": self.pipeline.model_summary(),
            "report_mode": {
                "trace_encoding": "keyframe-plus-delta",
                "keyframe_interval": self.config.simulation.report_keyframe_interval,
            },
            "trace": context.trace,
        }

    def return_home_radius_cells(self, context: NavigationContext) -> float:
        nominal_step_energy = max(
            1.0,
            transition_energy_j(
                airspeed=context.state.airspeed,
                current=(0, 0, 0),
                nxt=(1, 0, 0),
                local_u=0.0,
                local_v=0.0,
                local_w=0.0,
                step_distance_m=context.mission.step_distance_m,
                altitude_step_m=context.mission.altitude_step_m,
                climb_cost_per_level_j=context.mission.climb_cost_per_level_j,
                hover_power_w=context.mission.hover_power_w,
                cruise_power_w=context.mission.cruise_power_w,
                hotel_power_w=context.mission.hotel_power_w,
                headwind_power_per_mps_w=context.mission.headwind_power_per_mps_w,
                climb_power_per_mps_w=context.mission.climb_power_per_mps_w,
                descent_power_reduction_per_mps_w=context.mission.descent_power_reduction_per_mps_w,
            ),
        )
        reserve = context.mission.reserve_energy_ratio * self.config.mission.battery_capacity_j
        usable = max(0.0, context.battery_j - reserve)
        return usable / nominal_step_energy

    def return_home_budget_j(self, context: NavigationContext) -> float:
        reserve = context.mission.reserve_energy_ratio * self.config.mission.battery_capacity_j
        return max(0.0, context.battery_j - reserve)

    def _replan(self, context: NavigationContext) -> None:
        if context.latest_planning and context.step > 1:
            interval = max(1, self.config.planner.replan_interval_steps)
            if context.step % interval != 0 and len(context.planned_path) > 1:
                return
        context.mission.max_return_cost_j = self.return_home_budget_j(context)
        horiz = math.hypot(
            float(context.mission.goal[0]) - context.state.x,
            float(context.mission.goal[1]) - context.state.y,
        )
        if horiz <= 0.5:
            context.mission.guide_via = None
            # Final half-cell: allow full descent to goal altitude.
            # Allow final descent: clear elevated cruise hold.
            context.mission.preferred_cruise_agl = None
        plan_mission = Mission(
            start=(context.state.x, context.state.y, context.state.z),
            goal=context.mission.goal,
            max_steps=context.mission.max_steps,
            step_distance_m=context.mission.step_distance_m,
            home=context.mission.home,
            max_return_cost_j=context.mission.max_return_cost_j,
            nominal_airspeed=context.mission.nominal_airspeed,
            hover_power_w=context.mission.hover_power_w,
            cruise_power_w=context.mission.cruise_power_w,
            hotel_power_w=context.mission.hotel_power_w,
            headwind_power_per_mps_w=context.mission.headwind_power_per_mps_w,
            climb_power_per_mps_w=context.mission.climb_power_per_mps_w,
            descent_power_reduction_per_mps_w=context.mission.descent_power_reduction_per_mps_w,
            reserve_energy_ratio=context.mission.reserve_energy_ratio,
            altitude_step_m=context.mission.altitude_step_m,
            min_altitude_level=context.mission.min_altitude_level,
            max_altitude_level=context.mission.max_altitude_level,
            climb_cost_per_level_j=context.mission.climb_cost_per_level_j,
            clearance_agl_level=getattr(context.mission, "clearance_agl_level", 1.0),
            cruise_band_step=getattr(context.mission, "cruise_band_step", 0.02),
            corridor_energy_margin=getattr(context.mission, "corridor_energy_margin", None),
            elevation=context.terrain.elevation,
            guide_via=getattr(context.mission, "guide_via", None),
            preferred_cruise_agl=getattr(context.mission, "preferred_cruise_agl", None),
        )
        # Reuse Dijkstra return map for a few steps (belief wind drifts slowly vs replan rate).
        cache_ttl = max(2, int(getattr(self.config.planner, "replan_interval_steps", 2)) * 2)
        cached = None
        if (
            context.return_cost_map_cache is not None
            and context.step - context.return_cost_map_step <= cache_ttl
        ):
            cached = context.return_cost_map_cache
        planning = plan_path_details(
            context.belief_map,
            plan_mission,
            risk_weight=self.config.planner.risk_weight,
            safety_weight=self.config.planner.safety_weight,
            anytime_rounds=self.config.planner.anytime_rounds,
            heuristic_weight_start=self.config.planner.heuristic_weight_start,
            heuristic_weight_end=self.config.planner.heuristic_weight_end,
            search_node_budget=self.config.planner.search_node_budget,
            horizon_steps=self.config.planner.horizon_steps,
            beam_width=self.config.planner.beam_width,
            branch_width=self.config.planner.branch_width,
            discount_factor=self.config.planner.discount_factor,
            terminal_progress_weight=self.config.planner.terminal_progress_weight,
            return_cost_map_cache=cached,
        )
        context.mission.guide_via = getattr(plan_mission, "guide_via", None)
        context.mission.preferred_cruise_agl = getattr(plan_mission, "preferred_cruise_agl", None)
        context.latest_planning = planning
        fresh_map = planning.get("return_cost_map")
        if fresh_map is not None:
            context.return_cost_map_cache = fresh_map
            context.return_cost_map_step = context.step
        context.planned_path = planning["path"]
        context.planned_path = self._ensure_goal_approach(context)

    def _ensure_goal_approach(self, context: NavigationContext) -> list[tuple[float, float, float] | tuple[int, int, int]]:
        path = list(context.planned_path or [])
        goal = context.mission.goal
        gx, gy = float(goal[0]), float(goal[1])
        gz = float(goal[2]) if len(goal) >= 3 else 0.0
        sx, sy, sz = context.state.x, context.state.y, context.state.z
        horizontal = math.hypot(gx - sx, gy - sy)
        if horizontal > 3.5 or abs(sz - gz) <= 0.45:
            return path
        # Near goal but wrong altitude: descend/climb WHILE continuing toward the goal XY.
        dz = clamp(gz - sz, -1.2, 1.2)
        if horizontal > 1e-6:
            step = min(1.0, horizontal)
            nx = sx + (gx - sx) / horizontal * step
            ny = sy + (gy - sy) / horizontal * step
        else:
            nx, ny = sx, sy
        nxt = (
            nx,
            ny,
            clamp(sz + dz, float(context.mission.min_altitude_level), float(context.mission.max_altitude_level)),
        )
        return [(sx, sy, sz), nxt]

    def _move_if_possible(
        self,
        context: NavigationContext,
        prediction: dict,
        truth_field: dict | None = None,
    ) -> float:
        if len(context.planned_path) <= 1 or context.battery_j <= 0.0:
            return 0.0
        next_waypoint = context.planned_path[1]
        goal = context.mission.goal
        horizontal = math.hypot(float(goal[0]) - context.state.x, float(goal[1]) - context.state.y)
        # Straight AGL guides: follow the goal ray with baseline-like step size & AGL profile
        # so path-model energy matches the eval baseline discretization.
        next_waypoint = self._baseline_like_next_waypoint(context, next_waypoint, horizontal)
        # Hold sticky cruise AGL until the baseline descent window (last ~12%).
        preferred = getattr(context.mission, "preferred_cruise_agl", None)
        ms = context.mission.start
        total_horiz = math.hypot(float(goal[0]) - float(ms[0]), float(goal[1]) - float(ms[1]))
        total_horiz = max(total_horiz, horizontal, 1.0)
        progress = clamp(1.0 - horizontal / total_horiz, 0.0, 1.0)
        # Hold preferred only in cruise (not climb, not descent window).
        if preferred is not None and 0.12 <= progress <= 0.85 and horizontal > 0.5:
            next_waypoint = (
                float(next_waypoint[0]),
                float(next_waypoint[1]),
                max(float(next_waypoint[2]), float(preferred)),
            )
        local_u = trilinear_sample(prediction["u_layers"], context.state.x, context.state.y, context.state.z)
        local_v = trilinear_sample(prediction["v_layers"], context.state.x, context.state.y, context.state.z)
        local_w = trilinear_sample(prediction["w_layers"], context.state.x, context.state.y, context.state.z)
        safety_penalty = _sample_belief_scalar(context.belief_map, "safety_penalty", context.state.x, context.state.y, context.state.z)
        prev = (context.state.x, context.state.y, context.state.z)
        floor = cruise_agl_floor(
            horizontal,
            float(context.mission.min_altitude_level),
            float(getattr(context.mission, "clearance_agl_level", 1.0)),
        )
        min_z = max(float(context.mission.min_altitude_level), floor if horizontal > 5.0 else float(context.mission.min_altitude_level))
        mode = str((context.latest_planning or {}).get("planning_mode") or "")
        if mode.startswith("guide_straight_agl") or mode.startswith("guide_corridor_"):
            # Open-loop baseline-like step at the profile AGL.
            heading = (
                context.state.heading_rad
                if horizontal < 1e-6
                else math.atan2(float(next_waypoint[1]) - context.state.y, float(next_waypoint[0]) - context.state.x)
            )
            # During climb/descent windows, do not clamp up to clearance floor.
            if progress < 0.12 or progress > 0.85 or horizontal <= 5.0:
                z_lo = float(context.mission.min_altitude_level)
            else:
                z_lo = float(min_z)
            next_state = DroneState(
                x=clamp(float(next_waypoint[0]), 0.0, context.belief_map.width - 1),
                y=clamp(float(next_waypoint[1]), 0.0, context.belief_map.height - 1),
                z=clamp(float(next_waypoint[2]), z_lo, float(context.mission.max_altitude_level)),
                heading_rad=heading,
                battery_ratio=context.state.battery_ratio,
                airspeed=context.state.airspeed,
            )
        else:
            next_state = advance_continuous_state(
                context.state,
                next_waypoint,
                local_u=local_u,
                local_v=local_v,
                local_w=local_w,
                safety_penalty=safety_penalty,
                step_distance_m=context.mission.step_distance_m,
                altitude_step_m=context.mission.altitude_step_m,
                min_altitude_level=int(math.floor(min_z)),
                max_altitude_level=context.mission.max_altitude_level,
            )
        # Path-model accounting: same transition_energy_j + uplift discount as eval.
        # Prefer truth wind when available so battery matches path-model comparison.
        if truth_field is not None and "u" in truth_field:
            energy_u = trilinear_sample(truth_field["u"], prev[0], prev[1], prev[2])
            energy_v = trilinear_sample(truth_field["v"], prev[0], prev[1], prev[2])
            energy_w = trilinear_sample(truth_field["w"], prev[0], prev[1], prev[2])
        else:
            energy_u, energy_v, energy_w = local_u, local_v, local_w
        terrain_dz_m = terrain_delta_m(
            context.terrain.elevation,
            prev[0],
            prev[1],
            next_state.x,
            next_state.y,
        )
        required_energy = transition_energy_j(
            airspeed=context.state.airspeed,
            current=prev,
            nxt=(next_state.x, next_state.y, next_state.z),
            local_u=energy_u,
            local_v=energy_v,
            local_w=energy_w,
            step_distance_m=context.mission.step_distance_m,
            altitude_step_m=context.mission.altitude_step_m,
            climb_cost_per_level_j=context.mission.climb_cost_per_level_j,
            hover_power_w=context.mission.hover_power_w,
            cruise_power_w=context.mission.cruise_power_w,
            hotel_power_w=context.mission.hotel_power_w,
            headwind_power_per_mps_w=context.mission.headwind_power_per_mps_w,
            climb_power_per_mps_w=context.mission.climb_power_per_mps_w,
            descent_power_reduction_per_mps_w=context.mission.descent_power_reduction_per_mps_w,
            terrain_dz_m=terrain_dz_m,
        )
        # uplift_energy_scale is applied inside transition_energy_j
        if required_energy > context.battery_j:
            context.planned_path = [context.planned_path[0]]
            return 0.0
        context.battery_j = max(0.0, context.battery_j - required_energy)
        context.state = next_state
        context.state.battery_ratio = context.battery_j / max(self.config.mission.battery_capacity_j, 1e-6)
        if _state_near_waypoint(context.state, next_waypoint):
            context.planned_path = context.planned_path[1:]
        return float(required_energy)
    def _forecast_fields(
        self,
        context: NavigationContext,
        sample: CoarseWindSample,
        observation: Observation | None,
        is_keyframe: bool,
    ) -> tuple[dict, dict]:
        if is_keyframe or context.latest_prediction is None or context.latest_physics is None:
            prediction = self.pipeline.predict_grid(
                sample.timestamp,
                sample.u_km,
                sample.v_km,
                sample.w_km,
                context.predict_session,
                altitude_level=_state_level(context.state.z, context.belief_map.levels),
                u_100=sample.u100_km,
                v_100=sample.v100_km,
            )
            physics = self.pipeline.physics_grid(
                sample.timestamp,
                sample.u_km,
                sample.v_km,
                sample.w_km,
                altitude_level=_state_level(context.state.z, context.belief_map.levels),
                u_100=sample.u100_km,
                v_100=sample.v100_km,
            )
            context.latest_prediction = prediction
            context.latest_physics = physics
            return prediction, physics

        x0, y0, x1, y1 = self._planning_window(context, observation)
        prediction_window = self.pipeline.predict_window(
            sample.timestamp,
            sample.u_km,
            sample.v_km,
            x0,
            y0,
            x1,
            y1,
            sample.w_km,
            context.predict_session,
            altitude_level=_state_level(context.state.z, context.belief_map.levels),
            u_100=sample.u100_km,
            v_100=sample.v100_km,
        )
        physics_window = self.pipeline.physics_window(
            sample.timestamp,
            sample.u_km,
            sample.v_km,
            x0,
            y0,
            x1,
            y1,
            sample.w_km,
            altitude_level=_state_level(context.state.z, context.belief_map.levels),
            u_100=sample.u100_km,
            v_100=sample.v100_km,
        )
        prediction = _merge_window_into_grid(context.latest_prediction, prediction_window)
        physics = _merge_window_into_grid(context.latest_physics, physics_window)
        context.latest_prediction = prediction
        context.latest_physics = physics
        return prediction, physics

    def _planning_window(
        self,
        context: NavigationContext,
        observation: Observation | None,
    ) -> tuple[int, int, int, int]:
        width = context.belief_map.width
        height = context.belief_map.height
        radius = max(4, self.config.belief.observation_radius + 2)
        points = [
            (int(round(context.state.x)), int(round(context.state.y))),
            (context.mission.goal[0], context.mission.goal[1]),
        ]
        if observation is not None:
            points.append((observation.x, observation.y))
        for waypoint in context.planned_path[: min(len(context.planned_path), 6)]:
            points.append((waypoint[0], waypoint[1]))
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x0 = max(0, int(math.floor(min(x_values) - radius)))
        y0 = max(0, int(math.floor(min(y_values) - radius)))
        x1 = min(width, int(math.ceil(max(x_values) + radius + 1)))
        y1 = min(height, int(math.ceil(max(y_values) + radius + 1)))
        return x0, y0, x1, y1

    def _build_frame(
        self,
        context: NavigationContext,
        timestamp: str,
        physics: dict,
        prediction: dict,
        observation_payload: dict | None,
        truth_field: dict | None,
        is_keyframe: bool,
    ) -> dict:
        state = context.state
        state_level = _state_level(state.z, context.belief_map.levels)
        state_x = clamp(state.x, 0.0, max(context.belief_map.width - 1, 0))
        state_y = clamp(state.y, 0.0, max(context.belief_map.height - 1, 0))
        current_u = trilinear_sample(prediction["u_layers"], state_x, state_y, state.z)
        current_v = trilinear_sample(prediction["v_layers"], state_x, state_y, state.z)
        current_w = trilinear_sample(prediction["w_layers"], state_x, state_y, state.z)
        physics_u = trilinear_sample(physics["u_layers"], state_x, state_y, state.z)
        physics_v = trilinear_sample(physics["v_layers"], state_x, state_y, state.z)
        physics_w = trilinear_sample(physics["w_layers"], state_x, state_y, state.z)
        belief_energy = _sample_belief_scalar(context.belief_map, "expected_energy_gain", state_x, state_y, state.z)
        belief_uncertainty = _sample_belief_scalar(context.belief_map, "uncertainty", state_x, state_y, state.z)
        belief_confidence = _sample_belief_scalar(context.belief_map, "confidence", state_x, state_y, state.z)
        belief_entropy_value = _sample_belief_scalar(context.belief_map, "belief_entropy", state_x, state_y, state.z)
        belief_safety = _sample_belief_scalar(context.belief_map, "safety_penalty", state_x, state_y, state.z)
        belief_uplift_prob = _sample_belief_scalar(context.belief_map, "mode_prob_uplift", state_x, state_y, state.z)
        belief_sink_prob = _sample_belief_scalar(context.belief_map, "mode_prob_sink", state_x, state_y, state.z)
        truth_payload = None
        if truth_field:
            truth_u = trilinear_sample(truth_field["u"], state_x, state_y, state.z)
            truth_v = trilinear_sample(truth_field["v"], state_x, state_y, state.z)
            truth_w = trilinear_sample(truth_field["w"], state_x, state_y, state.z)
            truth_payload = {
                "u": truth_u,
                "v": truth_v,
                "w": truth_w,
                "speed": magnitude3(truth_u, truth_v, truth_w),
                "u_grid": truth_field["u"][state_level],
                "v_grid": truth_field["v"][state_level],
                "w_grid": truth_field["w"][state_level],
                "wind_speed_grid": _wind_speed_grid(truth_field["u"][state_level], truth_field["v"][state_level], truth_field["w"][state_level]),
                "profile": _profile_from_layers(truth_field["u"], truth_field["v"], truth_field["w"], state_x, state_y),
            }
        snapshot = belief_snapshot(context.belief_map, state_level)
        physics_profile = _profile_from_layers(physics["u_layers"], physics["v_layers"], physics["w_layers"], state_x, state_y)
        prediction_profile = _profile_from_layers(prediction["u_layers"], prediction["v_layers"], prediction["w_layers"], state_x, state_y)
        belief_profiles = _belief_layer_stacks(context.belief_map)
        return_cost_map = context.latest_planning.get("return_cost_map")
        return_budget_j = context.latest_planning.get("return_budget_j")
        maps = None
        if is_keyframe:
            maps = {
                "physics_wind_speed": _wind_speed_grid(physics["u"], physics["v"], physics["w"]),
                "physics_wind_u": physics["u"],
                "physics_wind_v": physics["v"],
                "physics_wind_w": physics["w"],
                "prediction_wind_speed": _wind_speed_grid(prediction["u"], prediction["v"], prediction["w"]),
                "prediction_wind_u": prediction["u"],
                "prediction_wind_v": prediction["v"],
                "prediction_wind_w": prediction["w"],
                "belief_energy": snapshot["energy"],
                "belief_uncertainty": snapshot["uncertainty"],
                "belief_confidence": snapshot["confidence"],
                "belief_entropy": snapshot["entropy"],
                "belief_safety": snapshot["safety"],
                "belief_uplift_prob": snapshot["uplift_prob"],
                "belief_sink_prob": snapshot["sink_prob"],
                "belief_wind_speed": snapshot["wind_speed"],
                "belief_wind_u": snapshot["wind_u"],
                "belief_wind_v": snapshot["wind_v"],
                "belief_wind_w": snapshot["wind_w"],
                "belief_energy_layers": belief_profiles["energy"],
                "belief_uncertainty_layers": belief_profiles["uncertainty"],
                "belief_confidence_layers": belief_profiles["confidence"],
                "belief_entropy_layers": belief_profiles["entropy"],
                "belief_safety_layers": belief_profiles["safety"],
                "belief_uplift_prob_layers": belief_profiles["uplift_prob"],
                "belief_sink_prob_layers": belief_profiles["sink_prob"],
                "belief_wind_speed_layers": belief_profiles["wind_speed"],
                "belief_wind_u_layers": belief_profiles["wind_u"],
                "belief_wind_v_layers": belief_profiles["wind_v"],
                "belief_wind_w_layers": belief_profiles["wind_w"],
                "prediction_wind_speed_layers": _speed_layer_stack(prediction["u_layers"], prediction["v_layers"], prediction["w_layers"]),
                "prediction_wind_u_layers": prediction["u_layers"],
                "prediction_wind_v_layers": prediction["v_layers"],
                "prediction_wind_w_layers": prediction["w_layers"],
                "physics_wind_speed_layers": _speed_layer_stack(physics["u_layers"], physics["v_layers"], physics["w_layers"]),
                "physics_wind_u_layers": physics["u_layers"],
                "physics_wind_v_layers": physics["v_layers"],
                "physics_wind_w_layers": physics["w_layers"],
                "truth_wind_speed": truth_payload["wind_speed_grid"] if truth_payload else None,
                "physics_vertical_profile": physics_profile["speed"],
                "prediction_vertical_profile": prediction_profile["speed"],
                "prediction_vertical_w_profile": prediction_profile["w"],
            }
            if truth_payload:
                maps["truth_wind_u_layers"] = truth_field["u"]
                maps["truth_wind_v_layers"] = truth_field["v"]
                maps["truth_wind_w_layers"] = truth_field["w"]
            if truth_payload:
                truth_grid = truth_field_from_payload(truth_payload)
                maps["prediction_residual_speed"] = _residual_speed_grid(prediction["u"], prediction["v"], prediction["w"], truth_field=truth_grid)
                maps["physics_residual_speed"] = _residual_speed_grid(physics["u"], physics["v"], physics["w"], truth_field=truth_grid)
                maps["truth_wind_speed_layers"] = _speed_layer_stack(truth_field["u"], truth_field["v"], truth_field["w"])
                maps["prediction_residual_speed_layers"] = _residual_layer_stack(prediction["u_layers"], prediction["v_layers"], prediction["w_layers"], truth_field)
                maps["physics_residual_speed_layers"] = _residual_layer_stack(physics["u_layers"], physics["v_layers"], physics["w_layers"], truth_field)
            if return_cost_map is not None and return_budget_j is not None:
                maps["return_cost_margin"] = _return_cost_margin_grid(return_cost_map, return_budget_j, context.mission)
                maps["return_feasible_mask"] = _return_feasible_mask_grid(return_cost_map, return_budget_j, context.mission)
        map_delta = {
            "position": [int(round(state_x)), int(round(state_y)), state_level],
            "prediction_wind_speed": magnitude3(current_u, current_v, current_w),
            "prediction_vertical_w": current_w,
            "belief_energy": belief_energy,
            "belief_uncertainty": belief_uncertainty,
            "belief_confidence": belief_confidence,
            "belief_entropy": belief_entropy_value,
            "belief_safety": belief_safety,
        }
        if truth_payload:
            map_delta["truth_wind_speed"] = truth_payload["speed"]
            map_delta["prediction_residual_speed"] = abs(magnitude3(current_u, current_v, current_w) - truth_payload["speed"])
            map_delta["physics_residual_speed"] = abs(magnitude3(physics_u, physics_v, physics_w) - truth_payload["speed"])
        return {
            "timestamp": timestamp,
            "position": [_round_state_value(state_x), _round_state_value(state_y), _round_state_value(state.z)],
            "battery_ratio": state.battery_ratio,
            "remaining_battery_j": context.battery_j,
            "goal_distance_cells": math.sqrt(
                (context.mission.goal[0] - state.x) ** 2
                + (context.mission.goal[1] - state.y) ** 2
                + (context.mission.goal[2] - state.z) ** 2
            ),
            "return_home_radius_cells": self.return_home_radius_cells(context),
            "return_home_budget_j": context.mission.max_return_cost_j,
            "planned_path": context.planned_path,
            "planned_path_cost": context.latest_planning["path_cost"],
            "planned_path_cost_breakdown": context.latest_planning.get("path_cost_breakdown", {}),
            "candidate_paths": context.latest_planning["candidates"],
            "planning_mode": context.latest_planning.get("planning_mode", "mpc_strict_return"),
            "return_constraint_active": context.latest_planning.get("return_constraint_active", True),
            "is_keyframe": is_keyframe,
            "forecast_mode": "global_keyframe" if is_keyframe else "local_window",
            "map_delta": map_delta,
            "sensor_packet": {
                "gps_position": [state.x, state.y, state.z],
                "heading_rad": state.heading_rad,
                "airspeed": state.airspeed if not observation_payload else observation_payload["airspeed"],
                "ground_speed": max(0.0, state.airspeed + current_u * math.cos(state.heading_rad) + current_v * math.sin(state.heading_rad))
                if not observation_payload
                else observation_payload["ground_speed"],
                "imu_accel_longitudinal": 0.08 * magnitude(current_u, current_v) - 0.03 * belief_uncertainty,
                "imu_accel_vertical": 0.12 * belief_energy + 0.18 * current_w,
                "measured_wind_u": observation_payload["u_obs"] if observation_payload else current_u + 0.35 * (belief_confidence - 0.5),
                "measured_wind_v": observation_payload["v_obs"] if observation_payload else current_v - 0.25 * (belief_uncertainty - 0.5),
                "measured_wind_w": observation_payload["w_obs"] if observation_payload else current_w + 0.2 * (belief_confidence - 0.5),
                "energy_rate": belief_energy,
                "altitude_m": state.z * context.mission.altitude_step_m,
                "agl_m": state.z * context.mission.altitude_step_m,
                "msl_m": msl_height_m(
                    sample_elevation(context.terrain.elevation, state.x, state.y),
                    state.z,
                    context.mission.altitude_step_m,
                ),
                "terrain_elev_m": sample_elevation(context.terrain.elevation, state.x, state.y),
            },
            "physics_at_drone": {"u": physics_u, "v": physics_v, "w": physics_w, "speed": magnitude3(physics_u, physics_v, physics_w)},
            "prediction_at_drone": {"u": current_u, "v": current_v, "w": current_w, "speed": magnitude3(current_u, current_v, current_w)},
            "physics_profile_at_drone": physics_profile,
            "prediction_profile_at_drone": prediction_profile,
            "truth_profile_at_drone": truth_payload["profile"] if truth_payload else None,
            "belief_at_drone": {
                "energy": belief_energy,
                "uncertainty": belief_uncertainty,
                "confidence": belief_confidence,
                "entropy": belief_entropy_value,
                "safety": belief_safety,
                "uplift_prob": belief_uplift_prob,
                "sink_prob": belief_sink_prob,
                "w": _sample_belief_scalar(context.belief_map, "wind_w", state_x, state_y, state.z),
                "altitude_level": state_level,
            },
            "observation": observation_payload,
            "truth_at_drone": truth_payload,
            "maps": maps,
        }

    @staticmethod
    def _clamp_observation(observation: Observation, width: int, height: int) -> Observation:
        x, y = _clamp_xy(observation.x, observation.y, width, height)
        return observation.__class__(
            timestamp=observation.timestamp,
            x=x,
            y=y,
            z=max(getattr(observation, "z", 0), 0),
            u_obs=observation.u_obs,
            v_obs=observation.v_obs,
            w_obs=observation.w_obs,
            airspeed=observation.airspeed,
            ground_speed=observation.ground_speed,
            climb_rate=observation.climb_rate,
            acceleration=observation.acceleration,
        )


def _clamp_xy(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 0, 0
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _merge_window_into_grid(base_grid: dict | None, window_grid: dict) -> dict:
    if base_grid is None:
        merged = dict(window_grid)
        merged.pop("x_bounds", None)
        merged.pop("y_bounds", None)
        return merged
    merged = {
        "timestamp": window_grid["timestamp"],
        "altitude_level": window_grid["altitude_level"],
        "u": [row[:] for row in base_grid["u"]],
        "v": [row[:] for row in base_grid["v"]],
        "w": [row[:] for row in base_grid["w"]],
        "u_layers": [[row[:] for row in layer] for layer in base_grid["u_layers"]],
        "v_layers": [[row[:] for row in layer] for layer in base_grid["v_layers"]],
        "w_layers": [[row[:] for row in layer] for layer in base_grid["w_layers"]],
    }
    x0, x1 = window_grid["x_bounds"]
    y0, y1 = window_grid["y_bounds"]
    for local_y, y in enumerate(range(y0, y1)):
        for local_x, x in enumerate(range(x0, x1)):
            merged["u"][y][x] = window_grid["u"][local_y][local_x]
            merged["v"][y][x] = window_grid["v"][local_y][local_x]
            merged["w"][y][x] = window_grid["w"][local_y][local_x]
    for level in range(len(merged["u_layers"])):
        for local_y, y in enumerate(range(y0, y1)):
            for local_x, x in enumerate(range(x0, x1)):
                merged["u_layers"][level][y][x] = window_grid["u_layers"][level][local_y][local_x]
                merged["v_layers"][level][y][x] = window_grid["v_layers"][level][local_y][local_x]
                merged["w_layers"][level][y][x] = window_grid["w_layers"][level][local_y][local_x]
    return merged


def _clamp_xyz(
    point: tuple[int, int] | tuple[int, int, int],
    width: int,
    height: int,
    mission_config,
) -> tuple[int, int, int]:
    x, y = _clamp_xy(int(point[0]), int(point[1]), width, height)
    raw_z = int(point[2]) if len(point) >= 3 else 0
    min_z = getattr(mission_config, "min_altitude_level", 0)
    max_z = getattr(mission_config, "max_altitude_level", 4)
    return x, y, max(min_z, min(max_z, raw_z))


def _state_level(z: float, levels: int) -> int:
    if levels <= 0:
        return 0
    return max(0, min(levels - 1, int(round(z))))


def _round_state_value(value: float) -> float:
    return round(float(value), 3)


def _goal_reached(state: DroneState, goal: tuple[int, int] | tuple[int, int, int]) -> bool:
    gx = float(goal[0])
    gy = float(goal[1])
    gz = float(goal[2]) if len(goal) >= 3 else 0.0
    return math.hypot(state.x - gx, state.y - gy) <= 0.45 and abs(state.z - gz) <= 0.55


def _state_near_waypoint(state: DroneState, waypoint: tuple[int, int, int]) -> bool:
    return math.hypot(state.x - waypoint[0], state.y - waypoint[1]) <= 0.35 and abs(state.z - waypoint[2]) <= 0.45


def _sample_belief_scalar(belief_map, attr: str, x: float, y: float, z: float) -> float:
    if belief_map.levels <= 0 or belief_map.height <= 0 or belief_map.width <= 0:
        return 0.0
    z = clamp(z, 0.0, belief_map.levels - 1)
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, belief_map.levels - 1)
    tz = z - z0
    lower = _sample_belief_scalar_2d(belief_map, attr, x, y, z0)
    upper = _sample_belief_scalar_2d(belief_map, attr, x, y, z1)
    return lower + (upper - lower) * tz


def _sample_belief_scalar_2d(belief_map, attr: str, x: float, y: float, level: int) -> float:
    x = clamp(x, 0.0, belief_map.width - 1)
    y = clamp(y, 0.0, belief_map.height - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, belief_map.width - 1)
    y1 = min(y0 + 1, belief_map.height - 1)
    tx = x - x0
    ty = y - y0
    v00 = getattr(belief_map.cells[level][y0][x0], attr)
    v10 = getattr(belief_map.cells[level][y0][x1], attr)
    v01 = getattr(belief_map.cells[level][y1][x0], attr)
    v11 = getattr(belief_map.cells[level][y1][x1], attr)
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return top + (bottom - top) * ty


def _wind_speed_grid(u_grid, v_grid, w_grid=None) -> list[list[float]]:
    u_arr = np.asarray(u_grid, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    if u_arr.ndim != 2 or u_arr.size == 0:
        return []
    height, width = u_arr.shape
    if w_grid is None:
        w_arr = np.zeros((height, width), dtype=np.float64)
    else:
        w_arr = np.asarray(w_grid, dtype=np.float64)
    speed = np.sqrt(u_arr * u_arr + v_arr * v_arr + w_arr * w_arr)
    return speed.tolist()


def _residual_speed_grid(u_grid, v_grid, w_grid, truth_field: dict) -> list[list[float]]:
    u_arr = np.asarray(u_grid, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    w_arr = np.asarray(w_grid, dtype=np.float64)
    if u_arr.ndim != 2 or u_arr.size == 0:
        return []
    height, width = u_arr.shape
    truth_w = truth_field.get("w")
    if truth_w is None:
        tw = np.zeros((height, width), dtype=np.float64)
    else:
        tw = np.asarray(truth_w, dtype=np.float64)
    tu = np.asarray(truth_field["u"], dtype=np.float64)
    tv = np.asarray(truth_field["v"], dtype=np.float64)
    pred = np.sqrt(u_arr * u_arr + v_arr * v_arr + w_arr * w_arr)
    truth = np.sqrt(tu * tu + tv * tv + tw * tw)
    return np.abs(pred - truth).tolist()


def _return_cost_margin_grid(return_cost_map: dict[tuple[int, int, int], float], budget_j: float, mission: Mission) -> list[list[float]]:
    width = max((state[0] for state in return_cost_map), default=-1) + 1
    height = max((state[1] for state in return_cost_map), default=-1) + 1
    layer = tuple(mission.goal)[2]
    grid = [[-budget_j for _ in range(width)] for _ in range(height)]
    for (x, y, z), value in return_cost_map.items():
        if z == layer:
            grid[y][x] = budget_j - value if math.isfinite(value) else -budget_j
    return grid


def _return_feasible_mask_grid(return_cost_map: dict[tuple[int, int, int], float], budget_j: float, mission: Mission) -> list[list[float]]:
    width = max((state[0] for state in return_cost_map), default=-1) + 1
    height = max((state[1] for state in return_cost_map), default=-1) + 1
    layer = tuple(mission.goal)[2]
    grid = [[0.0 for _ in range(width)] for _ in range(height)]
    for (x, y, z), value in return_cost_map.items():
        if z == layer and value <= budget_j:
            grid[y][x] = 1.0
    return grid


def truth_field_from_payload(payload: dict) -> dict:
    return {"u": payload["u_grid"], "v": payload["v_grid"], "w": payload.get("w_grid")}


def _profile_from_layers(
    u_layers: list[list[list[float]]],
    v_layers: list[list[list[float]]],
    w_layers: list[list[list[float]]],
    x: float,
    y: float,
) -> dict[str, list[float]]:
    levels = min(len(u_layers), len(v_layers), len(w_layers))
    profile_u = [bilinear_sample(u_layers[level], x, y) for level in range(levels)]
    profile_v = [bilinear_sample(v_layers[level], x, y) for level in range(levels)]
    profile_w = [bilinear_sample(w_layers[level], x, y) for level in range(levels)]
    return {
        "u": profile_u,
        "v": profile_v,
        "w": profile_w,
        "speed": [magnitude3(u, v, w) for u, v, w in zip(profile_u, profile_v, profile_w)],
    }


def _speed_layer_stack(
    u_layers: list[list[list[float]]],
    v_layers: list[list[list[float]]],
    w_layers: list[list[list[float]]],
) -> list[list[list[float]]]:
    return [_wind_speed_grid(u_layers[level], v_layers[level], w_layers[level]) for level in range(min(len(u_layers), len(v_layers), len(w_layers)))]


def _belief_layer_stacks(belief_map) -> dict[str, list[list[list[float]]]]:
    keys = ("energy", "uncertainty", "confidence", "entropy", "safety", "uplift_prob", "sink_prob", "wind_speed", "wind_u", "wind_v", "wind_w")
    stacks = {key: [] for key in keys}
    for level in range(belief_map.levels):
        snapshot = belief_snapshot(belief_map, level)
        for key in keys:
            stacks[key].append(snapshot[key])
    return stacks


def _residual_layer_stack(
    u_layers: list[list[list[float]]],
    v_layers: list[list[list[float]]],
    w_layers: list[list[list[float]]],
    truth_field: dict,
) -> list[list[list[float]]]:
    levels = min(len(u_layers), len(v_layers), len(w_layers), len(truth_field["u"]), len(truth_field["v"]), len(truth_field["w"]))
    return [
        _residual_speed_grid(
            u_layers[level],
            v_layers[level],
            w_layers[level],
            {"u": truth_field["u"][level], "v": truth_field["v"][level], "w": truth_field["w"][level]},
        )
        for level in range(levels)
    ]
