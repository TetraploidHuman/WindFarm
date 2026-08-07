from __future__ import annotations

from dataclasses import dataclass
import math

from .mathutils import magnitude, magnitude3
from .types import BeliefCell, BeliefMap, Observation, WindField


def create_belief_map(width: int, height: int, levels: int = 1) -> BeliefMap:
    cells = [[[BeliefCell() for _ in range(width)] for _ in range(height)] for _ in range(max(1, levels))]
    return BeliefMap(width=width, height=height, levels=max(1, levels), cells=cells)


def belief_snapshot(belief_map: BeliefMap, level: int = 0) -> dict[str, list[list[float]]]:
    active_level = _clamp_level(level, belief_map.levels)
    energy = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    uncertainty = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    confidence = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    entropy = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    safety = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    uplift_prob = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    sink_prob = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    wind_speed = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    wind_u = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    wind_v = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    wind_w = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    vertical_w = [[0.0 for _ in range(belief_map.width)] for _ in range(belief_map.height)]
    for y in range(belief_map.height):
        for x in range(belief_map.width):
            cell = belief_map.cells[active_level][y][x]
            energy[y][x] = cell.expected_energy_gain
            uncertainty[y][x] = cell.uncertainty
            confidence[y][x] = cell.confidence
            entropy[y][x] = cell.belief_entropy
            safety[y][x] = cell.safety_penalty
            uplift_prob[y][x] = cell.mode_prob_uplift
            sink_prob[y][x] = cell.mode_prob_sink
            wind_u[y][x] = cell.wind_u
            wind_v[y][x] = cell.wind_v
            wind_w[y][x] = cell.wind_w
            vertical_w[y][x] = cell.wind_w
            wind_speed[y][x] = magnitude3(cell.wind_u, cell.wind_v, cell.wind_w)
    return {
        "energy": energy,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "entropy": entropy,
        "safety": safety,
        "uplift_prob": uplift_prob,
        "sink_prob": sink_prob,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "wind_w": wind_w,
        "vertical_w": vertical_w,
        "wind_speed": wind_speed,
    }


@dataclass(slots=True)
class BeliefUpdater:
    observation_radius: int = 2
    advection_gain: float = 0.25
    decay_per_step: float = 0.04
    process_noise: float = 0.18
    observation_noise: float = 0.12
    advection_noise: float = 0.08

    def apply_prediction(self, belief_map: BeliefMap, wind: WindField, step: int) -> None:
        for z in range(belief_map.levels):
            for y in range(belief_map.height):
                for x in range(belief_map.width):
                    cell = belief_map.cells[z][y][x]
                    age = max(step - cell.last_update, 0)
                    process_var = self.process_noise * (1.0 + self.decay_per_step * age)
                    cell.wind_u = wind.u[z][y][x]
                    cell.wind_v = wind.v[z][y][x]
                    cell.wind_w = wind.w[z][y][x]
                    predicted_modes = _predict_mode_probs(cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift)
                    cell.mode_prob_sink = predicted_modes[0]
                    cell.mode_prob_neutral = predicted_modes[1]
                    cell.mode_prob_uplift = predicted_modes[2]
                    horizontal_speed = magnitude(cell.wind_u, cell.wind_v)
                    cell.wind_var_u = min(4.0, cell.wind_var_u + process_var)
                    cell.wind_var_v = min(4.0, cell.wind_var_v + process_var)
                    cell.wind_var_w = min(4.0, cell.wind_var_w + process_var)
                    cell.wind_cov_uv *= max(0.0, 1.0 - self.decay_per_step)
                    mode_entropy = _entropy(cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift)
                    cell.belief_entropy = mode_entropy
                    cell.uncertainty = ((cell.wind_var_u + cell.wind_var_v + cell.wind_var_w) / 3.0) + 0.25 * mode_entropy
                    cell.hazard_prob = _clamp01(
                        0.58 * cell.mode_prob_sink
                        + 0.18 * _logistic((-cell.wind_w - 0.25) * 1.7)
                        + 0.12 * _logistic(horizontal_speed - 5.0)
                        + 0.10 * _clamp01(cell.uncertainty / 3.0)
                    )
                    cell.safety_penalty = _clamp01(
                        0.52 * cell.hazard_prob
                        + 0.28 * _clamp01(cell.uncertainty / 3.0)
                        + 0.20 * _clamp01(abs(cell.wind_w) / 2.5)
                    )
                    cell.expected_energy_gain = _expected_energy_from_cell(cell, horizontal_speed)
                    cell.confidence = 1.0 / (1.0 + cell.uncertainty)

    def update_with_observation(
        self,
        belief_map: BeliefMap,
        observation: Observation,
        predicted_u: float,
        predicted_v: float,
        predicted_w: float,
        step: int,
    ) -> None:
        level = _clamp_level(observation.z, belief_map.levels)
        energy_gain = 0.5 * (observation.ground_speed - max(observation.airspeed, 0.0)) + observation.climb_rate
        observed_modes = _observe_mode_probs(observation.w_obs, energy_gain)
        for y in range(max(0, observation.y - self.observation_radius),
                       min(belief_map.height, observation.y + self.observation_radius + 1)):
            for x in range(max(0, observation.x - self.observation_radius),
                           min(belief_map.width, observation.x + self.observation_radius + 1)):
                distance = math.hypot(x - observation.x, y - observation.y)
                influence = max(0.0, 1.0 - distance / (self.observation_radius + 1))
                if influence <= 0.0:
                    continue
                cell = belief_map.cells[level][y][x]
                measurement_var = self.observation_noise / max(influence * influence, 1e-6)
                p00 = cell.wind_var_u
                p01 = cell.wind_cov_uv
                p10 = cell.wind_cov_uv
                p11 = cell.wind_var_v
                s00 = p00 + measurement_var
                s01 = p01
                s10 = p10
                s11 = p11 + measurement_var
                det_s = max((s00 * s11) - (s01 * s10), 1e-6)
                inv_s00 = s11 / det_s
                inv_s01 = -s01 / det_s
                inv_s10 = -s10 / det_s
                inv_s11 = s00 / det_s
                k00 = p00 * inv_s00 + p01 * inv_s10
                k01 = p00 * inv_s01 + p01 * inv_s11
                k10 = p10 * inv_s00 + p11 * inv_s10
                k11 = p10 * inv_s01 + p11 * inv_s11
                residual_u = observation.u_obs - cell.wind_u
                residual_v = observation.v_obs - cell.wind_v
                cell.wind_u = cell.wind_u + k00 * residual_u + k01 * residual_v
                cell.wind_v = cell.wind_v + k10 * residual_u + k11 * residual_v
                residual_w = observation.w_obs - cell.wind_w
                kalman_w = cell.wind_var_w / max(cell.wind_var_w + measurement_var, 1e-6)
                cell.wind_w = cell.wind_w + kalman_w * residual_w
                predicted_residual = magnitude3(
                    observation.u_obs - predicted_u,
                    observation.v_obs - predicted_v,
                    observation.w_obs - predicted_w,
                )
                updated_p00 = (1.0 - k00) * p00 - k01 * p10
                updated_p01 = (1.0 - k00) * p01 - k01 * p11
                updated_p10 = -k10 * p00 + (1.0 - k11) * p10
                updated_p11 = -k10 * p01 + (1.0 - k11) * p11
                cell.wind_var_u = max(0.02, updated_p00)
                cell.wind_var_v = max(0.02, updated_p11)
                cell.wind_var_w = max(0.02, (1.0 - kalman_w) * cell.wind_var_w)
                cell.wind_cov_uv = max(-1.0, min(1.0, 0.5 * (updated_p01 + updated_p10)))
                gain_scale = 0.5 * (k00 + k11)
                cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift = _blend_mode_probs(
                    (cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift),
                    observed_modes,
                    _clamp01(gain_scale * influence + 0.18),
                )
                mode_entropy = _entropy(cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift)
                cell.belief_entropy = mode_entropy
                horizontal_speed = magnitude(cell.wind_u, cell.wind_v)
                cell.expected_energy_gain = (
                    (1.0 - gain_scale * influence) * cell.expected_energy_gain
                    + (gain_scale * influence) * energy_gain
                )
                cell.expected_energy_gain = 0.5 * cell.expected_energy_gain + 0.5 * _expected_energy_from_cell(cell, horizontal_speed)
                cell.uncertainty = ((cell.wind_var_u + cell.wind_var_v + cell.wind_var_w) / 3.0) + 0.25 * mode_entropy
                local_shear = magnitude(cell.wind_u - predicted_u, cell.wind_v - predicted_v)
                observed_hazard = _clamp01(
                    0.42 * cell.mode_prob_sink
                    + 0.18 * _logistic((-observation.w_obs - 0.2) * 2.2)
                    + 0.16 * _clamp01(predicted_residual / 3.0)
                    + 0.12 * _clamp01(local_shear / 4.0)
                    + 0.12 * _clamp01(cell.uncertainty / 3.0)
                )
                cell.hazard_prob = 0.65 * cell.hazard_prob + 0.35 * observed_hazard
                cell.safety_penalty = _clamp01(
                    0.50 * cell.hazard_prob
                    + 0.25 * _clamp01(cell.uncertainty / 3.0)
                    + 0.15 * _clamp01(abs(observation.acceleration) / 0.35)
                    + 0.10 * _clamp01(max(0.0, -observation.climb_rate) / 1.4)
                )
                cell.confidence = 1.0 / (1.0 + cell.uncertainty)
                cell.last_update = step
        self._advect_along_wind(belief_map, observation.x, observation.y, level)

    def _advect_along_wind(self, belief_map: BeliefMap, x: int, y: int, z: int) -> None:
        origin = belief_map.cells[z][y][x]
        dx = 0 if abs(origin.wind_u) < 1e-6 else int(math.copysign(1, origin.wind_u))
        dy = 0 if abs(origin.wind_v) < 1e-6 else int(math.copysign(1, origin.wind_v))
        for step in range(1, 4):
            nx = x + dx * step
            ny = y + dy * step
            if 0 <= nx < belief_map.width and 0 <= ny < belief_map.height:
                cell = belief_map.cells[z][ny][nx]
                influence = self.advection_gain / step
                cell.wind_u = (1.0 - influence) * cell.wind_u + influence * origin.wind_u
                cell.wind_v = (1.0 - influence) * cell.wind_v + influence * origin.wind_v
                cell.wind_w = (1.0 - influence) * cell.wind_w + influence * origin.wind_w
                cell.wind_var_u = min(4.0, (1.0 - influence) * cell.wind_var_u + influence * (origin.wind_var_u + self.advection_noise))
                cell.wind_var_v = min(4.0, (1.0 - influence) * cell.wind_var_v + influence * (origin.wind_var_v + self.advection_noise))
                cell.wind_var_w = min(4.0, (1.0 - influence) * cell.wind_var_w + influence * (origin.wind_var_w + self.advection_noise))
                cell.wind_cov_uv = (1.0 - influence) * cell.wind_cov_uv + influence * origin.wind_cov_uv
                cell.expected_energy_gain = (1.0 - influence) * cell.expected_energy_gain + influence * origin.expected_energy_gain
                cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift = _blend_mode_probs(
                    (cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift),
                    (origin.mode_prob_sink, origin.mode_prob_neutral, origin.mode_prob_uplift),
                    influence,
                )
                cell.belief_entropy = _entropy(cell.mode_prob_sink, cell.mode_prob_neutral, cell.mode_prob_uplift)
                cell.hazard_prob = (1.0 - influence) * cell.hazard_prob + influence * origin.hazard_prob
                cell.safety_penalty = _clamp01((1.0 - influence) * cell.safety_penalty + influence * origin.safety_penalty)
                cell.uncertainty = ((cell.wind_var_u + cell.wind_var_v + cell.wind_var_w) / 3.0) + 0.25 * cell.belief_entropy
                cell.confidence = 1.0 / (1.0 + cell.uncertainty)


def _clamp_level(level: int, levels: int) -> int:
    if levels <= 0:
        return 0
    return max(0, min(levels - 1, int(level)))


def _predict_mode_probs(sink: float, neutral: float, uplift: float) -> tuple[float, float, float]:
    next_sink = 0.72 * sink + 0.18 * neutral + 0.06 * uplift
    next_neutral = 0.20 * sink + 0.64 * neutral + 0.20 * uplift
    next_uplift = 0.08 * sink + 0.18 * neutral + 0.74 * uplift
    return _normalize_modes(next_sink, next_neutral, next_uplift)


def _observe_mode_probs(vertical_w: float, energy_gain: float) -> tuple[float, float, float]:
    uplift_score = _logistic(2.6 * vertical_w + 0.9 * energy_gain)
    sink_score = _logistic(-2.4 * vertical_w - 0.7 * energy_gain)
    neutral_score = max(0.05, 1.0 - abs(vertical_w) - 0.22 * abs(energy_gain))
    return _normalize_modes(sink_score, neutral_score, uplift_score)


def _blend_mode_probs(
    prior: tuple[float, float, float],
    observed: tuple[float, float, float],
    update_gain: float,
) -> tuple[float, float, float]:
    sink = (1.0 - update_gain) * prior[0] + update_gain * observed[0]
    neutral = (1.0 - update_gain) * prior[1] + update_gain * observed[1]
    uplift = (1.0 - update_gain) * prior[2] + update_gain * observed[2]
    return _normalize_modes(sink, neutral, uplift)


def _normalize_modes(sink: float, neutral: float, uplift: float) -> tuple[float, float, float]:
    floor = 1e-6
    total = max(sink + neutral + uplift, floor * 3.0)
    return max(sink, floor) / total, max(neutral, floor) / total, max(uplift, floor) / total


def _entropy(*probs: float) -> float:
    total = 0.0
    for prob in probs:
        if prob > 1e-9:
            total -= prob * math.log(prob, 2)
    return total


def _expected_energy_from_cell(cell: BeliefCell, horizontal_speed: float) -> float:
    mode_bias = (1.25 * cell.mode_prob_uplift) - (0.95 * cell.mode_prob_sink)
    return 0.16 * horizontal_speed + 1.05 * cell.wind_w + mode_bias - 0.28 * cell.safety_penalty


def _logistic(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
