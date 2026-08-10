from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mathutils import magnitude, magnitude3
from .types import BeliefCell, BeliefMap, Observation, WindField


def create_belief_map(width: int, height: int, levels: int = 1) -> BeliefMap:
    z_n = max(1, levels)
    cells = [[[BeliefCell() for _ in range(width)] for _ in range(height)] for _ in range(z_n)]
    belief_map = BeliefMap(width=width, height=height, levels=z_n, cells=cells)
    belief_map.field_arrays = _default_field_arrays(z_n, height, width)
    return belief_map


def _default_field_arrays(z_n: int, height: int, width: int) -> dict[str, np.ndarray]:
    shape = (z_n, height, width)
    return {
        "wind_u": np.zeros(shape, dtype=np.float64),
        "wind_v": np.zeros(shape, dtype=np.float64),
        "wind_w": np.zeros(shape, dtype=np.float64),
        "mode_prob_sink": np.full(shape, 0.25, dtype=np.float64),
        "mode_prob_neutral": np.full(shape, 0.5, dtype=np.float64),
        "mode_prob_uplift": np.full(shape, 0.25, dtype=np.float64),
        "wind_var_u": np.ones(shape, dtype=np.float64),
        "wind_var_v": np.ones(shape, dtype=np.float64),
        "wind_var_w": np.ones(shape, dtype=np.float64),
        "wind_cov_uv": np.zeros(shape, dtype=np.float64),
        "belief_entropy": np.full(shape, 1.5, dtype=np.float64),
        "uncertainty": np.ones(shape, dtype=np.float64),
        "hazard_prob": np.full(shape, 0.15, dtype=np.float64),
        "safety_penalty": np.zeros(shape, dtype=np.float64),
        "expected_energy_gain": np.zeros(shape, dtype=np.float64),
        "confidence": np.zeros(shape, dtype=np.float64),
        "last_update": np.zeros(shape, dtype=np.float64),
    }


def sync_belief_cells_from_arrays(belief_map: BeliefMap) -> None:
    """Copy field_arrays → BeliefCell grid (dashboard / legacy cell readers)."""
    arrays = belief_map.field_arrays
    if not arrays:
        return
    z_n, h, w = belief_map.levels, belief_map.height, belief_map.width
    cells = belief_map.cells
    names = (
        "wind_u",
        "wind_v",
        "wind_w",
        "mode_prob_sink",
        "mode_prob_neutral",
        "mode_prob_uplift",
        "wind_var_u",
        "wind_var_v",
        "wind_var_w",
        "wind_cov_uv",
        "belief_entropy",
        "uncertainty",
        "hazard_prob",
        "safety_penalty",
        "expected_energy_gain",
        "confidence",
        "last_update",
    )
    packed = {name: np.asarray(arrays[name], dtype=np.float64) for name in names if name in arrays}
    for z in range(z_n):
        layer = cells[z]
        for y in range(h):
            row = layer[y]
            for x in range(w):
                cell = row[x]
                for name, arr in packed.items():
                    if name == "last_update":
                        cell.last_update = int(arr[z, y, x])
                    else:
                        setattr(cell, name, float(arr[z, y, x]))


def belief_snapshot(belief_map: BeliefMap, level: int = 0) -> dict[str, list[list[float]]]:
    active_level = _clamp_level(level, belief_map.levels)
    arrays = belief_map.field_arrays
    if arrays is not None and "wind_u" in arrays:
        def _layer(name: str) -> list[list[float]]:
            return np.asarray(arrays[name][active_level], dtype=np.float64).tolist()

        wind_u = _layer("wind_u")
        wind_v = _layer("wind_v")
        wind_w = _layer("wind_w")
        wu = np.asarray(arrays["wind_u"][active_level], dtype=np.float64)
        wv = np.asarray(arrays["wind_v"][active_level], dtype=np.float64)
        ww = np.asarray(arrays["wind_w"][active_level], dtype=np.float64)
        speed = np.sqrt(wu * wu + wv * wv + ww * ww).tolist()
        return {
            "energy": _layer("expected_energy_gain"),
            "uncertainty": _layer("uncertainty"),
            "confidence": _layer("confidence"),
            "entropy": _layer("belief_entropy"),
            "safety": _layer("safety_penalty"),
            "uplift_prob": _layer("mode_prob_uplift"),
            "sink_prob": _layer("mode_prob_sink"),
            "wind_u": wind_u,
            "wind_v": wind_v,
            "wind_w": wind_w,
            "vertical_w": wind_w,
            "wind_speed": speed,
        }

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


def pack_belief_field(belief_map: BeliefMap, attr: str) -> np.ndarray:
    """Pack a BeliefCell attribute into a (Z, H, W) float64 array."""
    arrays = belief_map.field_arrays
    if arrays is not None and attr in arrays:
        return np.asarray(arrays[attr], dtype=np.float64)
    z_n, h, w = belief_map.levels, belief_map.height, belief_map.width
    out = np.empty((z_n, h, w), dtype=np.float64)
    cells = belief_map.cells
    for z in range(z_n):
        layer = cells[z]
        for y in range(h):
            row = layer[y]
            for x in range(w):
                out[z, y, x] = getattr(row[x], attr)
    return out


def ensure_belief_field_arrays(belief_map: BeliefMap, attrs: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Return (and cache) packed belief fields used by the planner.

    Returns references into ``belief_map.field_arrays`` when possible (no copy).
    """
    needed = attrs or (
        "wind_u",
        "wind_v",
        "wind_w",
        "expected_energy_gain",
        "uncertainty",
        "safety_penalty",
        "mode_prob_uplift",
        "mode_prob_sink",
        "confidence",
        "belief_entropy",
    )
    arrays = belief_map.field_arrays
    if arrays is not None and all(name in arrays for name in needed):
        # Hot path: reuse existing ndarrays; avoid np.asarray copy every call.
        return {name: arrays[name] for name in needed}  # type: ignore[misc]
    packed = {name: pack_belief_field(belief_map, name) for name in needed}
    if belief_map.field_arrays is None:
        belief_map.field_arrays = packed
    else:
        belief_map.field_arrays.update(packed)
    return packed


@dataclass(slots=True)
class BeliefUpdater:
    observation_radius: int = 2
    advection_gain: float = 0.25
    decay_per_step: float = 0.04
    process_noise: float = 0.18
    observation_noise: float = 0.12
    advection_noise: float = 0.08

    def apply_prediction(self, belief_map: BeliefMap, wind: WindField, step: int) -> None:
        z_n, h, w = belief_map.levels, belief_map.height, belief_map.width
        if z_n <= 0 or h <= 0 or w <= 0:
            return

        wind_u = np.asarray(wind.u, dtype=np.float64)
        wind_v = np.asarray(wind.v, dtype=np.float64)
        wind_w = np.asarray(wind.w, dtype=np.float64)
        if wind_u.ndim != 3:
            wind_u = np.asarray(wind_u, dtype=np.float64).reshape(z_n, h, w)
            wind_v = np.asarray(wind_v, dtype=np.float64).reshape(z_n, h, w)
            wind_w = np.asarray(wind_w, dtype=np.float64).reshape(z_n, h, w)

        arrays = belief_map.field_arrays
        if arrays is not None and "last_update" in arrays and "wind_var_u" in arrays:
            last_update = np.asarray(arrays["last_update"], dtype=np.float64)
            var_u = np.asarray(arrays["wind_var_u"], dtype=np.float64).copy()
            var_v = np.asarray(arrays["wind_var_v"], dtype=np.float64).copy()
            var_w = np.asarray(arrays["wind_var_w"], dtype=np.float64).copy()
            cov_uv = np.asarray(arrays["wind_cov_uv"], dtype=np.float64).copy()
            mode_s = np.asarray(arrays["mode_prob_sink"], dtype=np.float64).copy()
            mode_n = np.asarray(arrays["mode_prob_neutral"], dtype=np.float64).copy()
            mode_u = np.asarray(arrays["mode_prob_uplift"], dtype=np.float64).copy()
        else:
            last_update = np.empty((z_n, h, w), dtype=np.float64)
            var_u = np.empty((z_n, h, w), dtype=np.float64)
            var_v = np.empty((z_n, h, w), dtype=np.float64)
            var_w = np.empty((z_n, h, w), dtype=np.float64)
            cov_uv = np.empty((z_n, h, w), dtype=np.float64)
            mode_s = np.empty((z_n, h, w), dtype=np.float64)
            mode_n = np.empty((z_n, h, w), dtype=np.float64)
            mode_u = np.empty((z_n, h, w), dtype=np.float64)
            cells = belief_map.cells
            for z in range(z_n):
                layer = cells[z]
                for y in range(h):
                    row = layer[y]
                    for x in range(w):
                        cell = row[x]
                        last_update[z, y, x] = cell.last_update
                        var_u[z, y, x] = cell.wind_var_u
                        var_v[z, y, x] = cell.wind_var_v
                        var_w[z, y, x] = cell.wind_var_w
                        cov_uv[z, y, x] = cell.wind_cov_uv
                        mode_s[z, y, x] = cell.mode_prob_sink
                        mode_n[z, y, x] = cell.mode_prob_neutral
                        mode_u[z, y, x] = cell.mode_prob_uplift

        age = np.maximum(step - last_update, 0.0)
        process_var = self.process_noise * (1.0 + self.decay_per_step * age)

        next_s = 0.72 * mode_s + 0.18 * mode_n + 0.06 * mode_u
        next_n = 0.20 * mode_s + 0.64 * mode_n + 0.20 * mode_u
        next_u = 0.08 * mode_s + 0.18 * mode_n + 0.74 * mode_u
        floor = 1e-6
        total = np.maximum(next_s + next_n + next_u, floor * 3.0)
        mode_s = np.maximum(next_s, floor) / total
        mode_n = np.maximum(next_n, floor) / total
        mode_u = np.maximum(next_u, floor) / total

        var_u = np.minimum(4.0, var_u + process_var)
        var_v = np.minimum(4.0, var_v + process_var)
        var_w = np.minimum(4.0, var_w + process_var)
        cov_uv = cov_uv * max(0.0, 1.0 - self.decay_per_step)

        # Fuse prediction with prior belief instead of overwriting (keeps obs / shear).
        pred_u = wind_u
        pred_v = wind_v
        pred_w = wind_w
        if arrays is not None and "wind_u" in arrays:
            prior_u = np.asarray(arrays["wind_u"], dtype=np.float64)
            prior_v = np.asarray(arrays["wind_v"], dtype=np.float64)
            prior_w = np.asarray(arrays["wind_w"], dtype=np.float64)
            prior_mag = np.abs(prior_u) + np.abs(prior_v) + np.abs(prior_w)
            informed = (last_update > 0) | (prior_mag > 1e-6)
            pred_noise = max(float(self.observation_noise), 0.10) + 0.05
            ku = var_u / (var_u + pred_noise)
            kv = var_v / (var_v + pred_noise)
            kw = var_w / (var_w + pred_noise)
            # Uninformed cells take the prediction fully (cold start).
            ku = np.where(informed, ku, 1.0)
            kv = np.where(informed, kv, 1.0)
            kw = np.where(informed, kw, 1.0)
            # Recently observed cells resist overwrite more.
            fresh = np.exp(-2.0 * self.decay_per_step * age)
            resist = np.where(last_update > 0, 1.0 - 0.55 * fresh, 1.0)
            # Shear / uplift lobes that disagree with a smooth forecast must not be
            # washed out (shanxi/jilin/taiwan-class edges). Observed cells get a
            # stronger horizontal hold; informed priors get a milder resist so
            # downscaled shear lobes survive early predict steps before first obs.
            innov_h = np.hypot(prior_u - pred_u, prior_v - pred_v)
            innov_w = np.abs(prior_w - pred_w)
            edge_protect_obs = np.clip(0.55 * innov_h + 0.35 * innov_w, 0.0, 0.70)
            edge_protect_prior = np.clip(0.28 * innov_h + 0.15 * innov_w, 0.0, 0.40)
            edge_protect = np.where(
                last_update > 0,
                edge_protect_obs,
                np.where(informed, edge_protect_prior, 0.0),
            )
            gain_scale = resist * (1.0 - edge_protect)
            ku = ku * gain_scale
            kv = kv * gain_scale
            kw = kw * gain_scale
            wind_u = prior_u + ku * (pred_u - prior_u)
            wind_v = prior_v + kv * (pred_v - prior_v)
            wind_w = prior_w + kw * (pred_w - prior_w)
            var_u = np.minimum(4.0, (1.0 - ku) * var_u + ku * pred_noise)
            var_v = np.minimum(4.0, (1.0 - kv) * var_v + kv * pred_noise)
            var_w = np.minimum(4.0, (1.0 - kw) * var_w + kw * pred_noise)

        mode_entropy = _entropy_np(mode_s, mode_n, mode_u)
        horizontal_speed = np.hypot(wind_u, wind_v)
        uncertainty = ((var_u + var_v + var_w) / 3.0) + 0.25 * mode_entropy
        hazard = _clamp01_np(
            0.58 * mode_s
            + 0.18 * _logistic_np((-wind_w - 0.25) * 1.7)
            + 0.12 * _logistic_np(horizontal_speed - 5.0)
            + 0.10 * _clamp01_np(uncertainty / 3.0)
        )
        # Punish downdraft / shear risk only — rising air is useful, not hazardous.
        safety = _clamp01_np(
            0.52 * hazard
            + 0.28 * _clamp01_np(uncertainty / 3.0)
            + 0.20 * _clamp01_np(np.maximum(-wind_w, 0.0) / 2.5)
        )
        # Direction-agnostic horizontal speed is NOT a gain (could be headwind).
        expected = (
            0.55 * np.maximum(wind_w, 0.0)
            - 0.85 * np.maximum(-wind_w, 0.0)
            + (1.25 * mode_u - 0.95 * mode_s)
            - 0.28 * safety
        )
        confidence = 1.0 / (1.0 + uncertainty)

        # Arrays are the hot-path source of truth; skip O(ZHW) cell writeback.
        belief_map.field_arrays = {
            "wind_u": wind_u,
            "wind_v": wind_v,
            "wind_w": wind_w,
            "mode_prob_sink": mode_s,
            "mode_prob_neutral": mode_n,
            "mode_prob_uplift": mode_u,
            "wind_var_u": var_u,
            "wind_var_v": var_v,
            "wind_var_w": var_w,
            "wind_cov_uv": cov_uv,
            "belief_entropy": mode_entropy,
            "uncertainty": uncertainty,
            "hazard_prob": hazard,
            "safety_penalty": safety,
            "expected_energy_gain": expected,
            "confidence": confidence,
            "last_update": last_update,
        }

    def update_with_observation(
        self,
        belief_map: BeliefMap,
        observation: Observation,
        predicted_u: float,
        predicted_v: float,
        predicted_w: float,
        step: int,
        *,
        observation_radius: int | None = None,
        advect: bool = True,
    ) -> None:
        level = _clamp_level(observation.z, belief_map.levels)
        energy_gain = 0.5 * (observation.ground_speed - max(observation.airspeed, 0.0)) + observation.climb_rate
        observed_modes = _observe_mode_probs(observation.w_obs, energy_gain)
        radius = self.observation_radius if observation_radius is None else max(0, int(observation_radius))
        arrays = belief_map.field_arrays
        if arrays is None or "wind_u" not in arrays:
            # Legacy cell-only path (tests without field_arrays).
            self._update_with_observation_cells(
                belief_map,
                observation,
                predicted_u,
                predicted_v,
                predicted_w,
                step,
                level,
                energy_gain,
                observed_modes,
                observation_radius=radius,
                advect=advect,
            )
            return

        y0 = max(0, observation.y - radius)
        y1 = min(belief_map.height, observation.y + radius + 1)
        x0 = max(0, observation.x - radius)
        x1 = min(belief_map.width, observation.x + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - observation.x, yy - observation.y)
        influence = np.maximum(0.0, 1.0 - distance / (radius + 1))
        mask = influence > 0.0
        if not np.any(mask):
            return

        wind_u = np.asarray(arrays["wind_u"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        wind_v = np.asarray(arrays["wind_v"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        wind_w = np.asarray(arrays["wind_w"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        var_u = np.asarray(arrays["wind_var_u"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        var_v = np.asarray(arrays["wind_var_v"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        var_w = np.asarray(arrays["wind_var_w"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        cov_uv = np.asarray(arrays["wind_cov_uv"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        mode_s = np.asarray(arrays["mode_prob_sink"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        mode_n = np.asarray(arrays["mode_prob_neutral"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        mode_u = np.asarray(arrays["mode_prob_uplift"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        expected = np.asarray(arrays["expected_energy_gain"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        hazard = np.asarray(arrays["hazard_prob"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        safety = np.asarray(arrays["safety_penalty"][level, y0:y1, x0:x1], dtype=np.float64).copy()

        measurement_var = self.observation_noise / np.maximum(influence * influence, 1e-6)
        p00, p01, p10, p11 = var_u, cov_uv, cov_uv, var_v
        s00 = p00 + measurement_var
        s01 = p01
        s10 = p10
        s11 = p11 + measurement_var
        det_s = np.maximum((s00 * s11) - (s01 * s10), 1e-6)
        inv_s00 = s11 / det_s
        inv_s01 = -s01 / det_s
        inv_s10 = -s10 / det_s
        inv_s11 = s00 / det_s
        k00 = p00 * inv_s00 + p01 * inv_s10
        k01 = p00 * inv_s01 + p01 * inv_s11
        k10 = p10 * inv_s00 + p11 * inv_s10
        k11 = p10 * inv_s01 + p11 * inv_s11
        residual_u = observation.u_obs - wind_u
        residual_v = observation.v_obs - wind_v
        wind_u = wind_u + k00 * residual_u + k01 * residual_v
        wind_v = wind_v + k10 * residual_u + k11 * residual_v
        residual_w = observation.w_obs - wind_w
        kalman_w = var_w / np.maximum(var_w + measurement_var, 1e-6)
        wind_w = wind_w + kalman_w * residual_w
        predicted_residual = magnitude3(
            observation.u_obs - predicted_u,
            observation.v_obs - predicted_v,
            observation.w_obs - predicted_w,
        )
        updated_p00 = (1.0 - k00) * p00 - k01 * p10
        updated_p01 = (1.0 - k00) * p01 - k01 * p11
        updated_p10 = -k10 * p00 + (1.0 - k11) * p10
        updated_p11 = -k10 * p01 + (1.0 - k11) * p11
        var_u = np.maximum(0.02, updated_p00)
        var_v = np.maximum(0.02, updated_p11)
        var_w = np.maximum(0.02, (1.0 - kalman_w) * var_w)
        cov_uv = np.clip(0.5 * (updated_p01 + updated_p10), -1.0, 1.0)
        gain_scale = 0.5 * (k00 + k11)
        blend = _clamp01_np(gain_scale * influence + 0.18)
        obs_s, obs_n, obs_u = observed_modes
        mode_s, mode_n, mode_u = _blend_mode_probs_np(
            mode_s, mode_n, mode_u, obs_s, obs_n, obs_u, blend
        )
        mode_entropy = _entropy_np(mode_s, mode_n, mode_u)
        expected = (1.0 - gain_scale * influence) * expected + (gain_scale * influence) * energy_gain
        expected_from_wind = (
            0.55 * np.maximum(wind_w, 0.0)
            - 0.85 * np.maximum(-wind_w, 0.0)
            + (1.25 * mode_u - 0.95 * mode_s)
            - 0.28 * safety
        )
        expected = 0.5 * expected + 0.5 * expected_from_wind
        uncertainty = ((var_u + var_v + var_w) / 3.0) + 0.25 * mode_entropy
        local_shear = np.hypot(wind_u - predicted_u, wind_v - predicted_v)
        observed_hazard = _clamp01_np(
            0.42 * mode_s
            + 0.18 * _logistic_np((-observation.w_obs - 0.2) * 2.2)
            + 0.16 * _clamp01_np(np.full_like(influence, predicted_residual / 3.0))
            + 0.12 * _clamp01_np(local_shear / 4.0)
            + 0.12 * _clamp01_np(uncertainty / 3.0)
        )
        hazard = 0.65 * hazard + 0.35 * observed_hazard
        safety = _clamp01_np(
            0.50 * hazard
            + 0.25 * _clamp01_np(uncertainty / 3.0)
            + 0.15 * _clamp01_np(abs(observation.acceleration) / 0.35)
            + 0.10 * _clamp01_np(max(0.0, -observation.climb_rate) / 1.4)
        )
        confidence = 1.0 / (1.0 + uncertainty)

        # Write only influenced cells back into the full arrays.
        def _put(name: str, patch: np.ndarray) -> None:
            arr = arrays[name]
            layer = arr[level, y0:y1, x0:x1]
            layer = np.asarray(layer, dtype=np.float64).copy()
            layer[mask] = patch[mask]
            arr[level, y0:y1, x0:x1] = layer

        _put("wind_u", wind_u)
        _put("wind_v", wind_v)
        _put("wind_w", wind_w)
        _put("wind_var_u", var_u)
        _put("wind_var_v", var_v)
        _put("wind_var_w", var_w)
        _put("wind_cov_uv", cov_uv)
        _put("mode_prob_sink", mode_s)
        _put("mode_prob_neutral", mode_n)
        _put("mode_prob_uplift", mode_u)
        _put("belief_entropy", mode_entropy)
        _put("uncertainty", uncertainty)
        _put("hazard_prob", hazard)
        _put("safety_penalty", safety)
        _put("expected_energy_gain", expected)
        _put("confidence", confidence)
        last = np.asarray(arrays["last_update"][level, y0:y1, x0:x1], dtype=np.float64).copy()
        last[mask] = float(step)
        arrays["last_update"][level, y0:y1, x0:x1] = last

        # Keep origin cell in sync for advection (reads cells).
        oy, ox = observation.y, observation.x
        if 0 <= oy < belief_map.height and 0 <= ox < belief_map.width:
            cell = belief_map.cells[level][oy][ox]
            cell.wind_u = float(arrays["wind_u"][level, oy, ox])
            cell.wind_v = float(arrays["wind_v"][level, oy, ox])
            cell.wind_w = float(arrays["wind_w"][level, oy, ox])
            cell.wind_var_u = float(arrays["wind_var_u"][level, oy, ox])
            cell.wind_var_v = float(arrays["wind_var_v"][level, oy, ox])
            cell.wind_var_w = float(arrays["wind_var_w"][level, oy, ox])
            cell.wind_cov_uv = float(arrays["wind_cov_uv"][level, oy, ox])
            cell.mode_prob_sink = float(arrays["mode_prob_sink"][level, oy, ox])
            cell.mode_prob_neutral = float(arrays["mode_prob_neutral"][level, oy, ox])
            cell.mode_prob_uplift = float(arrays["mode_prob_uplift"][level, oy, ox])
            cell.expected_energy_gain = float(arrays["expected_energy_gain"][level, oy, ox])
            cell.uncertainty = float(arrays["uncertainty"][level, oy, ox])
            cell.hazard_prob = float(arrays["hazard_prob"][level, oy, ox])
            cell.safety_penalty = float(arrays["safety_penalty"][level, oy, ox])
            cell.belief_entropy = float(arrays["belief_entropy"][level, oy, ox])
            cell.confidence = float(arrays["confidence"][level, oy, ox])
            cell.last_update = step
        if advect:
            self._advect_along_wind(belief_map, observation.x, observation.y, level)

    def _update_with_observation_cells(
        self,
        belief_map: BeliefMap,
        observation: Observation,
        predicted_u: float,
        predicted_v: float,
        predicted_w: float,
        step: int,
        level: int,
        energy_gain: float,
        observed_modes: tuple[float, float, float],
        *,
        observation_radius: int | None = None,
        advect: bool = True,
    ) -> None:
        radius = self.observation_radius if observation_radius is None else max(0, int(observation_radius))
        for y in range(max(0, observation.y - radius),
                       min(belief_map.height, observation.y + radius + 1)):
            for x in range(max(0, observation.x - radius),
                           min(belief_map.width, observation.x + radius + 1)):
                distance = math.hypot(x - observation.x, y - observation.y)
                influence = max(0.0, 1.0 - distance / (radius + 1))
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
        if advect:
            self._advect_along_wind(belief_map, observation.x, observation.y, level)

    def _advect_along_wind(self, belief_map: BeliefMap, x: int, y: int, z: int) -> None:
        arrays = belief_map.field_arrays
        origin = belief_map.cells[z][y][x]
        if arrays is not None and "wind_u" in arrays:
            origin.wind_u = float(arrays["wind_u"][z, y, x])
            origin.wind_v = float(arrays["wind_v"][z, y, x])
            origin.wind_w = float(arrays["wind_w"][z, y, x])
            origin.wind_var_u = float(arrays["wind_var_u"][z, y, x])
            origin.wind_var_v = float(arrays["wind_var_v"][z, y, x])
            origin.wind_var_w = float(arrays["wind_var_w"][z, y, x])
            origin.wind_cov_uv = float(arrays["wind_cov_uv"][z, y, x])
            origin.expected_energy_gain = float(arrays["expected_energy_gain"][z, y, x])
            origin.mode_prob_sink = float(arrays["mode_prob_sink"][z, y, x])
            origin.mode_prob_neutral = float(arrays["mode_prob_neutral"][z, y, x])
            origin.mode_prob_uplift = float(arrays["mode_prob_uplift"][z, y, x])
            origin.hazard_prob = float(arrays["hazard_prob"][z, y, x])
            origin.safety_penalty = float(arrays["safety_penalty"][z, y, x])
        dx = 0 if abs(origin.wind_u) < 1e-6 else int(math.copysign(1, origin.wind_u))
        dy = 0 if abs(origin.wind_v) < 1e-6 else int(math.copysign(1, origin.wind_v))
        for step in range(1, 4):
            nx = x + dx * step
            ny = y + dy * step
            if 0 <= nx < belief_map.width and 0 <= ny < belief_map.height:
                cell = belief_map.cells[z][ny][nx]
                if arrays is not None and "wind_u" in arrays:
                    cell.wind_u = float(arrays["wind_u"][z, ny, nx])
                    cell.wind_v = float(arrays["wind_v"][z, ny, nx])
                    cell.wind_w = float(arrays["wind_w"][z, ny, nx])
                    cell.wind_var_u = float(arrays["wind_var_u"][z, ny, nx])
                    cell.wind_var_v = float(arrays["wind_var_v"][z, ny, nx])
                    cell.wind_var_w = float(arrays["wind_var_w"][z, ny, nx])
                    cell.wind_cov_uv = float(arrays["wind_cov_uv"][z, ny, nx])
                    cell.expected_energy_gain = float(arrays["expected_energy_gain"][z, ny, nx])
                    cell.mode_prob_sink = float(arrays["mode_prob_sink"][z, ny, nx])
                    cell.mode_prob_neutral = float(arrays["mode_prob_neutral"][z, ny, nx])
                    cell.mode_prob_uplift = float(arrays["mode_prob_uplift"][z, ny, nx])
                    cell.hazard_prob = float(arrays["hazard_prob"][z, ny, nx])
                    cell.safety_penalty = float(arrays["safety_penalty"][z, ny, nx])
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
                arrays = belief_map.field_arrays
                if arrays is not None:
                    for name, value in (
                        ("wind_u", cell.wind_u),
                        ("wind_v", cell.wind_v),
                        ("wind_w", cell.wind_w),
                        ("mode_prob_sink", cell.mode_prob_sink),
                        ("mode_prob_neutral", cell.mode_prob_neutral),
                        ("mode_prob_uplift", cell.mode_prob_uplift),
                        ("wind_var_u", cell.wind_var_u),
                        ("wind_var_v", cell.wind_var_v),
                        ("wind_var_w", cell.wind_var_w),
                        ("wind_cov_uv", cell.wind_cov_uv),
                        ("belief_entropy", cell.belief_entropy),
                        ("uncertainty", cell.uncertainty),
                        ("hazard_prob", cell.hazard_prob),
                        ("safety_penalty", cell.safety_penalty),
                        ("expected_energy_gain", cell.expected_energy_gain),
                        ("confidence", cell.confidence),
                    ):
                        arr = arrays.get(name)
                        if arr is not None:
                            arr[z, ny, nx] = value


def _clamp_level(level: int, levels: int) -> int:
    if levels <= 0:
        return 0
    return max(0, min(levels - 1, int(level)))


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


def _blend_mode_probs_np(
    mode_s: np.ndarray,
    mode_n: np.ndarray,
    mode_u: np.ndarray,
    obs_s: float,
    obs_n: float,
    obs_u: float,
    update_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sink = (1.0 - update_gain) * mode_s + update_gain * obs_s
    neutral = (1.0 - update_gain) * mode_n + update_gain * obs_n
    uplift = (1.0 - update_gain) * mode_u + update_gain * obs_u
    floor = 1e-6
    total = np.maximum(sink + neutral + uplift, floor * 3.0)
    return np.maximum(sink, floor) / total, np.maximum(neutral, floor) / total, np.maximum(uplift, floor) / total


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


def _entropy_np(*probs: np.ndarray) -> np.ndarray:
    total = np.zeros_like(probs[0])
    for prob in probs:
        safe = np.maximum(prob, 1e-12)
        total = total - np.where(prob > 1e-9, prob * np.log2(safe), 0.0)
    return total


def _expected_energy_from_cell(cell: BeliefCell, horizontal_speed: float) -> float:
    del horizontal_speed  # unused: undirected speed is not an energy gain
    mode_bias = (1.25 * cell.mode_prob_uplift) - (0.95 * cell.mode_prob_sink)
    return (
        0.55 * max(cell.wind_w, 0.0)
        - 0.85 * max(-cell.wind_w, 0.0)
        + mode_bias
        - 0.28 * cell.safety_penalty
    )


def _logistic(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _logistic_np(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp01_np(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)
