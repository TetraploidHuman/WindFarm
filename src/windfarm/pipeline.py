from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
from pathlib import Path

import numpy as np

from .config import ModelConfig, TaskConfig, detect_cpu_count, resolve_n_jobs
from .execution import NavigationEngine, PredictSession, _goal_reached
from .io import (
    coarse_samples_from_dict,
    load_terrain_document,
    observations_from_dict,
    read_json,
    terrain_from_dict,
    training_samples_from_dict,
    write_json,
)
from .mathutils import hour_features, magnitude, magnitude3
from .ml import (
    ResidualWindModel,
    subsample_training_rows,
    subsample_training_samples,
    train_gradient_boosted_trees,
    train_lightgbm_regressor,
    train_ridge_regression,
    train_xgboost_regressor,
)
from .physics import downscale_wind
from .planner import path_distance_m
from .types import Mission, TerrainField, TrainingSample, WindField


FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "z_norm",
    "altitude_m",
    "elev",
    "slope",
    "aspect",
    "aspect_sin",
    "aspect_cos",
    "terrain_wave_sin",
    "terrain_wave_cos",
    "lee_wave_sin",
    "lee_wave_cos",
    "roughness",
    "u_km",
    "v_km",
    "w_km",
    "wind_speed_km",
    "u_phy",
    "v_phy",
    "w_phy",
    "u_km_slope",
    "v_km_slope",
    "w_km_slope",
    "f_slope",
    "f_rough",
    "delta_elev",
    "hour_sin",
    "hour_cos",
    "aspect_time_sin",
    "aspect_time_cos",
    "prev_u_km",
    "prev_v_km",
    "prev_w_km",
    "delta_u_km",
    "delta_v_km",
    "delta_w_km",
    "prev_local_u",
    "prev_local_v",
    "prev_local_w",
    "prev_local_speed",
    "local_trend_u",
    "local_trend_v",
    "local_trend_w",
    "prev2_u_km",
    "prev2_v_km",
    "prev2_w_km",
    "prev3_u_km",
    "prev3_v_km",
    "prev3_w_km",
    "coarse_u_mean",
    "coarse_v_mean",
    "coarse_w_mean",
    "prev2_local_u",
    "prev2_local_v",
    "prev2_local_w",
    "prev3_local_u",
    "prev3_local_v",
    "prev3_local_w",
    "local_u_mean",
    "local_v_mean",
    "local_w_mean",
]


@dataclass(slots=True)
class WindFarmPipeline:
    terrain: TerrainField
    residual_model: ResidualWindModel | None = None
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_metrics: dict = field(default_factory=dict)
    altitude_step_m: float = 50.0
    _physics_cache: dict[tuple[str, float, float, float, int], tuple[WindField, object]] = field(default_factory=dict)
    _static_feature_cache: dict[tuple[int, int, int], dict[str, float]] = field(default_factory=dict)
    _static_grids: dict[str, np.ndarray] | None = None

    def _ensure_static_grids(self) -> dict[str, np.ndarray]:
        if self._static_grids is not None:
            return self._static_grids
        elev = np.asarray(self.terrain.elevation, dtype=np.float64)
        slope = np.asarray(self.terrain.slope, dtype=np.float64)
        aspect = np.asarray(self.terrain.aspect, dtype=np.float64)
        roughness = np.asarray(self.terrain.roughness, dtype=np.float64)
        h, w = elev.shape if elev.ndim == 2 else (0, 0)
        yy, xx = np.mgrid[0:h, 0:w]
        terrain_wave = (xx + yy) / 7.5
        lee_wave = (xx - 0.6 * yy) / 6.0
        self._static_grids = {
            "elev": elev,
            "slope": slope,
            "aspect": aspect,
            "aspect_sin": np.sin(aspect),
            "aspect_cos": np.cos(aspect),
            "roughness": roughness,
            "x_norm": xx / max(w - 1, 1),
            "y_norm": yy / max(h - 1, 1),
            "terrain_wave_sin": np.sin(terrain_wave),
            "terrain_wave_cos": np.cos(terrain_wave),
            "lee_wave_sin": np.sin(lee_wave),
            "lee_wave_cos": np.cos(lee_wave),
        }
        return self._static_grids

    def _physics_bundle(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float = 0.0,
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ):
        alt_step = 50.0
        if getattr(self, "config", None) is not None:
            alt_step = float(getattr(self.config.mission, "altitude_step_m", 50.0))
        key = (
            timestamp,
            round(u_km, 6),
            round(v_km, 6),
            round(w_km, 6),
            None if u_100 is None else round(float(u_100), 6),
            None if v_100 is None else round(float(v_100), 6),
            round(alt_step, 3),
            self.model_config.altitude_levels,
        )
        cached = self._physics_cache.get(key)
        if cached is None:
            cached = downscale_wind(
                u_km,
                v_km,
                self.terrain,
                w_km,
                self.model_config.altitude_levels,
                altitude_step_m=alt_step,
                u_100=u_100,
                v_100=v_100,
            )
            self._physics_cache[key] = cached
        return cached

    def _feature_dict(
        self,
        timestamp: str,
        x: int,
        y: int,
        z: int,
        u_km: float,
        v_km: float,
        w_km: float = 0.0,
        temporal: dict[str, float] | None = None,
    ) -> dict[str, float]:
        wind_phy, diagnostics = self._physics_bundle(timestamp, u_km, v_km, w_km)
        hour_sin, hour_cos = hour_features(timestamp)
        level = _clamp_altitude_level(z, self.model_config.altitude_levels)
        features = dict(self._static_features(x, y, level))
        u_arr = np.asarray(wind_phy.u)
        v_arr = np.asarray(wind_phy.v)
        w_arr = np.asarray(wind_phy.w)
        u_phy = float(u_arr[level, y, x]) if u_arr.ndim == 3 else float(wind_phy.u[level][y][x])
        v_phy = float(v_arr[level, y, x]) if v_arr.ndim == 3 else float(wind_phy.v[level][y][x])
        w_phy = float(w_arr[level, y, x]) if w_arr.ndim == 3 else float(wind_phy.w[level][y][x])
        slope = features["slope"]
        aspect_sin = features["aspect_sin"]
        aspect_cos = features["aspect_cos"]
        f_slope = np.asarray(diagnostics.f_slope)
        f_rough = np.asarray(diagnostics.f_rough)
        delta_elev = np.asarray(diagnostics.delta_elev)
        features.update({
            "u_km": u_km,
            "v_km": v_km,
            "w_km": w_km,
            "wind_speed_km": magnitude3(u_km, v_km, w_km),
            "u_phy": u_phy,
            "v_phy": v_phy,
            "w_phy": w_phy,
            "u_km_slope": u_km * slope,
            "v_km_slope": v_km * slope,
            "w_km_slope": w_km * slope,
            "f_slope": float(f_slope[y, x]),
            "f_rough": float(f_rough[y, x]),
            "delta_elev": float(delta_elev[y, x]),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "aspect_time_sin": aspect_sin * hour_cos + aspect_cos * hour_sin,
            "aspect_time_cos": aspect_cos * hour_cos - aspect_sin * hour_sin,
            "prev_u_km": (temporal or {}).get("prev_u_km", u_km),
            "prev_v_km": (temporal or {}).get("prev_v_km", v_km),
            "prev_w_km": (temporal or {}).get("prev_w_km", w_km),
            "delta_u_km": (temporal or {}).get("delta_u_km", 0.0),
            "delta_v_km": (temporal or {}).get("delta_v_km", 0.0),
            "delta_w_km": (temporal or {}).get("delta_w_km", 0.0),
            "prev_local_u": (temporal or {}).get("prev_local_u", u_phy),
            "prev_local_v": (temporal or {}).get("prev_local_v", v_phy),
            "prev_local_w": (temporal or {}).get("prev_local_w", w_phy),
            "prev_local_speed": (temporal or {}).get("prev_local_speed", magnitude3(u_phy, v_phy, w_phy)),
            "local_trend_u": (temporal or {}).get("local_trend_u", 0.0),
            "local_trend_v": (temporal or {}).get("local_trend_v", 0.0),
            "local_trend_w": (temporal or {}).get("local_trend_w", 0.0),
            "prev2_u_km": (temporal or {}).get("prev2_u_km", u_km),
            "prev2_v_km": (temporal or {}).get("prev2_v_km", v_km),
            "prev2_w_km": (temporal or {}).get("prev2_w_km", w_km),
            "prev3_u_km": (temporal or {}).get("prev3_u_km", u_km),
            "prev3_v_km": (temporal or {}).get("prev3_v_km", v_km),
            "prev3_w_km": (temporal or {}).get("prev3_w_km", w_km),
            "coarse_u_mean": (temporal or {}).get("coarse_u_mean", u_km),
            "coarse_v_mean": (temporal or {}).get("coarse_v_mean", v_km),
            "coarse_w_mean": (temporal or {}).get("coarse_w_mean", w_km),
            "prev2_local_u": (temporal or {}).get("prev2_local_u", u_phy),
            "prev2_local_v": (temporal or {}).get("prev2_local_v", v_phy),
            "prev2_local_w": (temporal or {}).get("prev2_local_w", w_phy),
            "prev3_local_u": (temporal or {}).get("prev3_local_u", u_phy),
            "prev3_local_v": (temporal or {}).get("prev3_local_v", v_phy),
            "prev3_local_w": (temporal or {}).get("prev3_local_w", w_phy),
            "local_u_mean": (temporal or {}).get("local_u_mean", u_phy),
            "local_v_mean": (temporal or {}).get("local_v_mean", v_phy),
            "local_w_mean": (temporal or {}).get("local_w_mean", w_phy),
        })
        return features

    def _static_features(self, x: int, y: int, level: int) -> dict[str, float]:
        key = (x, y, level)
        cached = self._static_feature_cache.get(key)
        if cached is not None:
            return cached
        width = len(self.terrain.elevation[0]) if self.terrain.elevation else 1
        height = len(self.terrain.elevation)
        aspect = self.terrain.aspect[y][x]
        terrain_wave = (x + y) / 7.5
        lee_wave = (x - 0.6 * y) / 6.0
        cached = {
            "x_norm": x / max(width - 1, 1),
            "y_norm": y / max(height - 1, 1),
            "z_norm": level / max(self.model_config.altitude_levels - 1, 1),
            "altitude_m": level * float(self.altitude_step_m),
            "elev": self.terrain.elevation[y][x],
            "slope": self.terrain.slope[y][x],
            "aspect": aspect,
            "aspect_sin": math.sin(aspect),
            "aspect_cos": math.cos(aspect),
            "terrain_wave_sin": math.sin(terrain_wave),
            "terrain_wave_cos": math.cos(terrain_wave),
            "lee_wave_sin": math.sin(lee_wave),
            "lee_wave_cos": math.cos(lee_wave),
            "roughness": self.terrain.roughness[y][x],
        }
        self._static_feature_cache[key] = cached
        return cached

    def train(self, training_samples: list[TrainingSample], split_ratio: float | None = None) -> dict:
        # Training JSON is written time-ordered; skip O(N log N) sort when already sorted.
        if len(training_samples) < 2:
            ordered = list(training_samples)
        else:
            mid = len(training_samples) // 2
            if (
                training_samples[0].timestamp
                <= training_samples[mid].timestamp
                <= training_samples[-1].timestamp
            ):
                ordered = training_samples
            else:
                ordered = sorted(training_samples, key=lambda item: item.timestamp)
        ratio = split_ratio if split_ratio is not None else self.model_config.split_ratio
        split_idx = max(1, int(len(ordered) * ratio))
        train_samples = ordered[:split_idx]
        test_samples = ordered[split_idx:]
        # Subsample before feature extraction — avoids building features for ~1M unused rows.
        train_samples = subsample_training_samples(train_samples, self.model_config.max_training_samples)
        x_train: list[list[float]] = []
        y_u: list[float] = []
        y_v: list[float] = []
        y_w: list[float] = []
        train_temporal = self._sample_temporal_contexts(train_samples)
        for sample, temporal in zip(train_samples, train_temporal):
            features = self._feature_dict(sample.timestamp, sample.x, sample.y, sample.z, sample.u_km, sample.v_km, sample.w_km, temporal)
            x_train.append([features[name] for name in FEATURE_NAMES])
            y_u.append(sample.u_obs - features["u_phy"])
            y_v.append(sample.v_obs - features["v_phy"])
            y_w.append(sample.w_obs - features["w_phy"])
        x_train, y_u, y_v, y_w = subsample_training_rows(
            x_train,
            y_u,
            y_v,
            y_w,
            self.model_config.max_training_samples,
        )
        x_fit = x_train
        y_u_fit = y_u
        y_v_fit = y_v
        y_w_fit = y_w
        x_valid: list[list[float]] = []
        y_u_valid: list[float] = []
        y_v_valid: list[float] = []
        y_w_valid: list[float] = []
        validation_count = int(len(x_train) * self.model_config.validation_ratio)
        if self.model_config.model_type == "xgboost" and validation_count >= 32:
            split_train_idx = len(x_train) - validation_count
            x_fit = x_train[:split_train_idx]
            y_u_fit = y_u[:split_train_idx]
            y_v_fit = y_v[:split_train_idx]
            y_w_fit = y_w[:split_train_idx]
            x_valid = x_train[split_train_idx:]
            y_u_valid = y_u[split_train_idx:]
            y_v_valid = y_v[split_train_idx:]
            y_w_valid = y_w[split_train_idx:]
        actual_model_type = self.model_config.model_type
        if self.model_config.model_type == "ridge":
            u_model = train_ridge_regression(
                x_fit,
                y_u_fit,
                learning_rate=self.model_config.learning_rate,
                epochs=self.model_config.epochs,
                l2=self.model_config.l2,
            )
            v_model = train_ridge_regression(
                x_fit,
                y_v_fit,
                learning_rate=self.model_config.learning_rate,
                epochs=self.model_config.epochs,
                l2=self.model_config.l2,
            )
            w_model = train_ridge_regression(
                x_fit,
                y_w_fit,
                learning_rate=self.model_config.learning_rate,
                epochs=self.model_config.epochs,
                l2=self.model_config.l2,
            )
        elif self.model_config.model_type == "xgboost":
            try:
                u_model, v_model, w_model = self._fit_boosted_triplet(
                    train_xgboost_regressor,
                    x_fit,
                    y_u_fit,
                    y_v_fit,
                    y_w_fit,
                    x_valid,
                    y_u_valid,
                    y_v_valid,
                    y_w_valid,
                )
            except RuntimeError:
                try:
                    actual_model_type = "lightgbm"
                    u_model, v_model, w_model = self._fit_boosted_triplet(
                        train_lightgbm_regressor,
                        x_fit,
                        y_u_fit,
                        y_v_fit,
                        y_w_fit,
                        x_valid,
                        y_u_valid,
                        y_v_valid,
                        y_w_valid,
                    )
                except RuntimeError:
                    actual_model_type = "gbrt"
                    u_model = train_gradient_boosted_trees(
                        x_fit,
                        y_u_fit,
                        learning_rate=self.model_config.learning_rate,
                        num_estimators=self.model_config.num_estimators,
                        max_depth=self.model_config.max_depth,
                        min_samples_leaf=self.model_config.min_samples_leaf,
                        max_bins=self.model_config.max_bins,
                        feature_subsample_ratio=self.model_config.feature_subsample_ratio,
                    )
                    v_model = train_gradient_boosted_trees(
                        x_fit,
                        y_v_fit,
                        learning_rate=self.model_config.learning_rate,
                        num_estimators=self.model_config.num_estimators,
                        max_depth=self.model_config.max_depth,
                        min_samples_leaf=self.model_config.min_samples_leaf,
                        max_bins=self.model_config.max_bins,
                        feature_subsample_ratio=self.model_config.feature_subsample_ratio,
                    )
                    w_model = train_gradient_boosted_trees(
                        x_fit,
                        y_w_fit,
                        learning_rate=self.model_config.learning_rate,
                        num_estimators=self.model_config.num_estimators,
                        max_depth=self.model_config.max_depth,
                        min_samples_leaf=self.model_config.min_samples_leaf,
                        max_bins=self.model_config.max_bins,
                        feature_subsample_ratio=self.model_config.feature_subsample_ratio,
                    )
        elif self.model_config.model_type == "lightgbm":
            u_model, v_model, w_model = self._fit_boosted_triplet(
                train_lightgbm_regressor,
                x_fit,
                y_u_fit,
                y_v_fit,
                y_w_fit,
                x_valid,
                y_u_valid,
                y_v_valid,
                y_w_valid,
            )
        else:
            u_model = train_gradient_boosted_trees(
                x_fit,
                y_u_fit,
                learning_rate=self.model_config.learning_rate,
                num_estimators=self.model_config.num_estimators,
                max_depth=self.model_config.max_depth,
                min_samples_leaf=self.model_config.min_samples_leaf,
                max_bins=self.model_config.max_bins,
                feature_subsample_ratio=self.model_config.feature_subsample_ratio,
            )
            v_model = train_gradient_boosted_trees(
                x_fit,
                y_v_fit,
                learning_rate=self.model_config.learning_rate,
                num_estimators=self.model_config.num_estimators,
                max_depth=self.model_config.max_depth,
                min_samples_leaf=self.model_config.min_samples_leaf,
                max_bins=self.model_config.max_bins,
                feature_subsample_ratio=self.model_config.feature_subsample_ratio,
            )
            w_model = train_gradient_boosted_trees(
                x_fit,
                y_w_fit,
                learning_rate=self.model_config.learning_rate,
                num_estimators=self.model_config.num_estimators,
                max_depth=self.model_config.max_depth,
                min_samples_leaf=self.model_config.min_samples_leaf,
                max_bins=self.model_config.max_bins,
                feature_subsample_ratio=self.model_config.feature_subsample_ratio,
            )
        self.residual_model = ResidualWindModel(
            u_model=u_model,
            v_model=v_model,
            w_model=w_model,
            feature_names=FEATURE_NAMES,
        )
        # Cap eval set — full holdout can be ~80k rows and dominates wall clock.
        eval_cap = max(self.model_config.max_training_samples, 2000)
        eval_samples = test_samples or train_samples
        if len(eval_samples) > eval_cap:
            eval_samples = subsample_training_samples(eval_samples, eval_cap, seed=23)
        metrics = self.evaluate(eval_samples)
        metrics["train_size"] = len(train_samples)
        metrics["train_size_used"] = len(x_train)
        metrics["train_size_fit"] = len(x_fit)
        metrics["validation_size"] = len(x_valid)
        metrics["test_size"] = len(test_samples)
        metrics["eval_size_used"] = len(eval_samples)
        metrics["model_type"] = actual_model_type
        metrics["n_jobs_requested"] = self.model_config.n_jobs
        metrics["n_jobs_resolved"] = resolve_n_jobs(self.model_config.n_jobs)
        metrics["cpu_count"] = detect_cpu_count()
        if actual_model_type in {"xgboost", "lightgbm"}:
            metrics["best_iteration_u"] = getattr(u_model, "best_iteration", None)
            metrics["best_iteration_v"] = getattr(v_model, "best_iteration", None)
            metrics["best_iteration_w"] = getattr(w_model, "best_iteration", None)
            metrics["best_score_u"] = getattr(u_model, "best_score", None)
            metrics["best_score_v"] = getattr(v_model, "best_score", None)
            metrics["best_score_w"] = getattr(w_model, "best_score", None)
            metrics["feature_importance_gain_u"] = _top_feature_importance(getattr(u_model, "feature_importance_gain", None))
            metrics["feature_importance_gain_v"] = _top_feature_importance(getattr(v_model, "feature_importance_gain", None))
            metrics["feature_importance_gain_w"] = _top_feature_importance(getattr(w_model, "feature_importance_gain", None))
            metrics["validation_curve_u"] = getattr(u_model, "eval_history", {})
            metrics["validation_curve_v"] = getattr(v_model, "eval_history", {})
            metrics["validation_curve_w"] = getattr(w_model, "eval_history", {})
        return metrics

    def evaluate(self, samples: list[TrainingSample]) -> dict:
        if not self.residual_model:
            raise ValueError("Residual model is not trained.")
        if not samples:
            return {
                "rmse_physics": 0.0,
                "rmse_final": 0.0,
                "vertical_mae_physics": 0.0,
                "vertical_mae_final": 0.0,
                "direction_mae_rad": 0.0,
            }

        # Build feature rows once, then batch-predict residuals.
        eval_state = PredictSession(
            window_size=self.model_config.temporal_window_size,
            coarse_history=deque(maxlen=self.model_config.temporal_window_size),
        )
        feature_rows: list[list[float]] = []
        u_phy_list: list[float] = []
        v_phy_list: list[float] = []
        w_phy_list: list[float] = []
        u_obs_list: list[float] = []
        v_obs_list: list[float] = []
        w_obs_list: list[float] = []
        for sample in samples:
            wind_phy, _ = self._physics_bundle(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            level = _clamp_altitude_level(sample.z, self.model_config.altitude_levels)
            u_arr = np.asarray(wind_phy.u, dtype=np.float64)
            v_arr = np.asarray(wind_phy.v, dtype=np.float64)
            w_arr = np.asarray(wind_phy.w, dtype=np.float64)
            if u_arr.ndim == 3:
                u_phy = float(u_arr[level, sample.y, sample.x])
                v_phy = float(v_arr[level, sample.y, sample.x])
                w_phy = float(w_arr[level, sample.y, sample.x])
            else:
                u_phy = float(wind_phy.u[level][sample.y][sample.x])
                v_phy = float(wind_phy.v[level][sample.y][sample.x])
                w_phy = float(wind_phy.w[level][sample.y][sample.x])
            if eval_state.timestamp != sample.timestamp:
                eval_state.advance_header(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            temporal = eval_state.temporal_features(sample.x, sample.y, sample.z, u_phy, v_phy, w_phy)
            features = self._feature_dict(
                sample.timestamp, sample.x, sample.y, sample.z, sample.u_km, sample.v_km, sample.w_km, temporal
            )
            feature_rows.append([features[name] for name in FEATURE_NAMES])
            u_phy_list.append(u_phy)
            v_phy_list.append(v_phy)
            w_phy_list.append(w_phy)
            u_obs_list.append(sample.u_obs)
            v_obs_list.append(sample.v_obs)
            w_obs_list.append(sample.w_obs)
            # Keep temporal state coherent with online inference (use corrected wind).
            # Residuals filled after batch predict below — use physics here for history.
            eval_state.observe_cell(sample.x, sample.y, sample.z, u_phy, v_phy, w_phy)

        du, dv, dw = self.residual_model.predict_batch(feature_rows)
        du_a = np.asarray(du, dtype=np.float64)
        dv_a = np.asarray(dv, dtype=np.float64)
        dw_a = np.asarray(dw, dtype=np.float64)
        u_phy_a = np.asarray(u_phy_list, dtype=np.float64)
        v_phy_a = np.asarray(v_phy_list, dtype=np.float64)
        w_phy_a = np.asarray(w_phy_list, dtype=np.float64)
        u_obs_a = np.asarray(u_obs_list, dtype=np.float64)
        v_obs_a = np.asarray(v_obs_list, dtype=np.float64)
        w_obs_a = np.asarray(w_obs_list, dtype=np.float64)
        u_final = u_phy_a + du_a
        v_final = v_phy_a + dv_a
        w_final = w_phy_a + dw_a
        err_phy = np.sqrt((u_obs_a - u_phy_a) ** 2 + (v_obs_a - v_phy_a) ** 2 + (w_obs_a - w_phy_a) ** 2)
        err_final = np.sqrt((u_obs_a - u_final) ** 2 + (v_obs_a - v_final) ** 2 + (w_obs_a - w_final) ** 2)
        obs_dir = np.arctan2(u_obs_a, v_obs_a)
        pred_dir = np.arctan2(u_final, v_final)
        direction_err = np.abs(obs_dir - pred_dir)
        return {
            "rmse_physics": float(np.sqrt(np.mean(err_phy * err_phy))) if err_phy.size else 0.0,
            "rmse_final": float(np.sqrt(np.mean(err_final * err_final))) if err_final.size else 0.0,
            "vertical_mae_physics": float(np.mean(np.abs(w_obs_a - w_phy_a))) if w_obs_a.size else 0.0,
            "vertical_mae_final": float(np.mean(np.abs(w_obs_a - w_final))) if w_obs_a.size else 0.0,
            "direction_mae_rad": float(np.mean(direction_err)) if direction_err.size else 0.0,
        }

    def _fit_boosted_triplet(
        self,
        train_fn,
        x_fit: list[list[float]],
        y_u_fit: list[float],
        y_v_fit: list[float],
        y_w_fit: list[float],
        x_valid: list[list[float]],
        y_u_valid: list[float],
        y_v_valid: list[float],
        y_w_valid: list[float],
    ):
        total_threads = resolve_n_jobs(self.model_config.n_jobs)
        per_model_jobs = max(1, total_threads // 3)

        def _fit(y_fit: list[float], y_valid: list[float]):
            return train_fn(
                x_fit,
                y_fit,
                FEATURE_NAMES,
                learning_rate=self.model_config.learning_rate,
                num_estimators=self.model_config.num_estimators,
                max_depth=self.model_config.max_depth,
                feature_subsample_ratio=self.model_config.feature_subsample_ratio,
                validation_rows=x_valid,
                validation_targets=y_valid,
                early_stopping_rounds=self.model_config.early_stopping_rounds,
                n_jobs=per_model_jobs,
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            future_u = pool.submit(_fit, y_u_fit, y_u_valid)
            future_v = pool.submit(_fit, y_v_fit, y_v_valid)
            future_w = pool.submit(_fit, y_w_fit, y_w_valid)
            return future_u.result(), future_v.result(), future_w.result()

    def _sample_temporal_contexts(self, samples: list[TrainingSample]) -> list[dict[str, float]]:
        state = PredictSession(
            window_size=self.model_config.temporal_window_size,
            coarse_history=deque(maxlen=self.model_config.temporal_window_size),
        )
        contexts: list[dict[str, float]] = []
        for sample in samples:
            wind_phy, _ = self._physics_bundle(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            level = _clamp_altitude_level(sample.z, self.model_config.altitude_levels)
            u_arr = np.asarray(wind_phy.u, dtype=np.float64)
            v_arr = np.asarray(wind_phy.v, dtype=np.float64)
            w_arr = np.asarray(wind_phy.w, dtype=np.float64)
            if u_arr.ndim == 3:
                u_phy = float(u_arr[level, sample.y, sample.x])
                v_phy = float(v_arr[level, sample.y, sample.x])
                w_phy = float(w_arr[level, sample.y, sample.x])
            else:
                u_phy = wind_phy.u[level][sample.y][sample.x]
                v_phy = wind_phy.v[level][sample.y][sample.x]
                w_phy = wind_phy.w[level][sample.y][sample.x]
            if state.timestamp != sample.timestamp:
                state.advance_header(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            temporal = state.temporal_features(sample.x, sample.y, sample.z, u_phy, v_phy, w_phy)
            contexts.append(temporal)
            state.observe_cell(sample.x, sample.y, sample.z, sample.u_obs, sample.v_obs, sample.w_obs)
        return contexts

    def _predict_core(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float,
        active_session: PredictSession,
        active_level: int,
        x_values: list[int],
        y_values: list[int],
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> dict:
        wind_phy, diagnostics = self._physics_bundle(timestamp, u_km, v_km, w_km, u_100=u_100, v_100=v_100)
        u_arr = np.asarray(wind_phy.u, dtype=np.float64)
        v_arr = np.asarray(wind_phy.v, dtype=np.float64)
        w_arr = np.asarray(wind_phy.w, dtype=np.float64)
        level_count = int(u_arr.shape[0]) if u_arr.ndim == 3 else 0
        rows = len(y_values)
        cols = len(x_values)
        if active_session.timestamp != timestamp:
            active_session.advance_header(timestamp, u_km, v_km, w_km)

        if level_count == 0 or rows == 0 or cols == 0:
            return {
                "timestamp": timestamp,
                "altitude_level": active_level,
                "u": [],
                "v": [],
                "w": [],
                "u_layers": [],
                "v_layers": [],
                "w_layers": [],
                "x_bounds": [0, 0],
                "y_bounds": [0, 0],
            }

        ys = np.asarray(y_values, dtype=np.int64)
        xs = np.asarray(x_values, dtype=np.int64)
        u_win = u_arr[:, ys[:, None], xs[None, :]]
        v_win = v_arr[:, ys[:, None], xs[None, :]]
        w_win = w_arr[:, ys[:, None], xs[None, :]]

        if not self.residual_model:
            return {
                "timestamp": timestamp,
                "altitude_level": active_level,
                "u": u_win[active_level],
                "v": v_win[active_level],
                "w": w_win[active_level],
                "u_layers": u_win,
                "v_layers": v_win,
                "w_layers": w_win,
                "x_bounds": [x_values[0], x_values[-1] + 1],
                "y_bounds": [y_values[0], y_values[-1] + 1],
            }

        grids = self._ensure_static_grids()
        hour_sin, hour_cos = hour_features(timestamp)
        f_slope = np.asarray(diagnostics.f_slope, dtype=np.float64)
        f_rough = np.asarray(diagnostics.f_rough, dtype=np.float64)
        delta_elev = np.asarray(diagnostics.delta_elev, dtype=np.float64)
        n = level_count * rows * cols
        X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

        # Coarse temporal features are shared across the window.
        dummy = active_session.temporal_features(int(xs[0]), int(ys[0]), 0, 0.0, 0.0, 0.0)
        coarse_defaults = {
            "prev_u_km": dummy["prev_u_km"],
            "prev_v_km": dummy["prev_v_km"],
            "prev_w_km": dummy["prev_w_km"],
            "delta_u_km": dummy["delta_u_km"],
            "delta_v_km": dummy["delta_v_km"],
            "delta_w_km": dummy["delta_w_km"],
            "prev2_u_km": dummy["prev2_u_km"],
            "prev2_v_km": dummy["prev2_v_km"],
            "prev2_w_km": dummy["prev2_w_km"],
            "prev3_u_km": dummy["prev3_u_km"],
            "prev3_v_km": dummy["prev3_v_km"],
            "prev3_w_km": dummy["prev3_w_km"],
            "coarse_u_mean": dummy["coarse_u_mean"],
            "coarse_v_mean": dummy["coarse_v_mean"],
            "coarse_w_mean": dummy["coarse_w_mean"],
        }

        alt_levels = max(self.model_config.altitude_levels - 1, 1)
        wind_speed_km = magnitude3(u_km, v_km, w_km)
        # Sparse cell history: default prev_* = current physics, patch known keys only.
        hist = active_session.cell_history
        hist_win = active_session.cell_history_window
        prev_u = np.array(u_win, dtype=np.float64, copy=True)
        prev_v = np.array(v_win, dtype=np.float64, copy=True)
        prev_w = np.array(w_win, dtype=np.float64, copy=True)
        prev2_u = np.array(prev_u, copy=True)
        prev2_v = np.array(prev_v, copy=True)
        prev2_w = np.array(prev_w, copy=True)
        prev3_u = np.array(prev_u, copy=True)
        prev3_v = np.array(prev_v, copy=True)
        prev3_w = np.array(prev_w, copy=True)
        local_u_mean = np.array(prev_u, copy=True)
        local_v_mean = np.array(prev_v, copy=True)
        local_w_mean = np.array(prev_w, copy=True)
        x_index = {int(x): j for j, x in enumerate(x_values)}
        y_index = {int(y): i for i, y in enumerate(y_values)}
        for (hx, hy, hz), prev in hist.items():
            level = int(hz)
            if level < 0 or level >= level_count:
                continue
            li = y_index.get(int(hy))
            lj = x_index.get(int(hx))
            if li is None or lj is None:
                continue
            prev_u[level, li, lj] = prev["u_local"]
            prev_v[level, li, lj] = prev["v_local"]
            prev_w[level, li, lj] = prev["w_local"]
            window = list(hist_win.get((hx, hy, hz), ()))
            if window:
                local_u_mean[level, li, lj] = sum(item[0] for item in window) / len(window)
                local_v_mean[level, li, lj] = sum(item[1] for item in window) / len(window)
                local_w_mean[level, li, lj] = sum(item[2] for item in window) / len(window)
                prev2 = window[-2] if len(window) >= 2 else (prev["u_local"], prev["v_local"], prev["w_local"])
                prev3 = window[-3] if len(window) >= 3 else prev2
                prev2_u[level, li, lj], prev2_v[level, li, lj], prev2_w[level, li, lj] = prev2
                prev3_u[level, li, lj], prev3_v[level, li, lj], prev3_w[level, li, lj] = prev3

        # Build full (Z, H_win, W_win) feature volumes once (no per-level Python assembly).
        def _tile_hw(plane: np.ndarray) -> np.ndarray:
            return np.broadcast_to(plane[None, :, :], (level_count, rows, cols))

        slope = grids["slope"][ys][:, xs]
        aspect_sin = grids["aspect_sin"][ys][:, xs]
        aspect_cos = grids["aspect_cos"][ys][:, xs]
        levels = np.arange(level_count, dtype=np.float64)[:, None, None]
        block: dict[str, np.ndarray] = {
            "x_norm": _tile_hw(grids["x_norm"][ys][:, xs]),
            "y_norm": _tile_hw(grids["y_norm"][ys][:, xs]),
            "z_norm": np.broadcast_to(levels / alt_levels, (level_count, rows, cols)),
            "altitude_m": np.broadcast_to(levels * float(self.altitude_step_m), (level_count, rows, cols)),
            "elev": _tile_hw(grids["elev"][ys][:, xs]),
            "slope": _tile_hw(slope),
            "aspect": _tile_hw(grids["aspect"][ys][:, xs]),
            "aspect_sin": _tile_hw(aspect_sin),
            "aspect_cos": _tile_hw(aspect_cos),
            "terrain_wave_sin": _tile_hw(grids["terrain_wave_sin"][ys][:, xs]),
            "terrain_wave_cos": _tile_hw(grids["terrain_wave_cos"][ys][:, xs]),
            "lee_wave_sin": _tile_hw(grids["lee_wave_sin"][ys][:, xs]),
            "lee_wave_cos": _tile_hw(grids["lee_wave_cos"][ys][:, xs]),
            "roughness": _tile_hw(grids["roughness"][ys][:, xs]),
            "u_km": np.full((level_count, rows, cols), u_km, dtype=np.float64),
            "v_km": np.full((level_count, rows, cols), v_km, dtype=np.float64),
            "w_km": np.full((level_count, rows, cols), w_km, dtype=np.float64),
            "wind_speed_km": np.full((level_count, rows, cols), wind_speed_km, dtype=np.float64),
            "u_phy": u_win,
            "v_phy": v_win,
            "w_phy": w_win,
            "u_km_slope": _tile_hw(u_km * slope),
            "v_km_slope": _tile_hw(v_km * slope),
            "w_km_slope": _tile_hw(w_km * slope),
            "f_slope": _tile_hw(f_slope[ys][:, xs]),
            "f_rough": _tile_hw(f_rough[ys][:, xs]),
            "delta_elev": _tile_hw(delta_elev[ys][:, xs]),
            "hour_sin": np.full((level_count, rows, cols), hour_sin, dtype=np.float64),
            "hour_cos": np.full((level_count, rows, cols), hour_cos, dtype=np.float64),
            "aspect_time_sin": _tile_hw(aspect_sin * hour_cos + aspect_cos * hour_sin),
            "aspect_time_cos": _tile_hw(aspect_cos * hour_cos - aspect_sin * hour_sin),
            "prev_local_u": prev_u,
            "prev_local_v": prev_v,
            "prev_local_w": prev_w,
            "prev_local_speed": np.sqrt(prev_u * prev_u + prev_v * prev_v + prev_w * prev_w),
            "local_trend_u": u_win - prev_u,
            "local_trend_v": v_win - prev_v,
            "local_trend_w": w_win - prev_w,
            "prev2_local_u": prev2_u,
            "prev2_local_v": prev2_v,
            "prev2_local_w": prev2_w,
            "prev3_local_u": prev3_u,
            "prev3_local_v": prev3_v,
            "prev3_local_w": prev3_w,
            "local_u_mean": local_u_mean,
            "local_v_mean": local_v_mean,
            "local_w_mean": local_w_mean,
        }
        for key, value in coarse_defaults.items():
            block[key] = np.full((level_count, rows, cols), value, dtype=np.float64)

        for col_i, name in enumerate(FEATURE_NAMES):
            X[:, col_i] = np.asarray(block[name], dtype=np.float32).ravel()

        du, dv, dw = self.residual_model.predict_batch(X)
        du = np.asarray(du, dtype=np.float64).reshape(level_count, rows, cols)
        dv = np.asarray(dv, dtype=np.float64).reshape(level_count, rows, cols)
        dw = np.asarray(dw, dtype=np.float64).reshape(level_count, rows, cols)
        u_out = u_win + du
        v_out = v_win + dv
        w_out = w_win + dw
        # Keep (Z,H,W) ndarrays on the hot path; dashboard converts at report time.
        return {
            "timestamp": timestamp,
            "altitude_level": active_level,
            "u": u_out[active_level],
            "v": v_out[active_level],
            "w": w_out[active_level],
            "u_layers": u_out,
            "v_layers": v_out,
            "w_layers": w_out,
            "x_bounds": [x_values[0], x_values[-1] + 1],
            "y_bounds": [y_values[0], y_values[-1] + 1],
        }

    def predict_grid(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float = 0.0,
        session: PredictSession | None = None,
        altitude_level: int = 0,
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> dict:
        active_session = session or PredictSession(
            window_size=self.model_config.temporal_window_size,
            coarse_history=deque(maxlen=self.model_config.temporal_window_size),
        )
        row_count = len(self.terrain.elevation)
        col_count = len(self.terrain.elevation[0]) if row_count else 0
        active_level = _clamp_altitude_level(altitude_level, self.model_config.altitude_levels)
        result = self._predict_core(
            timestamp,
            u_km,
            v_km,
            w_km,
            active_session,
            active_level,
            list(range(col_count)),
            list(range(row_count)),
            u_100=u_100,
            v_100=v_100,
        )
        active_session.update_cell_history(result["u"], result["v"], result["w"], active_level)
        result.pop("x_bounds", None)
        result.pop("y_bounds", None)
        return result

    def predict_window(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        w_km: float = 0.0,
        session: PredictSession | None = None,
        altitude_level: int = 0,
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> dict:
        row_count = len(self.terrain.elevation)
        col_count = len(self.terrain.elevation[0]) if row_count else 0
        x0 = max(0, min(col_count, x_min))
        y0 = max(0, min(row_count, y_min))
        x1 = max(x0, min(col_count, x_max))
        y1 = max(y0, min(row_count, y_max))
        active_session = session or PredictSession(
            window_size=self.model_config.temporal_window_size,
            coarse_history=deque(maxlen=self.model_config.temporal_window_size),
        )
        active_level = _clamp_altitude_level(altitude_level, self.model_config.altitude_levels)
        return self._predict_core(
            timestamp,
            u_km,
            v_km,
            w_km,
            active_session,
            active_level,
            list(range(x0, x1)),
            list(range(y0, y1)),
            u_100=u_100,
            v_100=v_100,
        )

    def physics_grid(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float = 0.0,
        altitude_level: int = 0,
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> dict:
        wind_phy, _ = self._physics_bundle(timestamp, u_km, v_km, w_km, u_100=u_100, v_100=v_100)
        u = np.asarray(wind_phy.u, dtype=np.float64)
        v = np.asarray(wind_phy.v, dtype=np.float64)
        w = np.asarray(wind_phy.w, dtype=np.float64)
        level_count = int(u.shape[0]) if u.ndim == 3 else 0
        active_level = _clamp_altitude_level(altitude_level, level_count)
        return {
            "timestamp": timestamp,
            "altitude_level": active_level,
            "u": u[active_level] if level_count else [],
            "v": v[active_level] if level_count else [],
            "w": w[active_level] if level_count else [],
            "u_layers": u if level_count else [],
            "v_layers": v if level_count else [],
            "w_layers": w if level_count else [],
        }

    def physics_window(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        w_km: float = 0.0,
        altitude_level: int = 0,
        *,
        u_100: float | None = None,
        v_100: float | None = None,
    ) -> dict:
        wind_phy, _ = self._physics_bundle(timestamp, u_km, v_km, w_km, u_100=u_100, v_100=v_100)
        u = np.asarray(wind_phy.u, dtype=np.float64)
        v = np.asarray(wind_phy.v, dtype=np.float64)
        w = np.asarray(wind_phy.w, dtype=np.float64)
        if u.ndim != 3:
            return {
                "timestamp": timestamp,
                "altitude_level": 0,
                "u": [],
                "v": [],
                "w": [],
                "u_layers": [],
                "v_layers": [],
                "w_layers": [],
                "x_bounds": [0, 0],
                "y_bounds": [0, 0],
            }
        level_count, row_count, col_count = u.shape
        x0 = max(0, min(col_count, x_min))
        y0 = max(0, min(row_count, y_min))
        x1 = max(x0, min(col_count, x_max))
        y1 = max(y0, min(row_count, y_max))
        active_level = _clamp_altitude_level(altitude_level, level_count)
        u_win = u[:, y0:y1, x0:x1]
        v_win = v[:, y0:y1, x0:x1]
        w_win = w[:, y0:y1, x0:x1]
        u_layers = u_win.tolist()
        v_layers = v_win.tolist()
        w_layers = w_win.tolist()
        return {
            "timestamp": timestamp,
            "altitude_level": active_level,
            "u": u_layers[active_level] if u_layers else [],
            "v": v_layers[active_level] if v_layers else [],
            "w": w_layers[active_level] if w_layers else [],
            "u_layers": u_layers,
            "v_layers": v_layers,
            "w_layers": w_layers,
            "x_bounds": [x0, x1],
            "y_bounds": [y0, y1],
        }

    def online_plan(
        self,
        coarse_samples_path: str | Path,
        observations_path: str | Path,
        mission: Mission,
        task_config=None,
    ) -> dict:
        coarse_samples = coarse_samples_from_dict(read_json(coarse_samples_path)["coarse_wind"])
        observations = observations_from_dict(read_json(observations_path)["observations"])
        obs_by_time = {item.timestamp: item for item in observations}
        if task_config is None:
            from .config import TaskConfig
            task_config = TaskConfig(model=replace(self.model_config))
        engine_config = _derive_engine_config(task_config, mission)
        engine = NavigationEngine(self, engine_config)
        context = engine.create_context(mission.start, mission.goal)
        path_history = []
        for sample in coarse_samples:
            frame = engine.step(context, sample, obs_by_time.get(sample.timestamp), None)
            path_history.append(
                {
                    "timestamp": sample.timestamp,
                    "planned_path": frame["planned_path"],
                    "planned_path_cost": frame["planned_path_cost"],
                }
            )
            if _goal_reached(context.state, context.mission.goal) or context.step >= context.mission.max_steps:
                break

        return {
            "executed_path": [item["position"] for item in context.trace],
            "executed_distance_m": path_distance_m(
                [tuple(item["position"]) for item in context.trace],
                mission.step_distance_m,
                mission.altitude_step_m,
            ),
            "goal_reached": _goal_reached(context.state, context.mission.goal),
            "planning_history": path_history,
        }

    def save_model(self, path: str | Path) -> None:
        if not self.residual_model:
            raise ValueError("Residual model is not trained.")
        Path(path).write_text(self.residual_model.to_json(), encoding="utf-8")

    def model_summary(self) -> dict:
        if not self.residual_model:
            return {"model_type": "untrained"}
        actual_model_type = getattr(self.residual_model.u_model, "to_dict", lambda: {"model_type": self.model_config.model_type})().get("model_type", self.model_config.model_type)
        summary = {
            "model_type": actual_model_type,
            "feature_names": FEATURE_NAMES,
        }
        for prefix, model in (("u", self.residual_model.u_model), ("v", self.residual_model.v_model), ("w", self.residual_model.w_model)):
            summary[f"best_iteration_{prefix}"] = getattr(model, "best_iteration", None)
            summary[f"best_score_{prefix}"] = getattr(model, "best_score", None)
            summary[f"feature_importance_gain_{prefix}"] = _top_feature_importance(getattr(model, "feature_importance_gain", None))
            summary[f"validation_curve_{prefix}"] = getattr(model, "eval_history", {})
        if self.training_metrics:
            for key, value in self.training_metrics.items():
                if key not in summary or summary[key] in (None, {}, []):
                    summary[key] = value
        return summary

    @classmethod
    def from_paths(
        cls,
        terrain_path: str | Path,
        model_path: str | Path | None = None,
        model_config: ModelConfig | None = None,
        altitude_step_m: float | None = None,
    ) -> "WindFarmPipeline":
        terrain = terrain_from_dict(load_terrain_document(terrain_path)["terrain"])
        pipeline = cls(
            terrain=terrain,
            model_config=model_config or ModelConfig(),
            altitude_step_m=float(altitude_step_m if altitude_step_m is not None else 50.0),
        )
        if model_path:
            model_path = Path(model_path)
            pipeline.residual_model = ResidualWindModel.from_json(model_path.read_text(encoding="utf-8"))
            metrics_path = _infer_metrics_path(model_path)
            if metrics_path is not None and metrics_path.exists():
                pipeline.training_metrics = read_json(metrics_path)
        return pipeline


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / max(len(values), 1))


def _infer_metrics_path(model_path: Path) -> Path | None:
    stem = model_path.stem
    name = model_path.name
    parent = model_path.parent
    candidates = [
        parent / name.replace("model", "metrics", 1),
        parent / f"{stem.replace('model', 'metrics', 1)}.json",
        parent / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _top_feature_importance(feature_importance: dict[str, float] | None, limit: int = 8) -> dict[str, float]:
    if not feature_importance:
        return {}
    ranked = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    return {key: value for key, value in ranked[:limit]}


def _clamp_altitude_level(level: int, total_levels: int) -> int:
    if total_levels <= 0:
        return 0
    return max(0, min(total_levels - 1, int(level)))


def _derive_engine_config(task_config, mission: Mission):
    return TaskConfig(
        model=replace(task_config.model),
        belief=replace(task_config.belief),
        planner=replace(task_config.planner),
        simulation=replace(task_config.simulation),
        mission=replace(
            task_config.mission,
            start=mission.start,
            goal=mission.goal,
            max_steps=mission.max_steps,
            step_distance_m=mission.step_distance_m,
            altitude_step_m=mission.altitude_step_m,
            min_altitude_level=mission.min_altitude_level,
            max_altitude_level=mission.max_altitude_level,
            climb_cost_per_level_j=mission.climb_cost_per_level_j,
            clearance_agl_level=getattr(mission, "clearance_agl_level", 1.0),
            cruise_band_step=getattr(mission, "cruise_band_step", 0.025),
        ),
    )


def train_from_files(
    terrain_path: str | Path,
    training_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    model_config: ModelConfig | None = None,
) -> dict:
    pipeline = WindFarmPipeline.from_paths(terrain_path, model_config=model_config)
    training_samples = training_samples_from_dict(read_json(training_path)["samples"])
    metrics = pipeline.train(training_samples)
    pipeline.save_model(model_path)
    if metrics_path:
        write_json(metrics_path, metrics)
    return metrics
