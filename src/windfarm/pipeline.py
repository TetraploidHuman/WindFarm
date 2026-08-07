from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
from pathlib import Path

from .config import ModelConfig, TaskConfig, detect_cpu_count, resolve_n_jobs
from .execution import NavigationEngine, PredictSession, _goal_reached
from .io import coarse_samples_from_dict, observations_from_dict, read_json, terrain_from_dict, training_samples_from_dict, write_json
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
    _physics_cache: dict[tuple[str, float, float, float, int], tuple[WindField, object]] = field(default_factory=dict)
    _static_feature_cache: dict[tuple[int, int, int], dict[str, float]] = field(default_factory=dict)

    def _physics_bundle(self, timestamp: str, u_km: float, v_km: float, w_km: float = 0.0):
        key = (timestamp, round(u_km, 6), round(v_km, 6), round(w_km, 6), self.model_config.altitude_levels)
        cached = self._physics_cache.get(key)
        if cached is None:
            cached = downscale_wind(u_km, v_km, self.terrain, w_km, self.model_config.altitude_levels)
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
        u_phy = wind_phy.u[level][y][x]
        v_phy = wind_phy.v[level][y][x]
        w_phy = wind_phy.w[level][y][x]
        slope = features["slope"]
        aspect_sin = features["aspect_sin"]
        aspect_cos = features["aspect_cos"]
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
            "f_slope": diagnostics.f_slope[y][x],
            "f_rough": diagnostics.f_rough[y][x],
            "delta_elev": diagnostics.delta_elev[y][x],
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
            "altitude_m": level * 40.0,
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
        metrics = self.evaluate(test_samples or train_samples)
        metrics["train_size"] = len(train_samples)
        metrics["train_size_used"] = len(x_train)
        metrics["train_size_fit"] = len(x_fit)
        metrics["validation_size"] = len(x_valid)
        metrics["test_size"] = len(test_samples)
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
        errors_phy: list[float] = []
        errors_final: list[float] = []
        direction_errors: list[float] = []
        vertical_errors_phy: list[float] = []
        vertical_errors_final: list[float] = []
        eval_state = PredictSession(
            window_size=self.model_config.temporal_window_size,
            coarse_history=deque(maxlen=self.model_config.temporal_window_size),
        )
        for sample in samples:
            wind_phy, _ = self._physics_bundle(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            level = _clamp_altitude_level(sample.z, self.model_config.altitude_levels)
            u_phy = wind_phy.u[level][sample.y][sample.x]
            v_phy = wind_phy.v[level][sample.y][sample.x]
            w_phy = wind_phy.w[level][sample.y][sample.x]
            if eval_state.timestamp != sample.timestamp:
                eval_state.advance_header(sample.timestamp, sample.u_km, sample.v_km, sample.w_km)
            temporal = eval_state.temporal_features(sample.x, sample.y, sample.z, u_phy, v_phy, w_phy)
            features = self._feature_dict(sample.timestamp, sample.x, sample.y, sample.z, sample.u_km, sample.v_km, sample.w_km, temporal)
            du, dv, dw = self.residual_model.predict(features)
            u_final = u_phy + du
            v_final = v_phy + dv
            w_final = w_phy + dw
            errors_phy.append(magnitude3(sample.u_obs - u_phy, sample.v_obs - v_phy, sample.w_obs - w_phy))
            errors_final.append(magnitude3(sample.u_obs - u_final, sample.v_obs - v_final, sample.w_obs - w_final))
            vertical_errors_phy.append(abs(sample.w_obs - w_phy))
            vertical_errors_final.append(abs(sample.w_obs - w_final))
            obs_dir = math.atan2(sample.u_obs, sample.v_obs)
            pred_dir = math.atan2(u_final, v_final)
            direction_errors.append(abs(obs_dir - pred_dir))
            eval_state.observe_cell(sample.x, sample.y, sample.z, u_final, v_final, w_final)
        return {
            "rmse_physics": _rmse(errors_phy),
            "rmse_final": _rmse(errors_final),
            "vertical_mae_physics": sum(vertical_errors_phy) / max(len(vertical_errors_phy), 1),
            "vertical_mae_final": sum(vertical_errors_final) / max(len(vertical_errors_final), 1),
            "direction_mae_rad": sum(direction_errors) / max(len(direction_errors), 1),
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
    ) -> dict:
        wind_phy, _ = self._physics_bundle(timestamp, u_km, v_km, w_km)
        level_count = len(wind_phy.u)
        rows = len(y_values)
        cols = len(x_values)
        u_final = [[0.0 for _ in range(cols)] for _ in range(rows)]
        v_final = [[0.0 for _ in range(cols)] for _ in range(rows)]
        w_final = [[0.0 for _ in range(cols)] for _ in range(rows)]
        u_layers = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(level_count)]
        v_layers = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(level_count)]
        w_layers = [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(level_count)]
        if active_session.timestamp != timestamp:
            active_session.advance_header(timestamp, u_km, v_km, w_km)

        if not self.residual_model:
            for level in range(level_count):
                for local_y, y in enumerate(y_values):
                    for local_x, x in enumerate(x_values):
                        u_value = wind_phy.u[level][y][x]
                        v_value = wind_phy.v[level][y][x]
                        w_value = wind_phy.w[level][y][x]
                        u_layers[level][local_y][local_x] = u_value
                        v_layers[level][local_y][local_x] = v_value
                        w_layers[level][local_y][local_x] = w_value
                        if level == active_level:
                            u_final[local_y][local_x] = u_value
                            v_final[local_y][local_x] = v_value
                            w_final[local_y][local_x] = w_value
        else:
            feature_rows: list[list[float]] = []
            meta: list[tuple[int, int, int, float, float, float]] = []
            for level in range(level_count):
                for local_y, y in enumerate(y_values):
                    for local_x, x in enumerate(x_values):
                        u_phy = wind_phy.u[level][y][x]
                        v_phy = wind_phy.v[level][y][x]
                        w_phy = wind_phy.w[level][y][x]
                        temporal = active_session.temporal_features(x, y, level, u_phy, v_phy, w_phy)
                        features = self._feature_dict(timestamp, x, y, level, u_km, v_km, w_km, temporal)
                        feature_rows.append([features[name] for name in FEATURE_NAMES])
                        meta.append((level, local_y, local_x, u_phy, v_phy, w_phy))
            du_list, dv_list, dw_list = self.residual_model.predict_batch(feature_rows)
            for (level, local_y, local_x, u_phy, v_phy, w_phy), du, dv, dw in zip(meta, du_list, dv_list, dw_list):
                u_value = u_phy + du
                v_value = v_phy + dv
                w_value = w_phy + dw
                u_layers[level][local_y][local_x] = u_value
                v_layers[level][local_y][local_x] = v_value
                w_layers[level][local_y][local_x] = w_value
                if level == active_level:
                    u_final[local_y][local_x] = u_value
                    v_final[local_y][local_x] = v_value
                    w_final[local_y][local_x] = w_value

        return {
            "timestamp": timestamp,
            "altitude_level": active_level,
            "u": u_final,
            "v": v_final,
            "w": w_final,
            "u_layers": u_layers,
            "v_layers": v_layers,
            "w_layers": w_layers,
            "x_bounds": [x_values[0], x_values[-1] + 1] if x_values else [0, 0],
            "y_bounds": [y_values[0], y_values[-1] + 1] if y_values else [0, 0],
        }

    def predict_grid(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float = 0.0,
        session: PredictSession | None = None,
        altitude_level: int = 0,
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
        )

    def physics_grid(self, timestamp: str, u_km: float, v_km: float, w_km: float = 0.0, altitude_level: int = 0) -> dict:
        wind_phy, _ = self._physics_bundle(timestamp, u_km, v_km, w_km)
        level_count = len(wind_phy.u)
        active_level = _clamp_altitude_level(altitude_level, level_count)
        return {
            "timestamp": timestamp,
            "altitude_level": active_level,
            "u": wind_phy.u[active_level],
            "v": wind_phy.v[active_level],
            "w": wind_phy.w[active_level],
            "u_layers": wind_phy.u,
            "v_layers": wind_phy.v,
            "w_layers": wind_phy.w,
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
    ) -> dict:
        wind_phy, _ = self._physics_bundle(timestamp, u_km, v_km, w_km)
        level_count = len(wind_phy.u)
        row_count = len(wind_phy.u[0]) if level_count else 0
        col_count = len(wind_phy.u[0][0]) if row_count else 0
        x0 = max(0, min(col_count, x_min))
        y0 = max(0, min(row_count, y_min))
        x1 = max(x0, min(col_count, x_max))
        y1 = max(y0, min(row_count, y_max))
        active_level = _clamp_altitude_level(altitude_level, level_count)
        u_layers = [[row[x0:x1] for row in wind_phy.u[level][y0:y1]] for level in range(level_count)]
        v_layers = [[row[x0:x1] for row in wind_phy.v[level][y0:y1]] for level in range(level_count)]
        w_layers = [[row[x0:x1] for row in wind_phy.w[level][y0:y1]] for level in range(level_count)]
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
            "executed_distance_m": path_distance_m([tuple(item["position"]) for item in context.trace], mission.step_distance_m),
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
    ) -> "WindFarmPipeline":
        terrain = terrain_from_dict(read_json(terrain_path)["terrain"])
        pipeline = cls(terrain=terrain, model_config=model_config or ModelConfig())
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
