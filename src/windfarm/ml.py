from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import random

from .config import resolve_n_jobs

try:
    import numpy as np
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    np = None
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None


def _configure_xgboost_threads(booster: object, n_jobs: int | None = 0) -> int:
    threads = resolve_n_jobs(n_jobs)
    booster.set_param({"nthread": threads})
    return threads


def _configure_lightgbm_threads(booster: object, n_jobs: int | None = 0) -> int:
    threads = resolve_n_jobs(n_jobs)
    if hasattr(booster, "reset_parameter"):
        booster.reset_parameter({"num_threads": threads})
    return threads


@dataclass(slots=True)
class LinearRegressor:
    weights: list[float]
    bias: float
    means: list[float]
    scales: list[float]

    def predict(self, features: list[float]) -> float:
        total = self.bias
        for value, weight, mean, scale in zip(features, self.weights, self.means, self.scales):
            total += ((value - mean) / scale) * weight
        return total

    def to_dict(self) -> dict:
        return {
            "model_type": "linear",
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "scales": self.scales,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearRegressor":
        return cls(
            weights=[float(v) for v in payload["weights"]],
            bias=float(payload["bias"]),
            means=[float(v) for v in payload["means"]],
            scales=[float(v) for v in payload["scales"]],
        )


def train_ridge_regression(
    x_rows: list[list[float]],
    y_values: list[float],
    learning_rate: float = 0.03,
    epochs: int = 800,
    l2: float = 0.01,
) -> LinearRegressor:
    del learning_rate, epochs
    if not x_rows:
        raise ValueError("Training data is empty.")
    feature_count = len(x_rows[0])
    means = [0.0] * feature_count
    scales = [1.0] * feature_count
    for idx in range(feature_count):
        column = [row[idx] for row in x_rows]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        means[idx] = mean
        scales[idx] = max(math.sqrt(variance), 1e-6)
    normalized = []
    for row in x_rows:
        normalized.append([(value - mean) / scale for value, mean, scale in zip(row, means, scales)])

    design = [[1.0] + row for row in normalized]
    dim = feature_count + 1
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for row, target in zip(design, y_values):
        for i in range(dim):
            xty[i] += row[i] * target
            for j in range(dim):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, dim):
        xtx[i][i] += l2
    solution = _solve_linear_system(xtx, xty)
    return LinearRegressor(weights=solution[1:], bias=solution[0], means=means, scales=scales)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda idx: abs(augmented[idx][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            continue
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, size + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[idx][size] for idx in range(size)]


@dataclass(slots=True)
class TreeNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, features: list[float]) -> float:
        if self.feature_index is None or self.left is None or self.right is None or self.threshold is None:
            return self.value
        if features[self.feature_index] <= self.threshold:
            return self.left.predict(features)
        return self.right.predict(features)

    def to_dict(self) -> dict:
        payload = {"value": self.value}
        if self.feature_index is not None:
            payload["feature_index"] = self.feature_index
            payload["threshold"] = self.threshold
            payload["left"] = self.left.to_dict() if self.left else None
            payload["right"] = self.right.to_dict() if self.right else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TreeNode":
        return cls(
            value=float(payload["value"]),
            feature_index=payload.get("feature_index"),
            threshold=float(payload["threshold"]) if payload.get("threshold") is not None else None,
            left=cls.from_dict(payload["left"]) if payload.get("left") else None,
            right=cls.from_dict(payload["right"]) if payload.get("right") else None,
        )


@dataclass(slots=True)
class DecisionTreeRegressor:
    root: TreeNode

    def predict(self, features: list[float]) -> float:
        return self.root.predict(features)

    def to_dict(self) -> dict:
        return {"root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict) -> "DecisionTreeRegressor":
        return cls(root=TreeNode.from_dict(payload["root"]))


@dataclass(slots=True)
class GradientBoostedTreesRegressor:
    base_value: float
    learning_rate: float
    trees: list[DecisionTreeRegressor]

    def predict(self, features: list[float]) -> float:
        total = self.base_value
        for tree in self.trees:
            total += self.learning_rate * tree.predict(features)
        return total

    def to_dict(self) -> dict:
        return {
            "model_type": "gbrt",
            "base_value": self.base_value,
            "learning_rate": self.learning_rate,
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GradientBoostedTreesRegressor":
        return cls(
            base_value=float(payload["base_value"]),
            learning_rate=float(payload["learning_rate"]),
            trees=[DecisionTreeRegressor.from_dict(item) for item in payload["trees"]],
        )


@dataclass(slots=True)
class XGBoostRegressor:
    booster: object
    best_iteration: int | None = None
    best_score: float | None = None
    feature_importance_gain: dict[str, float] | None = None
    eval_history: dict[str, list[float]] | None = None
    n_jobs: int = 0

    def predict(self, features: list[float]) -> float:
        if np is None or xgb is None:
            raise RuntimeError("XGBoost runtime is not available.")
        array = np.asarray([features], dtype=np.float32)
        return float(self.booster.inplace_predict(array)[0])

    def predict_many(self, feature_rows: list[list[float]] | object) -> list[float] | object:
        if np is None or xgb is None:
            raise RuntimeError("XGBoost runtime is not available.")
        array = np.asarray(feature_rows, dtype=np.float32)
        if array.size == 0:
            return []
        if array.ndim == 1:
            array = array.reshape(1, -1)
        preds = self.booster.inplace_predict(array)
        # Keep ndarray on the hot path; callers that need lists can convert.
        return preds

    def to_dict(self) -> dict:
        raw = bytes(self.booster.save_raw())
        return {
            "model_type": "xgboost",
            "raw_model_b64": base64.b64encode(raw).decode("ascii"),
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "feature_importance_gain": self.feature_importance_gain or {},
            "eval_history": self.eval_history or {},
            "n_jobs": self.n_jobs,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "XGBoostRegressor":
        if xgb is None:
            raise RuntimeError("XGBoost runtime is not available.")
        booster = xgb.Booster()
        booster.load_model(bytearray(base64.b64decode(payload["raw_model_b64"])))
        n_jobs = int(payload.get("n_jobs", 0))
        _configure_xgboost_threads(booster, n_jobs)
        return cls(
            booster=booster,
            best_iteration=payload.get("best_iteration"),
            best_score=payload.get("best_score"),
            feature_importance_gain=dict(payload.get("feature_importance_gain", {})),
            eval_history={key: list(value) for key, value in payload.get("eval_history", {}).items()},
            n_jobs=n_jobs,
        )


@dataclass(slots=True)
class LightGBMRegressor:
    booster: object
    best_iteration: int | None = None
    best_score: float | None = None
    feature_importance_gain: dict[str, float] | None = None
    eval_history: dict[str, list[float]] | None = None
    n_jobs: int = 0

    def predict(self, features: list[float]) -> float:
        if lgb is None or np is None:
            raise RuntimeError("LightGBM runtime is not available.")
        array = np.asarray([features], dtype=np.float32)
        return float(self.booster.predict(array)[0])

    def predict_many(self, feature_rows: list[list[float]] | object) -> list[float] | object:
        if lgb is None or np is None:
            raise RuntimeError("LightGBM runtime is not available.")
        array = np.asarray(feature_rows, dtype=np.float32)
        if array.size == 0:
            return []
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return self.booster.predict(array)

    def to_dict(self) -> dict:
        return {
            "model_type": "lightgbm",
            "model_str": self.booster.model_to_string(),
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "feature_importance_gain": self.feature_importance_gain or {},
            "eval_history": self.eval_history or {},
            "n_jobs": self.n_jobs,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LightGBMRegressor":
        if lgb is None:
            raise RuntimeError("LightGBM runtime is not available.")
        booster = lgb.Booster(model_str=payload["model_str"])
        n_jobs = int(payload.get("n_jobs", 0))
        _configure_lightgbm_threads(booster, n_jobs)
        return cls(
            booster=booster,
            best_iteration=payload.get("best_iteration"),
            best_score=payload.get("best_score"),
            feature_importance_gain=dict(payload.get("feature_importance_gain", {})),
            eval_history={key: list(value) for key, value in payload.get("eval_history", {}).items()},
            n_jobs=n_jobs,
        )


def train_xgboost_regressor(
    x_rows: list[list[float]],
    y_values: list[float],
    feature_names: list[str],
    learning_rate: float = 0.1,
    num_estimators: int = 120,
    max_depth: int = 5,
    feature_subsample_ratio: float = 0.7,
    validation_rows: list[list[float]] | None = None,
    validation_targets: list[float] | None = None,
    early_stopping_rounds: int | None = None,
    n_jobs: int = 0,
) -> XGBoostRegressor:
    if np is None or xgb is None:
        raise RuntimeError("XGBoost is not installed.")
    threads = resolve_n_jobs(n_jobs)
    matrix = xgb.DMatrix(np.asarray(x_rows, dtype=np.float32), label=np.asarray(y_values, dtype=np.float32), feature_names=feature_names)
    evals = [(matrix, "train")]
    if validation_rows and validation_targets:
        valid_matrix = xgb.DMatrix(
            np.asarray(validation_rows, dtype=np.float32),
            label=np.asarray(validation_targets, dtype=np.float32),
            feature_names=feature_names,
        )
        evals.append((valid_matrix, "valid"))
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eta": learning_rate,
            "max_depth": max_depth,
            "subsample": 0.85,
            "colsample_bytree": feature_subsample_ratio,
            "min_child_weight": 4.0,
            "lambda": 1.0,
            "tree_method": "hist",
            "nthread": threads,
            "verbosity": 0,
        },
        dtrain=matrix,
        num_boost_round=num_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    _configure_xgboost_threads(booster, n_jobs)
    importance = booster.get_score(importance_type="gain")
    mapped_importance = {name: float(importance.get(name, 0.0)) for name in feature_names}
    best_score = None
    if hasattr(booster, "best_score"):
        try:
            best_score = float(booster.best_score)
        except (TypeError, ValueError):
            best_score = None
    eval_history = {}
    for dataset_name, metrics in evals_result.items():
        if "rmse" in metrics:
            eval_history[dataset_name] = [float(value) for value in metrics["rmse"]]
    return XGBoostRegressor(
        booster=booster,
        best_iteration=int(getattr(booster, "best_iteration", -1)) if hasattr(booster, "best_iteration") else None,
        best_score=best_score,
        feature_importance_gain=mapped_importance,
        eval_history=eval_history,
        n_jobs=n_jobs,
    )


def train_lightgbm_regressor(
    x_rows: list[list[float]],
    y_values: list[float],
    feature_names: list[str],
    learning_rate: float = 0.1,
    num_estimators: int = 120,
    max_depth: int = 5,
    feature_subsample_ratio: float = 0.7,
    validation_rows: list[list[float]] | None = None,
    validation_targets: list[float] | None = None,
    early_stopping_rounds: int | None = None,
    n_jobs: int = 0,
) -> LightGBMRegressor:
    if lgb is None or np is None:
        raise RuntimeError("LightGBM is not installed.")
    threads = resolve_n_jobs(n_jobs)
    train_dataset = lgb.Dataset(
        np.asarray(x_rows, dtype=np.float32),
        label=np.asarray(y_values, dtype=np.float32),
        feature_name=feature_names,
        free_raw_data=True,
    )
    valid_sets = [train_dataset]
    valid_names = ["train"]
    if validation_rows and validation_targets:
        valid_dataset = lgb.Dataset(
            np.asarray(validation_rows, dtype=np.float32),
            label=np.asarray(validation_targets, dtype=np.float32),
            feature_name=feature_names,
            reference=train_dataset,
            free_raw_data=True,
        )
        valid_sets.append(valid_dataset)
        valid_names.append("valid")
    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks = [lgb.record_evaluation(evals_result)]
    if early_stopping_rounds and len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": learning_rate,
            "num_leaves": max(8, min(63, 2 ** max_depth - 1 if max_depth > 0 else 31)),
            "max_depth": max_depth,
            "feature_fraction": feature_subsample_ratio,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 16,
            "lambda_l2": 1.0,
            "num_threads": threads,
            "verbosity": -1,
        },
        train_set=train_dataset,
        num_boost_round=num_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    _configure_lightgbm_threads(booster, n_jobs)
    importance = booster.feature_importance(importance_type="gain")
    mapped_importance = {name: float(value) for name, value in zip(feature_names, importance)}
    best_iteration = int(getattr(booster, "best_iteration", 0) or 0) or None
    best_score = None
    if "valid" in booster.best_score and "rmse" in booster.best_score["valid"]:
        best_score = float(booster.best_score["valid"]["rmse"])
    elif "train" in booster.best_score and "rmse" in booster.best_score["train"]:
        best_score = float(booster.best_score["train"]["rmse"])
    eval_history = {}
    for dataset_name, metrics in evals_result.items():
        if "rmse" in metrics:
            eval_history[dataset_name] = [float(value) for value in metrics["rmse"]]
    return LightGBMRegressor(
        booster=booster,
        best_iteration=best_iteration,
        best_score=best_score,
        feature_importance_gain=mapped_importance,
        eval_history=eval_history,
        n_jobs=n_jobs,
    )


def train_gradient_boosted_trees(
    x_rows: list[list[float]],
    y_values: list[float],
    learning_rate: float = 0.08,
    num_estimators: int = 28,
    max_depth: int = 4,
    min_samples_leaf: int = 24,
    max_bins: int = 16,
    feature_subsample_ratio: float = 0.7,
) -> GradientBoostedTreesRegressor:
    if not x_rows:
        raise ValueError("Training data is empty.")
    sample_count = len(x_rows)
    base_value = sum(y_values) / sample_count
    predictions = [base_value for _ in range(sample_count)]
    trees: list[DecisionTreeRegressor] = []
    rng = random.Random(23)

    for _ in range(num_estimators):
        residuals = [target - pred for target, pred in zip(y_values, predictions)]
        root = _build_tree(
            x_rows=x_rows,
            targets=residuals,
            indices=list(range(sample_count)),
            depth=0,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            feature_subsample_ratio=feature_subsample_ratio,
            rng=rng,
        )
        tree = DecisionTreeRegressor(root=root)
        trees.append(tree)
        for idx, features in enumerate(x_rows):
            predictions[idx] += learning_rate * tree.predict(features)

    return GradientBoostedTreesRegressor(base_value=base_value, learning_rate=learning_rate, trees=trees)


def _build_tree(
    x_rows: list[list[float]],
    targets: list[float],
    indices: list[int],
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_bins: int,
    feature_subsample_ratio: float,
    rng: random.Random,
) -> TreeNode:
    node_value = _mean([targets[idx] for idx in indices])
    if depth >= max_depth or len(indices) <= min_samples_leaf * 2:
        return TreeNode(value=node_value)

    feature_count = len(x_rows[0])
    feature_indices = list(range(feature_count))
    rng.shuffle(feature_indices)
    take = max(1, int(feature_count * feature_subsample_ratio))
    candidate_features = feature_indices[:take]

    best_feature = None
    best_threshold = None
    best_score = math.inf
    best_left: list[int] | None = None
    best_right: list[int] | None = None

    for feature_index in candidate_features:
        thresholds = _candidate_thresholds(x_rows, indices, feature_index, max_bins)
        for threshold in thresholds:
            left = [idx for idx in indices if x_rows[idx][feature_index] <= threshold]
            right = [idx for idx in indices if x_rows[idx][feature_index] > threshold]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            score = _squared_error(targets, left) + _squared_error(targets, right)
            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold
                best_left = left
                best_right = right

    if best_feature is None or best_left is None or best_right is None:
        return TreeNode(value=node_value)

    return TreeNode(
        value=node_value,
        feature_index=best_feature,
        threshold=best_threshold,
        left=_build_tree(
            x_rows,
            targets,
            best_left,
            depth + 1,
            max_depth,
            min_samples_leaf,
            max_bins,
            feature_subsample_ratio,
            rng,
        ),
        right=_build_tree(
            x_rows,
            targets,
            best_right,
            depth + 1,
            max_depth,
            min_samples_leaf,
            max_bins,
            feature_subsample_ratio,
            rng,
        ),
    )


def _candidate_thresholds(
    x_rows: list[list[float]],
    indices: list[int],
    feature_index: int,
    max_bins: int,
) -> list[float]:
    values = sorted(x_rows[idx][feature_index] for idx in indices)
    if len(values) < 2:
        return []
    thresholds: list[float] = []
    for bin_idx in range(1, max_bins + 1):
        pos = int(len(values) * bin_idx / (max_bins + 1))
        if pos <= 0 or pos >= len(values):
            continue
        left = values[pos - 1]
        right = values[pos]
        threshold = (left + right) * 0.5
        if not thresholds or abs(threshold - thresholds[-1]) > 1e-9:
            thresholds.append(threshold)
    return thresholds


def _squared_error(targets: list[float], indices: list[int]) -> float:
    if not indices:
        return 0.0
    mean = _mean([targets[idx] for idx in indices])
    return sum((targets[idx] - mean) ** 2 for idx in indices)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def subsample_training_rows(
    x_rows: list[list[float]],
    y_u: list[float],
    y_v: list[float],
    y_w: list[float],
    max_samples: int,
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    if max_samples <= 0 or len(x_rows) <= max_samples:
        return x_rows, y_u, y_v, y_w
    rng = random.Random(19)
    picked = sorted(rng.sample(range(len(x_rows)), max_samples))
    return (
        [x_rows[idx] for idx in picked],
        [y_u[idx] for idx in picked],
        [y_v[idx] for idx in picked],
        [y_w[idx] for idx in picked],
    )


def subsample_training_samples(samples: list, max_samples: int, seed: int = 19) -> list:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(samples)), max_samples))
    return [samples[idx] for idx in picked]


@dataclass(slots=True)
class ResidualWindModel:
    u_model: object
    v_model: object
    w_model: object
    feature_names: list[str]

    def predict(self, features: dict[str, float]) -> tuple[float, float, float]:
        vector = [features[name] for name in self.feature_names]
        return self.u_model.predict(vector), self.v_model.predict(vector), self.w_model.predict(vector)

    def predict_batch(self, feature_rows: list[list[float]] | object) -> tuple[list[float], list[float], list[float]]:
        if feature_rows is None:
            return [], [], []
        if hasattr(feature_rows, "__len__") and len(feature_rows) == 0:
            return [], [], []
        if hasattr(self.u_model, "predict_many"):
            # Parallel u/v/w when the feature matrix is large enough to amortize threads.
            n = len(feature_rows)
            if n >= 512:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=3) as pool:
                    fu = pool.submit(self.u_model.predict_many, feature_rows)
                    fv = pool.submit(self.v_model.predict_many, feature_rows)
                    fw = pool.submit(self.w_model.predict_many, feature_rows)
                    return fu.result(), fv.result(), fw.result()
            return (
                self.u_model.predict_many(feature_rows),
                self.v_model.predict_many(feature_rows),
                self.w_model.predict_many(feature_rows),
            )
        rows = feature_rows if isinstance(feature_rows, list) else np.asarray(feature_rows, dtype=np.float32).tolist()
        du = [self.u_model.predict(row) for row in rows]
        dv = [self.v_model.predict(row) for row in rows]
        dw = [self.w_model.predict(row) for row in rows]
        return du, dv, dw

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": self.feature_names,
                "u_model": self.u_model.to_dict(),
                "v_model": self.v_model.to_dict(),
                "w_model": self.w_model.to_dict(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> "ResidualWindModel":
        payload = json.loads(raw)
        feature_names = list(payload["feature_names"])
        w_payload = payload.get("w_model")
        if w_payload is None:
            # Backward compatibility for 2D models serialized before vertical wind support.
            w_payload = LinearRegressor(
                weights=[0.0] * len(feature_names),
                bias=0.0,
                means=[0.0] * len(feature_names),
                scales=[1.0] * len(feature_names),
            ).to_dict()
        return cls(
            u_model=_regressor_from_dict(payload["u_model"]),
            v_model=_regressor_from_dict(payload["v_model"]),
            w_model=_regressor_from_dict(w_payload),
            feature_names=feature_names,
        )


def _regressor_from_dict(payload: dict) -> object:
    model_type = payload.get("model_type", "linear")
    if model_type == "xgboost":
        return XGBoostRegressor.from_dict(payload)
    if model_type == "lightgbm":
        return LightGBMRegressor.from_dict(payload)
    if model_type == "gbrt":
        return GradientBoostedTreesRegressor.from_dict(payload)
    if model_type == "linear":
        return LinearRegressor.from_dict(payload)
    raise ValueError(f"Unsupported model type: {model_type}")
