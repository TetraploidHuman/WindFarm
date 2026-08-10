from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import CoarseWindSample, Observation, TerrainField, TrainingSample


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _sidecar_npz(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.npz")


def _cache_is_fresh(json_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists() or not json_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= json_path.stat().st_mtime
    except OSError:
        return False


def ensure_terrain_npz(terrain_json: str | Path) -> Path:
    """Build/refresh ``terrain.npz`` beside terrain.json (elevation/slope/aspect/roughness)."""
    json_path = Path(terrain_json)
    cache_path = _sidecar_npz(json_path)
    if _cache_is_fresh(json_path, cache_path):
        return cache_path
    payload = read_json(json_path)
    terrain = payload["terrain"]
    np.savez_compressed(
        cache_path,
        elevation=np.asarray(terrain["elevation"], dtype=np.float64),
        slope=np.asarray(terrain["slope"], dtype=np.float64),
        aspect=np.asarray(terrain["aspect"], dtype=np.float64),
        roughness=np.asarray(terrain["roughness"], dtype=np.float64),
    )
    return cache_path


def load_terrain_document(terrain_json: str | Path) -> dict:
    """Load terrain.json using a sidecar NPZ when available (same numeric content)."""
    json_path = Path(terrain_json)
    cache_path = _sidecar_npz(json_path)
    if _cache_is_fresh(json_path, cache_path):
        arrays = np.load(cache_path)
        return {
            "terrain": {
                "elevation": arrays["elevation"],
                "slope": arrays["slope"],
                "aspect": arrays["aspect"],
                "roughness": arrays["roughness"],
            }
        }
    payload = read_json(json_path)
    terrain = payload["terrain"]
    packed = {
        "elevation": np.asarray(terrain["elevation"], dtype=np.float64),
        "slope": np.asarray(terrain["slope"], dtype=np.float64),
        "aspect": np.asarray(terrain["aspect"], dtype=np.float64),
        "roughness": np.asarray(terrain["roughness"], dtype=np.float64),
    }
    np.savez_compressed(cache_path, **packed)
    out = dict(payload)
    out["terrain"] = packed
    return out


def ensure_truth_npz(truth_json: str | Path) -> Path:
    """Build/refresh ``truth.npz`` beside truth.json (stacked u/v/w + timestamps)."""
    json_path = Path(truth_json)
    cache_path = _sidecar_npz(json_path)
    if _cache_is_fresh(json_path, cache_path):
        return cache_path
    payload = read_json(json_path)
    fields = payload["truth_fields"]
    if not fields:
        np.savez_compressed(
            cache_path,
            timestamps=np.asarray([], dtype="U32"),
            u=np.zeros((0, 1, 1, 1), dtype=np.float64),
            v=np.zeros((0, 1, 1, 1), dtype=np.float64),
            w=np.zeros((0, 1, 1, 1), dtype=np.float64),
        )
        return cache_path
    timestamps = np.asarray([str(item["timestamp"]) for item in fields], dtype="U32")
    u = np.stack([np.asarray(item["u"], dtype=np.float64) for item in fields], axis=0)
    v = np.stack([np.asarray(item["v"], dtype=np.float64) for item in fields], axis=0)
    w = np.stack([np.asarray(item["w"], dtype=np.float64) for item in fields], axis=0)
    np.savez_compressed(cache_path, timestamps=timestamps, u=u, v=v, w=w)
    return cache_path


def load_truth_fields(truth_json: str | Path) -> list[dict]:
    """Load truth_fields list; prefers sidecar NPZ (views into stacked arrays)."""
    json_path = Path(truth_json)
    cache_path = ensure_truth_npz(json_path)
    data = np.load(cache_path)
    timestamps = data["timestamps"]
    u = data["u"]
    v = data["v"]
    w = data["w"]
    return [
        {
            "timestamp": str(timestamps[i]),
            "u": u[i],
            "v": v[i],
            "w": w[i],
        }
        for i in range(int(timestamps.shape[0]))
    ]


def terrain_from_dict(payload: dict) -> TerrainField:
    def _grid(value) -> list[list[float]]:
        # Keep list-of-lists for legacy ``elev[y][x]`` / truthiness checks.
        # Terrain grids are small; NPZ still avoids re-parsing the JSON text.
        return np.asarray(value, dtype=np.float64).tolist()

    return TerrainField(
        elevation=_grid(payload["elevation"]),
        slope=_grid(payload["slope"]),
        aspect=_grid(payload["aspect"]),
        roughness=_grid(payload["roughness"]),
    )


def coarse_samples_from_dict(items: list[dict]) -> list[CoarseWindSample]:
    fields = set(CoarseWindSample.__dataclass_fields__)
    out: list[CoarseWindSample] = []
    for item in items:
        payload = {k: item[k] for k in fields if k in item}
        out.append(CoarseWindSample(**payload))
    return out


def training_samples_from_dict(items: list[dict]) -> list[TrainingSample]:
    return [TrainingSample(**item) for item in items]


def observations_from_dict(items: list[dict]) -> list[Observation]:
    return [Observation(**item) for item in items]
