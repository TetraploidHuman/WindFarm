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


def terrain_from_dict(payload: dict) -> TerrainField:
    return TerrainField(
        elevation=payload["elevation"],
        slope=payload["slope"],
        aspect=payload["aspect"],
        roughness=payload["roughness"],
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

