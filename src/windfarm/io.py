from __future__ import annotations

import json
from pathlib import Path

from .types import CoarseWindSample, Observation, TerrainField, TrainingSample


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def terrain_from_dict(payload: dict) -> TerrainField:
    return TerrainField(
        elevation=payload["elevation"],
        slope=payload["slope"],
        aspect=payload["aspect"],
        roughness=payload["roughness"],
    )


def coarse_samples_from_dict(items: list[dict]) -> list[CoarseWindSample]:
    return [CoarseWindSample(**item) for item in items]


def training_samples_from_dict(items: list[dict]) -> list[TrainingSample]:
    return [TrainingSample(**item) for item in items]


def observations_from_dict(items: list[dict]) -> list[Observation]:
    return [Observation(**item) for item in items]

