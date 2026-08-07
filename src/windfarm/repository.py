from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .io import read_json, write_json
from .pipeline import WindFarmPipeline


@dataclass(slots=True)
class RunArtifacts:
    run_dir: Path
    terrain_path: Path
    model_path: Path
    metrics_path: Path
    mission_path: Path
    manifest_path: Path
    dashboard_path: Path
    live_dashboard_path: Path


class ModelRepository:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_name: str | None = None) -> RunArtifacts:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = run_name or f"run-{stamp}"
        run_dir = self.base_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(
            run_dir=run_dir,
            terrain_path=run_dir / "terrain.json",
            model_path=run_dir / "model.json",
            metrics_path=run_dir / "metrics.json",
            mission_path=run_dir / "mission_report.json",
            manifest_path=run_dir / "manifest.json",
            dashboard_path=run_dir / "dashboard.html",
            live_dashboard_path=run_dir / "live_dashboard.html",
        )

    def save_manifest(self, artifacts: RunArtifacts, payload: dict) -> None:
        write_json(artifacts.manifest_path, payload)

    def load_manifest(self, run_dir: str | Path) -> dict:
        return read_json(Path(run_dir) / "manifest.json")

    def load_pipeline(self, run_dir: str | Path) -> WindFarmPipeline:
        run_path = Path(run_dir)
        return WindFarmPipeline.from_paths(run_path / "terrain.json", run_path / "model.json")
