from __future__ import annotations

from pathlib import Path

from .execution import NavigationEngine, _goal_reached
from .io import coarse_samples_from_dict, load_truth_fields, observations_from_dict, read_json, write_json
from .types import Observation


class MissionRunner:
    def __init__(self, pipeline, config):
        self.pipeline = pipeline
        self.config = config
        self.engine = NavigationEngine(pipeline, config)

    def run(
        self,
        coarse_wind_path: str | Path,
        observations_path: str | Path,
        output_path: str | Path | None = None,
        truth_path: str | Path | None = None,
        start: tuple[int, int] | tuple[int, int, int] | None = None,
        goal: tuple[int, int] | tuple[int, int, int] | None = None,
    ) -> dict:
        coarse_samples = coarse_samples_from_dict(read_json(coarse_wind_path)["coarse_wind"])
        observations = observations_from_dict(read_json(observations_path)["observations"])
        obs_by_time = {item.timestamp: item for item in observations}
        truth_by_time = {}
        if truth_path:
            truth_by_time = {item["timestamp"]: item for item in load_truth_fields(truth_path)}

        requested_start = start or self.config.mission.start
        requested_goal = goal or self.config.mission.goal
        context = self.engine.create_context(requested_start, requested_goal)

        for sample in coarse_samples:
            observation: Observation | None = obs_by_time.get(sample.timestamp)
            truth_field = truth_by_time.get(sample.timestamp)
            self.engine.step(context, sample, observation, truth_field)
            if (
                _goal_reached(context.state, context.mission.goal)
                or context.battery_j <= 0.0
                or context.step >= context.mission.max_steps
            ):
                break

        report = self.engine.build_report(context, requested_start, requested_goal)
        if output_path:
            write_json(output_path, report)
        return report
