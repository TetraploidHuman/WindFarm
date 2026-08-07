from __future__ import annotations

from .config import TaskConfig
from .simulator import EnvironmentSimulator


def generate_sample_dataset(base_dir: str) -> None:
    config = TaskConfig()
    simulator = EnvironmentSimulator(seed=7)
    dataset = simulator.generate_dataset(
        width=config.simulation.width,
        height=config.simulation.height,
        resolution_m=config.simulation.resolution_m,
        time_steps=config.simulation.time_steps,
        start_time=config.simulation.start_time,
        sample_interval_seconds=config.simulation.sample_interval_seconds,
    )
    simulator.write_dataset(dataset, base_dir)
