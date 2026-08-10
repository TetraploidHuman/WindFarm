from __future__ import annotations

import argparse
import os
from pathlib import Path

from .api_server import serve_api
from .config import TaskConfig, load_task_config, save_task_config
from .dashboard import build_dashboard_from_report, build_live_dashboard_html
from .io import (
    ensure_terrain_npz,
    ensure_truth_npz,
    read_json,
    write_json,
)
from .live_server import serve_interactive_dashboard, serve_live_dashboard
from .mission_runner import MissionRunner
from .pipeline import WindFarmPipeline, train_from_files
from .repository import ModelRepository
from .sample_data import generate_sample_dataset
from .simulator import EnvironmentSimulator
from .types import Mission


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "True", "yes"}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="windfarm")
    sub = parser.add_subparsers(dest="command", required=True)

    sample_cmd = sub.add_parser("generate-sample-data")
    sample_cmd.add_argument("--output-dir", required=True)

    config_cmd = sub.add_parser("generate-config")
    config_cmd.add_argument("--output", required=True)

    simulate_cmd = sub.add_parser("simulate")
    simulate_cmd.add_argument("--config", required=True)
    simulate_cmd.add_argument("--output-dir", required=True)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--terrain", required=True)
    train_cmd.add_argument("--training", required=True)
    train_cmd.add_argument("--model-out", required=True)
    train_cmd.add_argument("--metrics-out")
    train_cmd.add_argument("--config")

    predict_cmd = sub.add_parser("predict")
    predict_cmd.add_argument("--terrain", required=True)
    predict_cmd.add_argument("--model")
    predict_cmd.add_argument("--timestamp", required=True)
    predict_cmd.add_argument("--u-km", required=True, type=float)
    predict_cmd.add_argument("--v-km", required=True, type=float)
    predict_cmd.add_argument("--w-km", type=float, default=0.0)
    predict_cmd.add_argument("--output", required=True)

    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--terrain", required=True)
    plan_cmd.add_argument("--model")
    plan_cmd.add_argument("--coarse-wind", required=True)
    plan_cmd.add_argument("--observations", required=True)
    plan_cmd.add_argument("--start", nargs="+", required=True, type=int)
    plan_cmd.add_argument("--goal", nargs="+", required=True, type=int)
    plan_cmd.add_argument("--max-steps", type=int, default=20)
    plan_cmd.add_argument("--step-distance-m", type=float, default=100.0)
    plan_cmd.add_argument("--output", required=True)

    mission_cmd = sub.add_parser("run-mission")
    mission_cmd.add_argument("--config", required=True)
    mission_cmd.add_argument("--terrain", required=True)
    mission_cmd.add_argument("--model", required=True)
    mission_cmd.add_argument("--coarse-wind", required=True)
    mission_cmd.add_argument("--observations", required=True)
    mission_cmd.add_argument("--truth")
    mission_cmd.add_argument("--start", nargs="+", type=int)
    mission_cmd.add_argument("--goal", nargs="+", type=int)
    mission_cmd.add_argument("--output", required=True)

    navigate_cmd = sub.add_parser("navigate")
    navigate_cmd.add_argument("--config", required=True)
    navigate_cmd.add_argument("--terrain", required=True)
    navigate_cmd.add_argument("--model", required=True)
    navigate_cmd.add_argument("--coarse-wind", required=True)
    navigate_cmd.add_argument("--observations", required=True)
    navigate_cmd.add_argument("--start", nargs="+", required=True, type=int)
    navigate_cmd.add_argument("--goal", nargs="+", required=True, type=int)
    navigate_cmd.add_argument("--truth")
    navigate_cmd.add_argument("--output", required=True)

    dashboard_cmd = sub.add_parser("build-dashboard")
    dashboard_cmd.add_argument("--report", required=True)
    dashboard_cmd.add_argument("--output", required=True)

    serve_cmd = sub.add_parser("serve-dashboard")
    serve_cmd.add_argument("--report", required=True)
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--interval-seconds", type=float, default=3.0)

    interactive_cmd = sub.add_parser("serve-navigate")
    interactive_cmd.add_argument("--config", required=True)
    interactive_cmd.add_argument("--terrain", required=True)
    interactive_cmd.add_argument("--model", required=True)
    interactive_cmd.add_argument("--host", default="127.0.0.1")
    interactive_cmd.add_argument("--port", type=int, default=8870)
    interactive_cmd.add_argument("--interval-seconds", type=float, default=1.0)
    interactive_cmd.add_argument("--sim-minutes-per-tick", type=int, default=1)

    api_cmd = sub.add_parser("serve-api")
    api_cmd.add_argument("--host", default="127.0.0.1")
    api_cmd.add_argument("--port", type=int, default=8780)

    repo_cmd = sub.add_parser("run-demo")
    repo_cmd.add_argument("--config", required=True)
    repo_cmd.add_argument("--runs-dir", required=True)
    repo_cmd.add_argument("--run-name")
    repo_cmd.add_argument(
        "--scenario",
        help="Use a prebuilt real scenario directory under scenarios/<name> (terrain+Open-Meteo wind)",
    )

    scenarios_cmd = sub.add_parser("build-scenarios", help="Download SRTM DEM + Open-Meteo wind and build scenario datasets")
    scenarios_cmd.add_argument("--output-dir", default="scenarios")
    scenarios_cmd.add_argument("--data-dir", default="data")
    scenarios_cmd.add_argument("--base-config", default="config_eval.json")
    scenarios_cmd.add_argument("--only", nargs="*", help="Optional subset of scenario names")
    scenarios_cmd.add_argument("--resolution-m", type=float, default=30.0, help="Native DEM crop resolution in meters")
    scenarios_cmd.add_argument("--width", type=int, default=None, help="Optional grid width override")
    scenarios_cmd.add_argument("--height", type=int, default=None, help="Optional grid height override")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-sample-data":
        generate_sample_dataset(args.output_dir)
        return

    if args.command == "generate-config":
        save_task_config(args.output, TaskConfig())
        return

    if args.command == "simulate":
        config = load_task_config(args.config)
        simulator = EnvironmentSimulator(seed=7)
        dataset = simulator.generate_dataset(
            width=config.simulation.width,
            height=config.simulation.height,
            resolution_m=config.simulation.resolution_m,
            time_steps=config.simulation.time_steps,
            start_time=config.simulation.start_time,
            sample_interval_seconds=config.simulation.sample_interval_seconds,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        simulator.write_dataset(dataset, args.output_dir)
        return

    if args.command == "train":
        config = load_task_config(args.config) if args.config else TaskConfig()
        train_from_files(args.terrain, args.training, args.model_out, args.metrics_out, config.model)
        return

    if args.command == "predict":
        pipeline = WindFarmPipeline.from_paths(args.terrain, args.model)
        payload = pipeline.predict_grid(args.timestamp, args.u_km, args.v_km, args.w_km)
        write_json(args.output, payload)
        return

    if args.command == "plan":
        pipeline = WindFarmPipeline.from_paths(args.terrain, args.model)
        mission = Mission(
            start=_parse_xyz(args.start, parser, "--start"),
            goal=_parse_xyz(args.goal, parser, "--goal"),
            max_steps=args.max_steps,
            step_distance_m=args.step_distance_m,
        )
        payload = pipeline.online_plan(args.coarse_wind, args.observations, mission)
        write_json(args.output, payload)
        return

    if args.command == "run-mission":
        config = load_task_config(args.config)
        pipeline = WindFarmPipeline.from_paths(args.terrain, args.model, config.model)
        start = _parse_xyz(args.start, parser, "--start") if args.start else None
        goal = _parse_xyz(args.goal, parser, "--goal") if args.goal else None
        MissionRunner(pipeline, config).run(args.coarse_wind, args.observations, args.output, args.truth, start, goal)
        return

    if args.command == "navigate":
        config = load_task_config(args.config)
        pipeline = WindFarmPipeline.from_paths(args.terrain, args.model, config.model)
        MissionRunner(pipeline, config).run(
            args.coarse_wind,
            args.observations,
            args.output,
            args.truth,
            _parse_xyz(args.start, parser, "--start"),
            _parse_xyz(args.goal, parser, "--goal"),
        )
        return

    if args.command == "build-dashboard":
        build_dashboard_from_report(args.report, args.output)
        return

    if args.command == "serve-dashboard":
        serve_live_dashboard(args.report, args.host, args.port, args.interval_seconds)
        return

    if args.command == "serve-navigate":
        config = load_task_config(args.config)
        serve_interactive_dashboard(
            args.terrain,
            args.model,
            config,
            args.host,
            args.port,
            args.interval_seconds,
            args.sim_minutes_per_tick,
        )
        return

    if args.command == "serve-api":
        serve_api(args.host, args.port)
        return

    if args.command == "build-scenarios":
        from .scenarios import build_default_scenarios

        summaries = build_default_scenarios(
            scenarios_dir=args.output_dir,
            data_dir=args.data_dir,
            base_config=args.base_config,
            only=args.only or None,
            resolution_m=args.resolution_m,
            width=args.width,
            height=args.height,
        )
        print(f"built {len(summaries)} scenarios under {args.output_dir}")
        return

    if args.command == "run-demo":
        import shutil

        from .perf import warmup_numeric_kernels

        warmup_numeric_kernels()
        config = load_task_config(args.config)
        repository = ModelRepository(args.runs_dir)
        artifacts = repository.create_run(args.run_name)
        scenario_dir = None
        if getattr(args, "scenario", None):
            scenario_dir = Path(args.scenario)
            if not scenario_dir.is_dir():
                candidate = Path("scenarios") / args.scenario
                if candidate.is_dir():
                    scenario_dir = candidate
                else:
                    raise SystemExit(f"Scenario directory not found: {args.scenario}")
            scenario_cfg = scenario_dir / "config.json"
            if scenario_cfg.exists():
                config = load_task_config(scenario_cfg)
        save_task_config(artifacts.run_dir / "config.json", config)
        if scenario_dir is not None:
            # training.json is only needed when (re)fitting the residual model.
            force_train = _env_flag("WINDFARM_FORCE_TRAIN")
            scenario_model = scenario_dir / "model.json"
            will_reuse_model = scenario_model.exists() and not force_train
            if _env_flag("WINDFARM_SKIP_TRAIN") and not will_reuse_model:
                raise SystemExit(
                    f"WINDFARM_SKIP_TRAIN=1 but cached model missing: {scenario_model}. "
                    "Run once without SKIP_TRAIN (caches model.json) or unset the flag."
                )
            asset_names = ["terrain.json", "coarse_wind.json", "observations.json", "truth.json"]
            if not will_reuse_model:
                asset_names.append("training.json")
            for name in asset_names:
                src = scenario_dir / name
                if not src.exists():
                    raise SystemExit(f"Scenario missing {name}: {src}")
                _link_or_copy(src, artifacts.run_dir / name)
            # Prefer binary sidecars beside the scenario JSON (shared across runs).
            if (scenario_dir / "truth.json").exists():
                ensure_truth_npz(scenario_dir / "truth.json")
            if (scenario_dir / "terrain.json").exists():
                ensure_terrain_npz(scenario_dir / "terrain.json")
            for stem in ("terrain", "truth"):
                cache_src = scenario_dir / f"{stem}.npz"
                if cache_src.exists():
                    _link_or_copy(cache_src, artifacts.run_dir / f"{stem}.npz")
            meta_src = scenario_dir / "scenario_meta.json"
            if meta_src.exists():
                shutil.copy2(meta_src, artifacts.run_dir / "scenario_meta.json")
        else:
            simulator = EnvironmentSimulator(seed=7)
            dataset = simulator.generate_dataset(
                width=config.simulation.width,
                height=config.simulation.height,
                resolution_m=config.simulation.resolution_m,
                time_steps=config.simulation.time_steps,
                start_time=config.simulation.start_time,
                sample_interval_seconds=config.simulation.sample_interval_seconds,
            )
            simulator.write_dataset(dataset, str(artifacts.run_dir))
            ensure_terrain_npz(artifacts.terrain_path)
            ensure_truth_npz(artifacts.run_dir / "truth.json")
            force_train = _env_flag("WINDFARM_FORCE_TRAIN")
            will_reuse_model = False
            scenario_model = None

        cache_model = _env_flag("WINDFARM_CACHE_MODEL", default=True)
        scenario_metrics = (scenario_dir / "metrics.json") if scenario_dir is not None else None
        reuse_model = bool(will_reuse_model)
        if reuse_model:
            assert scenario_model is not None
            _link_or_copy(scenario_model, artifacts.model_path)
            if scenario_metrics is not None and scenario_metrics.exists():
                _link_or_copy(scenario_metrics, artifacts.metrics_path)
                metrics = read_json(artifacts.metrics_path)
            else:
                metrics = {"reused_model": True, "source": str(scenario_model)}
                write_json(artifacts.metrics_path, metrics)
        else:
            metrics = train_from_files(
                artifacts.terrain_path,
                artifacts.run_dir / "training.json",
                artifacts.model_path,
                artifacts.metrics_path,
                config.model,
            )
            if (
                cache_model
                and scenario_dir is not None
                and artifacts.model_path.exists()
            ):
                _link_or_copy(artifacts.model_path, scenario_dir / "model.json")
                if artifacts.metrics_path.exists():
                    _link_or_copy(artifacts.metrics_path, scenario_dir / "metrics.json")

        pipeline = WindFarmPipeline.from_paths(
            artifacts.terrain_path,
            artifacts.model_path,
            config.model,
            altitude_step_m=config.mission.altitude_step_m,
        )
        report = MissionRunner(pipeline, config).run(
            artifacts.run_dir / "coarse_wind.json",
            artifacts.run_dir / "observations.json",
            artifacts.mission_path,
            artifacts.run_dir / "truth.json",
        )
        skip_dash = _env_flag("WINDFARM_SKIP_DASHBOARD")
        if not skip_dash:
            build_dashboard_from_report(artifacts.mission_path, artifacts.dashboard_path)
            artifacts.live_dashboard_path.write_text(build_live_dashboard_html(), encoding="utf-8")
        repository.save_manifest(
            artifacts,
            {
                "metrics": metrics,
                "mission": {
                    "goal_reached": report["goal_reached"],
                    "steps_executed": report["steps_executed"],
                    "battery_ratio": report["battery_ratio"],
                },
                "scenario": str(scenario_dir) if scenario_dir is not None else "synthetic",
                "artifacts": {
                    "dashboard": str(artifacts.dashboard_path) if not skip_dash else None,
                    "live_dashboard": str(artifacts.live_dashboard_path) if not skip_dash else None,
                    "mission_report": str(artifacts.mission_path),
                },
                "model_reused": bool(reuse_model),
            },
        )
        return


def _parse_xyz(values: list[int], parser: argparse.ArgumentParser, flag: str) -> tuple[int, int] | tuple[int, int, int]:
    if len(values) not in {2, 3}:
        parser.error(f"{flag} expects 2 or 3 integers")
    return tuple(values)  # type: ignore[return-value]


if __name__ == "__main__":
    main()
