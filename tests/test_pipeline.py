from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from windfarm.belief import belief_snapshot, create_belief_map
from windfarm.config import TaskConfig, save_task_config
from windfarm.api_server import _api_schema
from windfarm.dashboard import build_dashboard_from_report, build_live_dashboard_html
from windfarm.execution import NavigationEngine
from windfarm.io import coarse_samples_from_dict, observations_from_dict
from windfarm.mission_runner import MissionRunner
from windfarm.pipeline import WindFarmPipeline
from windfarm.planner import compute_return_cost_map, plan_path_details, transition_cost_breakdown
from windfarm.simulator import EnvironmentSimulator
from windfarm.types import Mission, TrainingSample


class PipelineTest(unittest.TestCase):
    def test_end_to_end_training_prediction_and_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = TaskConfig()
            config.model.model_type = "gbrt"
            config.model.num_estimators = 6
            config.model.max_depth = 2
            config.model.min_samples_leaf = 8
            config.model.max_training_samples = 300
            config.model.altitude_levels = 3
            config.simulation.width = 8
            config.simulation.height = 6
            config.simulation.time_steps = 5
            config.mission.goal = (5, 4)
            config.mission.max_steps = 10
            config.mission.max_altitude_level = 2
            save_task_config(Path(tmp) / "config.json", config)
            simulator = EnvironmentSimulator(seed=7)
            dataset = simulator.generate_dataset(
                width=config.simulation.width,
                height=config.simulation.height,
                resolution_m=config.simulation.resolution_m,
                time_steps=config.simulation.time_steps,
                start_time=config.simulation.start_time,
                sample_interval_seconds=config.simulation.sample_interval_seconds,
            )
            simulator.write_dataset(dataset, tmp)
            terrain = Path(tmp) / "terrain.json"
            training = Path(tmp) / "training.json"
            coarse = Path(tmp) / "coarse_wind.json"
            observations = Path(tmp) / "observations.json"
            truth = Path(tmp) / "truth.json"
            model = Path(tmp) / "model.json"

            pipeline = WindFarmPipeline.from_paths(terrain, model_config=config.model)
            training_samples = [TrainingSample(**item) for item in json.loads(training.read_text(encoding="utf-8"))["samples"]]
            metrics = pipeline.train(training_samples)
            self.assertLess(metrics["rmse_final"], metrics["rmse_physics"])
            pipeline.save_model(model)

            restored = WindFarmPipeline.from_paths(terrain, model, config.model)
            grid = restored.predict_grid("2026-03-21T12:00:00", 3.2, -4.4)
            self.assertEqual(len(grid["u"]), config.simulation.height)
            self.assertEqual(len(grid["u"][0]), config.simulation.width)
            self.assertEqual(len(grid["w"]), config.simulation.height)
            self.assertIn("u_layers", grid)

            window = restored.predict_window("2026-03-21T12:00:00", 3.2, -4.4, 2, 3, 7, 9, altitude_level=0)
            self.assertEqual(window["x_bounds"], [2, 7])
            self.assertEqual(window["y_bounds"], [3, 6])
            self.assertEqual(window["u"][0][0], grid["u"][3][2])
            self.assertEqual(window["u"][-1][-1], grid["u"][5][6])

            result = restored.online_plan(
                coarse,
                observations,
                Mission(start=(0, 0), goal=(6, 5), max_steps=10, step_distance_m=100.0, max_altitude_level=2),
            )
            self.assertGreaterEqual(len(result["executed_path"]), 1)
            self.assertIn("goal_reached", result)

            result_3d = restored.online_plan(
                coarse,
                observations,
                Mission(start=(0, 0, 0), goal=(5, 4, 2), max_steps=9, step_distance_m=100.0, max_altitude_level=2, climb_cost_per_level_j=50.0),
            )
            self.assertGreaterEqual(len(result_3d["executed_path"][0]), 3)
            self.assertAlmostEqual(result_3d["executed_path"][-1][2], 2.0, delta=0.45)

            report_path = Path(tmp) / "mission_report.json"
            report = MissionRunner(restored, config).run(coarse, observations, report_path, truth)
            self.assertIn("battery_ratio", report)
            self.assertGreater(report["steps_executed"], 0)
            self.assertIn("sensor_packet", report["trace"][0])
            self.assertIn("maps", report["trace"][0])
            self.assertIn("executed_path", report)
            self.assertEqual(report["route_summary"]["actual_goal"], [5, 4, 0])
            self.assertIn("model_summary", report)
            self.assertEqual(report["report_mode"]["trace_encoding"], "keyframe-plus-delta")
            self.assertIn("physics_at_drone", report["trace"][0])
            self.assertIn("candidate_paths", report["trace"][0])
            self.assertIn("map_delta", report["trace"][0])
            self.assertIn("w", report["trace"][0]["prediction_at_drone"])
            self.assertIn("measured_wind_w", report["trace"][0]["sensor_packet"])
            self.assertEqual(len(report["trace"][0]["position"]), 3)
            self.assertIn("prediction_profile_at_drone", report["trace"][0])
            self.assertGreaterEqual(len(report["trace"][0]["prediction_profile_at_drone"]["speed"]), 1)
            self.assertEqual(len(report["trace"][0]["prediction_profile_at_drone"]["speed"]), config.model.altitude_levels)
            self.assertIn("belief_wind_speed_layers", report["trace"][0]["maps"])
            self.assertIn("prediction_residual_speed_layers", report["trace"][0]["maps"])
            self.assertIn("belief_entropy_layers", report["trace"][0]["maps"])
            self.assertIn("belief_safety_layers", report["trace"][0]["maps"])
            self.assertIn("planned_path_cost_breakdown", report["trace"][0])
            self.assertIn("total_cost_j", report["trace"][0]["planned_path_cost_breakdown"])
            self.assertEqual(report["trace"][0]["forecast_mode"], "global_keyframe")
            self.assertIn("safety", report["trace"][0]["belief_at_drone"])
            self.assertIn("uplift_prob", report["trace"][0]["belief_at_drone"])
            mode0 = report["trace"][0]["planning_mode"]
            self.assertTrue(mode0.startswith("mpc_") or mode0.startswith("guide_"), mode0)

            custom_report = MissionRunner(restored, config).run(
                coarse,
                observations,
                Path(tmp) / "mission_report_custom.json",
                truth,
                start=(1, 1),
                goal=(6, 5),
            )
            self.assertEqual(custom_report["route_summary"]["actual_start"], [1, 1, 0])
            self.assertEqual(custom_report["route_summary"]["actual_goal"], [6, 5, 0])
            self.assertGreaterEqual(len(custom_report["executed_path"]), 1)

            dashboard_path = Path(tmp) / "dashboard.html"
            build_dashboard_from_report(report_path, dashboard_path)
            self.assertTrue(dashboard_path.exists())
            self.assertIn("残差热力图", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("训练摘要", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("物理场 vs 学习场 vs 真值", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("返航裕度", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("altitudeSelect", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("drawLayeredPath", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("forecastMode", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("costBreakdownBox", dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("/api/state", build_live_dashboard_html())
            self.assertIn("/api/v1/schema", str(_api_schema()))

    def test_planner_enforces_prefix_and_return_budget(self) -> None:
        belief_map = create_belief_map(3, 1, 1)
        mission = Mission(
            start=(0, 0, 0),
            goal=(2, 0, 0),
            max_steps=4,
            step_distance_m=100.0,
            home=(0, 0, 0),
            max_return_cost_j=200.0,
        )

        direct_return_cost = compute_return_cost_map(belief_map, mission)
        self.assertGreater(direct_return_cost[(1, 0, 0)], mission.max_return_cost_j)

        planning = plan_path_details(belief_map, mission)
        mode = planning["planning_mode"]
        self.assertTrue(
            mode in {"mpc_relaxed_return", "mpc_strict_return", "greedy_progress", "guide_straight_agl"}
            or mode.startswith("guide_straight_agl_")
            or mode.startswith("guide_corridor_"),
            msg=f"unexpected planning_mode={mode}",
        )
        self.assertEqual(planning["path"][0], (0, 0, 0))
        self.assertEqual(planning["path"][-1], (2, 0, 0))
        self.assertTrue(any(item.get("goal_reached") for item in planning["candidates"]) or planning["path"][-1] == (2, 0, 0))
        self.assertIn("path_cost_breakdown", planning)
        self.assertIn("energy_j", planning["path_cost_breakdown"])
        self.assertGreaterEqual(planning["candidates"][0]["horizon_steps"], 2)

    def test_transition_cost_breakdown_matches_total_cost(self) -> None:
        belief_map = create_belief_map(2, 1, 1)
        belief_map.cells[0][0][1].expected_energy_gain = 0.8
        belief_map.cells[0][0][1].uncertainty = 0.4
        belief_map.cells[0][0][1].safety_penalty = 0.2
        belief_map.cells[0][0][1].wind_u = 1.1
        belief_map.cells[0][0][1].wind_w = 0.3
        mission = Mission(start=(0, 0, 0), goal=(1, 0, 0), max_steps=2, step_distance_m=100.0)

        breakdown = transition_cost_breakdown(belief_map, (0, 0, 0), (1, 0, 0), mission)
        self.assertAlmostEqual(
            breakdown.total_cost_j,
            breakdown.energy_j
            - breakdown.progress_reward_j
            + breakdown.uncertainty_cost_j
            + breakdown.safety_cost_j
            + breakdown.altitude_bias_j,
            + breakdown.vertical_maneuver_cost_j,
        )

    def test_planner_does_not_loiter_indefinitely_inside_uplift_zone(self) -> None:
        belief_map = create_belief_map(7, 3, 1)
        for x in range(3):
            cell = belief_map.cells[0][1][x]
            cell.wind_w = 3.2
            cell.expected_energy_gain = 3.5
            cell.mode_prob_uplift = 0.98
            cell.mode_prob_sink = 0.01
            cell.uncertainty = 0.05
            cell.safety_penalty = 0.0
        mission = Mission(
            start=(0, 1, 0),
            goal=(6, 1, 0),
            max_steps=8,
            step_distance_m=100.0,
            nominal_airspeed=13.8,
            climb_cost_per_level_j=40.0,
            max_altitude_level=0,
        )

        planning = plan_path_details(
            belief_map,
            mission,
            anytime_rounds=3,
            horizon_steps=8,
            beam_width=24,
            branch_width=8,
        )

        rounded_path = [(round(x), round(y), round(z)) for x, y, z in planning["path"]]
        self.assertGreaterEqual(max(point[0] for point in rounded_path), 4)
        self.assertLess(
            abs(mission.goal[0] - planning["path"][-1][0]),
            abs(mission.goal[0] - planning["path"][0][0]),
        )
        self.assertGreaterEqual(len(set(rounded_path)), max(4, len(rounded_path) - 2))

    def test_planner_advances_along_primary_goal_axis_in_uplift(self) -> None:
        belief_map = create_belief_map(9, 5, 1)
        for x in range(3):
            for y in range(1, 4):
                cell = belief_map.cells[0][y][x]
                cell.wind_w = 4.5
                cell.expected_energy_gain = 4.0
                cell.mode_prob_uplift = 0.99
                cell.mode_prob_sink = 0.01
                cell.uncertainty = 0.03
                cell.safety_penalty = 0.0
        mission = Mission(
            start=(0, 2, 0),
            goal=(8, 0, 0),
            max_steps=10,
            step_distance_m=100.0,
            nominal_airspeed=13.8,
            climb_cost_per_level_j=35.0,
            max_altitude_level=0,
        )

        planning = plan_path_details(
            belief_map,
            mission,
            anytime_rounds=3,
            horizon_steps=8,
            beam_width=20,
            branch_width=8,
        )

        path = planning["path"]
        self.assertGreater(path[-1][0], path[0][0] + 1.5)
        self.assertLess(math.hypot(path[-1][0] - 8.0, path[-1][1] - 0.0), math.hypot(8.0, 2.0))

    def test_belief_snapshot_exposes_probabilities_and_safety(self) -> None:
        belief_map = create_belief_map(2, 2, 1)
        cell = belief_map.cells[0][0][0]
        cell.mode_prob_sink = 0.1
        cell.mode_prob_neutral = 0.3
        cell.mode_prob_uplift = 0.6
        cell.belief_entropy = 1.2
        cell.safety_penalty = 0.4

        snapshot = belief_snapshot(belief_map, 0)

        self.assertAlmostEqual(snapshot["uplift_prob"][0][0], 0.6)
        self.assertAlmostEqual(snapshot["sink_prob"][0][0], 0.1)
        self.assertAlmostEqual(snapshot["entropy"][0][0], 1.2)
        self.assertAlmostEqual(snapshot["safety"][0][0], 0.4)

    def test_navigation_engine_uses_local_window_between_keyframes(self) -> None:
        config = TaskConfig()
        config.model.model_type = "gbrt"
        config.model.num_estimators = 4
        config.model.max_depth = 2
        config.model.min_samples_leaf = 6
        config.model.max_training_samples = 120
        config.model.altitude_levels = 2
        config.simulation.width = 7
        config.simulation.height = 5
        config.simulation.time_steps = 4
        config.simulation.report_keyframe_interval = 3
        config.mission.goal = (6, 4)
        config.mission.max_steps = 6
        config.mission.max_altitude_level = 1

        simulator = EnvironmentSimulator(seed=7)
        dataset = simulator.generate_dataset(
            width=config.simulation.width,
            height=config.simulation.height,
            resolution_m=config.simulation.resolution_m,
            time_steps=config.simulation.time_steps,
            start_time=config.simulation.start_time,
            sample_interval_seconds=config.simulation.sample_interval_seconds,
        )
        pipeline = WindFarmPipeline(terrain=dataset.terrain, model_config=config.model)
        engine = NavigationEngine(pipeline, config)
        context = engine.create_context((0, 0), (6, 4))
        coarse_samples = coarse_samples_from_dict(dataset.coarse_wind)
        observations = observations_from_dict(dataset.observations)

        first = engine.step(context, coarse_samples[0], observations[0], None)
        second = engine.step(context, coarse_samples[1], observations[1], None)

        self.assertEqual(first["forecast_mode"], "global_keyframe")
        self.assertEqual(second["forecast_mode"], "local_window")


if __name__ == "__main__":
    unittest.main()
