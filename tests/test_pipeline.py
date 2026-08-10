from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from windfarm.belief import belief_snapshot, create_belief_map
from windfarm.config import TaskConfig, save_task_config
from windfarm.api_server import _api_schema
from windfarm.dashboard import build_dashboard_from_report, build_live_dashboard_html
from windfarm.belief import BeliefUpdater
from windfarm.execution import LATERAL_PROBE_OFFSET_M, NavigationContext, NavigationEngine
from windfarm.io import coarse_samples_from_dict, observations_from_dict
from windfarm.mission_runner import MissionRunner
from windfarm.pipeline import WindFarmPipeline
from windfarm.planner import (
    compute_return_cost_map,
    plan_path_details,
    transition_cost_breakdown,
    _energy_guide_paths,
    _select_preferred_cruise_band,
)
from windfarm.controller import speed_to_fly_airspeed
from windfarm.physics import downscale_wind
from windfarm.simulator import EnvironmentSimulator
from windfarm.types import DroneState, Mission, TerrainField, TrainingSample


class PipelineTest(unittest.TestCase):
    def test_speed_to_fly_pushes_into_headwind_not_calm(self) -> None:
        nominal = 16.5
        # Calm / sub-gate head: stay near nominal (execution also energy-gates boosts).
        self.assertAlmostEqual(speed_to_fly_airspeed(nominal, 0.0, 0.0), nominal, places=2)
        self.assertAlmostEqual(speed_to_fly_airspeed(nominal, 0.4, 0.0), nominal, places=2)
        self.assertLessEqual(speed_to_fly_airspeed(nominal, 0.2, 0.8), nominal + 0.05)
        # Clear headwind: MacCready-style boost proposal.
        fast = speed_to_fly_airspeed(nominal, 1.7, 0.6)
        self.assertGreater(fast, nominal + 1.0)
        self.assertLessEqual(fast, 22.0)

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
        arrays = belief_map.field_arrays
        assert arrays is not None
        arrays["mode_prob_sink"][0, 0, 0] = 0.1
        arrays["mode_prob_neutral"][0, 0, 0] = 0.3
        arrays["mode_prob_uplift"][0, 0, 0] = 0.6
        arrays["belief_entropy"][0, 0, 0] = 1.2
        arrays["safety_penalty"][0, 0, 0] = 0.4

        snapshot = belief_snapshot(belief_map, 0)

        self.assertAlmostEqual(snapshot["uplift_prob"][0][0], 0.6)
        self.assertAlmostEqual(snapshot["sink_prob"][0][0], 0.1)
        self.assertAlmostEqual(snapshot["entropy"][0][0], 1.2)
        self.assertAlmostEqual(snapshot["safety"][0][0], 0.4)

    def test_corridor_guides_disabled_when_margin_none(self) -> None:
        width, height, levels = 24, 14, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    # Stronger eastward wind on the north side → +y corridor is cheaper.
                    u = 12.0 if y >= height // 2 + 2 else -10.0
                    arrays["wind_u"][z, y, x] = u
                    arrays["uncertainty"][z, y, x] = 0.02
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = u
                    cell.uncertainty = 0.02
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, height / 2.0, 0.0)
        goal = (width - 3, height // 2, 0)
        mission = Mission(
            start=(2, height // 2, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=None,
            elevation=elev,
        )
        guides_off = _energy_guide_paths(start, goal, belief_map, mission)
        self.assertTrue(any(label.startswith("guide_straight_agl") for label, _ in guides_off))
        self.assertFalse(any(label.startswith("guide_corridor_") for label, _ in guides_off))

        mission.corridor_energy_margin = 1.02
        guides_on = _energy_guide_paths(start, goal, belief_map, mission)
        self.assertTrue(
            any(label.startswith("guide_corridor_") for label, _ in guides_on),
            msg=f"expected a corridor candidate, got {[l for l,_ in guides_on]}",
        )
        # With strong north-side tailwind, a corridor should beat straight in planning.
        mission.preferred_cruise_agl = 1.0
        plan = plan_path_details(
            belief_map,
            mission,
            risk_weight=1.0,
            safety_weight=1.0,
            anytime_rounds=2,
            heuristic_weight_start=2.0,
            heuristic_weight_end=1.0,
            search_node_budget=4000,
            horizon_steps=6,
            beam_width=16,
            branch_width=8,
            discount_factor=0.93,
            terminal_progress_weight=20.0,
        )
        self.assertTrue(
            str(plan.get("planning_mode", "")).startswith("guide_corridor"),
            msg=f"expected corridor planning mode, got {plan.get('planning_mode')}",
        )

    def test_corridor_guides_blocked_in_weak_wind(self) -> None:
        """Calm / uniform fields must not invent lateral corridors (sichuan-class failure)."""
        width, height, levels = 24, 14, 3

        def _run(u_south: float, u_north: float) -> list[str]:
            belief_map = create_belief_map(width, height, levels)
            arrays = belief_map.field_arrays
            assert arrays is not None
            for z in range(levels):
                for y in range(height):
                    for x in range(width):
                        u = u_north if y >= height // 2 + 2 else u_south
                        arrays["wind_u"][z, y, x] = u
                        arrays["wind_v"][z, y, x] = 0.05
                        arrays["wind_w"][z, y, x] = 0.0
                        arrays["mode_prob_uplift"][z, y, x] = 0.2
                        arrays["uncertainty"][z, y, x] = 0.05
                        cell = belief_map.cells[z][y][x]
                        cell.wind_u = u
                        cell.wind_v = 0.05
                        cell.wind_w = 0.0
                        cell.mode_prob_uplift = 0.2
                        cell.uncertainty = 0.05
            elev = [[10.0 for _ in range(width)] for _ in range(height)]
            start = (2.0, height / 2.0, 0.0)
            goal = (width - 3, height // 2, 0)
            mission = Mission(
                start=(2, height // 2, 0),
                goal=goal,
                max_steps=40,
                step_distance_m=50.0,
                altitude_step_m=50.0,
                clearance_agl_level=1.0,
                max_altitude_level=2,
                cruise_band_step=0.5,
                corridor_energy_margin=1.02,
                elevation=elev,
            )
            return [label for label, _ in _energy_guide_paths(start, goal, belief_map, mission)]

        weak = _run(0.4, 0.5)
        self.assertTrue(any(l.startswith("guide_straight_agl") for l in weak))
        self.assertFalse(any(l.startswith("guide_corridor_") for l in weak), msg=f"weak={weak}")
        # Below HARD_FLOOR (1.20) on flat DEM: no corridor (no terrain relief).
        below_hard = _run(0.8, 1.0)
        self.assertFalse(any(l.startswith("guide_corridor_") for l in below_hard), msg=f"below_hard={below_hard}")
        # Between hard/soft floors without a real lateral edge → still no corridor.
        subfloor = _run(1.4, 1.5)
        self.assertFalse(any(l.startswith("guide_corridor_") for l in subfloor), msg=f"subfloor={subfloor}")
        # Between floors WITH clear lateral shear that also wins Joules → unlock.
        sheared = _run(1.4, 6.0)
        self.assertTrue(
            any(l.startswith("guide_corridor_") for l in sheared),
            msg=f"expected shear-unlocked corridor, got {sheared}",
        )
        # Fake uplift in calm horizontal wind must still not invent corridors.
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    arrays["wind_u"][z, y, x] = 0.5
                    arrays["wind_v"][z, y, x] = 0.1
                    arrays["wind_w"][z, y, x] = 0.35
                    arrays["mode_prob_uplift"][z, y, x] = 0.7
                    arrays["uncertainty"][z, y, x] = 0.05
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = 0.5
                    cell.wind_v = 0.1
                    cell.wind_w = 0.35
                    cell.mode_prob_uplift = 0.7
                    cell.uncertainty = 0.05
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, height / 2.0, 0.0)
        goal = (width - 3, height // 2, 0)
        mission = Mission(
            start=(2, height // 2, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        fake_uplift = [label for label, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertFalse(
            any(l.startswith("guide_corridor_") for l in fake_uplift),
            msg=f"fake_uplift={fake_uplift}",
        )
        # Uniform moderate wind still has no lateral edge → no corridor.
        uniform = _run(2.5, 2.5)
        self.assertFalse(any(l.startswith("guide_corridor_") for l in uniform), msg=f"uniform={uniform}")

    def test_corridor_terrain_relief_unlocks_below_hard_floor(self) -> None:
        """Calm air + clear DEM shortcut may corridor; flat calm still must not."""
        width, height, levels = 28, 16, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    arrays["wind_u"][z, y, x] = 0.7
                    arrays["wind_v"][z, y, x] = 0.1
                    arrays["wind_w"][z, y, x] = 0.0
                    arrays["uncertainty"][z, y, x] = 0.05
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = 0.7
                    cell.wind_v = 0.1
                    cell.uncertainty = 0.05
        # Hill on the straight mid-route; +y bypass is lower → north via cuts climb.
        elev = []
        mid_y = height * 0.45
        for y in range(height):
            row = []
            for x in range(width):
                hill = 120.0 * math.exp(
                    -((x - width * 0.5) / 3.2) ** 2 - ((y - mid_y) / 1.6) ** 2
                )
                row.append(25.0 + hill)
            elev.append(row)
        start = (2.0, mid_y, 0.0)
        goal = (width - 3, int(mid_y), 0)
        mission = Mission(
            start=(2, int(mid_y), 0),
            goal=goal,
            max_steps=40,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertTrue(
            any(l.startswith("guide_corridor_") for l in labels),
            msg=f"expected terrain-relief corridor below hard floor, got {labels}",
        )
        # Same calm wind on flat DEM must stay blocked.
        flat = [[10.0 for _ in range(width)] for _ in range(height)]
        mission.elevation = flat
        flat_labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertFalse(
            any(l.startswith("guide_corridor_") for l in flat_labels),
            msg=f"flat calm must not corridor, got {flat_labels}",
        )

    def test_corridor_light_vertical_unlocks_below_hard_floor(self) -> None:
        """Light horizontal wind + clear vertical-wind lobe may unlock a corridor."""
        width, height, levels = 28, 16, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    arrays["wind_u"][z, y, x] = 0.55
                    arrays["wind_v"][z, y, x] = 0.05
                    # Stronger uplift on the +y side — light-air vertical corridor.
                    arrays["wind_w"][z, y, x] = 0.70 if y >= height // 2 + 1 else 0.02
                    arrays["uncertainty"][z, y, x] = 0.05
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = 0.55
                    cell.wind_v = 0.05
                    cell.wind_w = float(arrays["wind_w"][z, y, x])
                    cell.uncertainty = 0.05
        elev = [[12.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, height / 2.0, 0.0)
        goal = (width - 3, height // 2, 0)
        mission = Mission(
            start=(2, height // 2, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=30.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertTrue(
            any(l.startswith("guide_corridor_") for l in labels),
            msg=f"expected light-vertical corridor, got {labels}",
        )
        # Uniform weak vertical wind must stay blocked.
        for z in range(levels):
            arrays["wind_w"][z, :, :] = 0.12
            for y in range(height):
                for x in range(width):
                    belief_map.cells[z][y][x].wind_w = 0.12
        flat_w = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertFalse(
            any(l.startswith("guide_corridor_") for l in flat_w),
            msg=f"uniform light w must not corridor, got {flat_w}",
        )

    def test_corridor_light_speed_unlocks_below_hard_floor(self) -> None:
        """Light ambient + clear horizontal |speed| lobe may unlock without uplift."""
        width, height, levels = 32, 18, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        mid = height // 2
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    # Mid-band light headwind; +y lobe is light tailwind with |speed| edge.
                    if y >= mid + 2:
                        u = 2.2
                    else:
                        u = -0.85
                    arrays["wind_u"][z, y, x] = u
                    arrays["wind_v"][z, y, x] = 0.05
                    arrays["wind_w"][z, y, x] = 0.0
                    arrays["uncertainty"][z, y, x] = 0.04
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = u
                    cell.wind_v = 0.05
                    cell.wind_w = 0.0
                    cell.uncertainty = 0.04
        elev = [[12.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, float(mid), 0.0)
        goal = (width - 3, mid, 0)
        mission = Mission(
            start=(2, mid, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=30.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertTrue(
            any(l.startswith("guide_corridor_") for l in labels),
            msg=f"expected light-speed corridor, got {labels}",
        )
        # Uniform weak horizontal wind must stay blocked.
        for z in range(levels):
            arrays["wind_u"][z, :, :] = 0.7
            for y in range(height):
                for x in range(width):
                    belief_map.cells[z][y][x].wind_u = 0.7
        flat = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertFalse(
            any(l.startswith("guide_corridor_") for l in flat),
            msg=f"uniform light speed must not corridor, got {flat}",
        )

    def test_corridor_dual_band_probe_when_sticky_diverges(self) -> None:
        """Sticky cling vs raw argmin (>0.15) generates corridors on both layers.

        Sticky at z=1 stays within the 1.2% cling bar while raw argmin is z=2
        (slightly cheaper via mild aloft uplift) — dual probe must emit both zs.
        """
        width, height, levels = 28, 16, 4
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        mid = height // 2
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    # North lobe strong tailwind; aloft uplift so z=2 barely beats z=1
                    # (within sticky 1.2% cling) — dual probe must emit both zs.
                    u = 8.0 if y >= mid + 2 else -6.0
                    w = 0.75 if z >= 2 else 0.05
                    arrays["wind_u"][z, y, x] = u
                    arrays["wind_v"][z, y, x] = 0.1
                    arrays["wind_w"][z, y, x] = w
                    arrays["uncertainty"][z, y, x] = 0.03
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = u
                    cell.wind_v = 0.1
                    cell.wind_w = w
                    cell.uncertainty = 0.03
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, float(mid), 0.0)
        goal = (width - 3, mid, 0)
        mission = Mission(
            start=(2, mid, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=40.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=3,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
            preferred_cruise_agl=1.0,
        )
        labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        corridor_labels = [l for l in labels if l.startswith("guide_corridor_") and "locked" not in l]
        self.assertTrue(corridor_labels, msg=f"expected fresh corridors, got {labels}")
        zs = {l.split("_z")[-1] for l in corridor_labels if "_z" in l}
        self.assertGreaterEqual(
            len(zs),
            2,
            msg=f"dual-band should probe sticky and argmin, zs={zs} labels={corridor_labels}",
        )

    def test_locked_via_drops_when_fresh_corridor_cheaper(self) -> None:
        """Locked via must lose to a clearly better fresh corridor (dynamic swap)."""
        width, height, levels = 28, 16, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        mid = height // 2
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    u = 10.0 if y >= mid + 2 else -8.0
                    arrays["wind_u"][z, y, x] = u
                    arrays["wind_v"][z, y, x] = 0.05
                    arrays["wind_w"][z, y, x] = 0.0
                    arrays["uncertainty"][z, y, x] = 0.02
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = u
                    cell.wind_v = 0.05
                    cell.wind_w = 0.0
                    cell.uncertainty = 0.02
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, float(mid), 0.0)
        goal = (width - 3, mid, 0)
        # Lock a via on the *wrong* (south / headwind) side.
        mission = Mission(
            start=(2, mid, 0),
            goal=goal,
            max_steps=40,
            step_distance_m=40.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
            preferred_cruise_agl=1.0,
            guide_via=(width * 0.5, float(mid) - 4.0),
        )
        guides = _energy_guide_paths(start, goal, belief_map, mission)
        labels = [l for l, _ in guides]
        self.assertIsNone(mission.guide_via, msg="locked via should clear when fresh wins")
        self.assertFalse(
            any(l == "guide_corridor_locked" for l in labels),
            msg=f"locked must not remain in guides, got {labels}",
        )
        self.assertTrue(
            any(l.startswith("guide_corridor_") for l in labels),
            msg=f"fresh corridor should still be offered, got {labels}",
        )

    def test_multi_via_corridors_on_alternating_shear(self) -> None:
        """Opposite lateral shear lobes unlock 2-via S-curve candidates (strong ambient)."""
        width, height, levels = 36, 20, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        mid = height // 2
        for z in range(levels):
            for y in range(height):
                for x in range(width):
                    # First third: prefer north; last third: prefer south (needs S-curve).
                    if x < width / 3:
                        u = 9.0 if y >= mid + 2 else -7.0
                    elif x > 2 * width / 3:
                        u = 9.0 if y <= mid - 2 else -7.0
                    else:
                        u = 2.5
                    arrays["wind_u"][z, y, x] = u
                    arrays["wind_v"][z, y, x] = 0.05
                    arrays["wind_w"][z, y, x] = 0.0
                    arrays["uncertainty"][z, y, x] = 0.02
                    cell = belief_map.cells[z][y][x]
                    cell.wind_u = u
                    cell.wind_v = 0.05
                    cell.wind_w = 0.0
                    cell.uncertainty = 0.02
        elev = [[8.0 for _ in range(width)] for _ in range(height)]
        start = (2.0, float(mid), 0.0)
        goal = (width - 3, mid, 0)
        mission = Mission(
            start=(2, mid, 0),
            goal=goal,
            max_steps=50,
            step_distance_m=40.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        labels = [l for l, _ in _energy_guide_paths(start, goal, belief_map, mission)]
        self.assertTrue(
            any("2via" in l for l in labels),
            msg=f"expected 2-via corridor candidates, got {labels}",
        )

    def test_multi_via_blocked_in_weak_wind(self) -> None:
        """Calm ambient must not invent 2-via corridors."""
        width, height, levels = 28, 16, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            arrays["wind_u"][z, :, :] = 0.4
            arrays["uncertainty"][z, :, :] = 0.2
            for y in range(height):
                for x in range(width):
                    belief_map.cells[z][y][x].wind_u = 0.4
                    belief_map.cells[z][y][x].uncertainty = 0.2
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        mid = height // 2
        mission = Mission(
            start=(2, mid, 0),
            goal=(width - 3, mid, 0),
            max_steps=40,
            step_distance_m=40.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
        )
        labels = [l for l, _ in _energy_guide_paths((2.0, float(mid), 0.0), (width - 3, mid, 0), belief_map, mission)]
        self.assertFalse(any("2via" in l for l in labels), msg=f"calm 2via leak: {labels}")

    def test_mild_sticky_skips_mpc_even_with_guide_via(self) -> None:
        """Mild sticky + via: guides own XY; forcing MPC regresses Shanxi-class corridors."""
        width, height, levels = 20, 12, 3
        belief_map = create_belief_map(width, height, levels)
        arrays = belief_map.field_arrays
        assert arrays is not None
        for z in range(levels):
            arrays["wind_u"][z, :, :] = 3.0
            arrays["uncertainty"][z, :, :] = 0.05
            for y in range(height):
                for x in range(width):
                    belief_map.cells[z][y][x].wind_u = 3.0
                    belief_map.cells[z][y][x].uncertainty = 0.05
        elev = [[8.0 for _ in range(width)] for _ in range(height)]
        mission = Mission(
            start=(2, height // 2, 0),
            goal=(width - 3, height // 2, 0),
            max_steps=30,
            step_distance_m=40.0,
            altitude_step_m=50.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            cruise_band_step=0.5,
            corridor_energy_margin=1.02,
            elevation=elev,
            preferred_cruise_agl=1.0,
            guide_via=(width * 0.5, height * 0.5 + 2.0),
        )
        plan = plan_path_details(
            belief_map,
            mission,
            risk_weight=1.0,
            safety_weight=1.0,
            anytime_rounds=1,
            heuristic_weight_start=2.0,
            heuristic_weight_end=1.0,
            search_node_budget=800,
            horizon_steps=4,
            beam_width=8,
            branch_width=4,
            discount_factor=0.93,
            terminal_progress_weight=20.0,
        )
        notes = [c.get("note") for c in plan.get("candidates", []) if isinstance(c, dict)]
        self.assertIn(
            "sticky_cruise_skip_mpc",
            notes,
            msg=f"mild sticky should seed greedy (not MPC), candidates={plan.get('candidates')}",
        )

    def test_cruise_band_requires_evidence_to_leave_clearance(self) -> None:
        # Sub-5% "wins" at higher bands must not leave clearance (belief noise aloft).
        scores = {1.0: 1000.0, 1.35: 995.0, 2.0: 960.0}
        z = _select_preferred_cruise_band(scores, sticky=None, clearance=1.0, remaining_horiz=40.0, band_step=0.05)
        self.assertAlmostEqual(z, 1.0, places=5)
        # Clear ≥5% win vs clearance does climb.
        scores2 = {1.0: 1000.0, 1.5: 980.0, 2.5: 820.0}
        z2 = _select_preferred_cruise_band(scores2, sticky=None, clearance=1.0, remaining_horiz=40.0, band_step=0.05)
        self.assertAlmostEqual(z2, 2.5, places=5)
        # Sticky high layer with strong savings holds against weak lower argmin.
        scores3 = {1.0: 1000.0, 2.5: 820.0, 2.7: 818.0}
        z3 = _select_preferred_cruise_band(scores3, sticky=2.5, clearance=1.0, remaining_horiz=40.0, band_step=0.05)
        self.assertAlmostEqual(z3, 2.5, places=5)
        # Mistaken sticky ≥2% worse than clearance → corrective step down.
        scores4 = {1.0: 1000.0, 1.6: 1040.0}
        z4 = _select_preferred_cruise_band(scores4, sticky=1.6, clearance=1.0, remaining_horiz=40.0, band_step=0.05)
        self.assertLess(z4, 1.6)
        # Unearned sticky (better than clearance by only ~3%) still eases down.
        scores5 = {1.0: 1000.0, 2.2: 970.0}
        z5 = _select_preferred_cruise_band(
            scores5, sticky=2.2, clearance=1.0, remaining_horiz=40.0, band_step=0.05, climb_earned=False
        )
        self.assertLess(z5, 2.2)
        # Once earned, the same sticky may hold.
        z5b = _select_preferred_cruise_band(
            scores5, sticky=2.2, clearance=1.0, remaining_horiz=40.0, band_step=0.05, climb_earned=True
        )
        self.assertAlmostEqual(z5b, 2.2, places=5)
        # First step above clearance on rising DEM: 3% enough (cold start).
        scores6 = {1.0: 1000.0, 1.7: 960.0}
        z6 = _select_preferred_cruise_band(
            scores6,
            sticky=None,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            terrain_rise_m=80.0,
            altitude_step_m=50.0,
        )
        self.assertAlmostEqual(z6, 1.7, places=5)
        # Further climb above clearance+1 still needs the full 5% even with terrain rise.
        scores6b = {1.0: 1000.0, 1.5: 970.0, 2.5: 960.0}
        z6b = _select_preferred_cruise_band(
            scores6b,
            sticky=1.5,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            terrain_rise_m=80.0,
            altitude_step_m=50.0,
            climb_earned=True,
        )
        self.assertAlmostEqual(z6b, 1.5, places=5)
        # High band with ≥15% vs clearance unlocks soft ceiling; strong air may jump to argmin.
        scores7 = {1.0: 1000.0, 1.35: 990.0, 2.5: 840.0}
        z7 = _select_preferred_cruise_band(
            scores7,
            sticky=1.35,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.0,
        )
        self.assertGreaterEqual(z7, 2.4)
        self.assertLessEqual(z7, 2.5 + 1e-9)
        # Same 16% win in marginal air: not enough (≥20% required) → stay ≤ceiling.
        z7m = _select_preferred_cruise_band(
            scores7,
            sticky=1.35,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=1.5,
        )
        self.assertLessEqual(z7m, 1.35 + 1e-9)
        # 12% high-band win is not enough to unlock the soft ceiling.
        scores7w = {1.0: 1000.0, 1.35: 990.0, 2.5: 880.0}
        z7w = _select_preferred_cruise_band(
            scores7w, sticky=1.35, clearance=1.0, remaining_horiz=40.0, band_step=0.05, climb_earned=True
        )
        self.assertLessEqual(z7w, 1.35 + 1e-9)
        # Calm + flat: a 6% "win" at a mild higher band must not leave clearance.
        scores_calm = {1.0: 1000.0, 1.2: 940.0}
        z_calm = _select_preferred_cruise_band(
            scores_calm,
            sticky=None,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            ambient_wind_mps=0.6,
        )
        self.assertAlmostEqual(z_calm, 1.0, places=5)
        # Calm + rising DEM: ≥3% high-band win may unlock (aloft uplift, Hainan-class).
        scores_calm_relief = {1.0: 1000.0, 1.5: 990.0, 3.0: 965.0}
        z_calm_relief = _select_preferred_cruise_band(
            scores_calm_relief,
            sticky=None,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            ambient_wind_mps=0.6,
            terrain_rise_m=200.0,
            altitude_step_m=50.0,
        )
        self.assertGreaterEqual(z_calm_relief, 2.5)
        # Micro-layer on rising DEM: 1.5% enough; sub-1% still blocked.
        scores_micro_lo = {1.0: 1000.0, 1.033: 985.0}
        z_micro_lo = _select_preferred_cruise_band(
            scores_micro_lo,
            sticky=None,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            terrain_rise_m=100.0,
            altitude_step_m=50.0,
            ambient_wind_mps=2.2,
        )
        self.assertAlmostEqual(z_micro_lo, 1.033, places=5)
        scores_micro_noise = {1.0: 1000.0, 1.033: 992.0}
        z_micro_noise = _select_preferred_cruise_band(
            scores_micro_noise,
            sticky=None,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            terrain_rise_m=100.0,
            altitude_step_m=50.0,
            ambient_wind_mps=2.2,
        )
        self.assertAlmostEqual(z_micro_noise, 1.0, places=5)
        # Moderate-band catch-up (≤clearance+1.2) may jump farther in one step.
        scores7b = {1.0: 1000.0, 1.2: 990.0, 2.0: 850.0}
        z7b = _select_preferred_cruise_band(
            scores7b, sticky=1.2, clearance=1.0, remaining_horiz=40.0, band_step=0.05, climb_earned=True
        )
        self.assertGreaterEqual(z7b, 2.0 - 1e-9)
        # Strong air + earned high sticky: ~4% lower-band "win" must NOT dump (need ≥8%).
        scores_hold = {1.0: 1000.0, 1.3: 860.0, 2.57: 900.0}
        z_hold = _select_preferred_cruise_band(
            scores_hold,
            sticky=2.57,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.5,
            terrain_rise_m=80.0,
            altitude_step_m=50.0,
        )
        self.assertAlmostEqual(z_hold, 2.57, places=5)
        # ≥8% sticky-relative dump still allowed in strong air.
        scores_dump = {1.0: 1000.0, 1.3: 820.0, 2.57: 900.0}
        z_dump = _select_preferred_cruise_band(
            scores_dump,
            sticky=2.57,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.5,
            terrain_rise_m=80.0,
            altitude_step_m=50.0,
        )
        self.assertLess(z_dump, 2.57)
        # Weak air keeps the old ~2% dump bar (protect against bad high sticky).
        scores_weak_dump = {1.0: 1000.0, 1.3: 870.0, 2.0: 900.0}
        z_weak_dump = _select_preferred_cruise_band(
            scores_weak_dump,
            sticky=2.0,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=1.5,
        )
        self.assertLess(z_weak_dump, 2.0)
        # Approach + earned high sticky: hold cruise band (baseline polyline descends).
        scores_appr = {1.0: 1000.0, 2.567: 820.0, 2.6: 822.0}
        z_appr = _select_preferred_cruise_band(
            scores_appr,
            sticky=2.567,
            clearance=1.0,
            remaining_horiz=8.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.5,
        )
        self.assertAlmostEqual(z_appr, 2.567, places=5)
        # Earned+strong: clearly cheaper thin upper band → one micro hop (capped).
        scores_micro = {1.0: 1000.0, 2.567: 820.0, 2.667: 815.0}
        z_micro = _select_preferred_cruise_band(
            scores_micro,
            sticky=2.567,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.5,
        )
        self.assertGreater(z_micro, 2.567 + 1e-9)
        self.assertLessEqual(z_micro, 2.67 + 1e-9)
        # Weak/marginal air: same scores must not micro-climb upward.
        z_micro_w = _select_preferred_cruise_band(
            scores_micro,
            sticky=2.567,
            clearance=1.0,
            remaining_horiz=40.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=1.5,
        )
        self.assertLessEqual(z_micro_w, 2.567 + 1e-9)
        # Approach + low sticky near clearance: still ease down toward clearance.
        scores_appr_lo = {1.0: 1000.0, 1.35: 990.0}
        z_appr_lo = _select_preferred_cruise_band(
            scores_appr_lo,
            sticky=1.35,
            clearance=1.0,
            remaining_horiz=8.0,
            band_step=0.05,
            climb_earned=True,
            ambient_wind_mps=3.5,
        )
        self.assertLess(z_appr_lo, 1.35)

    def test_downscale_uses_100m_profile_aloft(self) -> None:
        elev = [[100.0, 110.0], [105.0, 115.0]]
        terrain = TerrainField(
            elevation=elev,
            slope=[[0.1, 0.1], [0.1, 0.1]],
            aspect=[[0.0, 0.0], [0.0, 0.0]],
            roughness=[[0.05, 0.05], [0.05, 0.05]],
        )
        wind_no, _ = downscale_wind(2.0, 0.0, terrain, 0.0, altitude_levels=5, altitude_step_m=30.0)
        wind_yes, _ = downscale_wind(
            2.0,
            0.0,
            terrain,
            0.0,
            altitude_levels=5,
            altitude_step_m=30.0,
            u_100=6.0,
            v_100=0.0,
        )
        import numpy as np

        u0 = float(np.asarray(wind_yes.u)[0].mean())
        u_hi = float(np.asarray(wind_yes.u)[-1].mean())
        u_hi_no = float(np.asarray(wind_no.u)[-1].mean())
        self.assertGreater(u_hi, u0)
        self.assertGreater(abs(u_hi), abs(u_hi_no))

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

    def test_lateral_truth_probes_update_off_track_belief(self) -> None:
        """Truth probes sample ±path-normal and fuse into belief off the ray."""
        width, height, levels = 40, 24, 3
        belief_map = create_belief_map(width, height, levels)
        elev = [[12.0 for _ in range(width)] for _ in range(height)]
        mission = Mission(
            start=(5, height // 2, 1),
            goal=(width - 5, height // 2, 1),
            max_steps=40,
            step_distance_m=30.0,
            altitude_step_m=30.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            elevation=elev,
        )
        mid_y = height // 2
        state = DroneState(
            x=20.0,
            y=float(mid_y),
            z=1.0,
            heading_rad=0.0,
            battery_ratio=1.0,
            airspeed=16.5,
        )
        context = NavigationContext(
            mission=mission,
            terrain=TerrainField(
                elevation=elev,
                slope=[[0.0] * width for _ in range(height)],
                aspect=[[0.0] * width for _ in range(height)],
                roughness=[[0.0] * width for _ in range(height)],
            ),
            state=state,
            battery_j=1e6,
            belief_map=belief_map,
            predict_session=None,  # type: ignore[arg-type]
            step=2,  # replan tick (default interval=2)
        )
        # Uniform forecast; sheared truth (north stronger). On-track ambient must
        # clear the hard floor so probes are not skipped (Beijing-class gate).
        flat = [[[2.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        zero = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        truth_u = []
        for _z in range(levels):
            layer = []
            for y in range(height):
                row = [6.0 if y > mid_y + 2 else 2.0 for _ in range(width)]
                layer.append(row)
            truth_u.append(layer)
        prediction = {"u_layers": flat, "v_layers": zero, "w_layers": zero}
        truth_field = {"u": truth_u, "v": zero, "w": zero}
        engine = object.__new__(NavigationEngine)
        engine.updater = BeliefUpdater(observation_radius=2)
        engine.config = TaskConfig()
        arrays = belief_map.field_arrays
        assert arrays is not None
        on_track_u_before = float(arrays["wind_u"][1, mid_y, 20])
        payload = engine._apply_truth_probes(
            context,
            prediction,
            truth_field,
            timestamp="2026-01-01T00:00:00Z",
        )
        self.assertIsNotNone(payload)
        assert payload is not None
        tags = {p["tag"] for p in payload}
        self.assertIn("left", tags)
        self.assertIn("right", tags)
        offset_cells = LATERAL_PROBE_OFFSET_M / 30.0
        left_y = int(round(mid_y + offset_cells))  # path east → left is +y
        right_y = int(round(mid_y - offset_cells))
        self.assertGreater(float(arrays["confidence"][1, left_y, 20]), 0.0)
        self.assertGreater(float(arrays["confidence"][1, right_y, 20]), 0.0)
        self.assertGreater(
            float(arrays["wind_u"][1, left_y, 20]),
            float(arrays["wind_u"][1, right_y, 20]),
        )
        # Cruise-band isolation: on-track cell must not move (radius=0, no advect).
        self.assertAlmostEqual(float(arrays["wind_u"][1, mid_y, 20]), on_track_u_before, places=6)

        # Hard-floor calm: no probe paint.
        calm_u = [[[0.5 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        calm_payload = engine._apply_truth_probes(
            context,
            prediction,
            {"u": calm_u, "v": zero, "w": zero},
            timestamp="2026-01-01T00:00:00Z",
        )
        self.assertIsNone(calm_payload)

        # Strong but uniform wind (Beijing plain): shear gate skips paint.
        flat_strong = [[[4.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        uniform_payload = engine._apply_truth_probes(
            context,
            {"u_layers": flat_strong, "v_layers": zero, "w_layers": zero},
            {"u": flat_strong, "v": zero, "w": zero},
            timestamp="2026-01-01T00:00:00Z",
        )
        self.assertIsNone(uniform_payload)

    def test_distill_truth_1via_picks_shear_lobe(self) -> None:
        """Dual-agree distill: via only when truth *and* belief beat straight."""
        from windfarm.planner import distill_truth_1via

        width, height, levels = 40, 24, 3
        belief_map = create_belief_map(width, height, levels)
        elev = [[12.0 for _ in range(width)] for _ in range(height)]
        mission = Mission(
            start=(5, height // 2, 1),
            goal=(width - 5, height // 2, 1),
            max_steps=40,
            step_distance_m=30.0,
            altitude_step_m=30.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            elevation=elev,
            corridor_energy_margin=1.02,
        )
        mid_y = height // 2
        truth_u = []
        for _z in range(levels):
            layer = []
            for y in range(height):
                layer.append([8.0 if y > mid_y + 1 else 1.5 for _ in range(width)])
            truth_u.append(layer)
        zero = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        # Belief matches truth shear → dual-agree accepts.
        belief_map.field_arrays["wind_u"][:] = np.asarray(truth_u, dtype=np.float64)
        via = distill_truth_1via(
            (5.0, float(mid_y), 1.0),
            (width - 5, mid_y, 1),
            belief_map,
            mission,
            {"u": truth_u, "v": zero, "w": zero},
            cruise_z=1.0,
            min_truth_save=0.002,
            min_belief_save=0.002,
        )
        self.assertIsNotNone(via)
        assert via is not None
        self.assertGreater(via[1], float(mid_y))
        # Belief still flat while truth has shear → reject (prevents Shanxi false win).
        flat_belief = create_belief_map(width, height, levels)
        none_via = distill_truth_1via(
            (5.0, float(mid_y), 1.0),
            (width - 5, mid_y, 1),
            flat_belief,
            mission,
            {"u": truth_u, "v": zero, "w": zero},
            cruise_z=1.0,
            min_truth_save=0.002,
            min_belief_save=0.002,
        )
        self.assertIsNone(none_via)

    def test_corridor_truth_gate_rejects_belief_only_win(self) -> None:
        """Truth gate blocks corridors that belief likes but truth does not."""
        from windfarm.planner import (
            CORRIDOR_TRUTH_WIN_NEED,
            _belief_map_from_truth_field,
            _corridor_truth_beats_straight,
            _agl_guide_polyline,
        )

        width, height, levels = 40, 24, 3
        belief_map = create_belief_map(width, height, levels)
        elev = [[10.0 for _ in range(width)] for _ in range(height)]
        mission = Mission(
            start=(5, height // 2, 1),
            goal=(width - 5, height // 2, 1),
            max_steps=40,
            step_distance_m=30.0,
            altitude_step_m=30.0,
            clearance_agl_level=1.0,
            max_altitude_level=2,
            elevation=elev,
        )
        mid_y = height // 2
        # Belief invents a north lobe; truth is uniform.
        for y in range(height):
            for x in range(width):
                belief_map.field_arrays["wind_u"][1, y, x] = 8.0 if y > mid_y else 1.0
        truth_u = [[[2.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        zero = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
        start = (5.0, float(mid_y), 1.0)
        goal = (width - 5, mid_y, 1)
        straight = _agl_guide_polyline(start, goal, belief_map, mission, cruise_z=1.0)
        via_path = _agl_guide_polyline(
            start, goal, belief_map, mission, via_xy=(20.0, float(mid_y + 4)), cruise_z=1.0
        )
        truth_belief = _belief_map_from_truth_field(
            belief_map, {"u": truth_u, "v": zero, "w": zero}
        )
        self.assertFalse(
            _corridor_truth_beats_straight(
                via_path, straight, truth_belief, mission, win_need=CORRIDOR_TRUTH_WIN_NEED
            )
        )
        # No truth → gate open (online / no-oracle mode).
        self.assertTrue(_corridor_truth_beats_straight(via_path, straight, None, mission))



if __name__ == "__main__":
    unittest.main()
