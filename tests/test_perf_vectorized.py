from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from windfarm.altitude import terrain_delta_m, terrain_delta_m_batch
from windfarm.belief import BeliefUpdater, create_belief_map
from windfarm.controller import transition_energy_batch, transition_energy_j
from windfarm.mathutils import trilinear_sample, trilinear_sample_batch
from windfarm.physics import downscale_wind
from windfarm.types import TerrainField, WindField


class VectorizedPhysicsTest(unittest.TestCase):
    def _tiny_terrain(self) -> TerrainField:
        elev = [[100.0, 120.0, 140.0], [110.0, 130.0, 150.0], [105.0, 125.0, 145.0]]
        slope = [[0.1, 0.2, 0.1], [0.15, 0.25, 0.15], [0.1, 0.2, 0.1]]
        aspect = [[0.0, 0.5, 1.0], [0.2, 0.7, 1.2], [0.1, 0.6, 1.1]]
        rough = [[0.2, 0.3, 0.2], [0.25, 0.4, 0.25], [0.2, 0.3, 0.2]]
        return TerrainField(elevation=elev, slope=slope, aspect=aspect, roughness=rough)

    def test_downscale_wind_shapes_and_finite(self) -> None:
        terrain = self._tiny_terrain()
        wind, diag = downscale_wind(4.0, -2.0, terrain, w_km=0.5, altitude_levels=5)
        u = np.asarray(wind.u)
        self.assertEqual(u.shape, (5, 3, 3))
        self.assertTrue(np.isfinite(u).all())
        self.assertTrue(np.isfinite(np.asarray(diag.f_elev)).all())

    def test_belief_prediction_uses_field_arrays(self) -> None:
        belief = create_belief_map(4, 3, levels=2)
        self.assertIsNotNone(belief.field_arrays)
        wind = WindField(
            u=np.ones((2, 3, 4)),
            v=np.zeros((2, 3, 4)),
            w=np.full((2, 3, 4), 0.2),
        )
        BeliefUpdater().apply_prediction(belief, wind, step=3)
        arrays = belief.field_arrays
        assert arrays is not None
        self.assertAlmostEqual(float(arrays["wind_u"][0, 1, 2]), 1.0)
        self.assertTrue(np.isfinite(arrays["expected_energy_gain"]).all())

    def test_belief_prediction_fuses_instead_of_overwrite(self) -> None:
        belief = create_belief_map(3, 2, levels=1)
        arrays = belief.field_arrays
        assert arrays is not None
        arrays["wind_u"][0, 0, 1] = 8.0
        arrays["wind_v"][0, 0, 1] = 0.0
        arrays["wind_w"][0, 0, 1] = 0.0
        arrays["last_update"][0, 0, 1] = 5
        arrays["wind_var_u"][0, 0, 1] = 0.05
        BeliefUpdater().apply_prediction(
            belief,
            WindField(
                u=np.zeros((1, 2, 3)),
                v=np.zeros((1, 2, 3)),
                w=np.zeros((1, 2, 3)),
            ),
            step=6,
        )
        fused = float(belief.field_arrays["wind_u"][0, 0, 1])
        # Recent observation must not be wiped to the zero prediction.
        self.assertGreater(fused, 4.0)

    def test_belief_prediction_preserves_observed_vertical_shear(self) -> None:
        """Observed uplift lobe vs calm forecast must retain a usable w edge."""
        belief = create_belief_map(4, 3, levels=1)
        arrays = belief.field_arrays
        assert arrays is not None
        arrays["wind_u"][...] = 0.6
        arrays["wind_v"][...] = 0.0
        arrays["wind_w"][...] = 0.05
        arrays["wind_w"][0, 1, 2] = 0.75
        arrays["last_update"][0, 1, 2] = 4
        arrays["wind_var_w"][0, 1, 2] = 0.08
        BeliefUpdater().apply_prediction(
            belief,
            WindField(
                u=np.full((1, 3, 4), 0.55),
                v=np.zeros((1, 3, 4)),
                w=np.full((1, 3, 4), 0.08),
            ),
            step=5,
        )
        kept = float(belief.field_arrays["wind_w"][0, 1, 2])
        calm = float(belief.field_arrays["wind_w"][0, 1, 0])
        self.assertGreater(kept, 0.40)
        self.assertGreater(kept - calm, 0.25)

    def test_trilinear_batch_matches_scalar(self) -> None:
        rng = np.random.default_rng(0)
        field = rng.normal(size=(4, 5, 6))
        xs = rng.uniform(0, 5, size=20)
        ys = rng.uniform(0, 4, size=20)
        zs = rng.uniform(0, 3, size=20)
        batch = trilinear_sample_batch(field, xs, ys, zs)
        layers = field.tolist()
        for i in range(20):
            self.assertAlmostEqual(batch[i], trilinear_sample(layers, xs[i], ys[i], zs[i]), places=9)

    def test_transition_energy_batch_matches_scalar(self) -> None:
        currents = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 1.0, 1.5]])
        nexts = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 1.5], [3.0, 1.0, 1.0]])
        u = np.array([1.0, -0.5, 0.2])
        v = np.array([0.0, 0.5, -0.3])
        w = np.array([0.1, 0.0, -0.2])
        terrain = np.array([0.0, 10.0, -5.0])
        batch = transition_energy_batch(
            airspeed=16.0,
            currents=currents,
            nexts=nexts,
            local_u=u,
            local_v=v,
            local_w=w,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            terrain_dz_m=terrain,
        )
        for i in range(3):
            scalar = transition_energy_j(
                airspeed=16.0,
                current=tuple(currents[i]),
                nxt=tuple(nexts[i]),
                local_u=float(u[i]),
                local_v=float(v[i]),
                local_w=float(w[i]),
                step_distance_m=50.0,
                altitude_step_m=50.0,
                climb_cost_per_level_j=225.0,
                terrain_dz_m=float(terrain[i]),
            )
            self.assertAlmostEqual(float(batch[i]), scalar, places=9)

    def test_terrain_climb_batch_matches_scalar(self) -> None:
        elev = [[0.0, 10.0, 30.0], [5.0, 20.0, 50.0], [8.0, 25.0, 60.0]]
        from windfarm.altitude import terrain_climb_along_line_batch, terrain_climb_along_line_m

        xs0 = np.array([0.0, 0.0, 1.0])
        ys0 = np.array([0.0, 1.0, 0.0])
        xs1 = np.array([2.0, 2.0, 2.0])
        ys1 = np.array([0.0, 1.0, 2.0])
        batch = terrain_climb_along_line_batch(elev, xs0, ys0, xs1, ys1, samples=4)
        for i in range(3):
            self.assertAlmostEqual(
                float(batch[i]),
                terrain_climb_along_line_m(elev, float(xs0[i]), float(ys0[i]), float(xs1[i]), float(ys1[i]), samples=4),
                places=9,
            )

    def test_polyline_batch_matches_scalar(self) -> None:
        from windfarm.planner import _polyline_model_energy_j, _polylines_model_energy_j
        from windfarm.types import Mission

        belief = create_belief_map(8, 6, levels=3)
        BeliefUpdater().apply_prediction(
            belief,
            WindField(
                u=np.ones((3, 6, 8)),
                v=np.zeros((3, 6, 8)),
                w=np.full((3, 6, 8), 0.15),
            ),
            step=1,
        )
        mission = Mission(
            start=(0, 0, 0),
            goal=(7, 5, 0),
            max_steps=20,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            elevation=[[float(y * 10 + x) for x in range(8)] for y in range(6)],
        )
        paths = [
            [(0.0, 0.0, 1.0), (3.0, 2.0, 1.0), (7.0, 5.0, 0.0)],
            [(0.0, 0.0, 1.2), (4.0, 1.0, 1.2), (7.0, 5.0, 0.0)],
            [(0.0, 0.0, 1.0)],
        ]
        batch = _polylines_model_energy_j(paths, belief, mission)
        for i, path in enumerate(paths):
            self.assertAlmostEqual(batch[i], _polyline_model_energy_j(path, belief, mission), places=9)

    def test_completed_plans_batch_matches_scalar(self) -> None:
        from windfarm.planner import _completed_plan_energy_j, _completed_plans_energy_j
        from windfarm.types import Mission

        belief = create_belief_map(8, 6, levels=3)
        BeliefUpdater().apply_prediction(
            belief,
            WindField(
                u=np.ones((3, 6, 8)),
                v=np.zeros((3, 6, 8)),
                w=np.full((3, 6, 8), 0.1),
            ),
            step=1,
        )
        mission = Mission(
            start=(0, 0, 0),
            goal=(7, 5, 0),
            max_steps=20,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            elevation=[[float(y * 5 + x) for x in range(8)] for y in range(6)],
        )
        goal = (7, 5, 0)
        paths = [
            [(0.0, 0.0, 1.0), (3.0, 2.0, 1.0), (7.0, 5.0, 0.0)],
            [(0.0, 0.0, 1.0), (4.0, 2.0, 1.0)],  # unfinished → completion tail
        ]
        batch = _completed_plans_energy_j(paths, goal, belief, mission)
        for i, path in enumerate(paths):
            self.assertAlmostEqual(batch[i], _completed_plan_energy_j(path, goal, belief, mission), places=9)

    def test_return_cost_coarse_and_budget_gate(self) -> None:
        from windfarm.planner import (
            ReturnCostLookup,
            _return_budget_is_tight,
            compute_return_cost_map,
            plan_path_details,
        )
        from windfarm.types import Mission

        belief = create_belief_map(48, 36, levels=5)
        BeliefUpdater().apply_prediction(
            belief,
            WindField(
                u=np.ones((5, 36, 48)),
                v=np.zeros((5, 36, 48)),
                w=np.full((5, 36, 48), 0.1),
            ),
            step=1,
        )
        elev = [[float(y + x) for x in range(48)] for y in range(36)]
        # Ample budget → no Dijkstra.
        mission_ample = Mission(
            start=(7, 25, 0),
            goal=(39, 13, 0),
            max_steps=80,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            home=(7, 25, 0),
            max_return_cost_j=500_000.0,
            elevation=elev,
            cruise_band_step=0.02,
        )
        self.assertFalse(
            _return_budget_is_tight(mission_ample, (7.0, 25.0, 0.0), (39, 13, 0))
        )
        plan = plan_path_details(belief, mission_ample, anytime_rounds=2, search_node_budget=3000)
        self.assertIsNone(plan["return_cost_map"])
        # Coarse map when forced.
        mission_tight = Mission(
            start=(7, 25, 0),
            goal=(39, 13, 0),
            max_steps=80,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            home=(7, 25, 0),
            max_return_cost_j=5_000.0,
            elevation=elev,
        )
        lookup = compute_return_cost_map(belief, mission_tight, xy_stride=2)
        self.assertIsInstance(lookup, ReturnCostLookup)
        self.assertEqual(lookup.stride, 2)
        self.assertGreater(len(lookup.costs), 100)
        self.assertLess(len(lookup.costs), 5000)
        self.assertTrue(math.isfinite(lookup.get((8, 26, 1))))

    def test_observation_update_vectorized(self) -> None:
        from windfarm.types import Observation

        belief = create_belief_map(6, 5, levels=2)
        wind = WindField(
            u=np.ones((2, 5, 6)),
            v=np.zeros((2, 5, 6)),
            w=np.full((2, 5, 6), 0.1),
        )
        updater = BeliefUpdater(observation_radius=2)
        updater.apply_prediction(belief, wind, step=1)
        before = float(belief.field_arrays["wind_u"][0, 2, 3])
        obs = Observation(
            timestamp="2024-01-01T08:00:00",
            x=3,
            y=2,
            z=0,
            u_obs=2.5,
            v_obs=0.5,
            w_obs=0.2,
            airspeed=14.0,
            ground_speed=14.5,
            climb_rate=0.1,
            acceleration=0.0,
        )
        updater.update_with_observation(belief, obs, predicted_u=1.0, predicted_v=0.0, predicted_w=0.1, step=2)
        after = float(belief.field_arrays["wind_u"][0, 2, 3])
        self.assertNotAlmostEqual(before, after)
        self.assertTrue(np.isfinite(belief.field_arrays["uncertainty"]).all())


if __name__ == "__main__":
    unittest.main()
