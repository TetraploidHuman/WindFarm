from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from windfarm.altitude import (
    cruise_agl_band_grid,
    cruise_agl_floor,
    effective_dz_levels,
    format_agl_band_token,
    msl_height_m,
    next_polyline_waypoint,
    sample_elevation,
    snap_cruise_agl,
    straight_agl_guide_polyline,
    terrain_climb_along_line_m,
    terrain_delta_m,
)
from windfarm.controller import transition_energy_j


class AltitudeModelTest(unittest.TestCase):
    def test_sample_and_msl(self) -> None:
        elev = [[100.0, 200.0], [300.0, 400.0]]
        self.assertAlmostEqual(sample_elevation(elev, 0.0, 0.0), 100.0)
        self.assertAlmostEqual(sample_elevation(elev, 1.0, 0.0), 200.0)
        self.assertAlmostEqual(msl_height_m(100.0, 1.0, 40.0), 140.0)

    def test_constant_agl_pays_terrain_climb(self) -> None:
        elev = [[0.0, 0.0, 80.0]]
        # Move at constant AGL z=1 across an 80 m rise.
        flat = transition_energy_j(
            airspeed=16.0,
            current=(0.0, 0.0, 1.0),
            nxt=(1.0, 0.0, 1.0),
            local_u=0.0,
            local_v=0.0,
            local_w=0.0,
            step_distance_m=100.0,
            altitude_step_m=40.0,
            climb_cost_per_level_j=180.0,
            terrain_dz_m=0.0,
        )
        hill = transition_energy_j(
            airspeed=16.0,
            current=(0.0, 0.0, 1.0),
            nxt=(2.0, 0.0, 1.0),
            local_u=0.0,
            local_v=0.0,
            local_w=0.0,
            step_distance_m=100.0,
            altitude_step_m=40.0,
            climb_cost_per_level_j=180.0,
            terrain_dz_m=terrain_delta_m(elev, 0.0, 0.0, 2.0, 0.0),
        )
        self.assertGreater(hill, flat)
        self.assertAlmostEqual(effective_dz_levels(0.0, 80.0, 40.0), 2.0)
        self.assertGreater(terrain_climb_along_line_m(elev, 0.0, 0.0, 2.0, 0.0), 70.0)

    def test_cruise_floor(self) -> None:
        self.assertEqual(cruise_agl_floor(20.0, 0.0, 1.0), 1.0)
        self.assertEqual(cruise_agl_floor(3.0, 0.0, 1.0), 0.0)

    def test_fine_cruise_band_grid(self) -> None:
        bands = cruise_agl_band_grid(1.0, 0.0, 4.0, step=0.025, span_levels=2.0)
        self.assertEqual(len(bands), 81)
        self.assertEqual(bands[0], 1.0)
        self.assertEqual(bands[-1], 3.0)
        self.assertIn(1.7, bands)
        self.assertIn(1.75, bands)
        self.assertEqual(format_agl_band_token(1.725), "1.725")
        self.assertAlmostEqual(
            snap_cruise_agl(1.73, clearance=1.0, min_level=0.0, max_level=4.0, step=0.025),
            1.725,
        )
        sticky = cruise_agl_band_grid(1.0, 0.0, 4.0, step=0.025, span_levels=2.0, sticky=2.12)
        self.assertIn(2.125, sticky)

    def test_straight_agl_guide_polyline_packing(self) -> None:
        pts = straight_agl_guide_polyline((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), agl_cruise=1.0)
        self.assertEqual(pts[0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pts[1][0], 1.2)  # 12% climb bite
        self.assertAlmostEqual(pts[1][2], 1.0)
        self.assertAlmostEqual(pts[-1][2], 0.0)
        nxt = next_polyline_waypoint(pts, 0.0, 0.0)
        self.assertIsNotNone(nxt)
        self.assertAlmostEqual(nxt[0], 1.2)


if __name__ == "__main__":
    unittest.main()
