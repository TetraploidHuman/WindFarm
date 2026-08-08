from __future__ import annotations

import unittest
from pathlib import Path

from windfarm.data_ingest import (
    DEFAULT_SCENARIOS,
    crop_srtm_to_grid,
    landcover_from_terrain,
    mission_endpoints,
    resample_coarse_wind,
    terrain_from_srtm,
)


ROOT = Path(__file__).resolve().parents[1]
HGT = ROOT / "data" / "N25E118.hgt"


@unittest.skipUnless(HGT.exists(), "SRTM tile data/N25E118.hgt not present")
class DataIngestTests(unittest.TestCase):
    def test_srtm_crop_has_orographic_relief(self) -> None:
        elev = crop_srtm_to_grid(HGT, 25.75, 118.60, width=24, height=18, resolution_m=100.0, tile_lat=25, tile_lon=118)
        flat = [v for row in elev for v in row]
        self.assertEqual(len(elev), 18)
        self.assertEqual(len(elev[0]), 24)
        self.assertGreater(max(flat) - min(flat), 200.0)

    def test_native_30m_crop_shape(self) -> None:
        from windfarm.scenarios import grid_dims_for_extent

        w, h = grid_dims_for_extent(2350.0, 1750.0, 30.0)
        self.assertEqual((w, h), (79, 59))
        elev = crop_srtm_to_grid(HGT, 25.75, 118.60, width=w, height=h, resolution_m=30.0, tile_lat=25, tile_lon=118)
        self.assertEqual(len(elev), h)
        self.assertEqual(len(elev[0]), w)
        flat = [v for row in elev for v in row]
        self.assertGreater(max(flat) - min(flat), 150.0)

    def test_terrain_prepare_and_mission(self) -> None:
        terrain = terrain_from_srtm(HGT, 25.75, 118.60, 24, 18, 100.0, 25, 118)
        self.assertEqual(len(terrain.slope), 18)
        self.assertEqual(len(terrain.roughness[0]), 24)
        landcover = landcover_from_terrain(terrain.elevation, 100.0)
        self.assertTrue(all(0 <= c <= 5 for row in landcover for c in row))
        start, goal = mission_endpoints(24, 18, (0.12, 0.78), (0.86, 0.28))
        self.assertNotEqual(start, goal)

    def test_resample_coarse_wind(self) -> None:
        hourly = [
            {"timestamp": "2024-03-20T08:00:00", "u_km": 1.0, "v_km": -2.0, "w_km": 0.1},
            {"timestamp": "2024-03-20T09:00:00", "u_km": 3.0, "v_km": -1.0, "w_km": 0.2},
        ]
        series = resample_coarse_wind(hourly, time_steps=5, start_time="2024-03-20T08:00:00", sample_interval_seconds=900)
        self.assertEqual(len(series), 5)
        self.assertAlmostEqual(series[0]["u_km"], 1.0)
        self.assertGreater(series[-1]["u_km"], 1.0)

    def test_default_scenario_catalog_diverse(self) -> None:
        names = {spec.name for spec in DEFAULT_SCENARIOS}
        self.assertGreaterEqual(len(names), 4)
        self.assertIn("fujian_hills", names)
        self.assertIn("beijing_plain", names)


if __name__ == "__main__":
    unittest.main()
