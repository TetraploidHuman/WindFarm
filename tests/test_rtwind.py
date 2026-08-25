from __future__ import annotations

import asyncio
import unittest

from windfarm.rtwind.hub import TelemetryHub
from windfarm.rtwind.types import RtwindConfig, TelemetryFrame


def _frame(source: str, lat: float = 1.0) -> TelemetryFrame:
    return TelemetryFrame(
        source=source,  # type: ignore[arg-type]
        vehicle_id="t1",
        t="2026-01-01T00:00:00+00:00",
        lat=lat,
        lon=2.0,
        alt_msl=100.0,
        heading=90.0,
        roll=0.0,
        pitch=0.0,
        yaw=90.0,
        airspeed=10.0,
        groundspeed=10.0,
        climb_rate=0.0,
    )


class RtwindHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_exclusive_publish(self) -> None:
        hub = TelemetryHub(RtwindConfig(active_source="sim"))
        ok_sim = await hub.publish(_frame("sim", 10.0))
        ok_live = await hub.publish(_frame("live", 20.0))
        self.assertTrue(ok_sim)
        self.assertFalse(ok_live)
        latest = hub.latest()
        assert latest is not None
        self.assertEqual(latest.lat, 10.0)

    async def test_switch_clears_track(self) -> None:
        hub = TelemetryHub(RtwindConfig(active_source="sim"))
        await hub.publish(_frame("sim"))
        self.assertEqual(len(hub.track()), 1)
        snap = await hub.set_active("live", clear_track=True)
        self.assertEqual(snap["active"], "live")
        self.assertEqual(len(hub.track()), 0)
        self.assertIsNone(hub.latest())


class MavlinkStateTests(unittest.TestCase):
    def test_merge_global_position_and_attitude(self) -> None:
        from windfarm.rtwind.mavlink_source import MavlinkState

        class Msg:
            def __init__(self, mtype: str, **kwargs):
                self._type = mtype
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def get_type(self) -> str:
                return self._type

        state = MavlinkState()
        state.update_from_msg(
            Msg(
                "GLOBAL_POSITION_INT",
                lat=int(27.908 * 1e7),
                lon=int(112.922 * 1e7),
                alt=120000,
                relative_alt=80000,
                hdg=9000,
                vx=500,
                vy=0,
                vz=-100,
            )
        )
        state.update_from_msg(Msg("ATTITUDE", roll=0.1, pitch=-0.05, yaw=1.57))
        frame = state.to_frame()
        assert frame is not None
        self.assertAlmostEqual(frame.lat, 27.908, places=3)
        self.assertAlmostEqual(frame.lon, 112.922, places=3)
        self.assertAlmostEqual(frame.alt_msl, 120.0, places=1)
        self.assertAlmostEqual(frame.alt_agl, 80.0, places=1)


class GeoGridTests(unittest.TestCase):
    def test_dem_grid_shape(self) -> None:
        from windfarm.rtwind.geo import GeoContext

        geo = GeoContext(data_dir="data")
        grid = geo.dem_grid(27.9, 112.9, 27.92, 112.94, nx=3, ny=4)
        self.assertEqual(grid["nx"], 3)
        self.assertEqual(grid["ny"], 4)
        self.assertEqual(len(grid["values"]), 4)
        self.assertEqual(len(grid["values"][0]), 3)


class SimSourceTests(unittest.TestCase):
    def test_snapshot_and_reset_origin(self) -> None:
        from windfarm.rtwind.sim_source import SimDroneSource
        from windfarm.rtwind.hub import TelemetryHub
        from windfarm.rtwind.types import RtwindConfig

        hub = TelemetryHub(RtwindConfig())
        sim = SimDroneSource(hub, RtwindConfig(sim_origin_lat=1.0, sim_origin_lon=2.0))
        snap = sim.snapshot()
        self.assertEqual(snap["origin_lat"], 1.0)
        sim.reset(lat=30.5, lon=114.3)
        snap2 = sim.snapshot()
        self.assertEqual(snap2["origin_lat"], 30.5)
        self.assertEqual(snap2["origin_lon"], 114.3)

    def test_heading_matches_orbit_tangent(self) -> None:
        import math

        from windfarm.rtwind.sim_source import SimDroneSource
        from windfarm.rtwind.hub import TelemetryHub
        from windfarm.rtwind.types import RtwindConfig

        sim = SimDroneSource(hub=TelemetryHub(RtwindConfig()), config=RtwindConfig())
        sim._phase = 0.0
        frame_east = sim._tick(0.0)
        self.assertAlmostEqual(frame_east.heading, 0.0, delta=0.1)
        sim._phase = math.pi / 2.0
        frame_north = sim._tick(0.0)
        self.assertAlmostEqual(frame_north.heading, 270.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
