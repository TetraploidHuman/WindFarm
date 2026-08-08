#!/usr/bin/env python3
"""Micro-benchmark for vectorized physics / belief / energy kernels."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfarm.belief import BeliefUpdater, create_belief_map
from windfarm.controller import transition_energy_batch, transition_energy_j
from windfarm.mathutils import trilinear_sample, trilinear_sample_batch
from windfarm.perf import warmup_numeric_kernels
from windfarm.physics import downscale_wind
from windfarm.planner import _polyline_model_energy_j, _polylines_model_energy_j, plan_path_details
from windfarm.types import Mission, TerrainField, WindField


def _terrain(h: int = 36, w: int = 48) -> TerrainField:
    yy, xx = np.mgrid[0:h, 0:w]
    elev = (1000.0 + 2.0 * xx + 3.0 * yy).astype(np.float64)
    slope = (0.05 + 0.001 * xx).astype(np.float64)
    aspect = (0.2 * xx).astype(np.float64)
    rough = np.full((h, w), 0.25, dtype=np.float64)
    return TerrainField(
        elevation=elev.tolist(),
        slope=slope.tolist(),
        aspect=aspect.tolist(),
        roughness=rough.tolist(),
    )


def _timed(label: str, fn, repeats: int = 5) -> float:
    # warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - t0) / repeats
    print(f"{label:42s} {dt * 1000:8.2f} ms")
    return dt


def main() -> None:
    warmup_numeric_kernels()
    terrain = _terrain()
    levels = 5

    def physics():
        return downscale_wind(4.0, -2.0, terrain, w_km=0.4, altitude_levels=levels)

    wind, _ = physics()
    belief = create_belief_map(48, 36, levels=levels)
    updater = BeliefUpdater()

    def belief_pred():
        updater.apply_prediction(belief, wind, step=3)

    field = np.asarray(wind.u, dtype=np.float64)
    rng = np.random.default_rng(0)
    n = 256
    xs = rng.uniform(0, 47, size=n)
    ys = rng.uniform(0, 35, size=n)
    zs = rng.uniform(0, 4, size=n)
    layers = field.tolist()

    def tri_scalar():
        for i in range(n):
            trilinear_sample(layers, float(xs[i]), float(ys[i]), float(zs[i]))

    def tri_batch():
        trilinear_sample_batch(field, xs, ys, zs)

    currents = np.column_stack([xs[:64], ys[:64], np.ones(64)])
    nexts = currents + np.array([1.0, 0.5, 0.0])
    u = rng.normal(size=64)
    v = rng.normal(size=64)
    w = rng.normal(size=64) * 0.2
    terrain_dz = rng.normal(size=64) * 5.0

    def energy_scalar():
        for i in range(64):
            transition_energy_j(
                airspeed=16.0,
                current=tuple(currents[i]),
                nxt=tuple(nexts[i]),
                local_u=float(u[i]),
                local_v=float(v[i]),
                local_w=float(w[i]),
                step_distance_m=50.0,
                altitude_step_m=50.0,
                climb_cost_per_level_j=225.0,
                terrain_dz_m=float(terrain_dz[i]),
            )

    def energy_batch():
        transition_energy_batch(
            airspeed=16.0,
            currents=currents,
            nexts=nexts,
            local_u=u,
            local_v=v,
            local_w=w,
            step_distance_m=50.0,
            altitude_step_m=50.0,
            climb_cost_per_level_j=225.0,
            terrain_dz_m=terrain_dz,
        )

    updater.apply_prediction(belief, wind, step=1)
    mission = Mission(
        start=(7, 25, 0),
        goal=(39, 13, 0),
        max_steps=80,
        step_distance_m=50.0,
        altitude_step_m=50.0,
        climb_cost_per_level_j=225.0,
        elevation=terrain.elevation,
        cruise_band_step=0.02,
        home=(7, 25, 0),
        max_return_cost_j=500_000.0,
    )
    paths = [
        [(7.0, 25.0, 1.0), (20.0, 18.0, 1.0), (39.0, 13.0, 0.0)],
        [(7.0, 25.0, 1.2), (22.0, 16.0, 1.2), (39.0, 13.0, 0.0)],
        [(7.0, 25.0, 1.5), (18.0, 20.0, 1.5), (39.0, 13.0, 0.0)],
        [(7.0, 25.0, 1.0), (25.0, 15.0, 1.0), (39.0, 13.0, 0.0)],
    ]

    def poly_scalar():
        for p in paths:
            _polyline_model_energy_j(p, belief, mission)

    def poly_batch():
        _polylines_model_energy_j(paths, belief, mission)

    def plan_once():
        plan_path_details(
            belief,
            mission,
            anytime_rounds=2,
            search_node_budget=3000,
            horizon_steps=7,
            beam_width=16,
            branch_width=8,
        )

    print("=== kernel microbench (48x36x5) ===")
    _timed("downscale_wind", physics)
    _timed("belief.apply_prediction", belief_pred)
    _timed("trilinear scalar x256", tri_scalar, repeats=3)
    _timed("trilinear batch x256", tri_batch)
    _timed("transition_energy scalar x64", energy_scalar, repeats=3)
    _timed("transition_energy batch x64", energy_batch)
    _timed("polyline energy scalar x4", poly_scalar, repeats=3)
    _timed("polyline energy batch x4", poly_batch)
    _timed("plan_path_details (light)", plan_once, repeats=2)


if __name__ == "__main__":
    main()
