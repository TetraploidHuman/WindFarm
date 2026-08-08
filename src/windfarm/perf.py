"""One-shot Numba/Numpy warmup so first planning step is not cold."""

from __future__ import annotations

import numpy as np


def warmup_numeric_kernels() -> None:
    """Compile Numba kernels and touch vectorized paths once."""
    from .controller import transition_energy_batch
    from .mathutils import trilinear_sample, trilinear_sample_batch

    field = np.zeros((3, 4, 5), dtype=np.float64)
    xs = np.array([0.5, 1.5], dtype=np.float64)
    ys = np.array([0.5, 1.0], dtype=np.float64)
    zs = np.array([0.0, 1.0], dtype=np.float64)
    trilinear_sample_batch(field, xs, ys, zs)
    trilinear_sample(field, 0.5, 0.5, 0.0)
    from .altitude import terrain_climb_along_line_batch

    elev = np.arange(36 * 48, dtype=np.float64).reshape(36, 48)
    zeros = np.zeros(8, dtype=np.float64)
    ones = np.ones(8, dtype=np.float64)
    terrain_climb_along_line_batch(elev, zeros, zeros, ones * 10, ones * 5, samples=4)
    currents = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    nexts = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float64)
    zeros = np.zeros(2, dtype=np.float64)
    transition_energy_batch(
        airspeed=16.0,
        currents=currents,
        nexts=nexts,
        local_u=zeros,
        local_v=zeros,
        local_w=zeros,
        step_distance_m=50.0,
        altitude_step_m=50.0,
        climb_cost_per_level_j=225.0,
        terrain_dz_m=zeros,
    )
