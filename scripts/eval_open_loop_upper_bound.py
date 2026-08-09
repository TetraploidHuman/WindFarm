#!/usr/bin/env python3
"""Open-loop truth oracles vs latest closed-loop eval — remaining headroom.

For each curated scenario, using truth wind (not belief):
  - best constant-AGL straight
  - best lateral via corridor (same offset set as planner light-air hunt)
  - same paths with headwind speed-to-fly airspeed

Compares against the newest eval-* run's path_model energy.
Does not tune per map; reports where ML / planning can still gain.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfarm.altitude import terrain_delta_m
from windfarm.belief import create_belief_map
from windfarm.config import load_task_config
from windfarm.controller import horizontal_headwind_mps, speed_to_fly_airspeed, transition_energy_j
from windfarm.io import read_json
from windfarm.mathutils import clamp, trilinear_sample
from windfarm.planner import _agl_guide_polyline, _polyline_model_energy_j
from windfarm.types import Mission

CORE = ("fujian_hills", "beijing_plain", "qinghai_ridge", "liaoning_coast")
HOLDOUT = (
    "xinjiang_gobi",
    "sichuan_foothills",
    "shanxi_loess",
    "taiwan_hills",
    "gansu_hexi",
    "guizhou_karst",
    "hainan_coast",
    "hubei_jianghan",
    "tibet_lhasa",
    "jilin_forest",
    "neimeng_grass",
    "yunnan_karst",
)
SCENARIOS = CORE + HOLDOUT
OFFSETS_M = (100.0, 150.0, 200.0, 250.0, 350.0, 450.0, 550.0)


def _mission_kw(m) -> dict:
    return dict(
        step_distance_m=m.step_distance_m,
        altitude_step_m=m.altitude_step_m,
        climb_cost_per_level_j=m.climb_cost_per_level_j,
        hover_power_w=m.hover_power_w,
        cruise_power_w=m.cruise_power_w,
        hotel_power_w=m.hotel_power_w,
        headwind_power_per_mps_w=m.headwind_power_per_mps_w,
        climb_power_per_mps_w=m.climb_power_per_mps_w,
        descent_power_reduction_per_mps_w=m.descent_power_reduction_per_mps_w,
    )


def _path_energy_truth(
    path,
    truth_field,
    elev,
    kw,
    airspeed_fn,
    *,
    nominal_airspeed: float | None = None,
    energy_gate: bool = False,
) -> float:
    """Truth-wind path energy. When energy_gate, keep proposed airspeed only if cheaper."""
    e = 0.0
    for a, b in zip(path, path[1:]):
        u = trilinear_sample(truth_field["u"], a[0], a[1], a[2])
        v = trilinear_sample(truth_field["v"], a[0], a[1], a[2])
        w = trilinear_sample(truth_field["w"], a[0], a[1], a[2])
        head = horizontal_headwind_mps(tuple(a), tuple(b), u, v)
        aspd = airspeed_fn(w, head)
        tdz = terrain_delta_m(elev, a[0], a[1], b[0], b[1])
        step_kw = dict(
            current=tuple(a),
            nxt=tuple(b),
            local_u=u,
            local_v=v,
            local_w=w,
            terrain_dz_m=tdz,
            **kw,
        )
        e_prop = transition_energy_j(airspeed=aspd, **step_kw)
        if (
            energy_gate
            and nominal_airspeed is not None
            and aspd > float(nominal_airspeed) + 1e-6
        ):
            e_nom = transition_energy_j(airspeed=float(nominal_airspeed), **step_kw)
            e += min(e_prop, e_nom)
        else:
            e += e_prop
    return e


def _latest_eval(runs: Path, name: str) -> Path | None:
    cands = sorted(runs.glob(f"eval-{name}-*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _eval_scenario(name: str, runs: Path) -> dict:
    run = _latest_eval(runs, name)
    if run is None:
        return {"scenario": name, "error": "no eval run"}
    cfg = load_task_config(run / "config.json")
    elev = read_json(run / "terrain.json")["terrain"]["elevation"]
    truth_fields = read_json(run / "truth.json")["truth_fields"]
    field = truth_fields[min(40, len(truth_fields) - 1)]
    report = read_json(run / "mission_report.json")
    closed_j = float(report["path_model_energy_j"])
    m = cfg.mission
    kw = _mission_kw(m)
    nominal = float(m.nominal_airspeed)
    h, w = len(elev), len(elev[0])
    belief = create_belief_map(w, h, cfg.model.altitude_levels)
    for attr, key in (("wind_u", "u"), ("wind_v", "v"), ("wind_w", "w")):
        arr = np.asarray(field[key], dtype=np.float64)
        belief.field_arrays[attr][: arr.shape[0]] = arr
    mission = Mission(
        start=tuple(m.start) + (0,),
        goal=tuple(m.goal) + (0,),
        max_steps=m.max_steps,
        step_distance_m=m.step_distance_m,
        clearance_agl_level=m.clearance_agl_level,
        max_altitude_level=m.max_altitude_level,
        altitude_step_m=m.altitude_step_m,
        cruise_band_step=m.cruise_band_step,
        elevation=elev,
    )
    start, goal = mission.start, mission.goal
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    dx, dy = gx - sx, gy - sy
    dist = math.hypot(dx, dy) or 1.0
    px, py = -dy / dist, dx / dist
    cell = float(m.step_distance_m)

    def const_as(_w, _h):
        return nominal

    def stf_as(ww, head):
        return speed_to_fly_airspeed(nominal, head, ww)

    best_straight = None
    for z in (1.0, 1.5, 2.0, 2.5, 3.0):
        path = _agl_guide_polyline(start, goal, belief, mission, cruise_z=z)
        e = _path_energy_truth(path, field, elev, kw, const_as)
        e_stf = _path_energy_truth(
            path, field, elev, kw, stf_as, nominal_airspeed=nominal, energy_gate=True
        )
        if best_straight is None or e < best_straight[0]:
            best_straight = (e, e_stf, z, path)

    se, se_stf, sz, sp = best_straight
    best_corr = None
    for z in (sz, 1.0, 1.5, 2.0, 2.5):
        for sign in (-1.0, 1.0):
            for off_m in OFFSETS_M:
                off = off_m / cell
                mx = clamp(sx + 0.5 * dx + sign * off * px, 0.0, w - 1)
                my = clamp(sy + 0.5 * dy + sign * off * py, 0.0, h - 1)
                path = _agl_guide_polyline(start, goal, belief, mission, via_xy=(mx, my), cruise_z=z)
                e = _path_energy_truth(path, field, elev, kw, const_as)
                e_stf = _path_energy_truth(
                    path, field, elev, kw, stf_as, nominal_airspeed=nominal, energy_gate=True
                )
                if best_corr is None or e < best_corr[0]:
                    best_corr = (e, e_stf, z, sign, off_m)

    oracle_j = min(se, best_corr[0] if best_corr else se)
    oracle_stf_j = min(se_stf, best_corr[1] if best_corr else se_stf)
    return {
        "scenario": name,
        "eval_run": run.name,
        "closed_loop_kJ": closed_j / 1000.0,
        "oracle_straight_kJ": se / 1000.0,
        "oracle_straight_stf_kJ": se_stf / 1000.0,
        "oracle_straight_z": sz,
        "oracle_corridor_kJ": (best_corr[0] / 1000.0) if best_corr else None,
        "oracle_corridor_stf_kJ": (best_corr[1] / 1000.0) if best_corr else None,
        "oracle_corridor_z": best_corr[2] if best_corr else None,
        "oracle_corridor_offset_m": best_corr[4] if best_corr else None,
        "oracle_best_kJ": oracle_j / 1000.0,
        "oracle_best_stf_kJ": oracle_stf_j / 1000.0,
        "gap_vs_oracle_pct": 100.0 * (closed_j - oracle_j) / max(oracle_j, 1e-9),
        "gap_vs_oracle_stf_pct": 100.0 * (closed_j - oracle_stf_j) / max(oracle_stf_j, 1e-9),
        "corridor_vs_straight_pct": 100.0 * (se - best_corr[0]) / max(se, 1e-9) if best_corr else 0.0,
    }


def main() -> None:
    runs = ROOT / "runs"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = runs / f"open-loop-upper-bound-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_eval_scenario(name, runs) for name in SCENARIOS]
    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Positive gap_vs_oracle_pct means closed-loop used more energy than truth open-loop best.",
        "results": rows,
    }
    out_dir.joinpath("upper_bound.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"{'scenario':20} {'closed':>8} {'ora_s':>8} {'ora_c':>8} {'ora+stf':>8} {'gap%':>8} {'gap_stf%':>9} {'corr%':>7}",
        "-" * 90,
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['scenario']:20} ERROR {r['error']}")
            continue
        lines.append(
            f"{r['scenario']:20} {r['closed_loop_kJ']:8.2f} {r['oracle_straight_kJ']:8.2f} "
            f"{(r['oracle_corridor_kJ'] or 0):8.2f} {r['oracle_best_stf_kJ']:8.2f} "
            f"{r['gap_vs_oracle_pct']:+8.1f} {r['gap_vs_oracle_stf_pct']:+9.1f} "
            f"{r['corridor_vs_straight_pct']:+7.1f}"
        )
    ok = [r for r in rows if "error" not in r]
    if ok:
        mg = sum(r["gap_vs_oracle_stf_pct"] for r in ok) / len(ok)
        lines.append("-" * 90)
        lines.append(f"MEAN gap vs oracle+stf: {mg:+.1f}%  (negative ⇒ closed-loop already beats this open-loop set)")
    text = "\n".join(lines) + "\n"
    out_dir.joinpath("upper_bound.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
