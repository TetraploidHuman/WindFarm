#!/usr/bin/env python3
"""Evaluate altitude/wind-aware planner energy vs constant-AGL straight baselines."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfarm.altitude import cruise_agl_band_grid, format_agl_band_token, sample_elevation, terrain_delta_m
from windfarm.controller import transition_energy_j
from windfarm.io import load_terrain_document, load_truth_fields, read_json
from windfarm.mathutils import trilinear_sample


# Core maps used during early tuning.
CORE_SCENARIOS = (
    "fujian_hills",
    "beijing_plain",
    "qinghai_ridge",
    "liaoning_coast",
)
# Holdout real-world maps (anti-overfit; do not tune on these in isolation).
HOLDOUT_SCENARIOS = (
    "xinjiang_gobi",
    "sichuan_foothills",
    "shanxi_loess",
    "taiwan_hills",
    # Expansion: arid / karst / tropical / plain / plateau / forest / steppe.
    "gansu_hexi",
    "guizhou_karst",
    "hainan_coast",
    "hubei_jianghan",
    "tibet_lhasa",
    "jilin_forest",
    "neimeng_grass",
    "yunnan_karst",
    # Fresh holdout (overfit probe — evaluate, do not tune on these alone).
    "zhejiang_hills",
    "shandong_taishan",
    "chongqing_hills",
    "guangxi_guilin",
)
SCENARIOS = CORE_SCENARIOS + HOLDOUT_SCENARIOS


def _discover_scenarios(scenarios_dir: Path) -> list[str]:
    """Return the curated SCENARIOS list that exist on disk (ignore leftover maps)."""
    ordered: list[str] = []
    for name in SCENARIOS:
        if (scenarios_dir / name / "terrain.json").exists():
            ordered.append(name)
    return ordered


def _mission_params(mission: dict) -> dict:
    return dict(
        airspeed=mission["nominal_airspeed"],
        step_distance_m=mission["step_distance_m"],
        altitude_step_m=mission["altitude_step_m"],
        climb_cost_per_level_j=mission["climb_cost_per_level_j"],
        hover_power_w=mission["hover_power_w"],
        cruise_power_w=mission["cruise_power_w"],
        hotel_power_w=mission["hotel_power_w"],
        headwind_power_per_mps_w=mission["headwind_power_per_mps_w"],
        climb_power_per_mps_w=mission["climb_power_per_mps_w"],
        descent_power_reduction_per_mps_w=mission["descent_power_reduction_per_mps_w"],
    )


def _wind_at(truth_fields: list, x: float, y: float, z: float, t: int) -> tuple[float, float, float]:
    field = truth_fields[max(0, min(int(t), len(truth_fields) - 1))]
    return (
        trilinear_sample(field["u"], x, y, z),
        trilinear_sample(field["v"], x, y, z),
        trilinear_sample(field["w"], x, y, z),
    )


def _step_energy(cur, nxt, u, v, w, kw: dict, elevation, apply_uplift_discount: bool = True) -> float:
    terrain_dz_m = terrain_delta_m(elevation, cur[0], cur[1], nxt[0], nxt[1])
    req = transition_energy_j(
        current=tuple(cur),
        nxt=tuple(nxt),
        local_u=u,
        local_v=v,
        local_w=w,
        terrain_dz_m=terrain_dz_m,
        **kw,
    )
    # uplift discount lives inside transition_energy_j; flag kept for API compat
    del apply_uplift_discount
    return req


def straight_agl_baseline(
    start,
    goal,
    truth_fields: list,
    kw: dict,
    elevation,
    agl_cruise: float = 1.0,
) -> dict:
    """Straight XY + constant AGL cruise (terrain-following in MSL).

    Profile: climb AGL → hold constant AGL while following DEM → descend AGL.
    Vertical energy includes terrain rises even when AGL is unchanged.
    """
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0
    gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2]) if len(goal) > 2 else 0.0
    dist = max(math.hypot(gx - sx, gy - sy), 1.0)
    climb_levels = max(0.0, agl_cruise - sz)
    desc_levels = max(0.0, agl_cruise - gz)
    climb_steps = max(1, int(math.ceil(climb_levels))) if climb_levels > 0.05 else 0
    desc_steps = max(1, int(math.ceil(desc_levels))) if desc_levels > 0.05 else 0
    n_cruise = max(1, int(math.ceil(dist)))
    pts = [(sx, sy, sz)]
    climb_frac = 0.12 if climb_steps else 0.0
    desc_frac = 0.12 if desc_steps else 0.0
    for i in range(1, climb_steps + 1):
        t = climb_frac * i / climb_steps
        z = sz + (agl_cruise - sz) * i / climb_steps
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, z))
    cruise_start = climb_frac
    cruise_end = 1.0 - desc_frac
    for i in range(1, n_cruise + 1):
        t = cruise_start + (cruise_end - cruise_start) * i / n_cruise
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, float(agl_cruise)))
    for i in range(1, desc_steps + 1):
        t = cruise_end + desc_frac * i / desc_steps
        z = agl_cruise + (gz - agl_cruise) * i / desc_steps
        pts.append((sx + (gx - sx) * t, sy + (gy - sy) * t, z))
    if math.hypot(pts[-1][0] - gx, pts[-1][1] - gy) > 0.01 or abs(pts[-1][2] - gz) > 0.01:
        pts.append((gx, gy, gz))
    total = 0.0
    terrain_climb_m = 0.0
    for i in range(len(pts) - 1):
        u, v, w = _wind_at(truth_fields, *pts[i], i)
        dz_terrain = terrain_delta_m(elevation, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        terrain_climb_m += max(0.0, dz_terrain)
        total += _step_energy(pts[i], pts[i + 1], u, v, w, kw, elevation)
    return {
        "energy_j": total,
        "agl_cruise": agl_cruise,
        "n_points": len(pts),
        "terrain_climb_m": terrain_climb_m,
    }


def analyze_run(run_dir: Path) -> dict:
    report = json.loads((run_dir / "mission_report.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    mission = config["mission"]
    active_route = mission.get("active_route") or {}
    route_id = active_route.get("id") or "r0"
    route_index = active_route.get("index", 0)
    truth_fields = load_truth_fields(run_dir / "truth.json")
    elevation = load_terrain_document(run_dir / "terrain.json")["terrain"]["elevation"]
    path = report["executed_path"]
    start = path[0]
    goal = (report.get("route_summary") or {}).get("actual_goal") or mission["goal"]
    if len(goal) < 3:
        goal = list(goal) + [0]
    cap = float(mission["battery_capacity_j"])
    battery_j = (1.0 - float(report["battery_ratio"])) * cap
    kw = _mission_params(mission)

    # Prefer the same per-step charges used by the live battery (truth-wind path model).
    trace = report.get("trace") or []
    step_energies = [float(fr["step_energy_j"]) for fr in trace if fr.get("step_energy_j") is not None]
    if report.get("path_model_energy_j") is not None:
        path_model_j = float(report["path_model_energy_j"])
    elif step_energies:
        path_model_j = sum(step_energies)
    else:
        path_model_j = 0.0
        for i in range(len(path) - 1):
            u, v, w = _wind_at(truth_fields, *path[i], i)
            path_model_j += _step_energy(path[i], path[i + 1], u, v, w, kw, elevation)

    path_terrain_climb_m = 0.0
    for i in range(len(path) - 1):
        path_terrain_climb_m += max(
            0.0, terrain_delta_m(elevation, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        )

    max_level = float(mission.get("max_altitude_level", 4))
    min_level = float(mission.get("min_altitude_level", 0))
    clearance = float(mission.get("clearance_agl_level", 1.0))
    step = float(mission.get("cruise_band_step", 0.02) or 0.02)
    cruise_levels = cruise_agl_band_grid(
        clearance,
        min_level,
        max_level,
        step=step,
        span_levels=2.0,
    )
    if not cruise_levels:
        cruise_levels = [float(min(1.0, max_level))]
    baselines = {
        f"agl_{format_agl_band_token(z)}": straight_agl_baseline(
            start, goal, truth_fields, kw, elevation, agl_cruise=z
        )
        for z in cruise_levels
    }
    nominal_token = format_agl_band_token(clearance)
    nominal_key = f"agl_{nominal_token}" if f"agl_{nominal_token}" in baselines else next(iter(baselines))
    nominal_j = baselines[nominal_key]["energy_j"]
    best_key = min(baselines, key=lambda k: baselines[k]["energy_j"])
    best_j = baselines[best_key]["energy_j"]

    zs = [p[2] for p in path]
    xy = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
    straight_xy = math.hypot(float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1]))
    elev_start = sample_elevation(elevation, start[0], start[1])
    elev_goal = sample_elevation(elevation, goal[0], goal[1])
    elev_path = [sample_elevation(elevation, p[0], p[1]) for p in path]

    save_model_vs_nominal = (nominal_j - path_model_j) / nominal_j if nominal_j > 0 else 0.0
    save_model_vs_best = (best_j - path_model_j) / best_j if best_j > 0 else 0.0
    save_battery_vs_nominal = (nominal_j - battery_j) / nominal_j if nominal_j > 0 else 0.0
    save_battery_vs_best = (best_j - battery_j) / best_j if best_j > 0 else 0.0

    meta = {}
    meta_path = run_dir / "scenario_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    map_name = meta.get("name") or run_dir.name.split("-r")[0].replace("eval-", "").rsplit("-", 2)[0]
    # Prefer explicit map name from meta; fall back carefully.
    if meta.get("name"):
        map_name = meta["name"]

    return {
        "run": run_dir.name,
        "scenario": f"{map_name}/{route_id}",
        "map": map_name,
        "route_id": route_id,
        "route_index": int(route_index) if route_index is not None else 0,
        "route_label": active_route.get("label"),
        "start": list(active_route.get("start") or mission.get("start") or start[:2]),
        "goal": list(active_route.get("goal") or mission.get("goal") or goal[:2]),
        "description": meta.get("description"),
        "goal_reached": bool(report["goal_reached"]),
        "steps": int(report["steps_executed"]),
        "battery_j": battery_j,
        "battery_kJ": battery_j / 1000.0,
        "path_model_kJ": path_model_j / 1000.0,
        "energy_kJ": path_model_j / 1000.0,  # primary comparable energy
        "battery_model_gap_kJ": (battery_j - path_model_j) / 1000.0,
        "baseline_nominal_kJ": nominal_j / 1000.0,
        "baseline_nominal_band": nominal_key,
        "baseline_best_kJ": best_j / 1000.0,
        "baseline_best_band": best_key,
        "baselines_kJ": {k: v["energy_j"] / 1000.0 for k, v in baselines.items()},
        "baselines_terrain_climb_m": {k: v["terrain_climb_m"] for k, v in baselines.items()},
        # Primary: path-model vs AGL (identical transition_energy_j + uplift discount accounting).
        "savings_vs_nominal_pct": 100.0 * save_model_vs_nominal,
        "savings_vs_best_pct": 100.0 * save_model_vs_best,
        "savings_model_vs_nominal_pct": 100.0 * save_model_vs_nominal,
        "savings_model_vs_best_pct": 100.0 * save_model_vs_best,
        # Diagnostic only: live battery (may still differ slightly due to micro-steps).
        "savings_battery_vs_nominal_pct": 100.0 * save_battery_vs_nominal,
        "savings_battery_vs_best_pct": 100.0 * save_battery_vs_best,
        "z_min": min(zs),
        "z_max": max(zs),
        "z_mean": sum(zs) / len(zs),
        "path_xy": xy,
        "straight_xy": straight_xy,
        "path_terrain_climb_m": path_terrain_climb_m,
        "elev_start_m": elev_start,
        "elev_goal_m": elev_goal,
        "elev_path_min_m": min(elev_path),
        "elev_path_max_m": max(elev_path),
        "final": report["final_position"],
        "elev_m": meta.get("elevation_m"),
        "wind": meta.get("wind"),
        "clearance_agl_level": clearance,
    }


def _scenario_routes(name: str) -> list[dict]:
    cfg = json.loads((ROOT / "scenarios" / name / "config.json").read_text())
    mission = cfg.get("mission", {})
    routes = mission.get("routes")
    if routes:
        return list(routes)
    from windfarm.data_ingest import mission_route_octet

    return mission_route_octet(mission["start"], mission["goal"])


def run_scenario(
    name: str,
    runs_dir: Path,
    *,
    route_index: int = 0,
    n_jobs: int | None = None,
) -> Path:
    stamp = datetime.now().strftime("%H%M%S")
    # Unique across parallel workers (same-second collisions otherwise).
    run_name = f"eval-{name}-r{route_index}-{stamp}-{os.getpid()}"
    cmd = [
        str(ROOT / ".venv-linux" / "bin" / "python"),
        "-m",
        "windfarm.cli",
        "run-demo",
        "--config",
        str(ROOT / "scenarios" / name / "config.json"),
        "--runs-dir",
        str(runs_dir),
        "--run-name",
        run_name,
        "--scenario",
        name,
        "--route-index",
        str(int(route_index)),
    ]
    env = dict(os.environ)
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/nix/store/f7pc134264jj4id4jnqpadzl0nnnayj7-ld-library-path/share/nix-ld/lib",
    )
    # Cap nested BLAS / XGB threads when many scenarios run concurrently.
    if n_jobs is not None:
        jobs = str(max(1, int(n_jobs)))
        env["WINDFARM_N_JOBS"] = jobs
        env["OMP_NUM_THREADS"] = jobs
        env["MKL_NUM_THREADS"] = jobs
        env["OPENBLAS_NUM_THREADS"] = jobs
        env["NUMEXPR_NUM_THREADS"] = jobs
    # Eval does not need 70MB+ HTML dashboards.
    env.setdefault("WINDFARM_SKIP_DASHBOARD", "1")
    print(f"\n=== running {name} route={route_index} → {run_name} ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return runs_dir / run_name


def main() -> None:
    runs_dir = ROOT / "runs"
    out_dir = runs_dir / f"multi-scenario-energy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: reanalyze existing run dirs without re-simulating.
    #   python scripts/eval_multi_scenario_energy.py --reanalyze runs/eval-foo runs/eval-bar ...
    reanalyze = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1] == "--reanalyze" else None
    reuse_dirs: list[Path] = []
    if reanalyze is not None:
        reuse_dirs = [Path(p) for p in sys.argv[2:]]
        if not reuse_dirs:
            raise SystemExit("usage: eval_multi_scenario_energy.py --reanalyze <run_dir>...")

    results = []
    if reuse_dirs:
        for run_dir in reuse_dirs:
            row = analyze_run(run_dir)
            results.append(row)
            print(
                f"  [{row['scenario']}] reached={row['goal_reached']} "
                f"model={row['path_model_kJ']:.2f} kJ agl={row['baseline_nominal_kJ']:.2f} "
                f"save={row['savings_vs_nominal_pct']:+.1f}% "
                f"(battery diag {row['savings_battery_vs_nominal_pct']:+.1f}%)",
                flush=True,
            )
    else:
        names = _discover_scenarios(ROOT / "scenarios")
        if not names:
            raise SystemExit("no scenarios with terrain.json found under scenarios/")
        # Optional filter: python scripts/eval_multi_scenario_energy.py --only a b c
        # Optional parallelism: --workers N  (default caps at 4; 18-way OOMs ~32GB hosts)
        # --primary-only: only route r0 (legacy single OD). Default: all mission.routes.
        # --catalog PATH --split train|val|sealed_holdout|tune: run jobs from a tune catalog.
        argv = sys.argv[1:]
        max_workers_arg: int | None = None
        only_names: list[str] | None = None
        primary_only = False
        octet_only = False
        catalog_path: Path | None = None
        catalog_split: str | None = None
        catalog_limit: int | None = None
        catalog_seed = 7
        cleanup_runs = False
        i = 0
        while i < len(argv):
            if argv[i] == "--only":
                i += 1
                only_names = []
                while i < len(argv) and not argv[i].startswith("--"):
                    only_names.append(argv[i])
                    i += 1
                continue
            if argv[i] == "--workers" and i + 1 < len(argv):
                max_workers_arg = max(1, int(argv[i + 1]))
                i += 2
                continue
            if argv[i] == "--output-dir" and i + 1 < len(argv):
                out_dir = Path(argv[i + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                i += 2
                continue
            if argv[i] == "--runs-dir" and i + 1 < len(argv):
                runs_dir = Path(argv[i + 1])
                runs_dir.mkdir(parents=True, exist_ok=True)
                i += 2
                continue
            if argv[i] == "--primary-only":
                primary_only = True
                i += 1
                continue
            if argv[i] == "--octet-only":
                octet_only = True
                i += 1
                continue
            if argv[i] == "--cleanup-runs":
                # Delete per-route demo artifacts after metrics are extracted.
                # Each run copies terrain/truth/model (~0.3–1.2GB); tune must use this.
                cleanup_runs = True
                i += 1
                continue
            if argv[i] == "--catalog" and i + 1 < len(argv):
                catalog_path = Path(argv[i + 1])
                i += 2
                continue
            if argv[i] == "--split" and i + 1 < len(argv):
                catalog_split = str(argv[i + 1])
                i += 2
                continue
            if argv[i] == "--limit" and i + 1 < len(argv):
                catalog_limit = max(1, int(argv[i + 1]))
                i += 2
                continue
            if argv[i] == "--seed" and i + 1 < len(argv):
                catalog_seed = int(argv[i + 1])
                i += 2
                continue
            i += 1
        if only_names is not None:
            wanted = set(only_names)
            names = [n for n in names if n in wanted]
            if not names:
                raise SystemExit("no matching scenarios for --only")
        jobs: list[tuple[str, int]] = []
        if catalog_path is not None:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            split = catalog_split or "train"
            selected = []
            for job in catalog.get("jobs", []):
                sp = str(job.get("split", ""))
                if split == "tune" and sp in ("train", "val"):
                    selected.append(job)
                elif sp == split:
                    selected.append(job)
            if not selected:
                raise SystemExit(f"no jobs for split={split} in {catalog_path}")
            if catalog_limit is not None and catalog_limit < len(selected):
                import random as _random

                rng = _random.Random(catalog_seed)
                selected = rng.sample(selected, catalog_limit)
            jobs = [(str(j["map"]), int(j["route_index"])) for j in selected]
            names = sorted({m for m, _ in jobs})
        else:
            for name in names:
                if primary_only:
                    route_count = 1
                elif octet_only:
                    route_count = min(8, len(_scenario_routes(name)))
                else:
                    route_count = len(_scenario_routes(name))
                for route_index in range(route_count):
                    jobs.append((name, route_index))
        # 8 curated maps fit in ~32GB at full parallelism; only cap when many more exist.
        default_cap = min(8, max(1, os.cpu_count() or 4))
        max_workers = min(len(jobs), max_workers_arg or default_cap)
        # Split cores across concurrent scenario processes to avoid XGB/OpenMP thrash.
        per_proc_jobs = max(1, (os.cpu_count() or 4) // max(max_workers, 1))
        print(
            f"parallel jobs: {len(jobs)} (maps={len(names)} routes_per_map="
            f"{'1' if primary_only else 'all'}) workers={max_workers} "
            f"WINDFARM_N_JOBS={per_proc_jobs} cleanup_runs={cleanup_runs} "
            f"runs_dir={runs_dir}",
            flush=True,
        )
        run_dirs: dict[str, Path] = {}
        # Thread pool is enough: each scenario already launches its own process.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_scenario, name, runs_dir, route_index=route_index, n_jobs=per_proc_jobs
                ): (name, route_index)
                for name, route_index in jobs
            }
            for fut in as_completed(futures):
                name, route_index = futures[fut]
                run_dir = fut.result()
                key = f"{name}/r{route_index}"
                run_dirs[key] = run_dir
                row = analyze_run(run_dir)
                results.append(row)
                print(
                    f"  [{row['scenario']}] reached={row['goal_reached']} model={row['path_model_kJ']:.2f} kJ "
                    f"agl={row['baseline_nominal_kJ']:.2f}({row['baseline_nominal_band']}) "
                    f"best={row['baseline_best_kJ']:.2f}({row['baseline_best_band']}) "
                    f"save_vs_agl={row['savings_vs_nominal_pct']:+.1f}% "
                    f"save_vs_best={row['savings_vs_best_pct']:+.1f}% "
                    f"bat_diag={row['savings_battery_vs_nominal_pct']:+.1f}% "
                    f"z=[{row['z_min']:.2f},{row['z_max']:.2f}] mean={row['z_mean']:.2f} "
                    f"terrain↑={row['path_terrain_climb_m']:.0f}m",
                    flush=True,
                )
                if cleanup_runs:
                    shutil.rmtree(run_dir, ignore_errors=True)
                    row["run"] = None
                    row["run_cleaned"] = True
        # Stable order: map order × route index.
        order = {f"{name}/r{ri}": i for i, (name, ri) in enumerate(jobs)}
        results.sort(
            key=lambda r: order.get(
                f"{r.get('map')}/{r.get('route_id')}",
                order.get(r["scenario"], 999),
            )
        )

    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "metric": "path-model energy vs constant-AGL straight (same transition_energy_j + uplift discount)",
        "savings_definition": "positive % means algorithm path-model used less energy than baseline",
        "primary_energy": "path_model_kJ",
        "battery_note": "battery_kJ / savings_battery_* are diagnostic only",
        "core_scenarios": list(CORE_SCENARIOS),
        "holdout_scenarios": list(HOLDOUT_SCENARIOS),
        "results": results,
    }

    def _aggregate(rows: list[dict]) -> dict:
        reached = [r for r in rows if r["goal_reached"]]
        return {
            "n_scenarios": len(rows),
            "n_reached": len(reached),
            "mean_savings_vs_nominal_pct": sum(r["savings_vs_nominal_pct"] for r in reached) / max(len(reached), 1),
            "mean_savings_vs_best_pct": sum(r["savings_vs_best_pct"] for r in reached) / max(len(reached), 1),
            "mean_path_model_kJ": sum(r["path_model_kJ"] for r in reached) / max(len(reached), 1),
            "mean_battery_kJ": sum(r["battery_kJ"] for r in reached) / max(len(reached), 1),
            "mean_baseline_nominal_kJ": sum(r["baseline_nominal_kJ"] for r in reached) / max(len(reached), 1),
            "mean_savings_battery_vs_nominal_pct": sum(r["savings_battery_vs_nominal_pct"] for r in reached)
            / max(len(reached), 1),
        }

    if results:
        summary["aggregate"] = _aggregate(results)
        core_rows = [r for r in results if r.get("map", r["scenario"].split("/")[0]) in CORE_SCENARIOS]
        hold_rows = [r for r in results if r.get("map", r["scenario"].split("/")[0]) in HOLDOUT_SCENARIOS]
        if core_rows:
            summary["aggregate_core"] = _aggregate(core_rows)
        if hold_rows:
            summary["aggregate_holdout"] = _aggregate(hold_rows)

    out_path = out_dir / "energy_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "scenario                reached  model_kJ  agl_kJ  best_kJ  save_agl%  save_best%  bat_diag%  zmax  terr↑m",
        "-" * 118,
    ]
    for r in results:
        lines.append(
            f"{r['scenario']:24s} {str(r['goal_reached']):7s} {r['path_model_kJ']:8.2f} "
            f"{r['baseline_nominal_kJ']:7.2f} {r['baseline_best_kJ']:7.2f} "
            f"{r['savings_vs_nominal_pct']:+9.1f} {r['savings_vs_best_pct']:+10.1f} "
            f"{r['savings_battery_vs_nominal_pct']:+9.1f} "
            f"{r['z_max']:5.2f} {r['path_terrain_climb_m']:7.0f}"
        )
    if "aggregate" in summary:
        a = summary["aggregate"]
        lines.append("-" * 110)
        lines.append(
            f"MEAN (reached {a['n_reached']}/{a['n_scenarios']})          "
            f"{a['mean_path_model_kJ']:8.2f} {a['mean_baseline_nominal_kJ']:7.2f}         "
            f"{a['mean_savings_vs_nominal_pct']:+9.1f} {a['mean_savings_vs_best_pct']:+10.1f} "
            f"{a['mean_savings_battery_vs_nominal_pct']:+9.1f}"
        )
        for key, label in (("aggregate_core", "CORE"), ("aggregate_holdout", "HOLDOUT")):
            if key in summary:
                b = summary[key]
                lines.append(
                    f"{label} (reached {b['n_reached']}/{b['n_scenarios']})        "
                    f"{b['mean_path_model_kJ']:8.2f} {b['mean_baseline_nominal_kJ']:7.2f}         "
                    f"{b['mean_savings_vs_nominal_pct']:+9.1f} {b['mean_savings_vs_best_pct']:+10.1f} "
                    f"{b['mean_savings_battery_vs_nominal_pct']:+9.1f}"
                )
    table = "\n".join(lines) + "\n"
    (out_dir / "energy_summary.txt").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
