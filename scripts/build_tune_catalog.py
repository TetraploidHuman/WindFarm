#!/usr/bin/env python3
"""Expand per-map mission.routes and write a train/val/sealed tune catalog.

Protocol
--------
- sealed_holdout: HOLDOUT maps × original octet r0–r7 only. Never use for tuning.
- tune pool: all other (map, route) pairs (core octets + every map's expanded routes).
- train / val: stratified 70/30 split of the tune pool (by map).

Example (≈1024 tune slots with 20 maps × 56 routes − 128 sealed):
  .venv-linux/bin/python scripts/build_tune_catalog.py --n-routes 56
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfarm.data_ingest import mission_route_bundle, mission_route_octet

# Keep in sync with scripts/eval_multi_scenario_energy.py
CORE_SCENARIOS = (
    "fujian_hills",
    "beijing_plain",
    "qinghai_ridge",
    "liaoning_coast",
)
HOLDOUT_SCENARIOS = (
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
    "zhejiang_hills",
    "shandong_taishan",
    "chongqing_hills",
    "guangxi_guilin",
)


def _discover_maps(scenarios_dir: Path) -> list[str]:
    ordered = list(CORE_SCENARIOS) + list(HOLDOUT_SCENARIOS)
    out: list[str] = []
    for name in ordered:
        if (scenarios_dir / name / "config.json").exists():
            out.append(name)
    for p in sorted(scenarios_dir.iterdir()):
        if p.is_dir() and (p / "config.json").exists() and p.name not in out:
            out.append(p.name)
    return out


def _expand_map_routes(map_dir: Path, *, n_routes: int, seed: int) -> list[dict]:
    cfg_path = map_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    mission = cfg.setdefault("mission", {})
    sim = cfg.get("simulation", {})
    width = int(sim.get("width") or 79)
    height = int(sim.get("height") or 59)
    start = mission.get("start")
    goal = mission.get("goal")
    if not start or not goal:
        raise SystemExit(f"missing start/goal in {cfg_path}")

    octet_path = map_dir / "routes_octet.json"
    existing = mission.get("routes") or []
    if not octet_path.exists():
        # Prefer the first 8 entries if already present; else rebuild from primary OD.
        if len(existing) >= 8 and all(str(r.get("id", "")).startswith("r") for r in existing[:8]):
            octet = existing[:8]
        else:
            octet = mission_route_octet(start, goal)
        octet_path.write_text(json.dumps(octet, indent=2), encoding="utf-8")
    else:
        octet = json.loads(octet_path.read_text(encoding="utf-8"))

    # Bundle always starts from primary start/goal (octet r0), not a random existing route.
    primary = octet[0]
    routes = mission_route_bundle(
        primary["start"],
        primary["goal"],
        width=width,
        height=height,
        n_routes=n_routes,
        seed=seed,
    )
    # Preserve octet labels / ids exactly from the backed-up octet.
    routes[: len(octet)] = octet
    for i, r in enumerate(routes):
        r["id"] = f"r{i}"
        if i >= 8:
            r.setdefault("label", f"expanded_{i}")
            r["expanded"] = True

    mission["routes"] = routes
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return routes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-dir", type=Path, default=ROOT / "scenarios")
    ap.add_argument("--n-routes", type=int, default=56, help="routes per map (octet + expanded)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scenarios" / "tune_catalog_1024.json",
    )
    ap.add_argument("--dry-run", action="store_true", help="do not rewrite config.json")
    args = ap.parse_args()

    maps = _discover_maps(args.scenarios_dir)
    if not maps:
        raise SystemExit(f"no maps under {args.scenarios_dir}")

    sealed: list[dict] = []
    tune: list[dict] = []
    per_map_counts: dict[str, int] = {}

    for name in maps:
        map_dir = args.scenarios_dir / name
        if args.dry_run:
            cfg = json.loads((map_dir / "config.json").read_text(encoding="utf-8"))
            mission = cfg["mission"]
            sim = cfg.get("simulation", {})
            routes = mission_route_bundle(
                mission["start"],
                mission["goal"],
                width=int(sim.get("width") or 79),
                height=int(sim.get("height") or 59),
                n_routes=args.n_routes,
                seed=args.seed,
            )
        else:
            routes = _expand_map_routes(map_dir, n_routes=args.n_routes, seed=args.seed)
        per_map_counts[name] = len(routes)
        is_holdout = name in HOLDOUT_SCENARIOS
        for ri, route in enumerate(routes):
            job = {
                "map": name,
                "route_index": ri,
                "route_id": route.get("id", f"r{ri}"),
                "expanded": bool(route.get("expanded", ri >= 8)),
            }
            if is_holdout and ri < 8:
                job["split"] = "sealed_holdout"
                sealed.append(job)
            else:
                tune.append(job)

    # Stratified train/val by map.
    rng = random.Random(args.seed)
    train: list[dict] = []
    val: list[dict] = []
    by_map: dict[str, list[dict]] = {}
    for job in tune:
        by_map.setdefault(job["map"], []).append(job)
    for name, jobs in by_map.items():
        rng.shuffle(jobs)
        n_train = int(round(len(jobs) * float(args.train_frac)))
        n_train = min(max(n_train, 1), len(jobs) - 1) if len(jobs) > 1 else len(jobs)
        for j in jobs[:n_train]:
            j = dict(j)
            j["split"] = "train"
            train.append(j)
        for j in jobs[n_train:]:
            j = dict(j)
            j["split"] = "val"
            val.append(j)

    catalog = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "sealed_holdout": "HOLDOUT maps × r0–r7 only. Do NOT tune on these.",
            "train": "Stratified subset of tune pool (core octets + all expanded).",
            "val": "Held-out subset of tune pool for model selection.",
            "final_gate": "After tuning, evaluate sealed_holdout once.",
        },
        "n_routes_per_map": args.n_routes,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "core_scenarios": list(CORE_SCENARIOS),
        "holdout_scenarios": list(HOLDOUT_SCENARIOS),
        "per_map_route_counts": per_map_counts,
        "counts": {
            "maps": len(maps),
            "sealed_holdout": len(sealed),
            "train": len(train),
            "val": len(val),
            "tune_pool": len(train) + len(val),
            "total_jobs": len(sealed) + len(train) + len(val),
        },
        "jobs": sealed + train + val,
    }
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(catalog["counts"], indent=2))
    print(f"wrote {args.output}" if not args.dry_run else "dry-run (no write)")


if __name__ == "__main__":
    main()
