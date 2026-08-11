#!/usr/bin/env python3
"""Constrained sticky-score retune (val-selected; sealed holdout forbidden).

Changes vs the first free search:
  - Bounds: w_along<=0, w_along2<=0, threshold>=0 (no permissive headwind sticky)
  - Planner also hard-vetoes outside the mild-along band
  - Select best by **val** objective, not train
  - Each trial evaluates train + val; hero-proxy penalty vs opt7e when present

Example:
  .venv-linux/bin/python scripts/tune_sticky_score_constrained.py \\
      --trials 16 --train-limit 400 --val-limit 128 --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_multi_scenario_energy.py"
PYTHON = ROOT / ".venv-linux" / "bin" / "python"
OPT7E = ROOT / "runs" / "multi-scenario-energy-opt7e-128-20260811-210249" / "energy_summary.json"

# Physically constrained box (no positive along weight, no negative threshold).
BOUNDS = [
    ("bias", -1.0, 1.5),
    ("w_along", -1.2, 0.0),
    ("w_along2", -0.4, 0.0),
    ("w_abs_cross", 0.0, 0.8),
    ("w_truth_margin", 1.0, 12.0),
    ("w_remain", 0.0, 1.0),
    ("w_aspect", -0.3, 0.3),
    ("threshold", 0.0, 1.2),
]

# Proxies that exist in tune pool / sealed; penalize regressions vs opt7e when present.
HERO_SCENARIOS = (
    "taiwan_hills/r0",
    "taiwan_hills/r6",
    "shanxi_loess/r1",
    "shanxi_loess/r2",
    "hainan_coast/r2",
    "hainan_coast/r3",
    "guizhou_karst/r4",
    "sichuan_foothills/r0",
    "fujian_hills/r2",
)


def _sample_weights(rng: random.Random) -> dict:
    cfg = {"mode": "logistic"}
    for name, lo, hi in BOUNDS:
        cfg[name] = float(rng.uniform(lo, hi))
    return cfg


def _seed_weights() -> dict:
    path = ROOT / "configs" / "sticky_score_constrained_seed.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["mode"] = "logistic"
        # Clamp into constrained box.
        for name, lo, hi in BOUNDS:
            if name in cfg:
                cfg[name] = float(min(hi, max(lo, float(cfg[name]))))
        return cfg
    return _sample_weights(random.Random(0))


def _run_eval(
    *,
    catalog: Path,
    split: str,
    limit: int | None,
    workers: int,
    score_json: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(EVAL),
        "--catalog",
        str(catalog),
        "--split",
        split,
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
        "--runs-dir",
        str(output_dir / "scratch_runs"),
        "--cleanup-runs",
        "--seed",
        str(seed),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    env = dict(os.environ)
    env["WINDFARM_STICKY_SCORE_JSON"] = str(score_json)
    env.setdefault("WINDFARM_SKIP_DASHBOARD", "1")
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/nix/store/f7pc134264jj4id4jnqpadzl0nnnayj7-ld-library-path/share/nix-ld/lib",
    )
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads((output_dir / "energy_summary.json").read_text(encoding="utf-8"))
    scratch = output_dir / "scratch_runs"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    return summary


def _load_opt7e_baseline() -> dict[str, float]:
    if not OPT7E.exists():
        return {}
    rows = json.loads(OPT7E.read_text(encoding="utf-8"))["results"]
    return {r["scenario"]: float(r["savings_vs_nominal_pct"]) for r in rows}


def _objective_from_summary(
    summary: dict,
    *,
    baseline: dict[str, float],
    hero_slack_pp: float = 0.5,
) -> tuple[float, dict]:
    rows = [r for r in summary.get("results", []) if r.get("goal_reached")]
    if not rows:
        return -1e9, {"mean": None, "n": 0, "n_hard": 0, "hero_pen": 0.0}
    saves = [float(r["savings_vs_nominal_pct"]) for r in rows]
    mean = float(sum(saves) / len(saves))
    n_hard = sum(1 for s in saves if s < -5.0)
    # Soft hard-lose penalty (lighter than v1 so mean can matter; hard still costly).
    score = mean - 4.0 * (n_hard / max(len(saves), 1)) * 100.0

    hero_pen = 0.0
    hero_hits = []
    by_scen = {r["scenario"]: float(r["savings_vs_nominal_pct"]) for r in rows}
    for scen in HERO_SCENARIOS:
        if scen not in by_scen or scen not in baseline:
            continue
        d = by_scen[scen] - baseline[scen]
        if d < -hero_slack_pp:
            # Quadratic-ish: -2pp → 8, -1pp → 2 beyond slack
            over = -d - hero_slack_pp
            hero_pen += 8.0 * over * over
            hero_hits.append({"scenario": scen, "dpp": d})
    score -= hero_pen
    return score, {
        "mean": mean,
        "n": len(saves),
        "n_hard": n_hard,
        "min": min(saves),
        "hero_pen": hero_pen,
        "hero_hits": hero_hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=ROOT / "scenarios" / "tune_catalog_1024.json")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--train-limit", type=int, default=400)
    ap.add_argument("--val-limit", type=int, default=128)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--eval-only",
        choices=("train", "val", "sealed_holdout", "tune"),
        default=None,
    )
    ap.add_argument("--score-json", type=Path, default=None)
    ap.add_argument("--allow-sealed", action="store_true")
    args = ap.parse_args()

    if not args.catalog.exists():
        raise SystemExit(f"missing catalog {args.catalog}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (ROOT / "runs" / f"sticky-tune-constrained-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_opt7e_baseline()

    if args.eval_only:
        if args.eval_only == "sealed_holdout" and not args.allow_sealed:
            raise SystemExit("refusing sealed_holdout without --allow-sealed")
        if args.score_json is None:
            raise SystemExit("--score-json required with --eval-only")
        split_dir = out_dir / f"eval_{args.eval_only}"
        summary = _run_eval(
            catalog=args.catalog,
            split=args.eval_only,
            limit=None if args.eval_only == "sealed_holdout" else args.val_limit,
            workers=args.workers,
            score_json=args.score_json,
            output_dir=split_dir,
            seed=args.seed,
        )
        score, meta = _objective_from_summary(summary, baseline=baseline)
        payload = {"split": args.eval_only, "objective": score, "meta": meta}
        print(json.dumps(payload, indent=2))
        (out_dir / "eval_only_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    rng = random.Random(args.seed)
    history: list[dict] = []
    best_val = -1e18
    best_cfg: dict | None = None

    candidates = [_seed_weights()] + [_sample_weights(rng) for _ in range(max(0, args.trials - 1))]

    for n, cfg in enumerate(candidates, start=1):
        trial_dir = out_dir / f"trial_{n:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        score_path = trial_dir / "sticky_score.json"
        score_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        try:
            train_summary = _run_eval(
                catalog=args.catalog,
                split="train",
                limit=args.train_limit,
                workers=args.workers,
                score_json=score_path,
                output_dir=trial_dir / "train_eval",
                seed=args.seed + n,
            )
            train_obj, train_meta = _objective_from_summary(train_summary, baseline=baseline)
            val_summary = _run_eval(
                catalog=args.catalog,
                split="val",
                limit=args.val_limit,
                workers=args.workers,
                score_json=score_path,
                output_dir=trial_dir / "val_eval",
                seed=args.seed + 10_000 + n,
            )
            val_obj, val_meta = _objective_from_summary(val_summary, baseline=baseline)
        except subprocess.CalledProcessError as exc:
            train_obj, train_meta = -1e9, {"error": str(exc)}
            val_obj, val_meta = -1e9, {"error": str(exc)}

        row = {
            "trial": n,
            "train_objective": train_obj,
            "val_objective": val_obj,
            "train_meta": train_meta,
            "val_meta": val_meta,
            "weights": cfg,
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[trial {n}/{len(candidates)}] val={val_obj:.3f} "
            f"(mean={val_meta.get('mean')} hard={val_meta.get('n_hard')} hero_pen={val_meta.get('hero_pen')}) "
            f"train={train_obj:.3f} w_along={cfg.get('w_along'):.3f} thr={cfg.get('threshold'):.3f}",
            flush=True,
        )
        if val_obj > best_val:
            best_val = val_obj
            best_cfg = cfg

    assert best_cfg is not None
    best_path = out_dir / "best_sticky_score.json"
    best_path.write_text(json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8")

    # Full val re-check of winner (same split, fresh subsample seed).
    final_val = _run_eval(
        catalog=args.catalog,
        split="val",
        limit=args.val_limit,
        workers=args.workers,
        score_json=best_path,
        output_dir=out_dir / "final_val",
        seed=args.seed + 777,
    )
    final_obj, final_meta = _objective_from_summary(final_val, baseline=baseline)
    report = {
        "protocol": "constrained + val-selected; sealed not used",
        "best_weights": best_cfg,
        "best_val_objective_during_search": best_val,
        "final_val": {"objective": final_obj, "meta": final_meta},
        "n_trials": len(candidates),
        "bounds": {name: [lo, hi] for name, lo, hi in BOUNDS},
    }
    (out_dir / "tune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"best score json: {best_path}")


if __name__ == "__main__":
    main()
