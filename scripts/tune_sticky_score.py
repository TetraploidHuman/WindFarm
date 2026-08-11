#!/usr/bin/env python3
"""Tune logistic prior-sticky score on the train catalog (sealed holdout forbidden).

Each trial writes a sticky-score JSON and runs a train subsample via
``eval_multi_scenario_energy.py``. Objective: mean savings_vs_nominal_pct
minus a hard-lose penalty. Validation uses the catalog val split only.

Example smoke:
  .venv-linux/bin/python scripts/build_tune_catalog.py --n-routes 56
  .venv-linux/bin/python scripts/tune_sticky_score.py --trials 2 --train-limit 8 --workers 8

Full-ish:
  .venv-linux/bin/python scripts/tune_sticky_score.py --trials 40 --train-limit 256 --workers 14

Final gate (once):
  .venv-linux/bin/python scripts/tune_sticky_score.py --eval-only sealed_holdout \\
      --allow-sealed --score-json runs/sticky-tune-.../best_sticky_score.json
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

# Search box for logistic sticky weights.
BOUNDS = [
    ("bias", -2.0, 2.0),
    ("w_along", -1.5, 0.5),
    ("w_along2", -0.5, 0.1),
    ("w_abs_cross", -0.2, 1.0),
    ("w_truth_margin", 0.0, 20.0),
    ("w_remain", -0.5, 1.5),
    ("w_aspect", -0.5, 0.5),
    ("threshold", -1.5, 1.5),
]


def _sample_weights(rng: random.Random) -> dict:
    cfg = {"mode": "logistic"}
    for name, lo, hi in BOUNDS:
        cfg[name] = float(rng.uniform(lo, hi))
    return cfg


def _seed_weights() -> dict:
    seed_path = ROOT / "configs" / "sticky_score_seed.json"
    if seed_path.exists():
        cfg = json.loads(seed_path.read_text(encoding="utf-8"))
        cfg["mode"] = "logistic"
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
    # Scratch demo dirs are removed per-route; drop the empty parent too.
    scratch = output_dir / "scratch_runs"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    return summary


def _objective_from_summary(summary: dict) -> tuple[float, dict]:
    rows = [r for r in summary.get("results", []) if r.get("goal_reached")]
    if not rows:
        return -1e9, {"mean": None, "n": 0, "n_hard": 0}
    saves = [float(r["savings_vs_nominal_pct"]) for r in rows]
    mean = float(sum(saves) / len(saves))
    n_hard = sum(1 for s in saves if s < -5.0)
    score = mean - 8.0 * (n_hard / max(len(saves), 1)) * 100.0
    return score, {"mean": mean, "n": len(saves), "n_hard": n_hard, "min": min(saves)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "scenarios" / "tune_catalog_1024.json",
    )
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--train-limit", type=int, default=128)
    ap.add_argument("--val-limit", type=int, default=128)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--eval-only",
        choices=("train", "val", "sealed_holdout", "tune"),
        default=None,
    )
    ap.add_argument("--score-json", type=Path, default=None)
    ap.add_argument(
        "--allow-sealed",
        action="store_true",
        help="Permit --eval-only sealed_holdout (final gate only).",
    )
    args = ap.parse_args()

    if not args.catalog.exists():
        raise SystemExit(f"missing catalog {args.catalog}; run scripts/build_tune_catalog.py first")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (ROOT / "runs" / f"sticky-tune-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if args.eval_only == "sealed_holdout" and not args.allow_sealed:
            raise SystemExit("refusing sealed_holdout without --allow-sealed (final gate only)")
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
        score, meta = _objective_from_summary(summary)
        payload = {"split": args.eval_only, "objective": score, "meta": meta}
        print(json.dumps(payload, indent=2))
        (out_dir / "eval_only_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    rng = random.Random(args.seed)
    history: list[dict] = []
    best_score = -1e18
    best_cfg: dict | None = None

    # Trial 1 always evaluates the seed curve so we have a baseline.
    candidates = [_seed_weights()] + [_sample_weights(rng) for _ in range(max(0, args.trials - 1))]

    for n, cfg in enumerate(candidates, start=1):
        trial_dir = out_dir / f"trial_{n:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        score_path = trial_dir / "sticky_score.json"
        score_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        try:
            summary = _run_eval(
                catalog=args.catalog,
                split="train",
                limit=args.train_limit,
                workers=args.workers,
                score_json=score_path,
                output_dir=trial_dir / "eval",
                seed=args.seed + n,
            )
            score, meta = _objective_from_summary(summary)
        except subprocess.CalledProcessError as exc:
            score, meta = -1e9, {"error": str(exc)}
        row = {"trial": n, "objective": score, "meta": meta, "weights": cfg}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[trial {n}/{len(candidates)}] obj={score:.4f} meta={meta} "
            f"w_along={cfg.get('w_along', 0):.3f} margin={cfg.get('w_truth_margin', 0):.2f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_cfg = cfg

    assert best_cfg is not None
    best_path = out_dir / "best_sticky_score.json"
    best_path.write_text(json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8")

    val_summary = _run_eval(
        catalog=args.catalog,
        split="val",
        limit=args.val_limit,
        workers=args.workers,
        score_json=best_path,
        output_dir=out_dir / "val_eval",
        seed=args.seed + 999,
    )
    val_score, val_meta = _objective_from_summary(val_summary)
    report = {
        "best_weights": best_cfg,
        "train_best_objective": best_score,
        "val": {"objective": val_score, "meta": val_meta},
        "n_trials": len(candidates),
        "protocol_note": "Do not run sealed_holdout until ready for a single final gate (--allow-sealed).",
    }
    (out_dir / "tune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"best score json: {best_path}")


if __name__ == "__main__":
    main()
