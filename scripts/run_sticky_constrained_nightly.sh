#!/usr/bin/env bash
# Full constrained sticky retune + sealed gate + verdict.
# Intended to run under systemd --user (survives SSH disconnect).
set -euo pipefail

ROOT="/home/miaox99/WindFarm/WindFarm20260617/WindFarm"
cd "$ROOT"

export WINDFARM_SKIP_DASHBOARD=1
export LD_LIBRARY_PATH="/nix/store/f7pc134264jj4id4jnqpadzl0nnnayj7-ld-library-path/share/nix-ld/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ROOT/.venv-linux/bin:/run/current-system/sw/bin:$PATH"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/runs/sticky-tune-constrained-nightly-$STAMP"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== sticky constrained nightly pipeline start $(date -Is) ==="
echo "OUT=$OUT"
echo "$OUT" > /tmp/sticky_constrained_nightly_out.txt

PY="$ROOT/.venv-linux/bin/python"
CATALOG="$ROOT/scenarios/tune_catalog_1024.json"
OPT7E="$ROOT/runs/multi-scenario-energy-opt7e-128-20260811-210249/energy_summary.json"

echo "=== [1/3] constrained tune (16 trials, train400/val128, val-selected) ==="
"$PY" "$ROOT/scripts/tune_sticky_score_constrained.py" \
  --catalog "$CATALOG" \
  --trials 16 \
  --train-limit 400 \
  --val-limit 128 \
  --workers 14 \
  --out-dir "$OUT/tune"

BEST="$OUT/tune/best_sticky_score.json"
if [[ ! -f "$BEST" ]]; then
  echo "FATAL: missing $BEST" >&2
  exit 2
fi
echo "best weights: $BEST"
cp -f "$BEST" "$OUT/best_sticky_score.json"

echo "=== [2/3] sealed holdout final gate (128) ==="
"$PY" "$ROOT/scripts/tune_sticky_score_constrained.py" \
  --catalog "$CATALOG" \
  --eval-only sealed_holdout \
  --allow-sealed \
  --score-json "$BEST" \
  --workers 14 \
  --out-dir "$OUT/sealed"

SEALED_SUMMARY="$OUT/sealed/eval_sealed_holdout/energy_summary.json"
if [[ ! -f "$SEALED_SUMMARY" ]]; then
  echo "FATAL: missing sealed summary" >&2
  exit 3
fi

echo "=== [3/3] compare vs opt7e + write verdict ==="
"$PY" - <<PY
import json
from pathlib import Path

out = Path("$OUT")
opt7 = Path("$OPT7E")
sealed = json.loads(Path("$SEALED_SUMMARY").read_text())
new = {r["scenario"]: r for r in sealed["results"]}
base = {r["scenario"]: r for r in json.loads(opt7.read_text())["results"]}
keys = sorted(set(new) & set(base))
dpp = [new[k]["savings_vs_nominal_pct"] - base[k]["savings_vs_nominal_pct"] for k in keys]
mean_n = sum(new[k]["savings_vs_nominal_pct"] for k in keys) / len(keys)
mean_b = sum(base[k]["savings_vs_nominal_pct"] for k in keys) / len(keys)
hard_n = sum(1 for k in keys if new[k]["savings_vs_nominal_pct"] < -5)
hard_b = sum(1 for k in keys if base[k]["savings_vs_nominal_pct"] < -5)
rows = sorted(
    (
        new[k]["savings_vs_nominal_pct"] - base[k]["savings_vs_nominal_pct"],
        k,
        base[k]["savings_vs_nominal_pct"],
        new[k]["savings_vs_nominal_pct"],
    )
    for k in keys
)
heroes = [
    "taiwan_hills/r0",
    "shanxi_loess/r1",
    "shanxi_loess/r2",
    "hainan_coast/r2",
    "taiwan_hills/r6",
    "guizhou_karst/r4",
    "sichuan_foothills/r0",
]
hero_reg = []
for h in heroes:
    if h in new and h in base:
        d = new[h]["savings_vs_nominal_pct"] - base[h]["savings_vs_nominal_pct"]
        if d < -0.5:
            hero_reg.append({"scenario": h, "dpp": d, "base": base[h]["savings_vs_nominal_pct"], "new": new[h]["savings_vs_nominal_pct"]})

# Ship only if mean improves and no hero drops >0.5pp and hard does not increase.
ship = (mean_n > mean_b + 0.02) and (hard_n <= hard_b) and (not hero_reg)

tune_report = {}
tr = out / "tune" / "tune_report.json"
if tr.exists():
    tune_report = json.loads(tr.read_text())

verdict = {
    "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    "out_dir": str(out),
    "vs_opt7e": {
        "n": len(keys),
        "mean_opt7e": mean_b,
        "mean_logistic": mean_n,
        "dpp": mean_n - mean_b,
        "hard_opt7e": hard_b,
        "hard_logistic": hard_n,
        "worst5": [
            {"scenario": r[1], "base": r[2], "new": r[3], "dpp": r[0]} for r in rows[:5]
        ],
        "best5": [
            {"scenario": r[1], "base": r[2], "new": r[3], "dpp": r[0]} for r in rows[-5:]
        ],
    },
    "hero_regressions_gt_0p5pp": hero_reg,
    "ship_replace_opt7e_default": ship,
    "reason": (
        "mean up, hard not worse, no hero <-0.5pp"
        if ship
        else "keep opt7e legacy default; logistic did not clear ship bar"
    ),
    "best_weights": tune_report.get("best_weights"),
    "tune_final_val": tune_report.get("final_val"),
}
(out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verdict, indent=2))
print("SHIP" if ship else "NO_SHIP")
PY

echo "=== pipeline done $(date -Is) ==="
echo "artifacts: $OUT/verdict.json $OUT/best_sticky_score.json $LOG"
