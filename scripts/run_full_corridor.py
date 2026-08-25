"""
Full corridor wind experiment — end-to-end pipeline.
1. Generate synthetic wind corridor data
2. Generate training samples + train XGBoost
3. Run mission with corridor wind
4. Generate all 24 charts
"""
import json, math, os, shutil, sys, time, random, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from windfarm.config import load_task_config, ModelConfig
from windfarm.pipeline import WindFarmPipeline
from windfarm.execution import NavigationEngine, _goal_reached
from windfarm.io import read_json, write_json, coarse_samples_from_dict, observations_from_dict, training_samples_from_dict

RUN = PROJECT / "runs" / "corridor"
RUN.mkdir(exist_ok=True)
LOG = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)

# ══════════════════════════════════════════════════════════════════
log("=" * 50)
log("STEP 1: Generate synthetic wind corridor data")
t0 = time.time()

h, w_grid, levels = 24, 32, 5
n_timesteps = 120
rng = random.Random(42)
start_time = datetime.fromisoformat("2026-03-21T08:30:00")

# ── Terrain (copy from natural2) ──
shutil.copy(str(PROJECT / "runs" / "natural2" / "terrain.json"), str(RUN / "terrain.json"))
terrain_data = read_json(str(RUN / "terrain.json"))["terrain"]

# ── Generate coarse_wind + truth + observations + training ──
log(f"Generating {n_timesteps} timesteps of wind corridor data...")
coarse_wind = []
truth_fields = []
observations = []
training_samples = []

for ti in range(n_timesteps):
    ts = (start_time + timedelta(seconds=ti * 15)).isoformat(timespec="seconds")

    # --- Coarse wind grid (6x8) ---
    cw_grid = []
    for row in range(6):
        crow = []
        for col in range(8):
            cy = (row + 0.5) * h / 6
            if 8 <= cy < 16:   ub, vb = -7.5, 1.5   # central headwind
            elif cy < 8:        ub, vb =  5.0, -1.0  # north tailwind
            else:               ub, vb =  5.5, -1.2  # south tailwind
            u = ub + 0.5 * math.sin(cy * 0.7 + ti / 30)
            v = vb + 0.4 * math.cos(cy * 0.6 + ti / 35)
            crow.append({"u": u, "v": v, "w": 0.1 * math.sin(cy * 0.5 + ti / 40),
                         "speed": math.hypot(u, v)})
        cw_grid.append(crow)

    uk = float(np.mean([c["u"] for r in cw_grid for c in r]))
    vk = float(np.mean([c["v"] for r in cw_grid for c in r]))
    wk = float(np.mean([c["w"] for r in cw_grid for c in r]))
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk, "grid": cw_grid})

    # --- Truth wind field (5 levels x 24 x 32) ---
    u_layers, v_layers, w_layers = [], [], []
    for level in range(levels):
        ug = np.zeros((h, w_grid)); vg = np.zeros((h, w_grid)); wg = np.zeros((h, w_grid))
        for y in range(h):
            for x in range(w_grid):
                if 8 <= y < 16:   ub, vb, wb = -7.5, 1.5, 0.0
                elif y < 8:        ub, vb, wb =  5.0, -1.0, -0.2
                else:               ub, vb, wb =  5.5, -1.2,  0.3
                lf = 0.7 + 0.3 * (level / max(levels - 1, 1))
                ug[y, x] = ub * lf + 0.5 * math.sin(x * 0.6 + y * 0.4 + ti / 25)
                vg[y, x] = vb * lf + 0.4 * math.cos(x * 0.5 + y * 0.5 + ti / 30)
                wg[y, x] = wb * lf + 0.2 * math.sin(x * 0.4 + y * 0.3 + ti / 20)
        u_layers.append(ug.tolist()); v_layers.append(vg.tolist()); w_layers.append(wg.tolist())
    truth_fields.append({"timestamp": ts, "u": u_layers, "v": v_layers, "w": w_layers})

    # --- Observations (1 per timestep) ---
    sx = min(w_grid - 4, 5 + ti // 10)
    sy = min(h - 4, 7 + ti // 12)
    sz = (ti // 45) % max(levels, 1)
    observations.append({
        "timestamp": ts, "x": sx, "y": sy, "z": sz,
        "u_obs": u_layers[sz][sy][sx] + rng.uniform(-0.05, 0.05),
        "v_obs": v_layers[sz][sy][sx] + rng.uniform(-0.05, 0.05),
        "w_obs": w_layers[sz][sy][sx] + rng.uniform(-0.04, 0.04),
        "airspeed": 13.6 + rng.uniform(-0.4, 0.4),
        "ground_speed": 14.1 + rng.uniform(-0.8, 1.0),
        "climb_rate": rng.uniform(-0.35, 0.9),
        "acceleration": rng.uniform(-0.2, 0.2),
    })

    # --- Training samples (50 per timestep = 6000 total) ---
    for _ in range(50):
        x = rng.randint(0, w_grid - 1); y = rng.randint(0, h - 1); z = rng.randint(0, levels - 1)
        training_samples.append({
            "timestamp": ts, "x": x, "y": y, "z": z,
            "u_km": uk, "v_km": vk, "w_km": wk,
            "u_obs": u_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "v_obs": v_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "w_obs": w_layers[z][y][x] + rng.uniform(-0.05, 0.05),
        })

log(f"  Coarse wind: {len(coarse_wind)} entries")
log(f"  Truth fields: {len(truth_fields)} entries")
log(f"  Observations: {len(observations)} entries")
log(f"  Training samples: {len(training_samples)} samples")
log(f"  Generated in {time.time() - t0:.1f}s")

# --- Save ---
write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

# Update config
cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 21)
cfg.mission.goal = (29, 2)
cfg.mission.max_steps = 100
cfg.simulation.report_keyframe_interval = 4
write_json(str(RUN / "config.json"), cfg.to_dict())

# ══════════════════════════════════════════════════════════════════
log("STEP 2: Train XGBoost model")
t0 = time.time()

model_cfg = ModelConfig(
    model_type="xgboost",
    learning_rate=0.15,
    num_estimators=50,
    max_depth=3,
    feature_subsample_ratio=0.6,
    max_training_samples=4000,
    altitude_levels=levels,
    early_stopping_rounds=10,
    validation_ratio=0.2,
)
log(f"  Config: {model_cfg.model_type}, {model_cfg.num_estimators} trees, max_depth={model_cfg.max_depth}")

try:
    pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=model_cfg)
    train_samples = training_samples_from_dict(training_samples)
    log(f"  Loaded {len(train_samples)} TrainingSample objects")

    metrics = pipeline.train(train_samples)
    pipeline.save_model(str(RUN / "model.json"))
    write_json(str(RUN / "metrics.json"), metrics)

    phys = metrics.get("rmse_physics", 0)
    final = metrics.get("rmse_final", 0)
    reduction = (1 - final / phys) * 100 if phys > 0 else 0
    log(f"  rmse_physics={phys:.4f} → rmse_final={final:.4f}")
    log(f"  Reduction: {reduction:.1f}%")
    log(f"  Trained in {time.time() - t0:.1f}s")
except Exception as e:
    log(f"  ERROR during training: {e}")
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
log("STEP 3: Run mission")
t0 = time.time()

cfg2 = load_task_config(str(RUN / "config.json"))
cfg2.model.model_type = "xgboost"
cfg2.simulation.report_keyframe_interval = 4

pipeline2 = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), str(RUN / "model.json"), cfg2.model)
cw_list = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by = {o.timestamp: o for o in obs_list}
truth_by = {t["timestamp"]: t for t in truth_fields}

log(f"  Start: {cfg2.mission.start} → Goal: {cfg2.mission.goal}")
engine = NavigationEngine(pipeline2, cfg2)
ctx = engine.create_context(cfg2.mission.start, cfg2.mission.goal)

last_log = 0
for i, s in enumerate(cw_list):
    engine.step(ctx, s, obs_by.get(s.timestamp), truth_by.get(s.timestamp))
    if i - last_log >= 10:
        elapsed = time.time() - t0
        log(f"  step {i+1}/{len(cw_list)} ({elapsed:.0f}s)  batt={ctx.state.battery_ratio*100:.1f}%")
        last_log = i
    if _goal_reached(ctx.state, cfg2.mission.goal):
        log(f"  GOAL REACHED at step {i+1}!")
        break
    if ctx.battery_j <= 0:
        log(f"  Battery exhausted at step {i+1}")
        break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# Quick path stats
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
sp, gp = cfg2.mission.start[:2], cfg2.mission.goal[:2]
sd = math.hypot(gp[0] - sp[0], gp[1] - sp[1])
ad = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
ys = [p[1] for p in path]

log(f"  Mission done in {elapsed:.0f}s")
log(f"  Steps: {len(ctx.trace)}  Goal: {_goal_reached(ctx.state, cfg2.mission.goal)}  Batt: {ctx.state.battery_ratio*100:.1f}%")
log(f"  Path: straight={sd:.1f} actual={ad:.1f} ({ad/sd:.3f}x)")
log(f"  Y: {path[0][1]:.1f}→{path[-1][1]:.1f} max={max(ys):.1f} min={min(ys):.1f}")

# ══════════════════════════════════════════════════════════════════
log("STEP 4: Generate charts")
t0 = time.time()
import subprocess
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
if result.returncode == 0:
    log(f"  Charts generated in {time.time() - t0:.1f}s")
else:
    log(f"  Chart error: {result.stderr[:500]}")

# ══════════════════════════════════════════════════════════════════
log("=" * 50)
log("ALL DONE. Key results:")
log(f"  RMSE: {phys:.4f} → {final:.4f} ({reduction:.1f}% reduction)")
log(f"  Mission: {len(ctx.trace)} steps, goal={'YES' if _goal_reached(ctx.state, cfg2.mission.goal) else 'NO'}")
log(f"  Battery: {ctx.state.battery_ratio*100:.1f}%")
log(f"  Path: straight={sd:.1f} actual={ad:.1f} ({ad/sd:.3f}x)")
log(f"  Data: {RUN}/")
log(f"  Charts: {PROJECT}/charts/")

# Save log
with open(str(RUN / "pipeline.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
