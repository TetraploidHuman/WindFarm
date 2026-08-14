"""
Synthetic wind corridor experiment — full pipeline (simplified).
"""
import json, math, os, shutil, sys, time, random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from windfarm.config import load_task_config
from windfarm.pipeline import WindFarmPipeline, train_from_files
from windfarm.execution import NavigationEngine, _goal_reached
from windfarm.io import read_json, write_json, coarse_samples_from_dict, observations_from_dict

RUN_DIR = PROJECT / "runs" / "corridor"
RUN_DIR.mkdir(exist_ok=True)
CFG = load_task_config("config.json")
# 路线: 从西北→东南, 穿过中央逆风带
CFG.mission.start = (2, 21)
CFG.mission.goal = (29, 2)

# ── Step 1: Copy terrain ────────────────────────────────────────
print("Step 1: Copying terrain...")
shutil.copy("runs/natural2/terrain.json", str(RUN_DIR / "terrain.json"))
terrain_data = read_json(str(RUN_DIR / "terrain.json"))["terrain"]
h, w_grid = len(terrain_data["elevation"]), len(terrain_data["elevation"][0])
print(f"  terrain: {h}x{w_grid}")

# ── Step 2: Generate synthetic wind data ─────────────────────────
print("Step 2: Generating synthetic wind data...")
n_steps = 120
levels = CFG.model.altitude_levels
start_time = datetime.fromisoformat("2026-03-21T08:30:00")
t0 = time.time()

coarse_wind = []
truth_fields = []
training_samples = []
observations = []
rng = random.Random(42)

for ti in range(n_steps):
    ts = (start_time + timedelta(seconds=ti * 15)).isoformat(timespec="seconds")

    # Grid averages for coarse wind scalar
    grid_rows, grid_cols = 6, 8
    cw_grid = []
    for row in range(grid_rows):
        crow = []
        for col in range(grid_cols):
            cy = (row + 0.5) * h / grid_rows
            if 8 <= cy < 16:
                ub, vb = -7.5, 1.5  # central headwind
            elif cy < 8:
                ub, vb = 5.0, -1.0  # north tailwind corridor
            else:
                ub, vb = 5.5, -1.2  # south tailwind corridor
            u = ub + 0.5 * math.sin(cy * 0.7 + ti / 30)
            v = vb + 0.4 * math.cos(cy * 0.6 + ti / 35)
            w = 0.1 * math.sin(cy * 0.5 + ti / 40)
            crow.append({"u": u, "v": v, "w": w, "speed": math.hypot(u, v)})
        cw_grid.append(crow)

    uk = float(np.mean([c["u"] for r in cw_grid for c in r]))
    vk = float(np.mean([c["v"] for r in cw_grid for c in r]))
    wk = float(np.mean([c["w"] for r in cw_grid for c in r]))

    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk, "grid": cw_grid})

    # Truth wind field
    u_layers, v_layers, w_layers = [], [], []
    for level in range(levels):
        ug = np.zeros((h, w_grid))
        vg = np.zeros((h, w_grid))
        wg = np.zeros((h, w_grid))
        for y in range(h):
            for x in range(w_grid):
                if 8 <= y < 16:
                    ub, vb, wb = -7.5, 1.5, 0.0
                elif y < 8:
                    ub, vb, wb = 5.0, -1.0, -0.2
                else:
                    ub, vb, wb = 5.5, -1.2, 0.3
                lf = 0.7 + 0.3 * (level / max(levels - 1, 1))
                ug[y, x] = ub * lf + 0.5 * math.sin(x * 0.6 + y * 0.4 + ti / 25)
                vg[y, x] = vb * lf + 0.4 * math.cos(x * 0.5 + y * 0.5 + ti / 30)
                wg[y, x] = wb * lf + 0.2 * math.sin(x * 0.4 + y * 0.3 + ti / 20)
        u_layers.append(ug.tolist())
        v_layers.append(vg.tolist())
        w_layers.append(wg.tolist())

    truth_fields.append({"timestamp": ts, "u": u_layers, "v": v_layers, "w": w_layers})

    # Generate ~70 training samples per timestep (8400 total for 120 steps)
    for _ in range(70):
        x = rng.randint(0, w_grid - 1)
        y = rng.randint(0, h - 1)
        z = rng.randint(0, levels - 1)
        training_samples.append({
            "timestamp": ts, "x": x, "y": y, "z": z,
            "u_km": uk, "v_km": vk, "w_km": wk,
            "u_obs": u_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "v_obs": v_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "w_obs": w_layers[z][y][x] + rng.uniform(-0.05, 0.05),
        })

    # One observation per timestep
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

print(f"  generated {n_steps} timesteps, {len(training_samples)} training samples, {len(observations)} obs in {time.time()-t0:.1f}s")

# ── Step 3: Save data ────────────────────────────────────────────
print("Step 3: Saving data...")
write_json(str(RUN_DIR / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN_DIR / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN_DIR / "training.json"), {"samples": training_samples})
write_json(str(RUN_DIR / "observations.json"), {"observations": observations})
shutil.copy("config.json", str(RUN_DIR / "config.json"))
# Update config start/goal
cfg_data = read_json("config.json")
cfg_data["mission"]["start"] = [2, 21]
cfg_data["mission"]["goal"] = [29, 2]
write_json(str(RUN_DIR / "config.json"), cfg_data)

# ── Step 4: Train model ──────────────────────────────────────────
print("Step 4: Training XGBoost model...")
t0 = time.time()
metrics = train_from_files(
    str(RUN_DIR / "terrain.json"),
    str(RUN_DIR / "training.json"),
    str(RUN_DIR / "model.json"),
    str(RUN_DIR / "metrics.json"),
    model_config=CFG.model,
)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  rmse_physics={metrics.get('rmse_physics', '?'):.4f} → rmse_final={metrics.get('rmse_final', '?'):.4f}")

# ── Step 5: Run mission ──────────────────────────────────────────
print("Step 5: Running mission...")
t0 = time.time()
pipeline = WindFarmPipeline.from_paths(
    str(RUN_DIR / "terrain.json"),
    str(RUN_DIR / "model.json"),
    CFG.model,
)
coarse_samples = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by_time = {o.timestamp: o for o in obs_list}
truth_by_time = {t["timestamp"]: t for t in truth_fields}

engine = NavigationEngine(pipeline, CFG)
context = engine.create_context(CFG.mission.start, CFG.mission.goal)

for i, sample in enumerate(coarse_samples):
    obs = obs_by_time.get(sample.timestamp)
    truth = truth_by_time.get(sample.timestamp)
    engine.step(context, sample, obs, truth)
    if _goal_reached(context.state, CFG.mission.goal) or context.battery_j <= 0.0:
        print(f"  {'goal!' if _goal_reached(context.state, CFG.mission.goal) else 'battery dead'} at step {i+1}")
        break

report = engine.build_report(context, CFG.mission.start, CFG.mission.goal)
write_json(str(RUN_DIR / "mission_report.json"), report)
print(f"  mission done in {time.time()-t0:.1f}s")
print(f"  steps={len(context.trace)} goal={_goal_reached(context.state, CFG.mission.goal)} batt={context.state.battery_ratio*100:.1f}%")

# ── Step 6: Print key metrics ────────────────────────────────────
print(f"\n{'='*50}")
print("DONE. Key results:")
print(f"  RMSE: {metrics['rmse_physics']:.4f} → {metrics['rmse_final']:.4f} ({(1-metrics['rmse_final']/metrics['rmse_physics'])*100:.1f}% reduction)")
print(f"  Mission: {len(context.trace)} steps, goal={'YES' if _goal_reached(context.state, CFG.mission.goal) else 'NO'}")
print(f"  Battery: {context.state.battery_ratio*100:.1f}% remaining")
print(f"  Data saved to {RUN_DIR}/")
