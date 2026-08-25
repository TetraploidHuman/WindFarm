"""
Terrain-based wind corridor experiment.
Uses terrain roughness to create wind corridors: middle band (y=8-16) = high roughness → slow wind.
North/South = low roughness → fast tailwind.
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
from windfarm.physics import downscale_wind
from windfarm.types import TerrainField

RUN = PROJECT / "runs" / "corridor"
RUN.mkdir(exist_ok=True)
LOG = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)

# ══════════════════════════════════════════════════════════════════
log("STEP 1: Build terrain with roughness corridor")
t0 = time.time()

# Load natural2 elevation, modify roughness
with open(str(PROJECT / "runs" / "natural2" / "terrain.json")) as f:
    base_terrain = json.load(f)["terrain"]

h, w_grid = len(base_terrain["elevation"]), len(base_terrain["elevation"][0])
levels = 5

new_roughness = []
new_slope = []
for y in range(h):
    rrow = []; srow = []
    for x in range(w_grid):
        if 8 <= y < 16:
            rrow.append(3.0); srow.append(0.06)   # dead zone middle
        else:
            rrow.append(0.02); srow.append(0.01)  # fast corridors
    new_roughness.append(rrow); new_slope.append(srow)

terrain_dict = {
    "elevation": base_terrain["elevation"],
    "slope": new_slope,
    "aspect": base_terrain["aspect"],
    "roughness": new_roughness,
}
terrain = TerrainField(**terrain_dict)
write_json(str(RUN / "terrain.json"), {"terrain": terrain_dict})
log(f"  Terrain: {h}x{w_grid}, middle roughness=0.85, corridor roughness=0.05")

# ══════════════════════════════════════════════════════════════════
log("STEP 2: Generate coarse_wind, truth, observations, training")
n_timesteps = 120
rng = random.Random(42)
start_time = datetime.fromisoformat("2026-03-21T08:30:00")

# Use uniform coarse wind — corridor comes from TERRAIN modulation
u_km_const, v_km_const, w_km_const = 4.0, -1.0, 0.0

coarse_wind = []
truth_fields = []
observations = []
training_samples = []

for ti in range(n_timesteps):
    ts = (start_time + timedelta(seconds=ti * 15)).isoformat(timespec="seconds")

    # Small temporal variation around constant wind
    uk = u_km_const + 0.3 * math.sin(ti / 30)
    vk = v_km_const + 0.2 * math.cos(ti / 35)
    wk = w_km_const + 0.05 * math.sin(ti / 40)

    # Coarse wind grid (6x8) — uniform pattern, corridor from terrain
    cw_grid = []
    for row in range(6):
        crow = []
        for col in range(8):
            u = uk + 0.3 * math.sin(row * 0.5 + col * 0.4)
            v = vk + 0.2 * math.cos(row * 0.4 + col * 0.5)
            crow.append({"u": u, "v": v, "w": wk, "speed": math.hypot(u, v)})
        cw_grid.append(crow)
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk, "grid": cw_grid})

    # Truth wind field = physics + random perturbations (ML can learn these)
    wind_field, _ = downscale_wind(uk, vk, terrain, wk, levels)
    u_layers = []; v_layers = []; w_layers = []
    for lv in range(levels):
        ug = [row[:] for row in wind_field.u[lv]]
        vg = [row[:] for row in wind_field.v[lv]]
        wg = [row[:] for row in wind_field.w[lv]]
        for y in range(h):
            for x in range(w_grid):
                # Add terrain-correlated perturbations (simulate local effects physics misses)
                local = 1.5 * math.sin(x*0.8 + y*0.6) * math.cos(y*0.3)
                ug[y][x] += local + rng.uniform(-0.3, 0.3)
                vg[y][x] += local*0.7 + rng.uniform(-0.2, 0.2)
                wg[y][x] += 0.3*math.sin(x*0.5 + y*0.7) + rng.uniform(-0.15, 0.15)
        u_layers.append(ug); v_layers.append(vg); w_layers.append(wg)
    truth_fields.append({"timestamp": ts, "u": u_layers, "v": v_layers, "w": w_layers})

    # Observation (1 per timestep)
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

    # Training samples (50 per timestep)
    for _ in range(50):
        x, y, z = rng.randint(0, w_grid - 1), rng.randint(0, h - 1), rng.randint(0, levels - 1)
        training_samples.append({
            "timestamp": ts, "x": x, "y": y, "z": z,
            "u_km": uk, "v_km": vk, "w_km": wk,
            "u_obs": u_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "v_obs": v_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "w_obs": w_layers[z][y][x] + rng.uniform(-0.05, 0.05),
        })

log(f"  Generated {n_timesteps} timesteps, {len(training_samples)} training samples in {time.time()-t0:.1f}s")

# Save
write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

# Config
cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 21)
cfg.mission.goal = (29, 2)
cfg.mission.max_steps = 100
cfg.simulation.report_keyframe_interval = 4
cfg.planner.horizon_steps = 15
cfg.planner.terminal_progress_weight = 60.0
write_json(str(RUN / "config.json"), cfg.to_dict())

# ══════════════════════════════════════════════════════════════════
log("STEP 3: Train XGBoost model")
t0 = time.time()

model_cfg = ModelConfig(
    model_type="xgboost", learning_rate=0.15, num_estimators=50, max_depth=3,
    feature_subsample_ratio=0.6, max_training_samples=4000, altitude_levels=levels,
    early_stopping_rounds=10, validation_ratio=0.2,
)

pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=model_cfg)
train_samples = training_samples_from_dict(training_samples)
metrics = pipeline.train(train_samples)
pipeline.save_model(str(RUN / "model.json"))
write_json(str(RUN / "metrics.json"), metrics)

phys = metrics.get("rmse_physics", 0)
final = metrics.get("rmse_final", 0)
reduction = (1 - final / phys) * 100 if phys > 0 else 0
log(f"  rmse_physics={phys:.4f} → rmse_final={final:.4f} ({reduction:.1f}% reduction)")
log(f"  Trained in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════
log("STEP 4: Run mission")
t0 = time.time()

cfg2 = load_task_config(str(RUN / "config.json"))
cfg2.model.model_type = "xgboost"
cfg2.simulation.report_keyframe_interval = 4
cfg2.planner.horizon_steps = 15
cfg2.planner.terminal_progress_weight = 60.0

pipeline2 = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), str(RUN / "model.json"), cfg2.model)
cw_list = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by = {o.timestamp: o for o in obs_list}
truth_by = {t["timestamp"]: t for t in truth_fields}

log(f"  Mission: {cfg2.mission.start} → {cfg2.mission.goal}")
engine = NavigationEngine(pipeline2, cfg2)
ctx = engine.create_context(cfg2.mission.start, cfg2.mission.goal)

last_log = 0
for i, s in enumerate(cw_list):
    engine.step(ctx, s, obs_by.get(s.timestamp), truth_by.get(s.timestamp))
    if i - last_log >= 10:
        elapsed = time.time() - t0
        log(f"  step {i+1}/{len(cw_list)} ({elapsed:.0f}s) batt={ctx.state.battery_ratio*100:.1f}%")
        last_log = i
    if _goal_reached(ctx.state, cfg2.mission.goal):
        log(f"  GOAL at step {i+1}!")
        break
    if ctx.battery_j <= 0:
        log(f"  Battery dead at step {i+1}")
        break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# Path stats
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
sp, gp = cfg2.mission.start[:2], cfg2.mission.goal[:2]
sd = math.hypot(gp[0] - sp[0], gp[1] - sp[1])
ad = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
ys = [p[1] for p in path]

log(f"  Done in {elapsed:.0f}s: steps={len(ctx.trace)} goal={_goal_reached(ctx.state, cfg2.mission.goal)} batt={ctx.state.battery_ratio*100:.1f}%")
log(f"  Path: straight={sd:.1f} actual={ad:.1f} ({ad/sd:.3f}x)")
log(f"  Y: start={path[0][1]:.1f} → end={path[-1][1]:.1f} max={max(ys):.1f} min={min(ys):.1f}")

# Check path deviation from straight line
straight_y_at_x = lambda x: sp[1] + (x - sp[0]) * (gp[1] - sp[1]) / (gp[0] - sp[0])
deviations = [abs(p[1] - straight_y_at_x(p[0])) for p in path]
log(f"  Max deviation from straight: {max(deviations):.2f} cells")
log(f"  Mean deviation: {np.mean(deviations):.2f} cells")

# ══════════════════════════════════════════════════════════════════
log("STEP 5: Generate charts")
t0 = time.time()
import subprocess
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
log(f"  Charts: {'OK' if result.returncode == 0 else 'FAILED'} in {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════════
log("=" * 50)
log("ALL DONE")
log(f"  RMSE: {phys:.4f} → {final:.4f} ({reduction:.1f}% reduction)")
log(f"  Mission: {len(ctx.trace)} steps, goal={'YES' if _goal_reached(ctx.state, cfg2.mission.goal) else 'NO'}")
log(f"  Path: straight={sd:.1f} actual={ad:.1f} ({ad/sd:.3f}x)")
log(f"  Deviation: mean={np.mean(deviations):.2f} max={max(deviations):.2f} cells")

with open(str(RUN / "pipeline.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
