"""Vertical corridor: x=10-22 headwind band, west/east tailwind corridors."""
import json, math, os, shutil, sys, time, random
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
from windfarm.controller import transition_energy_j

RUN = PROJECT / "runs" / "corridor"
RUN.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load real data ──
shutil.copy(str(PROJECT / "runs" / "natural2" / "terrain.json"), str(RUN / "terrain.json"))
terrain_data = read_json(str(RUN / "terrain.json"))["terrain"]
terrain = TerrainField(**terrain_data)
h = len(terrain.elevation); w_grid = len(terrain.elevation[0]); levels = 5

real_coarse = read_json(str(PROJECT / "runs" / "natural2" / "coarse_wind.json"))["coarse_wind"]
n_steps = min(120, len(real_coarse))
rng = random.Random(42)

# ── Vertical corridor truth ──
# x=10-22: strong headwind. x<10 & x>=22: tailwind corridors.
# Route: SW(2,2)→NE(29,21), straight line passes through x=10-22 headwind
# Optimal detour: go north around the headwind band (via x<10 corridor, then east via x>=22 corridor)
log("Generating vertical corridor truth...")
log("  Central BLOCK (x=10-22,y=8-16): HEADWIND  |  Surrounding: TAILWIND")

coarse_wind, truth_fields, observations, training_samples = [], [], [], []

for ti in range(n_steps):
    ts = real_coarse[ti]["timestamp"]
    uk = real_coarse[ti]["u_km"]; vk = real_coarse[ti]["v_km"]
    wk = real_coarse[ti].get("w_km", 0.0)
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk})

    phys, _ = downscale_wind(uk, vk, terrain, wk, levels)
    ul, vl, wl = [], [], []
    for lv in range(levels):
        ug = [r[:] for r in phys.u[lv]]
        vg = [r[:] for r in phys.v[lv]]
        wg_row = [r[:] for r in phys.w[lv]]
        lf = 0.7 + 0.3 * (lv / max(levels - 1, 1))
        for y in range(h):
            for x in range(w_grid):
                if 10 <= x < 22 and 8 <= y < 16:
                    cu, cv, cw = -5.0 * lf, 2.5 * lf, -0.3 * lf  # central headwind BLOCK
                else:
                    cu, cv, cw = 4.0 * lf, -2.0 * lf, 0.2 * lf   # surrounding tailwind
                ug[y][x] += cu + 0.5 * math.sin(x * 0.7 + y * 0.5) + rng.uniform(-0.3, 0.3)
                vg[y][x] += cv + 0.4 * math.cos(x * 0.6 + y * 0.6) + rng.uniform(-0.2, 0.2)
                wg_row[y][x] += cw + 0.2 * math.sin(x * 0.4 + y * 0.4) + rng.uniform(-0.15, 0.15)
        ul.append(ug); vl.append(vg); wl.append(wg_row)
    truth_fields.append({"timestamp": ts, "u": ul, "v": vl, "w": wl})

    sx = min(w_grid - 4, 5 + ti // 10); sy = min(h - 4, 7 + ti // 12)
    sz = (ti // 45) % max(levels, 1)
    observations.append({"timestamp": ts, "x": sx, "y": sy, "z": sz,
        "u_obs": ul[sz][sy][sx] + rng.uniform(-0.05, 0.05),
        "v_obs": vl[sz][sy][sx] + rng.uniform(-0.05, 0.05),
        "w_obs": wl[sz][sy][sx] + rng.uniform(-0.04, 0.04),
        "airspeed": 13.6 + rng.uniform(-0.4, 0.4), "ground_speed": 14.1 + rng.uniform(-0.8, 1.0),
        "climb_rate": rng.uniform(-0.35, 0.9), "acceleration": rng.uniform(-0.2, 0.2)})

    for _ in range(40):
        x, y, z = rng.randint(0, w_grid - 1), rng.randint(0, h - 1), rng.randint(0, levels - 1)
        training_samples.append({"timestamp": ts, "x": x, "y": y, "z": z,
            "u_km": uk, "v_km": vk, "w_km": wk,
            "u_obs": ul[z][y][x] + rng.uniform(-0.08, 0.08),
            "v_obs": vl[z][y][x] + rng.uniform(-0.08, 0.08),
            "w_obs": wl[z][y][x] + rng.uniform(-0.05, 0.05)})

# Verify
u0 = np.array(truth_fields[0]["u"][0])
block_u = u0[8:16, 10:22].mean()
surround_u = (u0.sum() - u0[8:16, 10:22].sum()) / (u0.size - u0[8:16, 10:22].size)
log(f"Truth: block(x=10-22,y=8-16) u={block_u:.1f}  surround u={surround_u:.1f}  ratio={surround_u/max(block_u,0.01):.1f}x")

write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

# Config: SW→NE, straight line crosses headwind at x=10-22
cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 20); cfg.mission.goal = (29, 4)
cfg.mission.max_steps = 80
cfg.simulation.report_keyframe_interval = 4
cfg.planner.horizon_steps = 25
cfg.planner.terminal_progress_weight = 30.0
write_json(str(RUN / "config.json"), cfg.to_dict())
log(f"Route: {cfg.mission.start} -> {cfg.mission.goal}")

# Train XGBoost
log("Training XGBoost...")
t0 = time.time()
model_cfg = ModelConfig(model_type="xgboost", learning_rate=0.15, num_estimators=60, max_depth=4,
    feature_subsample_ratio=0.6, max_training_samples=4000, altitude_levels=levels,
    early_stopping_rounds=12, validation_ratio=0.2)
pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=model_cfg)
metrics = pipeline.train(training_samples_from_dict(training_samples))
pipeline.save_model(str(RUN / "model.json"))
write_json(str(RUN / "metrics.json"), metrics)
phys = metrics.get("rmse_physics", 0); final = metrics.get("rmse_final", 0)
log(f"RMSE: {phys:.4f} -> {final:.4f} ({(1-final/phys)*100:.1f}%) in {time.time()-t0:.0f}s")

# Run mission
log("Running mission...")
t0 = time.time()
cfg2 = load_task_config(str(RUN / "config.json"))
cfg2.model.model_type = "xgboost"
cfg2.planner.horizon_steps = 14
cfg2.planner.terminal_progress_weight = 50.0
pipeline2 = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), str(RUN / "model.json"), cfg2.model)
cw_list = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by = {o.timestamp: o for o in obs_list}
truth_by = {t["timestamp"]: t for t in truth_fields}
engine = NavigationEngine(pipeline2, cfg2)
ctx = engine.create_context(cfg2.mission.start, cfg2.mission.goal)

for i, s in enumerate(cw_list):
    engine.step(ctx, s, obs_by.get(s.timestamp), truth_by.get(s.timestamp))
    if i % 15 == 0:
        log(f"  step {i+1} ({time.time()-t0:.0f}s) batt={ctx.state.battery_ratio*100:.1f}%")
    if _goal_reached(ctx.state, cfg2.mission.goal):
        log(f"  GOAL at step {i+1}!"); break
    if ctx.battery_j <= 0:
        log(f"  BATT DEAD at step {i+1}"); break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# Energy comparison
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
sp, gp = cfg2.mission.start[:2], cfg2.mission.goal[:2]
sd = math.hypot(gp[0] - sp[0], gp[1] - sp[1])
ad = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
phys_wind = [(f.get("physics_at_drone", {}).get("u", 0), f.get("physics_at_drone", {}).get("v", 0), f.get("physics_at_drone", {}).get("w", 0)) for f in ctx.trace]
sd_m = report["grid"]["resolution_m"]; alt_m = report["mission"]["altitude_step_m"]
n = len(ctx.trace)

def path_energy(pts, wind, n_steps):
    e = 0.0
    for i in range(1, len(pts)):
        u, v, w = wind[min(i - 1, len(wind) - 1)]
        e += transition_energy_j(airspeed=16.5, current=(pts[i-1][0], pts[i-1][1], 0.0),
            nxt=(pts[i][0], pts[i][1], 0.0), local_u=u, local_v=v, local_w=w,
            step_distance_m=sd_m, altitude_step_m=alt_m,
            climb_cost_per_level_j=180.0, hover_power_w=105.0, cruise_power_w=150.0,
            hotel_power_w=18.0, headwind_power_per_mps_w=14.0, climb_power_per_mps_w=125.0,
            descent_power_reduction_per_mps_w=58.0)
    return e

straight_pts = [(sp[0] + i/n*(gp[0]-sp[0]), sp[1] + i/n*(gp[1]-sp[1])) for i in range(n+1)]
se = path_energy(straight_pts, phys_wind, n)
ae = path_energy(path, phys_wind, n)

xs = [p[0] for p in path]; ys = [p[1] for p in path]
extra_dist = (ad - sd) / sd * 100
savings = (se - ae) / se * 100

log(f"Steps={n} Goal={_goal_reached(ctx.state, cfg2.mission.goal)} Batt={ctx.state.battery_ratio*100:.1f}%")
log(f"Straight={sd:.1f} Actual={ad:.1f} Extra={extra_dist:+.1f}%")
log(f"Energy: straight={se:.0f}J actual={ae:.0f}J Savings={savings:+.1f}%")
log(f"Path x: {xs[0]:.1f}->{xs[-1]:.1f} max={max(xs):.1f} min={min(xs):.1f}")
log(f"Path y: {ys[0]:.1f}->{ys[-1]:.1f} max={max(ys):.1f} min={min(ys):.1f}")

# Generate charts
log("Generating charts...")
import subprocess
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
log(f"Charts: {'OK' if result.returncode == 0 else 'FAILED'}")

log(f"\nFINAL: RMSE {phys:.4f}->{final:.4f} ({(1-final/phys)*100:.1f}%) | Extra dist {extra_dist:+.1f}% | Energy savings {savings:+.1f}%")
