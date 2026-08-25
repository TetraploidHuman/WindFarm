"""Final experiment: path-aware estimate + belief energy bonus + full charts."""
import json, math, os, shutil, sys, time, random, subprocess
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
from windfarm.types import TerrainField, Observation, WindField
from windfarm.belief import BeliefUpdater
from windfarm.controller import transition_energy_j

RUN = PROJECT / "runs" / "corridor"
RUN.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════
log("STEP 1: Load real data + generate corridor truth")
shutil.copy(str(PROJECT / "runs" / "natural2" / "terrain.json"), str(RUN / "terrain.json"))
terrain_data = read_json(str(RUN / "terrain.json"))["terrain"]
terrain = TerrainField(**terrain_data)
h, w_grid, levels = len(terrain.elevation), len(terrain.elevation[0]), 5

real_coarse = read_json(str(PROJECT / "runs" / "natural2" / "coarse_wind.json"))["coarse_wind"]
n_steps = min(120, len(real_coarse))
rng = random.Random(42)

# Central block: x=10-22, y=8-16 = strong headwind. Surround = tailwind.
log("Central BLOCK (x=10-22, y=8-16): HEADWIND | Surround: TAILWIND")

coarse_wind, truth_fields, observations, training_samples = [], [], [], []
for ti in range(n_steps):
    ts = real_coarse[ti]["timestamp"]
    uk = real_coarse[ti]["u_km"]; vk = real_coarse[ti]["v_km"]; wk = real_coarse[ti].get("w_km", 0.0)
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk})

    phys, _ = downscale_wind(uk, vk, terrain, wk, levels)
    ul, vl, wl = [], [], []
    for lv in range(levels):
        ug = [r[:] for r in phys.u[lv]]; vg = [r[:] for r in phys.v[lv]]; wg_row = [r[:] for r in phys.w[lv]]
        lf = 0.7 + 0.3 * (lv / max(levels - 1, 1))
        for y in range(h):
            for x in range(w_grid):
                if 14 <= x < 18 and 8 <= y < 16:
                    cu, cv, cw = -15.0 * lf, 8.0 * lf, -0.5 * lf
                else:
                    cu, cv, cw = 4.0 * lf, -2.0 * lf, 0.2 * lf
                ug[y][x] += cu + 0.5 * math.sin(x * 0.7 + y * 0.5) + rng.uniform(-0.3, 0.3)
                vg[y][x] += cv + 0.4 * math.cos(x * 0.6 + y * 0.6) + rng.uniform(-0.2, 0.2)
                wg_row[y][x] += cw + 0.2 * math.sin(x * 0.4 + y * 0.4) + rng.uniform(-0.15, 0.15)
        ul.append(ug); vl.append(vg); wl.append(wg_row)
    truth_fields.append({"timestamp": ts, "u": ul, "v": vl, "w": wl})

    sx = min(w_grid - 4, 5 + ti // 10); sy = min(h - 4, 7 + ti // 12); sz = (ti // 45) % max(levels, 1)
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

block_u = np.array(truth_fields[0]["u"][0])[8:16, 10:22].mean()
surround_u = (np.array(truth_fields[0]["u"][0]).sum() - np.array(truth_fields[0]["u"][0])[8:16, 10:22].sum()) / (h*w_grid - 8*12)
log(f"Truth: block_u={block_u:.1f} surround_u={surround_u:.1f}")

write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 12); cfg.mission.goal = (29, 12)
cfg.mission.max_steps = 50
cfg.simulation.report_keyframe_interval = 4
cfg.planner.horizon_steps = 20; cfg.planner.terminal_progress_weight = 5.0
write_json(str(RUN / "config.json"), cfg.to_dict())
log(f"Route: {cfg.mission.start} -> {cfg.mission.goal}")

# ══════════════════════════════════════════════════════════════════
log("STEP 2: Train XGBoost")
t0 = time.time()
model_cfg = ModelConfig(model_type="xgboost", learning_rate=0.15, num_estimators=60, max_depth=4,
    feature_subsample_ratio=0.6, max_training_samples=4000, altitude_levels=levels,
    early_stopping_rounds=12, validation_ratio=0.2)
pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=model_cfg)
metrics = pipeline.train(training_samples_from_dict(training_samples))
pipeline.save_model(str(RUN / "model.json"))
write_json(str(RUN / "metrics.json"), metrics)
phys = metrics.get("rmse_physics", 0); final = metrics.get("rmse_final", 0)
reduction = (1 - final / phys) * 100 if phys > 0 else 0
log(f"RMSE: {phys:.4f} -> {final:.4f} ({reduction:.1f}%) in {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════════
log("STEP 3: Run mission with wind-aware planner")
t0 = time.time()
cfg2 = load_task_config(str(RUN / "config.json"))
cfg2.model.model_type = "xgboost"
cfg2.planner.horizon_steps = 20; cfg2.planner.terminal_progress_weight = 5.0
pipeline2 = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), str(RUN / "model.json"), cfg2.model)
cw_list = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by = {o.timestamp: o for o in obs_list}
truth_by = {t["timestamp"]: t for t in truth_fields}

engine = NavigationEngine(pipeline2, cfg2)
ctx = engine.create_context(cfg2.mission.start, cfg2.mission.goal)

# Inject truth observations for belief initialization
truth0 = truth_by[truth_fields[0]["timestamp"]]
u0 = np.array(truth0["u"][0]); v0 = np.array(truth0["v"][0]); w0 = np.array(truth0["w"][0])
updater = BeliefUpdater(observation_radius=cfg2.belief.observation_radius,
    advection_gain=cfg2.belief.advection_gain, decay_per_step=cfg2.belief.decay_per_step,
    process_noise=cfg2.belief.process_noise, observation_noise=cfg2.belief.observation_noise,
    advection_noise=cfg2.belief.advection_noise)
sample0 = cw_list[0]
pred, phys_w = engine._forecast_fields(ctx, sample0, None, True)
updater.apply_prediction(ctx.belief_map, WindField(u=pred["u_layers"], v=pred["v_layers"], w=pred["w_layers"]), 1)
for y in range(0, h, 3):
    for x in range(0, w_grid, 3):
        obs = Observation(timestamp=sample0.timestamp, x=x, y=y, z=0,
            u_obs=float(u0[y, x]), v_obs=float(v0[y, x]), w_obs=float(w0[y, x]))
        updater.update_with_observation(ctx.belief_map, obs, pred["u"][y][x], pred["v"][y][x], pred["w"][y][x], 1)

for i, s in enumerate(cw_list):
    obs = obs_by.get(s.timestamp); truth = truth_by.get(s.timestamp)
    engine.step(ctx, s, obs, truth)
    if i % 10 == 0:
        elapsed = time.time() - t0
        log(f"  step {i+1} ({elapsed:.0f}s) batt={ctx.state.battery_ratio*100:.1f}%")
    if _goal_reached(ctx.state, cfg2.mission.goal) or ctx.battery_j <= 0 or ctx.step >= cfg2.mission.max_steps:
        log(f"  STOP: goal={_goal_reached(ctx.state, cfg2.mission.goal)} step={ctx.step}")
        break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# Stats
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
sp, gp = cfg2.mission.start[:2], cfg2.mission.goal[:2]
sd = math.hypot(gp[0] - sp[0], gp[1] - sp[1])
ad = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
phys_wind = [(f.get("physics_at_drone", {}).get("u", 0), f.get("physics_at_drone", {}).get("v", 0),
              f.get("physics_at_drone", {}).get("w", 0)) for f in ctx.trace]
n = len(ctx.trace)
se = sum(transition_energy_j(airspeed=16.5, current=(sp[0] + (i - 1) / n * (gp[0] - sp[0]), sp[1] + (i - 1) / n * (gp[1] - sp[1]), 0.0),
    nxt=(sp[0] + i / n * (gp[0] - sp[0]), sp[1] + i / n * (gp[1] - sp[1]), 0.0),
    local_u=phys_wind[min(i - 1, n - 1)][0], local_v=phys_wind[min(i - 1, n - 1)][1],
    local_w=phys_wind[min(i - 1, n - 1)][2], step_distance_m=report["grid"]["resolution_m"],
    altitude_step_m=report["mission"]["altitude_step_m"], climb_cost_per_level_j=180.0,
    hover_power_w=105.0, cruise_power_w=150.0, hotel_power_w=18.0, headwind_power_per_mps_w=14.0,
    climb_power_per_mps_w=125.0, descent_power_reduction_per_mps_w=58.0) for i in range(1, n + 1))
ae = sum(transition_energy_j(airspeed=16.5, current=(path[i - 1][0], path[i - 1][1], 0.0),
    nxt=(path[i][0], path[i][1], 0.0), local_u=phys_wind[min(i - 1, n - 1)][0],
    local_v=phys_wind[min(i - 1, n - 1)][1], local_w=phys_wind[min(i - 1, n - 1)][2],
    step_distance_m=report["grid"]["resolution_m"], altitude_step_m=report["mission"]["altitude_step_m"],
    climb_cost_per_level_j=180.0, hover_power_w=105.0, cruise_power_w=150.0, hotel_power_w=18.0,
    headwind_power_per_mps_w=14.0, climb_power_per_mps_w=125.0, descent_power_reduction_per_mps_w=58.0)
    for i in range(1, len(path)))

xs = [p[0] for p in path]; ys = [p[1] for p in path]
extra = (ad - sd) / sd * 100; savings = (se - ae) / se * 100
goes_through = any(10 <= p[0] < 22 and 8 <= p[1] < 16 for p in path)

log(f"Steps={n} Goal={_goal_reached(ctx.state, cfg2.mission.goal)} Batt={ctx.state.battery_ratio*100:.1f}%")
log(f"Straight={sd:.1f} Actual={ad:.1f} Extra={extra:+.1f}% Savings={savings:+.1f}%")
log(f"Through block? {goes_through}")
log(f"Path y: {ys[0]:.1f}->{ys[-1]:.1f} min={min(ys):.1f} max={max(ys):.1f}")

# ══════════════════════════════════════════════════════════════════
log("STEP 4: Generate charts")
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
log(f"Charts: {'OK' if result.returncode == 0 else 'FAILED'}")

# ══════════════════════════════════════════════════════════════════
log(f"\nFINAL: RMSE {phys:.4f}->{final:.4f} ({reduction:.1f}%) | Path {extra:+.1f}% extra | Energy {savings:+.1f}% savings")
