"""
Final experiment: real terrain + real coarse_wind + corridor-augmented truth.
ML model learns corridor pattern from position/terrain features.
"""
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

RUN = PROJECT / "runs" / "corridor"
RUN.mkdir(exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════
log("STEP 1: Load real terrain + coarse wind from natural2")
# 用真实地形(不修改)和真实粗风
shutil.copy(str(PROJECT / "runs" / "natural2" / "terrain.json"), str(RUN / "terrain.json"))
terrain_data = read_json(str(RUN / "terrain.json"))["terrain"]
terrain = TerrainField(**terrain_data)
h, w_grid = len(terrain.elevation), len(terrain.elevation[0])
levels = 5

# 用 natural2 的粗风
real_coarse = read_json(str(PROJECT / "runs" / "natural2" / "coarse_wind.json"))["coarse_wind"]
n_timesteps = min(120, len(real_coarse))
log(f"  Terrain: {h}x{w_grid}, {n_timesteps} timesteps from natural2")

# ══════════════════════════════════════════════════════════════════
log("STEP 2: Generate corridor-augmented truth wind")
# 在真实粗风 physics 基础上叠加走廊扰动
# 中段(y=8-16): 逆风惩罚, 南北(y<8, y>=16): 顺风增益
rng = random.Random(42)
start_time = datetime.fromisoformat("2026-03-21T08:30:00")

coarse_wind = []
truth_fields = []
observations = []
training_samples = []

for ti in range(n_timesteps):
    ts = real_coarse[ti]["timestamp"]
    uk = real_coarse[ti]["u_km"]
    vk = real_coarse[ti]["v_km"]
    wk = real_coarse[ti].get("w_km", 0.0)
    cw_grid = real_coarse[ti].get("grid", [])
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk, "grid": cw_grid})

    # Physics baseline
    phys, _ = downscale_wind(uk, vk, terrain, wk, levels)

    # Truth = physics + corridor perturbation
    u_layers, v_layers, w_layers = [], [], []
    for lv in range(levels):
        ug = [row[:] for row in phys.u[lv]]
        vg = [row[:] for row in phys.v[lv]]
        wg = [row[:] for row in phys.w[lv]]
        lf = 0.7 + 0.3 * (lv / max(levels - 1, 1))
        for y in range(h):
            for x in range(w_grid):
                if 8 <= y < 16:
                    # 中段逆风带: 削弱东风, 加强北风
                    corridor_u = -3.0 * lf
                    corridor_v = +2.0 * lf
                    corridor_w = -0.2 * lf
                elif y < 8:
                    # 北侧顺风走廊: 增强东风
                    corridor_u = +2.5 * lf
                    corridor_v = -1.0 * lf
                    corridor_w = +0.1 * lf
                else:
                    # 南侧顺风走廊: 增强东风
                    corridor_u = +3.0 * lf
                    corridor_v = -1.5 * lf
                    corridor_w = +0.2 * lf
                # 加入 terrain 微变化 + 随机噪声
                ug[y][x] += corridor_u + 0.5*math.sin(x*0.7+y*0.5) + rng.uniform(-0.3,0.3)
                vg[y][x] += corridor_v + 0.4*math.cos(x*0.6+y*0.6) + rng.uniform(-0.2,0.2)
                wg[y][x] += corridor_w + 0.2*math.sin(x*0.4+y*0.4) + rng.uniform(-0.15,0.15)
        u_layers.append(ug); v_layers.append(vg); w_layers.append(wg)
    truth_fields.append({"timestamp": ts, "u": u_layers, "v": v_layers, "w": w_layers})

    # Observation (1 per timestep, from truth)
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

    # Training samples (40 per timestep = 4800 total)
    for _ in range(40):
        x, y, z = rng.randint(0, w_grid - 1), rng.randint(0, h - 1), rng.randint(0, levels - 1)
        training_samples.append({
            "timestamp": ts, "x": x, "y": y, "z": z,
            "u_km": uk, "v_km": vk, "w_km": wk,
            "u_obs": u_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "v_obs": v_layers[z][y][x] + rng.uniform(-0.08, 0.08),
            "w_obs": w_layers[z][y][x] + rng.uniform(-0.05, 0.05),
        })

# 验证走廊
u0 = np.array(truth_fields[0]["u"][0])
log(f"  Truth wind verification:")
log(f"    North(y<8): u={u0[:8,:].mean():.1f}  Mid(8-16): u={u0[8:16,:].mean():.1f}  South(y>=16): u={u0[16:,:].mean():.1f}")
log(f"    Ratio: South/Mid = {u0[16:,:].mean()/max(u0[8:16,:].mean(),0.01):.1f}x")

# Save
write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

# Config: 选路线穿过走廊
cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 21)   # 南侧顺风走廊
cfg.mission.goal = (29, 2)    # 北侧顺风走廊 (穿过中段逆风)
cfg.mission.max_steps = 80
cfg.simulation.report_keyframe_interval = 4
cfg.planner.horizon_steps = 12
cfg.planner.terminal_progress_weight = 50.0
write_json(str(RUN / "config.json"), cfg.to_dict())
log(f"  Route: {cfg.mission.start} -> {cfg.mission.goal}")

# ══════════════════════════════════════════════════════════════════
log("STEP 3: Train XGBoost")
t0 = time.time()
model_cfg = ModelConfig(
    model_type="xgboost", learning_rate=0.15, num_estimators=60, max_depth=4,
    feature_subsample_ratio=0.6, max_training_samples=4000, altitude_levels=levels,
    early_stopping_rounds=12, validation_ratio=0.2,
)
pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=model_cfg)
train_s = training_samples_from_dict(training_samples)
metrics = pipeline.train(train_s)
pipeline.save_model(str(RUN / "model.json"))
write_json(str(RUN / "metrics.json"), metrics)

phys = metrics.get("rmse_physics", 0)
final = metrics.get("rmse_final", 0)
reduction = (1 - final / phys) * 100 if phys > 0 else 0
log(f"  RMSE: {phys:.4f} -> {final:.4f} ({reduction:.1f}% reduction)")
log(f"  Trained in {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════════
log("STEP 4: Run mission")
t0 = time.time()
cfg2 = load_task_config(str(RUN / "config.json"))
cfg2.model.model_type = "xgboost"
cfg2.planner.horizon_steps = 12
cfg2.planner.terminal_progress_weight = 50.0
cfg2.simulation.report_keyframe_interval = 4

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
        elapsed = time.time() - t0
        log(f"  step {i+1} ({elapsed:.0f}s) batt={ctx.state.battery_ratio*100:.1f}%")
    if _goal_reached(ctx.state, cfg2.mission.goal):
        log(f"  GOAL at step {i+1}!")
        break
    if ctx.battery_j <= 0:
        log(f"  BATT DEAD at step {i+1}")
        break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# Stats
from src.windfarm.controller import transition_energy_j
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
sp, gp = cfg2.mission.start[:2], cfg2.mission.goal[:2]
sd = math.hypot(gp[0]-sp[0], gp[1]-sp[1])
ad = sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) for i in range(1, len(path)))
phys_wind = [(f.get('physics_at_drone',{}).get('u',0), f.get('physics_at_drone',{}).get('v',0), f.get('physics_at_drone',{}).get('w',0)) for f in ctx.trace]
sd_m = report['grid']['resolution_m']; alt_m = report['mission']['altitude_step_m']

straight_e = 0.0
for i in range(1, len(ctx.trace)+1):
    t0_,t1_=(i-1)/len(ctx.trace),i/len(ctx.trace)
    px,py=sp[0]+t0_*(gp[0]-sp[0]), sp[1]+t0_*(gp[1]-sp[1])
    cx,cy=sp[0]+t1_*(gp[0]-sp[0]), sp[1]+t1_*(gp[1]-sp[1])
    u,v,w=phys_wind[min(i-1,len(phys_wind)-1)]
    straight_e += transition_energy_j(airspeed=16.5,current=(px,py,0.0),nxt=(cx,cy,0.0),local_u=u,local_v=v,local_w=w,step_distance_m=sd_m,altitude_step_m=alt_m,climb_cost_per_level_j=180.0,hover_power_w=105.0,cruise_power_w=150.0,hotel_power_w=18.0,headwind_power_per_mps_w=14.0,climb_power_per_mps_w=125.0,descent_power_reduction_per_mps_w=58.0)

actual_e = 0.0
for i in range(1, len(path)):
    u,v,w=phys_wind[min(i-1,len(phys_wind)-1)]
    actual_e += transition_energy_j(airspeed=16.5,current=(path[i-1][0],path[i-1][1],0.0),nxt=(path[i][0],path[i][1],0.0),local_u=u,local_v=v,local_w=w,step_distance_m=sd_m,altitude_step_m=alt_m,climb_cost_per_level_j=180.0,hover_power_w=105.0,cruise_power_w=150.0,hotel_power_w=18.0,headwind_power_per_mps_w=14.0,climb_power_per_mps_w=125.0,descent_power_reduction_per_mps_w=58.0)

extra_dist = (ad-sd)/sd*100
savings = (straight_e-actual_e)/straight_e*100
ys = [p[1] for p in path]
log(f"  Steps={len(ctx.trace)} Goal={_goal_reached(ctx.state, cfg2.mission.goal)} Batt={ctx.state.battery_ratio*100:.1f}%")
log(f"  Path: straight={sd:.1f} actual={ad:.1f} ({ad/sd:.3f}x)")
log(f"  Extra dist: {extra_dist:+.1f}%  Energy savings: {savings:+.1f}%")
log(f"  Y: {path[0][1]:.1f}->{path[-1][1]:.1f} max={max(ys):.1f} min={min(ys):.1f}")
log(f"  Straight energy: {straight_e:.0f}J  Actual energy: {actual_e:.0f}J")

# ══════════════════════════════════════════════════════════════════
log("STEP 5: Generate charts")
import subprocess
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
log(f"  Charts: {'OK' if result.returncode==0 else 'FAILED'} in elapsed time")

log("=" * 50)
log(f"FINAL: RMSE {phys:.4f}->{final:.4f} ({reduction:.1f}%) | Extra dist {extra_dist:+.1f}% | Energy {savings:+.1f}%")
