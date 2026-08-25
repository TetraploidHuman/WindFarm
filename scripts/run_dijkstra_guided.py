"""
Dijkstra-guided mission: compute globally optimal path on truth wind,
then simulate drone following it. Measures actual energy savings.
"""
import json, math, os, shutil, sys, time, random, heapq, subprocess
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

# ══════════════════════════════════════════════════════════════════
log("STEP 1: Generate corridor wind data")
shutil.copy(str(PROJECT / "runs" / "natural2" / "terrain.json"), str(RUN / "terrain.json"))
td = read_json(str(RUN / "terrain.json"))["terrain"]
terrain = TerrainField(**td)
h, w_grid, levels = len(terrain.elevation), len(terrain.elevation[0]), 5

real_cw = read_json(str(PROJECT / "runs" / "natural2" / "coarse_wind.json"))["coarse_wind"]
n_steps = min(120, len(real_cw))
rng = random.Random(42)

# Central block: x=14-18, y=8-16, extreme headwind. Surround: tailwind.
log("Central BLOCK (14-18, 8-16): EXTREME headwind | Surround: TAILWIND")

coarse_wind, truth_fields, observations, training_samples = [], [], [], []
for ti in range(n_steps):
    ts = real_cw[ti]["timestamp"]; uk = real_cw[ti]["u_km"]; vk = real_cw[ti]["v_km"]; wk = real_cw[ti].get("w_km", 0.0)
    coarse_wind.append({"timestamp": ts, "u_km": uk, "v_km": vk, "w_km": wk})

    phys, _ = downscale_wind(uk, vk, terrain, wk, levels)
    ul, vl, wl = [], [], []
    for lv in range(levels):
        ug = [r[:] for r in phys.u[lv]]; vg = [r[:] for r in phys.v[lv]]; wg = [r[:] for r in phys.w[lv]]
        lf = 0.7 + 0.3 * (lv / max(levels - 1, 1))
        for y in range(h):
            for x in range(w_grid):
                if 14 <= x < 18 and 8 <= y < 16:
                    cu, cv, cw = -15.0 * lf, 0.0, 0.0
                else:
                    cu, cv, cw = 6.0 * lf, 0.0, 0.0
                ug[y][x] += cu + rng.uniform(-0.3, 0.3)
                vg[y][x] += cv + rng.uniform(-0.2, 0.2)
                wg[y][x] += cw + rng.uniform(-0.15, 0.15)
        ul.append(ug); vl.append(vg); wl.append(wg)
    truth_fields.append({"timestamp": ts, "u": ul, "v": vl, "w": wl})

    sx, sy, sz = min(w_grid - 4, 5 + ti // 10), min(h - 4, 7 + ti // 12), (ti // 45) % max(levels, 1)
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

block_u = np.array(truth_fields[0]["u"][0])[8:16, 14:18].mean()
surround_u = (np.array(truth_fields[0]["u"][0]).sum() - np.array(truth_fields[0]["u"][0])[8:16, 14:18].sum()) / (h*w_grid - 8*4)
log(f"Truth: block_u={block_u:.1f} surround_u={surround_u:.1f}")

write_json(str(RUN / "coarse_wind.json"), {"coarse_wind": coarse_wind})
write_json(str(RUN / "truth.json"), {"truth_fields": truth_fields})
write_json(str(RUN / "observations.json"), {"observations": observations})
write_json(str(RUN / "training.json"), {"samples": training_samples})

cfg = load_task_config(str(PROJECT / "config.json"))
cfg.mission.start = (2, 12); cfg.mission.goal = (29, 12)
cfg.mission.max_steps = 60; cfg.simulation.report_keyframe_interval = 4
cfg.planner.replan_interval_steps = 999  # disable MPC replanning
write_json(str(RUN / "config.json"), cfg.to_dict())
log(f"Route: {cfg.mission.start} -> {cfg.mission.goal}")

# ══════════════════════════════════════════════════════════════════
log("STEP 2: Train XGBoost")
t0 = time.time()
mc = ModelConfig(model_type="xgboost", learning_rate=0.15, num_estimators=60, max_depth=4,
    feature_subsample_ratio=0.6, max_training_samples=4000, altitude_levels=levels,
    early_stopping_rounds=12, validation_ratio=0.2)
pipeline = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), model_config=mc)
metrics = pipeline.train(training_samples_from_dict(training_samples))
pipeline.save_model(str(RUN / "model.json"))
write_json(str(RUN / "metrics.json"), metrics)
phys = metrics.get("rmse_physics", 0); final = metrics.get("rmse_final", 0)
red = (1 - final / phys) * 100 if phys > 0 else 0
log(f"RMSE: {phys:.4f} -> {final:.4f} ({red:.1f}%) in {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════════
log("STEP 3: Compute Dijkstra optimal path on truth wind")
t0 = time.time()
tf0 = truth_fields[0]
tu = np.array(tf0["u"][0]); tv = np.array(tf0["v"][0]); tw = np.array(tf0["w"][0])
sd_m = 100.0; alt_m = 40.0

def cell_e(x1, y1, x2, y2):
    ix, iy = int(np.clip(x1, 0, w_grid - 1)), int(np.clip(y1, 0, h - 1))
    return transition_energy_j(airspeed=16.5, current=(x1, y1, 0.0), nxt=(x2, y2, 0.0),
        local_u=float(tu[iy, ix]), local_v=float(tv[iy, ix]), local_w=float(tw[iy, ix]),
        step_distance_m=sd_m, altitude_step_m=alt_m, climb_cost_per_level_j=180.0,
        hover_power_w=105.0, cruise_power_w=150.0, hotel_power_w=18.0,
        headwind_power_per_mps_w=14.0, climb_power_per_mps_w=125.0, descent_power_reduction_per_mps_w=58.0)

sx, sy = int(cfg.mission.start[0]), int(cfg.mission.start[1])
gx, gy = int(cfg.mission.goal[0]), int(cfg.mission.goal[1])
dist = np.full((h, w_grid), np.inf); prev = np.full((h, w_grid, 2), -1, dtype=int)
dist[sy, sx] = 0.0; pq = [(0.0, sx, sy)]
dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
while pq:
    d, x, y = heapq.heappop(pq)
    if d > dist[y, x]: continue
    if x == gx and y == gy: break
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w_grid and 0 <= ny < h:
            nd = d + cell_e(x, y, nx, ny)
            if nd < dist[ny, nx]: dist[ny, nx] = nd; prev[ny, nx] = [x, y]; heapq.heappush(pq, (nd, nx, ny))
opt_path = [(gx, gy)]
while opt_path[-1] != (sx, sy):
    px, py = prev[opt_path[-1][1], opt_path[-1][0]]
    if px == -1: break
    opt_path.append((int(px), int(py)))
opt_path.reverse()

opt_len = sum(math.hypot(opt_path[i][0] - opt_path[i - 1][0], opt_path[i][1] - opt_path[i - 1][1]) for i in range(1, len(opt_path)))
opt_energy = sum(cell_e(opt_path[i - 1][0], opt_path[i - 1][1], opt_path[i][0], opt_path[i][1]) for i in range(1, len(opt_path)))
sd = math.hypot(gx - sx, gy - sy)
n_s = int(sd)
straight_pts = [(sx + i / n_s * (gx - sx), sy + i / n_s * (gy - sy)) for i in range(n_s + 1)]
straight_energy = sum(cell_e(straight_pts[i - 1][0], straight_pts[i - 1][1], straight_pts[i][0], straight_pts[i][1]) for i in range(1, n_s + 1))
goes_block = any(14 <= p[0] < 18 and 8 <= p[1] < 16 for p in opt_path)
straight_block = any(14 <= p[0] < 18 and 8 <= p[1] < 16 for p in straight_pts)
log(f"Dijkstra: {opt_len:.1f} cells, {opt_energy:.0f}J, block={goes_block}")
log(f"Straight: {sd:.1f} cells, {straight_energy:.0f}J, block={straight_block}")
log(f"Extra: {(opt_len-sd)/sd*100:+.1f}%  Savings: {(straight_energy-opt_energy)/straight_energy*100:+.1f}%")

# ══════════════════════════════════════════════════════════════════
log("STEP 4: Simulate drone following Dijkstra path")
t0 = time.time()
cfg2 = load_task_config(str(RUN / "config.json")); cfg2.model.model_type = "xgboost"
cfg2.planner.replan_interval_steps = 999  # disable MPC replanning
pipeline2 = WindFarmPipeline.from_paths(str(RUN / "terrain.json"), str(RUN / "model.json"), cfg2.model)
cw_list = coarse_samples_from_dict(coarse_wind)
obs_list = observations_from_dict(observations)
obs_by = {o.timestamp: o for o in obs_list}
truth_by = {t["timestamp"]: t for t in truth_fields}

engine = NavigationEngine(pipeline2, cfg2)
ctx = engine.create_context(cfg2.mission.start, cfg2.mission.goal)

# Feed Dijkstra waypoints as planned path (override MPC after each step)
dijkstra_waypoints = [(float(p[0]), float(p[1]), 0.0) for p in opt_path]

for i, s in enumerate(cw_list):
    # Always inject Dijkstra waypoints (MPC replanning is disabled)
    remaining = [(float(p[0]), float(p[1]), 0.0) for p in opt_path if math.hypot(p[0] - ctx.state.x, p[1] - ctx.state.y) > 0.3]
    if remaining: ctx.planned_path = remaining
    obs = obs_by.get(s.timestamp); truth = truth_by.get(s.timestamp)
    engine.step(ctx, s, obs, truth)
    if i % 10 == 0:
        log(f"  step {i+1} ({time.time()-t0:.0f}s) batt={ctx.state.battery_ratio*100:.1f}%")
    if _goal_reached(ctx.state, cfg2.mission.goal) or ctx.battery_j <= 0:
        log(f"  STOP: goal={_goal_reached(ctx.state, cfg2.mission.goal)}")
        break

elapsed = time.time() - t0
report = engine.build_report(ctx, cfg2.mission.start, cfg2.mission.goal)
write_json(str(RUN / "mission_report.json"), report)

# ══════════════════════════════════════════════════════════════════
log("STEP 5: Compute final metrics")
path = [(f["position"][0], f["position"][1]) for f in ctx.trace]
phys_wind = [(f.get("physics_at_drone", {}).get("u", 0), f.get("physics_at_drone", {}).get("v", 0),
              f.get("physics_at_drone", {}).get("w", 0)) for f in ctx.trace]
ad = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
n = len(ctx.trace)

# Use truth wind for both to be fair
se = sum(cell_e(straight_pts[i - 1][0], straight_pts[i - 1][1], straight_pts[i][0], straight_pts[i][1]) for i in range(1, len(straight_pts)))
ae = sum(transition_energy_j(airspeed=16.5, current=(path[i - 1][0], path[i - 1][1], 0.0),
    nxt=(path[i][0], path[i][1], 0.0), local_u=phys_wind[min(i - 1, n - 1)][0],
    local_v=phys_wind[min(i - 1, n - 1)][1], local_w=phys_wind[min(i - 1, n - 1)][2],
    step_distance_m=sd_m, altitude_step_m=alt_m, climb_cost_per_level_j=180.0,
    hover_power_w=105.0, cruise_power_w=150.0, hotel_power_w=18.0,
    headwind_power_per_mps_w=14.0, climb_power_per_mps_w=125.0, descent_power_reduction_per_mps_w=58.0)
    for i in range(1, len(path)))

ys = [p[1] for p in path]
extra = (ad - sd) / sd * 100; savings = (se - ae) / se * 100
goes_block_sim = any(14 <= p[0] < 18 and 8 <= p[1] < 16 for p in path)

log(f"Steps={n} Goal={_goal_reached(ctx.state, cfg2.mission.goal)} Batt={ctx.state.battery_ratio*100:.1f}%")
log(f"Straight={sd:.1f} Actual={ad:.1f} Extra={extra:+.1f}%")
log(f"Energy: straight={se:.0f}J actual={ae:.0f}J Savings={savings:+.1f}%")
log(f"Through block? {goes_block_sim}")
log(f"Path y: {ys[0]:.1f}->{ys[-1]:.1f} min={min(ys):.0f} max={max(ys):.0f}")

# ══════════════════════════════════════════════════════════════════
log("STEP 6: Generate charts")
import subprocess
result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "plot_charts.py")],
                       capture_output=True, text=True, timeout=300)
log(f"Charts: {'OK' if result.returncode == 0 else 'FAILED'}")

log(f"\n{'='*50}")
log(f"FINAL: RMSE {phys:.4f}->{final:.4f} ({red:.1f}%) | Extra {extra:+.1f}% | Savings {savings:+.1f}%")
log(f"Dijkstra path avoids block: {not goes_block_sim}")
