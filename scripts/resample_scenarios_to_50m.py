#!/usr/bin/env python3
"""Upsample scenario grids from 100 m XY to 50 m isotropic cubes (no SRTM rebuild).

Elevation / roughness: bilinear 2x. Slope / aspect: recomputed at 50 m.
Truth wind (z,y,x): bilinear in y,x per altitude level.
Training / observations: map cell centers (x,y) -> (2x+0.5, 2y+0.5).
Mission start/goal and max_steps scale with the finer XY grid.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfarm.terrain_tools import derive_slope_aspect  # noqa: E402

OLD_RES = 100.0
NEW_RES = 50.0
SCALE = int(round(OLD_RES / NEW_RES))  # 2
OLD_ALT = 40.0
NEW_ALT = 50.0


def _upsample_2d(grid: list[list[float]], scale: int = SCALE) -> list[list[float]]:
    """Bilinear upsample by integer factor (cell-centered)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return []
    out_h, out_w = h * scale, w * scale
    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    for j in range(out_h):
        # Map output cell center into old continuous coords [0, w-1] x [0, h-1]
        y = (j + 0.5) / scale - 0.5
        y = min(max(y, 0.0), h - 1.0)
        y0 = int(math.floor(y))
        y1 = min(y0 + 1, h - 1)
        ty = y - y0
        for i in range(out_w):
            x = (i + 0.5) / scale - 0.5
            x = min(max(x, 0.0), w - 1.0)
            x0 = int(math.floor(x))
            x1 = min(x0 + 1, w - 1)
            tx = x - x0
            v00 = float(grid[y0][x0])
            v10 = float(grid[y0][x1])
            v01 = float(grid[y1][x0])
            v11 = float(grid[y1][x1])
            out[j][i] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
    return out


def _upsample_wind_zyx(field: list[list[list[float]]], scale: int = SCALE) -> list[list[list[float]]]:
    """Upsample wind [z][y][x] in y,x only (altitude levels unchanged)."""
    return [_upsample_2d(level, scale) for level in field]


def _scale_xy_point(x: float | int, y: float | int) -> tuple[int, int]:
    return int(x) * SCALE, int(y) * SCALE


def _remap_wind_z(
    field: list[list[list[float]]],
    *,
    old_step: float = OLD_ALT,
    new_step: float = NEW_ALT,
) -> list[list[list[float]]]:
    """Interpolate wind levels so index z means z*new_step meters AGL."""
    n_old = len(field)
    if n_old == 0:
        return field
    h = len(field[0])
    w = len(field[0][0]) if h else 0
    out: list[list[list[float]]] = [[[0.0] * w for _ in range(h)] for _ in range(n_old)]
    for zi in range(n_old):
        z_old = (zi * new_step) / max(old_step, 1e-6)
        z0 = int(math.floor(z_old))
        z1 = min(z0 + 1, n_old - 1)
        t = z_old - z0
        if z0 >= n_old - 1:
            z0 = z1 = n_old - 1
            t = 0.0
        for y in range(h):
            for x in range(w):
                out[zi][y][x] = (1.0 - t) * float(field[z0][y][x]) + t * float(field[z1][y][x])
    return out


def _elev_stats(elevation: list[list[float]]) -> dict[str, float]:
    vals = [float(v) for row in elevation for v in row]
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / max(len(vals), 1),
    }


def _update_config(cfg: dict) -> dict:
    sim = cfg["simulation"]
    mission = cfg["mission"]
    old_w, old_h = int(sim["width"]), int(sim["height"])
    sim["width"] = old_w * SCALE
    sim["height"] = old_h * SCALE
    sim["resolution_m"] = NEW_RES

    start = mission["start"]
    goal = mission["goal"]
    # Integer cell indices: map old cell -> finer cell near center of 2x2 block.
    mission["start"] = [int(start[0]) * SCALE + SCALE // 2, int(start[1]) * SCALE + SCALE // 2]
    mission["goal"] = [int(goal[0]) * SCALE + SCALE // 2, int(goal[1]) * SCALE + SCALE // 2]
    if len(start) > 2:
        mission["start"].append(start[2])
    if len(goal) > 2:
        mission["goal"].append(goal[2])

    mission["step_distance_m"] = NEW_RES
    mission["altitude_step_m"] = NEW_ALT
    mission["climb_cost_per_level_j"] = float(mission.get("climb_cost_per_level_j", 180.0)) * (NEW_ALT / OLD_ALT)
    # ~1 m physical cruise grid: 1/50 = 0.02 levels
    mission["cruise_band_step"] = 0.02
    mission["clearance_agl_level"] = float(mission.get("clearance_agl_level", 1.0))
    mission["max_steps"] = int(mission.get("max_steps", 40)) * SCALE

    belief = cfg.setdefault("belief", {})
    # Keep ~200 m observation radius in physical meters.
    old_r = int(belief.get("observation_radius", 2))
    belief["observation_radius"] = max(1, int(round(old_r * (OLD_RES / NEW_RES))))
    return cfg


def _resample_scenario(path: Path) -> None:
    print(f"resampling {path.name} ...", flush=True)
    terrain_path = path / "terrain.json"
    terrain_doc = json.loads(terrain_path.read_text())
    terr = terrain_doc["terrain"]
    elev = _upsample_2d(terr["elevation"])
    slope, aspect = derive_slope_aspect(elev, NEW_RES)
    rough = _upsample_2d(terr["roughness"])
    terr["elevation"] = elev
    terr["slope"] = slope
    terr["aspect"] = aspect
    terr["roughness"] = rough
    terrain_path.write_text(json.dumps(terrain_doc, indent=2) + "\n")

    truth_path = path / "truth.json"
    truth_doc = json.loads(truth_path.read_text())
    for field in truth_doc["truth_fields"]:
        for comp in ("u", "v", "w"):
            field[comp] = _remap_wind_z(_upsample_wind_zyx(field[comp]))
    truth_path.write_text(json.dumps(truth_doc) + "\n")

    train_path = path / "training.json"
    train_doc = json.loads(train_path.read_text())
    for sample in train_doc["samples"]:
        sx, sy = _scale_xy_point(sample["x"], sample["y"])
        sample["x"] = sx
        sample["y"] = sy
        # Keep integer level index; truth wind planes are remapped to new altitude_step.
        sample["z"] = int(sample["z"])
    train_path.write_text(json.dumps(train_doc) + "\n")

    obs_path = path / "observations.json"
    obs_doc = json.loads(obs_path.read_text())
    for sample in obs_doc["observations"]:
        sx, sy = _scale_xy_point(sample["x"], sample["y"])
        sample["x"] = sx
        sample["y"] = sy
        sample["z"] = int(sample["z"])
    obs_path.write_text(json.dumps(obs_doc) + "\n")

    cfg_path = path / "config.json"
    cfg = _update_config(json.loads(cfg_path.read_text()))
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    meta_path = path / "scenario_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["grid"] = {
            "width": cfg["simulation"]["width"],
            "height": cfg["simulation"]["height"],
            "resolution_m": NEW_RES,
        }
        meta["mission"] = {"start": cfg["mission"]["start"], "goal": cfg["mission"]["goal"]}
        meta["elevation_m"] = _elev_stats(elev)
        meta["resampled_to_m"] = NEW_RES
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"  -> {cfg['simulation']['width']}x{cfg['simulation']['height']} @ {NEW_RES} m, "
        f"start={cfg['mission']['start']} goal={cfg['mission']['goal']}",
        flush=True,
    )


def main() -> None:
    scenarios = sorted(p for p in (ROOT / "scenarios").iterdir() if p.is_dir() and (p / "config.json").exists())
    for path in scenarios:
        cfg = json.loads((path / "config.json").read_text())
        res = float(cfg["simulation"]["resolution_m"])
        if abs(res - NEW_RES) < 1e-6:
            print(f"skip {path.name} (already {NEW_RES} m)")
            continue
        if abs(res - OLD_RES) > 1e-6:
            raise SystemExit(f"{path.name}: unexpected resolution_m={res}")
        _resample_scenario(path)
    print("done")


if __name__ == "__main__":
    main()
