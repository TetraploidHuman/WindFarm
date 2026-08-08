"""Ingest real DEM (SRTM .hgt) and coarse wind (Open-Meteo archive)."""

from __future__ import annotations

import gzip
import math
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .io import write_json
from .terrain_tools import prepare_terrain
from .types import TerrainField

SRTM_SIZE = 3601
SRTM_SKADI_BASE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A named real-world crop used for multi-scenario evaluation."""

    name: str
    description: str
    lat: float
    lon: float
    # Southwest corner of the SRTM 1° tile that contains (lat, lon).
    tile_lat: int
    tile_lon: int
    # Optional distinct wind fetch dates (synoptic diversity).
    wind_start_date: str = "2024-03-20"
    wind_end_date: str = "2024-03-21"
    # Mission corridor fractions in [0, 1].
    start_frac: tuple[float, float] = (0.15, 0.75)
    goal_frac: tuple[float, float] = (0.85, 0.30)


# Diverse sites: coastal hills, flat plains, alpine ridge, windy coast.
DEFAULT_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="fujian_hills",
        description="Fujian coastal hills / valley (SRTM N25E118)",
        lat=25.75,
        lon=118.60,
        tile_lat=25,
        tile_lon=118,
        wind_start_date="2024-03-20",
        wind_end_date="2024-03-21",
        start_frac=(0.12, 0.78),
        goal_frac=(0.86, 0.28),
    ),
    ScenarioSpec(
        name="beijing_plain",
        description="North China plain near Beijing (SRTM N39E116)",
        lat=39.55,
        lon=116.45,
        tile_lat=39,
        tile_lon=116,
        wind_start_date="2024-04-12",
        wind_end_date="2024-04-13",
        start_frac=(0.18, 0.72),
        goal_frac=(0.82, 0.32),
    ),
    ScenarioSpec(
        name="qinghai_ridge",
        description="NE Tibetan plateau foothills / ridge (SRTM N36E100)",
        lat=36.50,
        lon=100.60,
        tile_lat=36,
        tile_lon=100,
        wind_start_date="2024-01-18",
        wind_end_date="2024-01-19",
        start_frac=(0.14, 0.70),
        goal_frac=(0.84, 0.35),
    ),
    ScenarioSpec(
        name="qingdao_coast",
        description="Shandong windy coast (SRTM N36E120)",
        lat=36.20,
        lon=120.65,
        tile_lat=36,
        tile_lon=120,
        wind_start_date="2024-11-05",
        wind_end_date="2024-11-06",
        start_frac=(0.16, 0.68),
        goal_frac=(0.80, 0.34),
    ),
)


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def hgt_path_for_tile(data_dir: str | Path, lat: int, lon: int) -> Path:
    return Path(data_dir) / f"{tile_name(lat, lon)}.hgt"


def skadi_url(lat: int, lon: int) -> str:
    name = tile_name(lat, lon)
    folder = name[:3]  # e.g. N25
    return f"{SRTM_SKADI_BASE}/{folder}/{name}.hgt.gz"


def ensure_srtm_tile(data_dir: str | Path, lat: int, lon: int, timeout_s: float = 180.0) -> Path:
    """Download SRTM 1° tile if missing; return path to uncompressed .hgt."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = hgt_path_for_tile(data_dir, lat, lon)
    if path.exists() and path.stat().st_size >= SRTM_SIZE * SRTM_SIZE * 2:
        return path
    url = skadi_url(lat, lon)
    request = urllib.request.Request(url, headers={"User-Agent": "WindFarm/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    if url.endswith(".gz") or payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    if len(payload) != SRTM_SIZE * SRTM_SIZE * 2:
        raise ValueError(f"Unexpected HGT size for {path.name}: {len(payload)}")
    path.write_bytes(payload)
    return path


def read_hgt_elevation(path: str | Path) -> list[list[float]]:
    """Read a 1°-SRTM .hgt file as a 3601×3601 elevation grid (meters)."""
    raw = Path(path).read_bytes()
    expected = SRTM_SIZE * SRTM_SIZE * 2
    if len(raw) != expected:
        raise ValueError(f"HGT {path} has {len(raw)} bytes, expected {expected}")
    elevation: list[list[float]] = []
    offset = 0
    for _ in range(SRTM_SIZE):
        row: list[float] = []
        for _ in range(SRTM_SIZE):
            value = struct.unpack_from(">h", raw, offset)[0]
            offset += 2
            # SRTM void is -32768; replace with neighbor-ish zero for tooling safety.
            row.append(0.0 if value == -32768 else float(value))
        elevation.append(row)
    return elevation


def _bilinear(grid: list[list[float]], x: float, y: float) -> float:
    height = len(grid)
    width = len(grid[0]) if height else 0
    if width == 0 or height == 0:
        return 0.0
    x = min(max(x, 0.0), width - 1.001)
    y = min(max(y, 0.0), height - 1.001)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    v00 = grid[y0][x0]
    v10 = grid[y0][x1]
    v01 = grid[y1][x0]
    v11 = grid[y1][x1]
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return top + (bottom - top) * ty


def crop_srtm_to_grid(
    hgt_path: str | Path,
    center_lat: float,
    center_lon: float,
    width: int,
    height: int,
    resolution_m: float,
    tile_lat: int | None = None,
    tile_lon: int | None = None,
) -> list[list[float]]:
    """Crop/resample an SRTM tile around a lat/lon into a planning grid."""
    elevation_full = read_hgt_elevation(hgt_path)
    # Infer tile SW corner from filename if not provided.
    if tile_lat is None or tile_lon is None:
        stem = Path(hgt_path).stem  # N25E118
        tile_lat = int(stem[1:3]) * (1 if stem[0] == "N" else -1)
        tile_lon = int(stem[4:7]) * (1 if stem[3] == "E" else -1)

    # SRTM row 0 is north edge of the 1° tile.
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
    half_w_m = 0.5 * (width - 1) * resolution_m
    half_h_m = 0.5 * (height - 1) * resolution_m

    elev: list[list[float]] = []
    for j in range(height):
        # Grid y increases southward in this project (row 0 = north-ish mission y small?);
        # match existing real_terrain convention: y index increases with decreasing lat.
        lat = center_lat + (half_h_m - j * resolution_m) / meters_per_deg_lat
        row: list[float] = []
        for i in range(width):
            lon = center_lon + (i * resolution_m - half_w_m) / meters_per_deg_lon
            # Pixel coords inside the 1° tile (lon west→east, lat north→south).
            px = (lon - tile_lon) * (SRTM_SIZE - 1)
            py = (tile_lat + 1 - lat) * (SRTM_SIZE - 1)
            row.append(_bilinear(elevation_full, px, py))
        elev.append(row)
    return elev


def landcover_from_terrain(elevation: list[list[float]], resolution_m: float) -> list[list[int]]:
    """Heuristic landcover classes from slope / relative elevation (no external LULC)."""
    height = len(elevation)
    width = len(elevation[0]) if height else 0
    flat = [value for row in elevation for value in row]
    p20 = sorted(flat)[max(0, int(0.20 * len(flat)) - 1)] if flat else 0.0
    p80 = sorted(flat)[min(len(flat) - 1, int(0.80 * len(flat)))] if flat else 0.0
    landcover: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            left = elevation[y][max(0, x - 1)]
            right = elevation[y][min(width - 1, x + 1)]
            up = elevation[max(0, y - 1)][x]
            down = elevation[min(height - 1, y + 1)][x]
            dzdx = (right - left) / max(2.0 * resolution_m, 1e-6)
            dzdy = (down - up) / max(2.0 * resolution_m, 1e-6)
            slope = math.atan(math.hypot(dzdx, dzdy))
            z = elevation[y][x]
            if slope < 0.04 and z <= p20:
                row.append(0)  # valley / water-adjacent
            elif slope > 0.35:
                row.append(3)  # rocky / sparse
            elif z >= p80 and slope > 0.12:
                row.append(1)  # ridge scrub
            elif slope < 0.08:
                row.append(2)  # farmland / open
            else:
                row.append(2)
        landcover.append(row)
    return landcover


def terrain_from_srtm(
    hgt_path: str | Path,
    center_lat: float,
    center_lon: float,
    width: int,
    height: int,
    resolution_m: float,
    tile_lat: int | None = None,
    tile_lon: int | None = None,
) -> TerrainField:
    elevation = crop_srtm_to_grid(
        hgt_path,
        center_lat,
        center_lon,
        width,
        height,
        resolution_m,
        tile_lat=tile_lat,
        tile_lon=tile_lon,
    )
    landcover = landcover_from_terrain(elevation, resolution_m)
    return prepare_terrain(elevation, landcover, resolution_m)


def _http_json(url: str, timeout_s: float = 90.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "WindFarm/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            import json

            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:300]}") from exc


def fetch_open_meteo_hourly(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timeout_s: float = 90.0,
) -> list[dict]:
    """Fetch hourly 10 m + 100 m wind from Open-Meteo archive."""
    query = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
    )
    payload = _http_json(f"{OPEN_METEO_ARCHIVE}?{query}", timeout_s=timeout_s)
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    speeds = hourly.get("wind_speed_10m") or []
    dirs = hourly.get("wind_direction_10m") or []
    speeds_100 = hourly.get("wind_speed_100m") or [None] * len(times)
    dirs_100 = hourly.get("wind_direction_100m") or [None] * len(times)
    if not times or len(times) != len(speeds) or len(times) != len(dirs):
        raise RuntimeError("Open-Meteo response missing hourly wind fields")
    samples: list[dict] = []
    for stamp, speed, direction, speed100, direction100 in zip(times, speeds, dirs, speeds_100, dirs_100):
        if speed is None or direction is None:
            continue
        rad = math.radians(float(direction))
        u_km = -float(speed) * math.sin(rad)
        v_km = -float(speed) * math.cos(rad)
        if speed100 is not None and direction100 is not None:
            rad100 = math.radians(float(direction100))
            u100_km = -float(speed100) * math.sin(rad100)
            v100_km = -float(speed100) * math.cos(rad100)
        else:
            u100_km = u_km
            v100_km = v_km
        hour = int(stamp[11:13]) if len(stamp) >= 13 else 12
        w_km = 0.18 * max(0.0, math.sin(math.pi * (hour - 8) / 10.0))
        iso = stamp if "T" in stamp else stamp
        if len(iso) == 16:
            iso = iso + ":00"
        samples.append(
            {
                "timestamp": iso,
                "u_km": u_km,
                "v_km": v_km,
                "w_km": w_km,
                "u100_km": u100_km,
                "v100_km": v100_km,
                "speed_mps": float(speed),
                "direction_deg": float(direction),
            }
        )
    if not samples:
        raise RuntimeError("Open-Meteo returned no usable wind samples")
    return samples


def resample_coarse_wind(
    hourly: list[dict],
    time_steps: int,
    start_time: str,
    sample_interval_seconds: int,
) -> list[dict]:
    """Interpolate hourly Open-Meteo samples onto the simulation timeline."""
    if not hourly:
        raise ValueError("hourly wind series is empty")
    base = [datetime.fromisoformat(item["timestamp"]) for item in hourly]
    u = [float(item["u_km"]) for item in hourly]
    v = [float(item["v_km"]) for item in hourly]
    w = [float(item.get("w_km", 0.0)) for item in hourly]
    u100 = [float(item["u100_km"]) if item.get("u100_km") is not None else float(item["u_km"]) for item in hourly]
    v100 = [float(item["v100_km"]) if item.get("v100_km") is not None else float(item["v_km"]) for item in hourly]
    t0 = datetime.fromisoformat(start_time)
    out: list[dict] = []
    for step in range(time_steps):
        t = t0 + timedelta(seconds=step * sample_interval_seconds)
        if t <= base[0]:
            idx = 0
            frac = 0.0
        elif t >= base[-1]:
            idx = len(base) - 2
            frac = 1.0
        else:
            idx = 0
            while idx < len(base) - 2 and base[idx + 1] < t:
                idx += 1
            span = (base[idx + 1] - base[idx]).total_seconds()
            frac = 0.0 if span <= 0 else (t - base[idx]).total_seconds() / span
        u_km = u[idx] + (u[idx + 1] - u[idx]) * frac
        v_km = v[idx] + (v[idx + 1] - v[idx]) * frac
        w_km = w[idx] + (w[idx + 1] - w[idx]) * frac
        u100_km = u100[idx] + (u100[idx + 1] - u100[idx]) * frac
        v100_km = v100[idx] + (v100[idx + 1] - v100[idx]) * frac
        out.append(
            {
                "timestamp": t.isoformat(timespec="seconds"),
                "u_km": u_km,
                "v_km": v_km,
                "w_km": w_km,
                "u100_km": u100_km,
                "v100_km": v100_km,
            }
        )
    return out


def mission_endpoints(
    width: int,
    height: int,
    start_frac: tuple[float, float],
    goal_frac: tuple[float, float],
) -> tuple[tuple[int, int], tuple[int, int]]:
    start = (
        int(clamp_int(round(start_frac[0] * (width - 1)), 0, width - 1)),
        int(clamp_int(round(start_frac[1] * (height - 1)), 0, height - 1)),
    )
    goal = (
        int(clamp_int(round(goal_frac[0] * (width - 1)), 0, width - 1)),
        int(clamp_int(round(goal_frac[1] * (height - 1)), 0, height - 1)),
    )
    return start, goal


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def write_terrain_json(path: str | Path, terrain: TerrainField, meta: dict | None = None) -> None:
    payload: dict = {"terrain": asdict(terrain)}
    if meta:
        payload["meta"] = meta
    write_json(path, payload)


def write_coarse_wind_json(path: str | Path, coarse_wind: list[dict], meta: dict | None = None) -> None:
    payload: dict = {"coarse_wind": coarse_wind}
    if meta:
        payload["meta"] = meta
    write_json(path, payload)
