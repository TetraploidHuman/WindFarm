"""Ingest real DEM (SRTM .hgt) and coarse wind (Open-Meteo archive)."""

from __future__ import annotations

import gzip
import math
import os
import ssl
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


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that finds system CA bundles on NixOS/Linux.

    CPython's default often looks for /etc/ssl/cert.pem; many distros only ship
    /etc/ssl/certs/ca-certificates.crt, which causes CERTIFICATE_VERIFY_FAILED.
    """
    candidates: list[str] = []
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(value)
    try:
        import certifi

        candidates.append(certifi.where())
    except Exception:
        pass
    candidates.extend(
        [
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/cert.pem",
        ]
    )
    for path in candidates:
        if path and Path(path).is_file():
            return ssl.create_default_context(cafile=path)
    return ssl.create_default_context()


def _detect_local_clash_proxy() -> str:
    """Pick a reachable Clash mixed-port if env vars are unset.

    Prefer LAN gateway Clash (172.20.128.142:7897), then localhost / FlClash variants.
    """
    import socket

    hosts = ("172.20.128.142", "127.0.0.1", "172.20.128.1")
    ports = (7897, 7890, 6088)
    for host in hosts:
        for port in ports:
            try:
                with socket.create_connection((host, port), timeout=0.6):
                    return f"http://{host}:{port}"
            except OSError:
                continue
    return ""


def _proxy_opener() -> urllib.request.OpenerDirector:
    """Build a URL opener that honors Clash/system proxy env vars + system CAs.

    Supported (first non-empty wins for https):
      WINDFARM_HTTP_PROXY, HTTPS_PROXY, https_proxy, HTTP_PROXY, http_proxy, ALL_PROXY, all_proxy

    If none are set, probe 127.0.0.1:7897 / 7890 and 172.20.128.1:7897 / 7890.
    """
    keys = (
        "WINDFARM_HTTP_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    )
    proxy = next((os.environ[k].strip() for k in keys if os.environ.get(k, "").strip()), "")
    if not proxy:
        proxy = _detect_local_clash_proxy()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=_ssl_context()),
    ]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _urlopen(request: urllib.request.Request, timeout: float):
    return _proxy_opener().open(request, timeout=timeout)


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


# Eight representative real-world maps (curated for diversity + hard negatives).
DEFAULT_SCENARIOS: tuple[ScenarioSpec, ...] = (
    # --- core (early tuning) ---
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
        name="liaoning_coast",
        description="Liaodong peninsula coast near Dalian (SRTM N39E121)",
        lat=39.15,
        lon=121.65,
        tile_lat=39,
        tile_lon=121,
        wind_start_date="2024-11-18",
        wind_end_date="2024-11-19",
        start_frac=(0.15, 0.72),
        goal_frac=(0.85, 0.31),
    ),
    # --- holdout (anti-overfit; do not tune on these in isolation) ---
    ScenarioSpec(
        name="xinjiang_gobi",
        description="Hami Gobi corridor / arid basin (SRTM N42E093)",
        lat=42.83,
        lon=93.52,
        tile_lat=42,
        tile_lon=93,
        wind_start_date="2024-02-10",
        wind_end_date="2024-02-11",
        start_frac=(0.16, 0.70),
        goal_frac=(0.83, 0.33),
    ),
    ScenarioSpec(
        name="sichuan_foothills",
        description="West Sichuan basin-edge foothills (SRTM N30E103)",
        lat=30.65,
        lon=103.45,
        tile_lat=30,
        tile_lon=103,
        wind_start_date="2024-08-22",
        wind_end_date="2024-08-23",
        start_frac=(0.15, 0.73),
        goal_frac=(0.85, 0.30),
    ),
    ScenarioSpec(
        name="shanxi_loess",
        description="Loess plateau gullies near Yan'an (SRTM N36E109)",
        lat=36.65,
        lon=109.45,
        tile_lat=36,
        tile_lon=109,
        wind_start_date="2024-10-08",
        wind_end_date="2024-10-09",
        start_frac=(0.15, 0.72),
        goal_frac=(0.85, 0.30),
    ),
    ScenarioSpec(
        name="taiwan_hills",
        description="Central Taiwan foothills / windward slopes (SRTM N23E121)",
        lat=23.55,
        lon=121.15,
        tile_lat=23,
        tile_lon=121,
        wind_start_date="2024-09-03",
        wind_end_date="2024-09-04",
        start_frac=(0.14, 0.73),
        goal_frac=(0.84, 0.30),
    ),
    # --- holdout expansion (regime diversity; already built on disk) ---
    ScenarioSpec(
        name="gansu_hexi",
        description="Hexi corridor arid wind gap near Jiayuguan (SRTM N39E098)",
        lat=39.75,
        lon=98.5,
        tile_lat=39,
        tile_lon=98,
        wind_start_date="2024-03-05",
        wind_end_date="2024-03-06",
        start_frac=(0.15, 0.69),
        goal_frac=(0.81, 0.34),
    ),
    ScenarioSpec(
        name="guizhou_karst",
        description="Guizhou plateau karst near Guiyang (SRTM N26E106)",
        lat=26.55,
        lon=106.7,
        tile_lat=26,
        tile_lon=106,
        wind_start_date="2024-06-20",
        wind_end_date="2024-06-21",
        start_frac=(0.14, 0.73),
        goal_frac=(0.84, 0.29),
    ),
    ScenarioSpec(
        name="hainan_coast",
        description="Hainan tropical hills / coastal fringe (SRTM N18E109)",
        lat=18.85,
        lon=109.55,
        tile_lat=18,
        tile_lon=109,
        wind_start_date="2024-07-12",
        wind_end_date="2024-07-13",
        start_frac=(0.14, 0.73),
        goal_frac=(0.84, 0.31),
    ),
    ScenarioSpec(
        name="hubei_jianghan",
        description="Jianghan plain / humid flatland near Wuhan (SRTM N30E114)",
        lat=30.55,
        lon=114.3,
        tile_lat=30,
        tile_lon=114,
        wind_start_date="2024-05-28",
        wind_end_date="2024-05-29",
        start_frac=(0.16, 0.69),
        goal_frac=(0.82, 0.32),
    ),
    ScenarioSpec(
        name="tibet_lhasa",
        description="Lhasa valley / high plateau corridor (SRTM N29E091)",
        lat=29.65,
        lon=91.15,
        tile_lat=29,
        tile_lon=91,
        wind_start_date="2024-01-25",
        wind_end_date="2024-01-26",
        start_frac=(0.15, 0.69),
        goal_frac=(0.81, 0.34),
    ),
    ScenarioSpec(
        name="jilin_forest",
        description="Changbai foothills / forested hills (SRTM N42E127)",
        lat=42.45,
        lon=127.25,
        tile_lat=42,
        tile_lon=127,
        wind_start_date="2024-09-18",
        wind_end_date="2024-09-19",
        start_frac=(0.14, 0.71),
        goal_frac=(0.85, 0.31),
    ),
    ScenarioSpec(
        name="neimeng_grass",
        description="Xilingol grassland / open steppe (SRTM N43E116)",
        lat=43.95,
        lon=116.08,
        tile_lat=43,
        tile_lon=116,
        wind_start_date="2024-06-08",
        wind_end_date="2024-06-09",
        start_frac=(0.16, 0.69),
        goal_frac=(0.82, 0.32),
    ),
    ScenarioSpec(
        name="yunnan_karst",
        description="Central Yunnan karst hills near Kunming (SRTM N25E102)",
        lat=25.05,
        lon=102.72,
        tile_lat=25,
        tile_lon=102,
        wind_start_date="2024-05-15",
        wind_end_date="2024-05-16",
        start_frac=(0.13, 0.75),
        goal_frac=(0.84, 0.29),
    ),
    # --- fresh holdout (overfit probe; do not tune on these) ---
    ScenarioSpec(
        name="zhejiang_hills",
        description="West Zhejiang humid hills near Quzhou (SRTM N29E119)",
        lat=29.55,
        lon=119.55,
        tile_lat=29,
        tile_lon=119,
        wind_start_date="2024-04-08",
        wind_end_date="2024-04-09",
        start_frac=(0.14, 0.72),
        goal_frac=(0.84, 0.30),
    ),
    ScenarioSpec(
        name="shandong_taishan",
        description="Tai Shan foothills / N China hills (SRTM N36E117)",
        lat=36.25,
        lon=117.15,
        tile_lat=36,
        tile_lon=117,
        wind_start_date="2024-10-22",
        wind_end_date="2024-10-23",
        start_frac=(0.15, 0.71),
        goal_frac=(0.84, 0.31),
    ),
    ScenarioSpec(
        name="chongqing_hills",
        description="Chongqing basin-edge humid hills (SRTM N29E106)",
        lat=29.55,
        lon=106.55,
        tile_lat=29,
        tile_lon=106,
        wind_start_date="2024-07-28",
        wind_end_date="2024-07-29",
        start_frac=(0.14, 0.73),
        goal_frac=(0.85, 0.30),
    ),
    ScenarioSpec(
        name="guangxi_guilin",
        description="Guilin karst tower plain / S China (SRTM N25E110)",
        lat=25.25,
        lon=110.35,
        tile_lat=25,
        tile_lon=110,
        wind_start_date="2024-08-14",
        wind_end_date="2024-08-15",
        start_frac=(0.14, 0.72),
        goal_frac=(0.84, 0.30),
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
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("WINDFARM_HTTP_PROXY") or ""
    if proxy:
        print(f"[srtm] downloading {path.name} via proxy {proxy}")
    else:
        print(f"[srtm] downloading {path.name} (direct; set HTTPS_PROXY for Clash)")
    with _urlopen(request, timeout_s) as response:
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
        with _urlopen(request, timeout_s) as response:
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


def mission_route_quartet(
    start: tuple[int, int] | list[int],
    goal: tuple[int, int] | list[int],
) -> list[dict]:
    """First four OD pairs (diagonals) of :func:`mission_route_octet`."""
    return mission_route_octet(start, goal)[:4]


def mission_route_octet(
    start: tuple[int, int] | list[int],
    goal: tuple[int, int] | list[int],
) -> list[dict]:
    """Eight OD pairs on the axis-aligned box spanned by the original start/goal.

    r0–r3: both diagonals, both directions (same L1 length as the primary mission).
    r4–r5: horizontal mid-edge crossing (west↔east at mid_y).
    r6–r7: vertical mid-edge crossing (south↔north at mid_x).
    """
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    c0 = (sx, sy)
    c1 = (gx, gy)
    c2 = (sx, gy)
    c3 = (gx, sy)
    mx = int(round(0.5 * (sx + gx)))
    my = int(round(0.5 * (sy + gy)))
    # Avoid degenerate midpoints collapsing onto a corner when start/goal share a coordinate.
    if mx == sx or mx == gx:
        mx = sx + (1 if gx > sx else -1)
    if my == sy or my == gy:
        my = sy + (1 if gy > sy else -1)
    west = (sx, my)
    east = (gx, my)
    south = (mx, sy)
    north = (mx, gy)
    return [
        {"id": "r0", "label": "primary", "start": list(c0), "goal": list(c1)},
        {"id": "r1", "label": "primary_rev", "start": list(c1), "goal": list(c0)},
        {"id": "r2", "label": "cross", "start": list(c2), "goal": list(c3)},
        {"id": "r3", "label": "cross_rev", "start": list(c3), "goal": list(c2)},
        {"id": "r4", "label": "horizontal", "start": list(west), "goal": list(east)},
        {"id": "r5", "label": "horizontal_rev", "start": list(east), "goal": list(west)},
        {"id": "r6", "label": "vertical", "start": list(south), "goal": list(north)},
        {"id": "r7", "label": "vertical_rev", "start": list(north), "goal": list(south)},
    ]


def mission_route_bundle(
    start: tuple[int, int] | list[int],
    goal: tuple[int, int] | list[int],
    *,
    width: int,
    height: int,
    n_routes: int = 56,
    seed: int = 7,
    margin: int = 4,
) -> list[dict]:
    """Octet plus stratified random ODs for large-scale tune catalogs.

    Keeps r0–r7 identical to :func:`mission_route_octet` (sealed holdout safe).
    Extra routes (r8+) sample headings and lengths near the primary L1 span,
    with endpoints clamped inside the map margin.
    """
    import random as _random

    base = mission_route_octet(start, goal)
    n_routes = max(int(n_routes), len(base))
    if n_routes == len(base):
        return base

    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    primary_l1 = max(abs(gx - sx) + abs(gy - sy), 8)
    lo_x, hi_x = int(margin), int(width) - 1 - int(margin)
    lo_y, hi_y = int(margin), int(height) - 1 - int(margin)
    if hi_x <= lo_x or hi_y <= lo_y:
        lo_x, hi_x = 1, max(int(width) - 2, 2)
        lo_y, hi_y = 1, max(int(height) - 2, 2)

    rng = _random.Random(int(seed) ^ (sx * 131 + sy * 17 + gx * 3 + gy))
    seen: set[tuple[int, int, int, int]] = set()
    for r in base:
        s0, g0 = r["start"], r["goal"]
        seen.add((int(s0[0]), int(s0[1]), int(g0[0]), int(g0[1])))

    out = list(base)
    # Length bands × direction strata for diversity (not pure i.i.d. noise).
    length_scales = (0.70, 0.85, 1.00, 1.15)
    attempts = 0
    max_attempts = max(2000, 40 * (n_routes - len(base)))
    while len(out) < n_routes and attempts < max_attempts:
        attempts += 1
        scale = length_scales[(len(out) - len(base)) % len(length_scales)]
        target = max(8.0, float(primary_l1) * float(scale))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        # Random start in the playable box, then step toward goal.
        x0 = rng.randint(lo_x, hi_x)
        y0 = rng.randint(lo_y, hi_y)
        x1 = int(round(x0 + target * math.cos(ang)))
        y1 = int(round(y0 + target * math.sin(ang)))
        x1 = clamp_int(x1, lo_x, hi_x)
        y1 = clamp_int(y1, lo_y, hi_y)
        if abs(x1 - x0) + abs(y1 - y0) < 8:
            continue
        key = (x0, y0, x1, y1)
        rev = (x1, y1, x0, y0)
        if key in seen or rev in seen:
            continue
        seen.add(key)
        idx = len(out)
        out.append(
            {
                "id": f"r{idx}",
                "label": f"expanded_{idx}",
                "start": [x0, y0],
                "goal": [x1, y1],
                "expanded": True,
            }
        )
    return out


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
