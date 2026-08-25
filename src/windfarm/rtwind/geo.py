from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data_ingest import (
    SRTM_SIZE,
    _bilinear,
    _urlopen,
    ensure_srtm_tile,
    read_hgt_elevation,
    tile_name,
)
from .types import EnvAtPoint

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


class GeoContext:
    """Real DEM (SRTM) + Open-Meteo forecast at a lat/lon (shared by Live/Sim)."""

    def __init__(self, data_dir: str | Path = "data", cache_ttl_s: float = 600.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_s = cache_ttl_s
        self._weather_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
        self._dem_cache: dict[str, list[list[float]]] = {}
        self._last_dem_error: str | None = None

    def _tile_corners(self, lat: float, lon: float) -> tuple[int, int]:
        tile_lat = math.floor(lat)
        tile_lon = math.floor(lon)
        return tile_lat, tile_lon

    def elevation_msl(self, lat: float, lon: float) -> float | None:
        try:
            tile_lat, tile_lon = self._tile_corners(lat, lon)
            key = tile_name(tile_lat, tile_lon)
            if key not in self._dem_cache:
                path = ensure_srtm_tile(self.data_dir, tile_lat, tile_lon)
                self._dem_cache[key] = read_hgt_elevation(path)
            grid = self._dem_cache[key]
            px = (lon - tile_lon) * (SRTM_SIZE - 1)
            py = (tile_lat + 1 - lat) * (SRTM_SIZE - 1)
            self._last_dem_error = None
            return float(_bilinear(grid, px, py))
        except Exception as exc:
            self._last_dem_error = f"dem: {exc}"
            return None

    def _fetch_weather(self, lat: float, lon: float) -> dict[str, Any]:
        qlat = round(lat, 3)
        qlon = round(lon, 3)
        cached = self._weather_cache.get((qlat, qlon))
        now = time.monotonic()
        if cached and now - cached[0] < self.cache_ttl_s:
            return {**cached[1], "_cached": True}
        params = urllib.parse.urlencode(
            {
                "latitude": qlat,
                "longitude": qlon,
                "current": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
        )
        req = urllib.request.Request(
            f"{OPEN_METEO_FORECAST}?{params}",
            headers={"User-Agent": "WindFarm-rtwind/0.1"},
        )
        with _urlopen(req, timeout=12.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        current = payload.get("current") or {}
        spd = current.get("wind_speed_10m")
        deg = current.get("wind_direction_10m")
        temp = current.get("temperature_2m")
        u = v = None
        if spd is not None and deg is not None:
            # Meteorological from-direction → to-vector (east/north)
            rad = math.radians(float(deg) + 180.0)
            u = float(spd) * math.sin(rad)
            v = float(spd) * math.cos(rad)
        result = {
            "wind_speed_mps": float(spd) if spd is not None else None,
            "wind_dir_deg": float(deg) if deg is not None else None,
            "wind_u": u,
            "wind_v": v,
            "temperature_c": float(temp) if temp is not None else None,
            "_cached": False,
        }
        self._weather_cache[(qlat, qlon)] = (now, result)
        return result

    def env_at(self, lat: float, lon: float) -> EnvAtPoint:
        fetched = datetime.now(timezone.utc).isoformat()
        dem = self.elevation_msl(lat, lon)
        errs: list[str] = []
        if dem is None and self._last_dem_error:
            errs.append(self._last_dem_error)
        weather: dict[str, Any] = {}
        stale = False
        try:
            weather = self._fetch_weather(lat, lon)
            stale = bool(weather.get("_cached"))
        except Exception as exc:
            errs.append(f"weather: {exc}")
            # try last cache even if expired
            qlat, qlon = round(lat, 3), round(lon, 3)
            cached = self._weather_cache.get((qlat, qlon))
            if cached:
                weather = cached[1]
                stale = True
        return EnvAtPoint(
            lat=lat,
            lon=lon,
            dem_msl=dem,
            wind_speed_mps=weather.get("wind_speed_mps"),
            wind_dir_deg=weather.get("wind_dir_deg"),
            wind_u=weather.get("wind_u"),
            wind_v=weather.get("wind_v"),
            temperature_c=weather.get("temperature_c"),
            fetched_at=fetched,
            stale=stale,
            error="; ".join(errs) if errs else None,
        )

    def dem_grid(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        nx: int = 24,
        ny: int = 24,
    ) -> dict[str, Any]:
        nx = max(2, min(int(nx), 64))
        ny = max(2, min(int(ny), 64))
        south, north = min(south, north), max(south, north)
        west, east = min(west, east), max(west, east)
        lats = [south + (north - south) * i / (ny - 1) for i in range(ny)]
        lons = [west + (east - west) * j / (nx - 1) for j in range(nx)]
        values: list[list[float | None]] = []
        vmin: float | None = None
        vmax: float | None = None
        for lat in lats:
            row: list[float | None] = []
            for lon in lons:
                v = self.elevation_msl(lat, lon)
                row.append(v)
                if v is not None:
                    vmin = v if vmin is None else min(vmin, v)
                    vmax = v if vmax is None else max(vmax, v)
            values.append(row)
        return {
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "nx": nx,
            "ny": ny,
            "lats": lats,
            "lons": lons,
            "values": values,
            "min": vmin,
            "max": vmax,
            "source": "srtm1",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def wind_field(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        nx: int = 6,
        ny: int = 6,
    ) -> dict[str, Any]:
        nx = max(2, min(int(nx), 12))
        ny = max(2, min(int(ny), 12))
        south, north = min(south, north), max(south, north)
        west, east = min(west, east), max(west, east)
        lats = [south + (north - south) * i / (ny - 1) for i in range(ny)]
        lons = [west + (east - west) * j / (nx - 1) for j in range(nx)]
        # One Open-Meteo call at bbox center (small areas: wind is near-uniform at 10 m).
        stale_any = False
        center_lat = (south + north) / 2.0
        center_lon = (west + east) / 2.0
        try:
            w = self._fetch_weather(center_lat, center_lon)
            stale_any = bool(w.get("_cached"))
        except Exception:
            w = {}
        vectors: list[dict[str, Any]] = []
        for lat in lats:
            for lon in lons:
                vectors.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "u": w.get("wind_u"),
                        "v": w.get("wind_v"),
                        "speed_mps": w.get("wind_speed_mps"),
                        "dir_deg": w.get("wind_dir_deg"),
                    }
                )
        return {
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "nx": nx,
            "ny": ny,
            "vectors": vectors,
            "stale": stale_any,
            "source": "open-meteo-forecast",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
