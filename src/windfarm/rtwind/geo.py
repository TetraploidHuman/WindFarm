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
MISSION_SIZE_KM = 10.0


def bbox_km(center_lat: float, center_lon: float, size_km: float = MISSION_SIZE_KM) -> dict[str, float]:
    """Square bbox centered on (lat, lon) with side length size_km."""
    half_m = 0.5 * size_km * 1000.0
    mlat = 111_320.0
    mlon = 111_320.0 * math.cos(math.radians(center_lat))
    dlat = half_m / mlat
    dlon = half_m / mlon
    return {
        "south": center_lat - dlat,
        "north": center_lat + dlat,
        "west": center_lon - dlon,
        "east": center_lon + dlon,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "size_km": size_km,
    }


class GeoContext:
    """Real DEM (SRTM) + Open-Meteo forecast at a lat/lon (shared by Live/Sim)."""

    def __init__(self, data_dir: str | Path = "data", cache_ttl_s: float = 600.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_s = cache_ttl_s
        self._weather_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
        self._dem_cache: dict[str, list[list[float]]] = {}
        self._last_dem_error: str | None = None
        self._region_status: dict[str, Any] | None = None

    def _tile_corners(self, lat: float, lon: float) -> tuple[int, int]:
        tile_lat = math.floor(lat)
        tile_lon = math.floor(lon)
        return tile_lat, tile_lon

    def list_srtm_tiles(self, south: float, west: float, north: float, east: float) -> list[tuple[int, int]]:
        south, north = min(south, north), max(south, north)
        west, east = min(west, east), max(west, east)
        lat0 = math.floor(south)
        lat1 = math.floor(max(south, north - 1e-9))
        lon0 = math.floor(west)
        lon1 = math.floor(max(west, east - 1e-9))
        tiles: list[tuple[int, int]] = []
        for tlat in range(lat0, lat1 + 1):
            for tlon in range(lon0, lon1 + 1):
                tiles.append((tlat, tlon))
        return tiles

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

    def _weather_from_current(self, current: dict[str, Any]) -> dict[str, Any]:
        spd = current.get("wind_speed_10m")
        deg = current.get("wind_direction_10m")
        temp = current.get("temperature_2m")
        u = v = None
        if spd is not None and deg is not None:
            rad = math.radians(float(deg) + 180.0)
            u = float(spd) * math.sin(rad)
            v = float(spd) * math.cos(rad)
        return {
            "wind_speed_mps": float(spd) if spd is not None else None,
            "wind_dir_deg": float(deg) if deg is not None else None,
            "wind_u": u,
            "wind_v": v,
            "temperature_c": float(temp) if temp is not None else None,
            "_cached": False,
        }

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
        result = self._weather_from_current(current)
        self._weather_cache[(qlat, qlon)] = (now, result)
        return result

    def _fetch_weather_many(self, points: list[tuple[float, float]]) -> dict[tuple[float, float], dict[str, Any]]:
        """Fetch weather for unique rounded points; uses cache and Open-Meteo multi-location."""
        now = time.monotonic()
        out: dict[tuple[float, float], dict[str, Any]] = {}
        need: list[tuple[float, float]] = []
        for lat, lon in points:
            key = (round(lat, 3), round(lon, 3))
            if key in out:
                continue
            cached = self._weather_cache.get(key)
            if cached and now - cached[0] < self.cache_ttl_s:
                out[key] = {**cached[1], "_cached": True}
            elif key not in need:
                need.append(key)

        chunk_size = 40
        for i in range(0, len(need), chunk_size):
            chunk = need[i : i + chunk_size]
            if len(chunk) == 1:
                qlat, qlon = chunk[0]
                try:
                    out[(qlat, qlon)] = self._fetch_weather(qlat, qlon)
                except Exception:
                    out[(qlat, qlon)] = {}
                continue
            lats = ",".join(str(p[0]) for p in chunk)
            lons = ",".join(str(p[1]) for p in chunk)
            params = urllib.parse.urlencode(
                {
                    "latitude": lats,
                    "longitude": lons,
                    "current": "temperature_2m,wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms",
                    "timezone": "UTC",
                }
            )
            req = urllib.request.Request(
                f"{OPEN_METEO_FORECAST}?{params}",
                headers={"User-Agent": "WindFarm-rtwind/0.1"},
            )
            try:
                with _urlopen(req, timeout=20.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, list):
                    rows = payload
                else:
                    cur = payload.get("current") or {}
                    n = len(chunk)
                    rows = []
                    for j in range(n):
                        row_cur: dict[str, Any] = {}
                        for k in ("temperature_2m", "wind_speed_10m", "wind_direction_10m"):
                            val = cur.get(k)
                            if isinstance(val, list) and j < len(val):
                                row_cur[k] = val[j]
                            elif not isinstance(val, list):
                                row_cur[k] = val
                        rows.append({"current": row_cur})
                for j, point in enumerate(chunk):
                    if j >= len(rows):
                        out[point] = {}
                        continue
                    row = rows[j]
                    current = row.get("current") if isinstance(row, dict) else {}
                    if not isinstance(current, dict):
                        current = {}
                    result = self._weather_from_current(current)
                    self._weather_cache[point] = (now, result)
                    out[point] = result
            except Exception:
                for point in chunk:
                    if point in out:
                        continue
                    try:
                        out[point] = self._fetch_weather(point[0], point[1])
                    except Exception:
                        out[point] = {}
        return out

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
        missing = 0
        for lat in lats:
            row: list[float | None] = []
            for lon in lons:
                v = self.elevation_msl(lat, lon)
                row.append(v)
                if v is None:
                    missing += 1
                else:
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
            "missing": missing,
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
        points = [(lat, lon) for lat in lats for lon in lons]
        weather_map = self._fetch_weather_many(points)
        stale_any = False
        missing = 0
        vectors: list[dict[str, Any]] = []
        for lat, lon in points:
            key = (round(lat, 3), round(lon, 3))
            w = weather_map.get(key) or {}
            if w.get("_cached"):
                stale_any = True
            if w.get("wind_u") is None and w.get("wind_v") is None:
                missing += 1
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
            "missing": missing,
            "source": "open-meteo-forecast",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def prefetch_region_progress(
        self,
        *,
        center_lat: float,
        center_lon: float,
        size_km: float = MISSION_SIZE_KM,
        weather_nx: int = 11,
        weather_ny: int = 11,
        dem: bool = True,
        weather: bool = True,
    ):
        """Yield progress dicts, then a final ``done`` event with the result payload."""
        box = bbox_km(center_lat, center_lon, size_km)
        south, north = box["south"], box["north"]
        west, east = box["west"], box["east"]
        lat_span = north - south
        lon_span = east - west
        if lat_span > 3.5 or lon_span > 3.5:
            yield {
                "type": "done",
                "ok": False,
                "error": f"区域过大（{lat_span:.2f}°×{lon_span:.2f}°），请检查中心点",
                "bounds": {"south": south, "west": west, "north": north, "east": east},
                "pct": 100,
            }
            return

        tiles = self.list_srtm_tiles(south, west, north, east) if dem else []
        weather_nx = max(2, min(int(weather_nx), 15))
        weather_ny = max(2, min(int(weather_ny), 15))
        weather_steps = 1 if weather else 0
        total = max(len(tiles) + weather_steps, 1)
        done_steps = 0

        yield {
            "type": "start",
            "pct": 0,
            "message": f"准备下载 {size_km:g}×{size_km:g} km 任务区",
            "center_lat": center_lat,
            "center_lon": center_lon,
            "size_km": size_km,
            "dem_tiles": len(tiles),
            "weather_points": weather_nx * weather_ny if weather else 0,
        }

        tile_results: list[dict[str, Any]] = []
        for tlat, tlon in tiles:
            name = tile_name(tlat, tlon)
            yield {
                "type": "dem",
                "pct": int(100 * done_steps / total),
                "message": f"地形瓦片 {name}…",
                "tile": name,
            }
            try:
                path = ensure_srtm_tile(self.data_dir, tlat, tlon)
                if name not in self._dem_cache:
                    self._dem_cache[name] = read_hgt_elevation(path)
                tile_results.append({"tile": name, "ok": True, "path": str(path.name)})
            except Exception as exc:
                tile_results.append({"tile": name, "ok": False, "error": str(exc)})
            done_steps += 1
            yield {
                "type": "dem",
                "pct": int(100 * done_steps / total),
                "message": f"地形瓦片 {name} 完成",
                "tile": name,
                "ok": tile_results[-1].get("ok", False),
            }

        weather_ok = 0
        weather_fail = 0
        if weather:
            yield {
                "type": "weather",
                "pct": int(100 * done_steps / total),
                "message": f"气象网格 {weather_nx}×{weather_ny}…",
            }
            lats = [south + (north - south) * i / (weather_ny - 1) for i in range(weather_ny)]
            lons = [west + (east - west) * j / (weather_nx - 1) for j in range(weather_nx)]
            points = [(lat, lon) for lat in lats for lon in lons]
            got = self._fetch_weather_many(points)
            for lat, lon in points:
                key = (round(lat, 3), round(lon, 3))
                w = got.get(key) or {}
                if w.get("wind_u") is not None or w.get("wind_speed_mps") is not None:
                    weather_ok += 1
                else:
                    weather_fail += 1
            done_steps += 1
            yield {
                "type": "weather",
                "pct": int(100 * done_steps / total),
                "message": f"气象完成 {weather_ok}/{weather_ok + weather_fail}",
            }

        dem_ok = sum(1 for t in tile_results if t.get("ok"))
        dem_fail = sum(1 for t in tile_results if not t.get("ok"))
        partial = dem_ok > 0 or weather_ok > 0
        result = {
            "ok": dem_fail == 0 and (not weather or weather_fail == 0),
            "partial": partial and not (dem_fail == 0 and (not weather or weather_fail == 0)),
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "center_lat": center_lat,
            "center_lon": center_lon,
            "size_km": size_km,
            "dem": {
                "requested": len(tile_results),
                "ok": dem_ok,
                "failed": dem_fail,
                "tiles": tile_results,
            },
            "weather": {
                "requested": weather_nx * weather_ny if weather else 0,
                "ok": weather_ok,
                "failed": weather_fail,
                "nx": weather_nx,
                "ny": weather_ny,
            },
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        self._region_status = result
        yield {"type": "done", "pct": 100, "message": "下载完成", **result}

    def prefetch_region(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        weather_nx: int = 11,
        weather_ny: int = 11,
        dem: bool = True,
        weather: bool = True,
        center_lat: float | None = None,
        center_lon: float | None = None,
        size_km: float = MISSION_SIZE_KM,
    ) -> dict[str, Any]:
        """Download SRTM tiles + weather samples for a mission patch (default 10×10 km)."""
        if center_lat is None or center_lon is None:
            center_lat = (min(south, north) + max(south, north)) / 2.0
            center_lon = (min(west, east) + max(west, east)) / 2.0
        final: dict[str, Any] = {"ok": False}
        for ev in self.prefetch_region_progress(
            center_lat=center_lat,
            center_lon=center_lon,
            size_km=size_km,
            weather_nx=weather_nx,
            weather_ny=weather_ny,
            dem=dem,
            weather=weather,
        ):
            if ev.get("type") == "done":
                final = ev
        return {k: v for k, v in final.items() if k not in {"type", "pct", "message"}}

    def region_status(self) -> dict[str, Any]:
        if self._region_status is None:
            return {"prefetched": False}
        return {"prefetched": True, **self._region_status}
