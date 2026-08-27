from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SourceKind = Literal["live", "sim"]
LinkState = Literal["ok", "waiting", "stale", "offline"]


@dataclass(slots=True)
class TelemetryFrame:
    source: SourceKind
    vehicle_id: str
    t: str
    heading: float
    roll: float
    pitch: float
    yaw: float
    airspeed: float
    groundspeed: float
    climb_rate: float
    lat: float | None = None
    lon: float | None = None
    alt_msl: float | None = None
    battery: float | None = None
    link: LinkState = "ok"
    seq: int = 0
    alt_agl: float | None = None
    client_ts: float | None = None  # phone unix epoch seconds for e2e latency

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrackPoint:
    t: str
    lat: float
    lon: float
    alt_msl: float
    heading: float
    seq: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnvAtPoint:
    lat: float
    lon: float
    dem_msl: float | None
    wind_speed_mps: float | None
    wind_dir_deg: float | None
    wind_u: float | None
    wind_v: float | None
    temperature_c: float | None
    fetched_at: str
    stale: bool = False
    dem_source: str = "srtm1"
    weather_source: str = "open-meteo-forecast"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RtwindConfig:
    host: str = "127.0.0.1"
    port: int = 8877
    active_source: SourceKind = "sim"
    sim_enabled: bool = True
    sim_origin_lat: float = 27.908
    sim_origin_lon: float = 112.922
    sim_hz: float = 5.0
    track_max: int = 2400
    live_stale_seconds: float = 3.0
    mavlink_enabled: bool = True
    mavlink_bind: str = "0.0.0.0"
    mavlink_udp_port: int = 14550
    mavlink_publish_hz: float = 10.0
    data_dir: str = "data"
    root_path: str = ""
    carto_basemap_key: str = ""
