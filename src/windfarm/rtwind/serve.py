from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .app import create_app
from .types import RtwindConfig, SourceKind


def _load_rtwind_local_env() -> None:
    """Load project-root `.rtwind.local.env` when env vars are not already set."""
    root = Path(__file__).resolve().parents[3]
    for name in (".rtwind.local.env", "rtwind.local.env"):
        path = root / name
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def serve_rtwind(
    host: str = "127.0.0.1",
    port: int = 8877,
    *,
    active_source: SourceKind = "sim",
    sim_enabled: bool = True,
    sim_origin_lat: float = 27.908,
    sim_origin_lon: float = 112.922,
    data_dir: str | Path = "data",
    root_path: str = "",
    mavlink_enabled: bool = True,
    mavlink_bind: str = "0.0.0.0",
    mavlink_udp_port: int = 14550,
    carto_basemap_key: str | None = None,
) -> None:
    _load_rtwind_local_env()
    carto_key = carto_basemap_key if carto_basemap_key is not None else os.environ.get("RTWIND_CARTO_BASEMAP_KEY", "")
    config = RtwindConfig(
        host=host,
        port=port,
        active_source=active_source,
        sim_enabled=sim_enabled,
        sim_origin_lat=sim_origin_lat,
        sim_origin_lon=sim_origin_lon,
        data_dir=str(data_dir),
        root_path=root_path,
        mavlink_enabled=mavlink_enabled,
        mavlink_bind=mavlink_bind,
        mavlink_udp_port=mavlink_udp_port,
        carto_basemap_key=carto_key,
    )
    app = create_app(config)
    uvicorn.run(app, host=host, port=port, root_path=root_path or "", log_level="info")
