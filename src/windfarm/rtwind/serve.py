from __future__ import annotations

from pathlib import Path

import uvicorn

from .app import create_app
from .types import RtwindConfig, SourceKind


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
) -> None:
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
    )
    app = create_app(config)
    uvicorn.run(app, host=host, port=port, root_path=root_path or "", log_level="info")
