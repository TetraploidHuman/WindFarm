from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .geo import GeoContext
from .hub import TelemetryHub
from .live_source import LiveIngestSource
from .sim_source import SimDroneSource
from .types import RtwindConfig, SourceKind


class SourceBody(BaseModel):
    active: SourceKind


class SimControlBody(BaseModel):
    action: Literal["reset", "start", "stop"] = "reset"
    lat: float | None = None
    lon: float | None = None


class IngestBody(BaseModel):
    lat: float
    lon: float
    alt_msl: float | None = None
    alt: float | None = None
    heading: float | None = None
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float | None = None
    airspeed: float = 0.0
    groundspeed: float | None = None
    climb_rate: float = 0.0
    battery: float | None = None
    vehicle_id: str = "live-1"
    t: str | None = None
    alt_agl: float | None = None


def create_app(config: RtwindConfig | None = None) -> FastAPI:
    config = config or RtwindConfig()
    hub = TelemetryHub(config)
    geo = GeoContext(config.data_dir)
    sim = SimDroneSource(hub, config)
    live = LiveIngestSource(hub, config)
    static_dir = Path(__file__).resolve().parent / "static"

    async def _on_source_change(previous: SourceKind, current: SourceKind) -> None:
        if previous == "sim":
            await sim.stop()
        if previous == "live":
            await live.stop()
        if current == "sim" and config.sim_enabled:
            await sim.start()
        if current == "live":
            await live.start()

    hub.add_source_listener(_on_source_change)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.active_source == "sim" and config.sim_enabled:
            await sim.start()
        elif config.active_source == "live":
            await live.start()
        heartbeat = asyncio.create_task(_heartbeat_loop(hub), name="rtwind-hb")
        try:
            yield
        finally:
            heartbeat.cancel()
            await sim.stop()
            await live.stop()

    app = FastAPI(title="WindFarm rtwind", lifespan=lifespan)
    app.state.hub = hub
    app.state.geo = geo
    app.state.sim = sim
    app.state.live = live
    app.state.config = config

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        payload = hub.health()
        payload["live"] = live.status()
        return payload

    @app.get("/api/source")
    async def get_source() -> dict[str, Any]:
        snap = hub.source_snapshot()
        snap["live"] = live.status()
        return snap

    @app.post("/api/source")
    async def post_source(body: SourceBody) -> dict[str, Any]:
        if body.active == "sim" and not config.sim_enabled:
            return {"error": "sim disabled", **hub.source_snapshot()}
        return await hub.set_active(body.active, clear_track=True)

    @app.get("/api/telemetry/latest")
    async def telemetry_latest() -> dict[str, Any]:
        frame = hub.latest()
        return {"active": hub.active, "frame": frame.to_dict() if frame else None}

    @app.get("/api/telemetry/track")
    async def telemetry_track(since: int = -1, limit: int = 500) -> dict[str, Any]:
        return {"active": hub.active, "track": hub.track(since=since, limit=limit)}

    @app.get("/api/env/at")
    async def env_at(lat: float, lon: float) -> dict[str, Any]:
        env = await asyncio.to_thread(geo.env_at, lat, lon)
        # Enrich AGL on latest if positions close
        frame = hub.latest()
        payload = env.to_dict()
        if frame and env.dem_msl is not None:
            payload["alt_agl_estimate"] = frame.alt_msl - env.dem_msl
        return payload

    @app.get("/api/env/dem")
    async def env_dem(
        south: float,
        west: float,
        north: float,
        east: float,
        nx: int = 24,
        ny: int = 24,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(geo.dem_grid, south, west, north, east, nx=nx, ny=ny)

    @app.get("/api/env/wind-field")
    async def env_wind_field(
        south: float,
        west: float,
        north: float,
        east: float,
        nx: int = 6,
        ny: int = 6,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(geo.wind_field, south, west, north, east, nx=nx, ny=ny)

    @app.post("/api/ingest")
    async def ingest(body: IngestBody) -> dict[str, Any]:
        data = body.model_dump(exclude_none=True)
        if "alt_msl" not in data and "alt" in data:
            data["alt_msl"] = data["alt"]
        frame = await live.ingest(data)
        return {
            "accepted": frame is not None,
            "active": hub.active,
            "frame": frame.to_dict() if frame else None,
            "hint": None if hub.active == "live" else "switch active source to live to display ingest",
        }

    @app.get("/api/sim")
    async def get_sim() -> dict[str, Any]:
        return {"ok": True, **sim.snapshot(), "active": hub.active}

    @app.post("/api/sim/control")
    async def sim_control(body: SimControlBody) -> dict[str, Any]:
        if body.action == "reset":
            sim.reset(lat=body.lat, lon=body.lon)
            if hub.active == "sim":
                await hub.set_active("sim", clear_track=True)
                await sim.start()
        elif body.action == "start" and config.sim_enabled:
            if hub.active != "sim":
                await hub.set_active("sim", clear_track=True)
            else:
                await sim.start()
        elif body.action == "stop":
            await sim.stop()
        return {"ok": True, "sim_running": sim.running, **sim.snapshot(), **hub.source_snapshot()}

    @app.websocket("/api/ws/ingest")
    async def ws_ingest_endpoint(websocket: WebSocket) -> None:
        """UAV/phone upload-only socket: no dashboard fan-out on this connection."""
        await websocket.accept()
        try:
            await websocket.send_json({"type": "hello", "role": "ingest", "active": hub.active})
            while True:
                msg = await websocket.receive_json()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "ingest":
                    frame_body = msg.get("frame", msg)
                    if not isinstance(frame_body, dict):
                        continue
                    payload = dict(frame_body)
                elif "lat" in msg and "lon" in msg:
                    payload = dict(msg)
                else:
                    continue
                if "alt_msl" not in payload and "alt" in payload:
                    payload["alt_msl"] = payload["alt"]
                await live.ingest(payload)
        except WebSocketDisconnect:
            pass

    @app.websocket("/api/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        q = hub.subscribe()

        async def _ingest_reader() -> None:
            """Backward-compatible: dashboard WS can also accept ingest frames."""
            while True:
                msg = await websocket.receive_json()
                if not isinstance(msg, dict):
                    continue
                payload: dict[str, Any]
                if msg.get("type") == "ingest":
                    frame_body = msg.get("frame", msg)
                    if not isinstance(frame_body, dict):
                        continue
                    payload = dict(frame_body)
                elif "lat" in msg and "lon" in msg:
                    payload = dict(msg)
                else:
                    continue
                if "alt_msl" not in payload and "alt" in payload:
                    payload["alt_msl"] = payload["alt"]
                await live.ingest(payload)

        reader_task = asyncio.create_task(_ingest_reader(), name="rtwind-ws-ingest")
        try:
            await websocket.send_json({"type": "hello", **hub.source_snapshot(), "health": hub.health()})
            latest = hub.latest()
            if latest:
                await websocket.send_json({"type": "telemetry", "frame": latest.to_dict()})
            track = hub.track(since=-1, limit=400)
            if track:
                await websocket.send_json({"type": "track_snapshot", "track": track})
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat", "active": hub.active})
        except WebSocketDisconnect:
            pass
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            hub.unsubscribe(q)

    @app.get("/")
    async def index() -> Response:
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-cache"},
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


async def _heartbeat_loop(hub: TelemetryHub) -> None:
    while True:
        await asyncio.sleep(20.0)
        await hub._broadcast({"type": "heartbeat", "active": hub.active})
