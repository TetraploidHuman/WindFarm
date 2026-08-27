from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .belief_runtime import BeliefRuntime
from .camera_hub import CameraHub
from .geo import GeoContext
from .hub import TelemetryHub
from .live_source import LiveIngestSource
from .plan_runtime import PlanRuntime
from .sim_source import SimDroneSource
from .types import RtwindConfig, SourceKind, TelemetryFrame


class SourceBody(BaseModel):
    active: SourceKind


class SimControlBody(BaseModel):
    action: Literal["reset", "start", "stop"] = "reset"
    lat: float | None = None
    lon: float | None = None


class BeliefResetBody(BaseModel):
    lat: float | None = None
    lon: float | None = None


class PrefetchBody(BaseModel):
    center_lat: float
    center_lon: float
    size_km: float = Field(default=10.0, ge=1.0, le=30.0)
    weather_nx: int = Field(default=11, ge=2, le=15)
    weather_ny: int = Field(default=11, ge=2, le=15)
    dem: bool = True
    weather: bool = False


class IngestBody(BaseModel):
    lat: float | None = None
    lon: float | None = None
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
    client_ts: float | None = None


def create_app(config: RtwindConfig | None = None) -> FastAPI:
    config = config or RtwindConfig()
    hub = TelemetryHub(config)
    geo = GeoContext(config.data_dir)
    sim = SimDroneSource(hub, config)
    live = LiveIngestSource(hub, config)
    belief = BeliefRuntime()
    plan = PlanRuntime()
    camera = CameraHub()
    static_dir = Path(__file__).resolve().parent / "static"

    async def _on_source_change(previous: SourceKind, current: SourceKind) -> None:
        plan.reset()
        if current == "sim":
            belief.reset(lat=sim.origin_lat, lon=sim.origin_lon)
            await camera.clear()
        else:
            belief.reset()
        if previous == "sim":
            await sim.stop()
        if previous == "live":
            await live.stop()
        if current == "sim" and config.sim_enabled:
            await sim.start()
        if current == "live":
            await live.start()

    _last_plan_check_mono = 0.0
    _last_belief_observe_mono = 0.0
    _last_belief_tick_mono = 0.0

    async def _on_frame(frame: TelemetryFrame) -> None:
        nonlocal _last_belief_observe_mono, _last_belief_tick_mono
        now = time.monotonic()
        if now - _last_belief_observe_mono >= 0.25:
            _last_belief_observe_mono = now
            await asyncio.to_thread(belief.observe_frame, frame)
        st = belief.status()
        obs = int(st.get("obs_count") or 0)
        if obs > 0 and obs % 32 == 0 and now - _last_belief_tick_mono >= 3.0:
            _last_belief_tick_mono = now
            await hub._broadcast({"type": "belief_tick", "status": st})
        if frame.lat is None or frame.lon is None:
            return
        nonlocal _last_plan_check_mono
        now = time.monotonic()
        if now - _last_plan_check_mono < 2.0:
            return
        _last_plan_check_mono = now
        upd = await asyncio.to_thread(
            plan.maybe_replan,
            frame,
            belief=belief,
            geo=geo,
            origin_lat=sim.origin_lat if hub.active == "sim" else None,
            origin_lon=sim.origin_lon if hub.active == "sim" else None,
        )
        if upd:
            await hub._broadcast(upd)
        elif plan.status().get("ready"):
            lam = await asyncio.to_thread(plan.update_lambda, frame.lat, frame.lon)
            await hub._broadcast({"type": "plan_progress", "lambda": round(lam, 4)})

    hub.add_source_listener(_on_source_change)
    hub.add_frame_listener(_on_frame)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.active_source == "sim" and config.sim_enabled:
            belief.reset(lat=sim.origin_lat, lon=sim.origin_lon)
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
    app.state.belief = belief
    app.state.plan = plan
    app.state.camera = camera
    app.state.config = config

    def _perception_status() -> dict[str, Any]:
        region = geo.region_status()
        prefetched = bool(region.get("prefetched"))
        dem = region.get("dem") or {}
        weather = region.get("weather") or {}
        dem_ok = prefetched and int(dem.get("ok") or 0) > 0
        weather_ok = prefetched and int(weather.get("ok") or 0) > 0
        return {
            "ready": dem_ok and weather_ok,
            "prefetched": prefetched,
            "dem_ok": dem_ok,
            "weather_ok": weather_ok,
            "partial": bool(region.get("partial")),
            "size_km": region.get("size_km"),
        }

    @app.get("/api/config/public")
    async def config_public() -> dict[str, Any]:
        return {
            "carto_basemap_key": config.carto_basemap_key or "",
        }

    @app.get("/api/system/layers")
    async def system_layers() -> dict[str, Any]:
        return {
            "perception": _perception_status(),
            "cognition": belief.status(),
            "planning": plan.status(),
        }

    @app.get("/api/plan/path")
    async def plan_path() -> dict[str, Any]:
        return plan.path_payload()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        payload = hub.health()
        payload["live"] = live.status()
        payload["belief"] = belief.status()
        payload["plan"] = plan.status()
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
    async def env_at(lat: float, lon: float, weather: bool = False) -> dict[str, Any]:
        """DEM from server; weather defaults off (browser fetches Open-Meteo directly)."""
        env = await asyncio.to_thread(geo.env_at, lat, lon, weather=weather)
        frame = hub.latest()
        payload = env.to_dict()
        if frame and env.dem_msl is not None and frame.alt_msl is not None:
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

    @app.post("/api/env/prefetch")
    async def env_prefetch(body: PrefetchBody) -> dict[str, Any]:
        """Pre-download a fixed mission patch (default 10×10 km) of DEM + weather."""
        result = await asyncio.to_thread(
            geo.prefetch_region,
            0.0,
            0.0,
            0.0,
            0.0,
            center_lat=body.center_lat,
            center_lon=body.center_lon,
            size_km=body.size_km,
            weather_nx=body.weather_nx,
            weather_ny=body.weather_ny,
            dem=body.dem,
            weather=body.weather,
        )
        if result.get("ok") or result.get("partial"):
            st = belief.status()
            clat, clon = st.get("center_lat"), st.get("center_lon")
            moved = (
                clat is None
                or clon is None
                or abs(clat - body.center_lat) > 0.0005
                or abs(clon - body.center_lon) > 0.0005
            )
            if moved:
                belief.reset(lat=body.center_lat, lon=body.center_lon)
                plan.reset()
        return result

    @app.post("/api/env/prefetch/stream")
    async def env_prefetch_stream(body: PrefetchBody) -> StreamingResponse:
        """NDJSON progress stream for mission-patch prefetch."""

        def _maybe_realign_belief(result: dict[str, Any]) -> None:
            if not (result.get("ok") or result.get("partial")):
                return
            st = belief.status()
            clat, clon = st.get("center_lat"), st.get("center_lon")
            moved = (
                clat is None
                or clon is None
                or abs(clat - body.center_lat) > 0.0005
                or abs(clon - body.center_lon) > 0.0005
            )
            if moved:
                belief.reset(lat=body.center_lat, lon=body.center_lon)
                plan.reset()

        async def gen():
            import json as _json
            from queue import Queue
            from threading import Thread

            q: Queue = Queue()

            def worker() -> None:
                try:
                    for ev in geo.prefetch_region_progress(
                        center_lat=body.center_lat,
                        center_lon=body.center_lon,
                        size_km=body.size_km,
                        weather_nx=body.weather_nx,
                        weather_ny=body.weather_ny,
                        dem=body.dem,
                        weather=body.weather,
                    ):
                        q.put(ev)
                except Exception as exc:
                    q.put({"type": "done", "ok": False, "error": str(exc), "pct": 100})
                finally:
                    q.put(None)

            Thread(target=worker, daemon=True).start()
            while True:
                ev = await asyncio.to_thread(q.get)
                if ev is None:
                    break
                if ev.get("type") == "done":
                    _maybe_realign_belief(ev)
                yield _json.dumps(ev, ensure_ascii=False) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.get("/api/env/region")
    async def env_region() -> dict[str, Any]:
        return geo.region_status()

    @app.get("/api/camera/status")
    async def camera_status() -> dict[str, Any]:
        return await camera.status()

    @app.get("/api/camera/latest")
    async def camera_latest() -> Response:
        data = await camera.latest_bytes()
        if not data:
            return Response(status_code=404, content="no camera frame yet")
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/camera/upload")
    async def camera_upload(request: Request, vehicle_id: str = "live-1") -> dict[str, Any]:
        body = await request.body()
        return await camera.put(body, vehicle_id=vehicle_id)

    @app.get("/api/camera/stream")
    async def camera_stream() -> StreamingResponse:
        async def gen():
            while True:
                data = await camera.latest_bytes()
                if data:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    )
                await asyncio.sleep(0.15)

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/belief/status")
    async def belief_status() -> dict[str, Any]:
        return belief.status()

    @app.get("/api/belief/field")
    async def belief_field(layer: str = "energy", level: int = 0) -> dict[str, Any]:
        return await asyncio.to_thread(belief.field_geo, layer, level=level)

    @app.post("/api/belief/reset")
    async def belief_reset(body: BeliefResetBody | None = None) -> dict[str, Any]:
        body = body or BeliefResetBody()
        belief.reset(lat=body.lat, lon=body.lon)
        plan.reset()
        return {"ok": True, **belief.status()}

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
            belief.reset(lat=sim.origin_lat, lon=sim.origin_lon)
            plan.reset()
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
