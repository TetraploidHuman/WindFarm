from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_MOUNT_PREFIXES = ("/wind",)


def _public_path(path: str) -> str:
    for prefix in _MOUNT_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            stripped = path[len(prefix):]
            return stripped if stripped else "/"
    return path

from .config import TaskConfig
from .dashboard import build_live_dashboard_html
from .execution import NavigationEngine, _round_state_value, _state_level
from .io import read_json
from .pipeline import WindFarmPipeline
from .simulator import EnvironmentSimulator
from .types import CoarseWindSample, Observation


class LiveReplay:
    def __init__(self, report: dict, interval_seconds: float):
        self.report = report
        self.interval_seconds = interval_seconds
        self._published = 1 if report.get("trace") else 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        trace = self.report.get("trace", [])
        while self._running and self._published < len(trace):
            time.sleep(self.interval_seconds)
            with self._lock:
                self._published = min(self._published + 1, len(trace))

    def snapshot(self) -> dict:
        with self._lock:
            published = self._published
        partial = dict(self.report)
        partial["trace"] = self.report.get("trace", [])[:published]
        summary = _summary_from_trace(partial)
        return {
            "report": _report_meta(partial),
            "summary": summary,
            "trace": partial["trace"],
            "trace_append": [],
            "latest_seq": published - 1,
            "stream_done": published >= len(self.report.get("trace", [])),
            "capabilities": {"interactive_goal": False},
        }

    def snapshot_since(self, since: int) -> dict:
        snapshot = self.snapshot()
        trace = snapshot.pop("trace")
        start = max(0, since + 1)
        return {
            "summary": snapshot["summary"],
            "report": snapshot["report"],
            "trace_append": trace[start:],
            "latest_seq": snapshot["latest_seq"],
            "stream_done": snapshot["stream_done"],
            "capabilities": snapshot["capabilities"],
        }

    def stop(self) -> None:
        self._running = False


class InteractiveSimulation:
    def __init__(
        self,
        pipeline: WindFarmPipeline,
        simulator: EnvironmentSimulator,
        config: TaskConfig,
        interval_seconds: float,
        sim_minutes_per_tick: int,
    ):
        self.pipeline = pipeline
        self.simulator = simulator
        self.config = config
        self.engine = NavigationEngine(pipeline, config)
        self.interval_seconds = interval_seconds
        self.sim_minutes_per_tick = max(1, sim_minutes_per_tick)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._random = random.Random(11)
        self._boot()

    def _boot(self) -> None:
        terrain = self.pipeline.terrain
        self.width = len(terrain.elevation[0]) if terrain.elevation else 0
        self.height = len(terrain.elevation)
        self.current_time = datetime.fromisoformat(self.config.simulation.start_time)
        self.minute_index = self.current_time.hour * 60 + self.current_time.minute
        self.context = self.engine.create_context(self.config.mission.start, self.config.mission.goal)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def set_goal(self, goal: tuple[int, int] | tuple[int, int, int]) -> None:
        with self._lock:
            self.context.mission.goal = _clamp_goal(goal, self.width, self.height, self.config)
            self.engine._replan(self.context)
            self._refresh_latest_trace_frame()

    def snapshot(self) -> dict:
        with self._lock:
            report = self._report()
            latest_frame = self.context.trace[-1:] if self.context.trace else []
            return {
                "report": _report_meta(report),
                "summary": _summary_from_trace(report),
                "trace": latest_frame,
                "trace_append": [],
                "latest_seq": len(self.context.trace) - 1,
                "stream_done": False,
                "capabilities": {"interactive_goal": True},
            }

    def snapshot_since(self, since: int) -> dict:
        with self._lock:
            report = self._report()
            start = max(0, since + 1)
            return {
                "report": _report_meta(report),
                "summary": _summary_from_trace(report),
                "trace_append": self.context.trace[start:],
                "latest_seq": len(self.context.trace) - 1,
                "stream_done": False,
                "capabilities": {"interactive_goal": True},
            }

    def command_snapshot(self) -> dict:
        with self._lock:
            report = self._report()
            latest_seq = len(self.context.trace) - 1
            latest_frame = self.context.trace[-1] if self.context.trace else None
            return {
                "ok": True,
                "goal": list(self.context.mission.goal),
                "report": _report_meta(report),
                "summary": _summary_from_trace(report),
                "latest_frame": latest_frame,
                "latest_seq": latest_seq,
                "capabilities": {"interactive_goal": True},
            }

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                self._advance_one_tick()
            time.sleep(self.interval_seconds)

    def _advance_one_tick(self) -> None:
        timestamp = self.current_time.isoformat(timespec="seconds")
        u_km, v_km, w_km = self.simulator.coarse_wind_at(self.minute_index)
        next_step = self.context.step + 1
        keyframe_interval = max(1, self.config.simulation.report_keyframe_interval)
        is_keyframe = next_step == 1 or next_step % keyframe_interval == 0
        truth_payload = None
        observation = self._observe_current_cell(timestamp, u_km, v_km, w_km)
        if is_keyframe:
            truth_u, truth_v, truth_w = self.simulator.truth_field(
                self.pipeline.terrain,
                u_km,
                v_km,
                self.minute_index,
                w_km,
                self.config.model.altitude_levels,
            )
            truth_payload = {"timestamp": timestamp, "u": truth_u, "v": truth_v, "w": truth_w}
        self.engine.step(
            self.context,
            CoarseWindSample(timestamp=timestamp, u_km=u_km, v_km=v_km, w_km=w_km),
            observation,
            truth_payload,
        )
        if len(self.context.trace) > 720:
            self.context.trace = self.context.trace[-720:]
        self.current_time += timedelta(minutes=self.sim_minutes_per_tick)
        self.minute_index += self.sim_minutes_per_tick

    def _observe_current_cell(
        self,
        timestamp: str,
        u_km: float,
        v_km: float,
        w_km: float,
    ) -> Observation:
        x = int(round(self.context.state.x))
        y = int(round(self.context.state.y))
        z = _state_level(self.context.state.z, self.config.model.altitude_levels)
        baseline = self.pipeline.physics_window(
            timestamp,
            u_km,
            v_km,
            x,
            y,
            x + 1,
            y + 1,
            w_km,
            altitude_level=z,
        )
        true_u, true_v, true_w = self.simulator.truth_at_cell(
            self.pipeline.terrain,
            u_km,
            v_km,
            self.minute_index,
            x,
            y,
            z,
            baseline["u"][0][0],
            baseline["v"][0][0],
            baseline["w"][0][0],
            self.config.model.altitude_levels,
        )
        return Observation(
            timestamp=timestamp,
            x=x,
            y=y,
            z=z,
            u_obs=true_u + self._random.uniform(-0.06, 0.06),
            v_obs=true_v + self._random.uniform(-0.06, 0.06),
            w_obs=true_w + self._random.uniform(-0.04, 0.04),
            airspeed=self.context.state.airspeed + self._random.uniform(-0.3, 0.3),
            ground_speed=max(4.0, self.context.state.airspeed + 0.4 * true_u + self._random.uniform(-0.5, 0.7)),
            climb_rate=self._random.uniform(-0.2, 0.8),
            acceleration=self._random.uniform(-0.15, 0.15),
        )

    def _report(self) -> dict:
        return self.engine.build_report(self.context, self.config.mission.start, self.context.mission.goal)

    def _refresh_latest_trace_frame(self) -> None:
        if not self.context.trace:
            return
        frame = self.context.trace[-1]
        goal = self.context.mission.goal
        state = self.context.state
        frame["planned_path"] = self.context.planned_path
        frame["planned_path_cost"] = self.context.latest_planning.get("path_cost", 0.0)
        frame["planned_path_cost_breakdown"] = self.context.latest_planning.get("path_cost_breakdown", {})
        frame["candidate_paths"] = self.context.latest_planning.get("candidates", [])
        frame["planning_mode"] = self.context.latest_planning.get("planning_mode", "mpc_strict_return")
        frame["return_constraint_active"] = self.context.latest_planning.get("return_constraint_active", True)
        frame["position"] = [_round_state_value(state.x), _round_state_value(state.y), _round_state_value(state.z)]
        frame["goal_distance_cells"] = math.sqrt(
            (goal[0] - state.x) ** 2
            + (goal[1] - state.y) ** 2
            + (goal[2] - state.z) ** 2
        )


def serve_live_dashboard(report_path: str | Path, host: str = "127.0.0.1", port: int = 8765, interval_seconds: float = 3.0) -> None:
    report = read_json(report_path)
    replay = LiveReplay(report, interval_seconds)
    replay.start()
    html = build_live_dashboard_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            path = _public_path(parsed.path)
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/api/bootstrap":
                self._send_json(replay.snapshot())
                return
            if path == "/api/state":
                since = int(params.get("since", ["-1"])[0])
                self._send_json(replay.snapshot_since(since))
                return
            self.send_response(404)
            self.end_headers()

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = _public_path(parsed.path)
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if path in ("/api/bootstrap", "/api/state"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        replay.stop()
        server.server_close()


def serve_interactive_dashboard(
    terrain_path: str | Path,
    model_path: str | Path,
    config: TaskConfig,
    host: str = "127.0.0.1",
    port: int = 8870,
    interval_seconds: float = 1.0,
    sim_minutes_per_tick: int = 1,
) -> None:
    pipeline = WindFarmPipeline.from_paths(terrain_path, model_path, config.model)
    simulation = InteractiveSimulation(
        pipeline=pipeline,
        simulator=EnvironmentSimulator(seed=7),
        config=config,
        interval_seconds=interval_seconds,
        sim_minutes_per_tick=sim_minutes_per_tick,
    )
    simulation.start()
    html = build_live_dashboard_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            path = _public_path(parsed.path)
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/api/bootstrap":
                self._send_json(simulation.snapshot())
                return
            if path == "/api/state":
                since = int(params.get("since", ["-1"])[0])
                self._send_json(simulation.snapshot_since(since))
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if _public_path(parsed.path) != "/api/command":
                self.send_response(404)
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            payload = json.loads(body.decode("utf-8") or "{}")
            goal = payload.get("goal")
            if not isinstance(goal, list) or len(goal) not in {2, 3}:
                self.send_response(400)
                self.end_headers()
                return
            simulation.set_goal(tuple(int(item) for item in goal))
            self._send_json(simulation.command_snapshot())

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        simulation.stop()
        server.server_close()
def _clamp_xy(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 0, 0
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _clamp_goal(goal: tuple[int, int] | tuple[int, int, int], width: int, height: int, config: TaskConfig) -> tuple[int, int, int]:
    x, y = _clamp_xy(int(goal[0]), int(goal[1]), width, height)
    z = int(goal[2]) if len(goal) >= 3 else 0
    z = max(config.mission.min_altitude_level, min(config.mission.max_altitude_level, z))
    return x, y, z


def _report_meta(report: dict) -> dict:
    return {
        "grid": report["grid"],
        "mission": report["mission"],
        "terrain": report["terrain"],
        "model_summary": report.get("model_summary", {}),
        "report_mode": report.get("report_mode", {}),
    }


def _summary_from_trace(report: dict) -> dict:
    trace = report.get("trace", [])
    last = trace[-1] if trace else None
    return {
        "goal_reached": report.get("goal_reached", False),
        "final_position": report.get("final_position", last["position"] if last else None),
        "battery_ratio": report.get("battery_ratio", last["battery_ratio"] if last else None),
        "steps_executed": report.get("steps_executed", len(trace)),
    }
