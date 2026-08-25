from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .hub import TelemetryHub
from .types import RtwindConfig, TelemetryFrame


class MavlinkState:
    """Thread-safe partial MAVLink telemetry merge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._last_rx = 0.0

    def touch(self) -> None:
        with self._lock:
            self._last_rx = time.monotonic()

    def last_rx(self) -> float:
        with self._lock:
            return self._last_rx

    def update_from_msg(self, msg: Any) -> None:
        mtype = msg.get_type()
        with self._lock:
            if mtype == "GLOBAL_POSITION_INT":
                self._data["lat"] = msg.lat / 1e7
                self._data["lon"] = msg.lon / 1e7
                self._data["alt_msl"] = msg.alt / 1000.0
                self._data["relative_alt"] = msg.relative_alt / 1000.0
                if msg.hdg != 65535:
                    self._data["heading"] = msg.hdg / 100.0
                self._data["vx"] = msg.vx / 100.0
                self._data["vy"] = msg.vy / 100.0
                self._data["vz"] = msg.vz / 100.0
            elif mtype == "ATTITUDE":
                self._data["roll"] = math.degrees(msg.roll)
                self._data["pitch"] = math.degrees(msg.pitch)
                self._data["yaw"] = math.degrees(msg.yaw) % 360.0
                if "heading" not in self._data:
                    self._data["heading"] = self._data["yaw"]
            elif mtype == "VFR_HUD":
                self._data["airspeed"] = float(msg.airspeed)
                self._data["groundspeed"] = float(msg.groundspeed)
                self._data["heading"] = float(msg.heading)
                self._data["climb_rate"] = float(msg.climb)
            elif mtype == "SYS_STATUS":
                if msg.battery_remaining != 255:
                    self._data["battery"] = max(0.0, min(1.0, msg.battery_remaining / 100.0))
            self._last_rx = time.monotonic()

    def to_frame(self, vehicle_id: str = "mavlink-1") -> TelemetryFrame | None:
        with self._lock:
            if "lat" not in self._data or "lon" not in self._data:
                return None
            d = dict(self._data)
        climb = d.get("climb_rate")
        if climb is None and "vz" in d:
            climb = -float(d["vz"])
        return TelemetryFrame(
            source="live",
            vehicle_id=vehicle_id,
            t=datetime.now(timezone.utc).isoformat(),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            alt_msl=float(d.get("alt_msl", 0.0)),
            heading=float(d.get("heading", d.get("yaw", 0.0))),
            roll=float(d.get("roll", 0.0)),
            pitch=float(d.get("pitch", 0.0)),
            yaw=float(d.get("yaw", d.get("heading", 0.0))),
            airspeed=float(d.get("airspeed", 0.0)),
            groundspeed=float(d.get("groundspeed", d.get("airspeed", 0.0))),
            climb_rate=float(climb or 0.0),
            battery=d.get("battery"),
            link="ok",
            alt_agl=float(d["relative_alt"]) if d.get("relative_alt") is not None else None,
        )


class MavlinkUdpListener:
    """Background UDP MAVLink reader (ArduPilot/PX4 dialect via pymavlink)."""

    def __init__(self, config: RtwindConfig, state: MavlinkState):
        self.config = config
        self.state = state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None
        self._conn: Any = None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def listening(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not self.config.mavlink_enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtwind-mavlink-udp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            self._error = f"pymavlink missing: {exc}"
            return
        bind = self.config.mavlink_bind
        port = self.config.mavlink_udp_port
        url = f"udpin:{bind}:{port}"
        try:
            self._conn = mavutil.mavlink_connection(url, dialect="ardupilotmega")
            self._error = None
        except Exception as exc:
            self._error = str(exc)
            return
        while not self._stop.is_set():
            try:
                msg = self._conn.recv_match(blocking=True, timeout=1.0)
            except Exception as exc:
                if not self._stop.is_set():
                    self._error = str(exc)
                break
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype in {"GLOBAL_POSITION_INT", "ATTITUDE", "VFR_HUD", "SYS_STATUS", "HEARTBEAT"}:
                self.state.update_from_msg(msg)
                self.state.touch()
