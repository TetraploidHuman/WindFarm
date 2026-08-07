from __future__ import annotations

from dataclasses import dataclass, field
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import traceback
from urllib.parse import urlparse
from uuid import uuid4

from .config import TaskConfig, load_task_config
from .mission_runner import MissionRunner
from .pipeline import WindFarmPipeline, train_from_files


@dataclass(slots=True)
class JobRecord:
    job_id: str
    kind: str
    status: str = "queued"
    result: dict | None = None
    error: dict | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> JobRecord:
        job = JobRecord(job_id=str(uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)


def serve_api(host: str = "127.0.0.1", port: int = 8780) -> None:
    jobs = JobStore()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/v1/health":
                self._send_json({"status": "ok"})
                return
            if parsed.path == "/api/v1/schema":
                self._send_json(_api_schema())
                return
            if parsed.path.startswith("/api/v1/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                job = jobs.get(job_id)
                if not job:
                    self._send_json({"error": {"code": "not_found", "message": "job not found"}}, status=404)
                    return
                self._send_json(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "status": job.status,
                        "result": job.result,
                        "error": job.error,
                    }
                )
                return
            self._send_json({"error": {"code": "not_found", "message": "route not found"}}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/v1/train":
                self._enqueue("train", payload, _run_train)
                return
            if parsed.path == "/api/v1/predict":
                self._enqueue("predict", payload, _run_predict)
                return
            if parsed.path == "/api/v1/predict-sync":
                self._run_sync("predict", payload, _run_predict)
                return
            if parsed.path == "/api/v1/navigate":
                self._enqueue("navigate", payload, _run_navigate)
                return
            self._send_json({"error": {"code": "not_found", "message": "route not found"}}, status=404)

        def log_message(self, format: str, *args) -> None:
            return

        def _enqueue(self, kind: str, payload: dict, runner) -> None:
            try:
                _validate_payload(kind, payload)
            except ValueError as exc:
                self._send_json({"error": {"code": "bad_request", "message": str(exc)}}, status=400)
                return
            job = jobs.create(kind)
            thread = threading.Thread(target=_run_job, args=(jobs, job.job_id, runner, payload), daemon=True)
            thread.start()
            self._send_json({"job_id": job.job_id, "status": job.status}, status=202)

        def _run_sync(self, kind: str, payload: dict, runner) -> None:
            try:
                _validate_payload(kind, payload)
                self._send_json({"status": "completed", "result": runner(payload)})
            except ValueError as exc:
                self._send_json({"error": {"code": "bad_request", "message": str(exc)}}, status=400)
            except Exception as exc:  # pragma: no cover
                self._send_json(
                    {
                        "error": {
                            "code": "internal_error",
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=6),
                        }
                    },
                    status=500,
                )

        def _read_json(self) -> dict:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            return json.loads(raw.decode("utf-8") or "{}")

        def _send_json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _run_job(store: JobStore, job_id: str, runner, payload: dict) -> None:
    store.update(job_id, status="running")
    try:
        result = runner(payload)
        store.update(job_id, status="completed", result=result)
    except Exception as exc:  # pragma: no cover
        store.update(
            job_id,
            status="failed",
            error={"message": str(exc), "traceback": traceback.format_exc(limit=6)},
        )


def _run_train(payload: dict) -> dict:
    config = load_task_config(payload["config"]) if payload.get("config") else TaskConfig()
    metrics = train_from_files(
        payload["terrain"],
        payload["training"],
        payload["model_out"],
        payload.get("metrics_out"),
        config.model,
    )
    return {"metrics": metrics}


def _run_predict(payload: dict) -> dict:
    config = load_task_config(payload["config"]) if payload.get("config") else TaskConfig()
    pipeline = WindFarmPipeline.from_paths(payload["terrain"], payload.get("model"), config.model)
    altitude_level = int(payload.get("altitude_level", 0))
    physics = pipeline.physics_grid(payload["timestamp"], float(payload["u_km"]), float(payload["v_km"]), float(payload.get("w_km", 0.0)), altitude_level)
    prediction = pipeline.predict_grid(payload["timestamp"], float(payload["u_km"]), float(payload["v_km"]), float(payload.get("w_km", 0.0)), altitude_level=altitude_level)
    return {"physics": physics, "prediction": prediction, "model_summary": pipeline.model_summary()}


def _run_navigate(payload: dict) -> dict:
    config = load_task_config(payload["config"])
    pipeline = WindFarmPipeline.from_paths(payload["terrain"], payload["model"], config.model)
    report = MissionRunner(pipeline, config).run(
        payload["coarse_wind"],
        payload["observations"],
        payload.get("output"),
        payload.get("truth"),
        tuple(payload["start"]) if payload.get("start") else None,
        tuple(payload["goal"]) if payload.get("goal") else None,
    )
    return {"report": report}


def _validate_payload(kind: str, payload: dict) -> None:
    required = {
        "train": ["terrain", "training", "model_out"],
        "predict": ["terrain", "timestamp", "u_km", "v_km"],
        "navigate": ["config", "terrain", "model", "coarse_wind", "observations"],
    }[kind]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if kind == "predict":
        _validate_int(payload.get("altitude_level", 0), "altitude_level")
    if kind == "navigate":
        if "start" in payload:
            _validate_point(payload["start"], "start")
        if "goal" in payload:
            _validate_point(payload["goal"], "goal")


def _validate_point(value, field: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(f"{field} must be a 2- or 3-element coordinate")
    for item in value:
        _validate_int(item, field)


def _validate_int(value, field: str) -> None:
    try:
        int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer") from None


def _api_schema() -> dict:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "WindFarm API",
            "version": "v1",
            "description": "3D layered wind prediction, belief update, and navigation API.",
        },
        "paths": {
            "/api/v1/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {"200": {"description": "Service is available"}},
                }
            },
            "/api/v1/schema": {
                "get": {
                    "summary": "Return this schema document",
                    "responses": {"200": {"description": "Schema payload"}},
                }
            },
            "/api/v1/jobs/{job_id}": {
                "get": {
                    "summary": "Get async job status",
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Job state"}, "404": {"description": "Job not found"}},
                }
            },
            "/api/v1/train": {
                "post": {
                    "summary": "Train residual wind models",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["terrain", "training", "model_out"],
                                    "properties": {
                                        "terrain": {"type": "string"},
                                        "training": {"type": "string"},
                                        "model_out": {"type": "string"},
                                        "metrics_out": {"type": "string"},
                                        "config": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"202": {"description": "Training job accepted"}},
                }
            },
            "/api/v1/predict": {
                "post": {
                    "summary": "Run async layered wind prediction",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": _predict_request_schema()}}},
                    "responses": {
                        "202": {"description": "Prediction job accepted"},
                        "400": {"description": "Invalid request"},
                    },
                }
            },
            "/api/v1/predict-sync": {
                "post": {
                    "summary": "Run synchronous layered wind prediction",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": _predict_request_schema()}}},
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["completed"]},
                                            "result": {
                                                "type": "object",
                                                "properties": {
                                                    "physics": {"type": "object"},
                                                    "prediction": {"type": "object"},
                                                    "model_summary": {"type": "object"},
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid request"},
                    },
                }
            },
            "/api/v1/navigate": {
                "post": {
                    "summary": "Run async 3D navigation mission",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["config", "terrain", "model", "coarse_wind", "observations"],
                                    "properties": {
                                        "config": {"type": "string"},
                                        "terrain": {"type": "string"},
                                        "model": {"type": "string"},
                                        "coarse_wind": {"type": "string"},
                                        "observations": {"type": "string"},
                                        "truth": {"type": "string"},
                                        "output": {"type": "string"},
                                        "start": _point_schema(),
                                        "goal": _point_schema(),
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"202": {"description": "Navigation job accepted"}, "400": {"description": "Invalid request"}},
                }
            },
        },
    }


def _predict_request_schema() -> dict:
    return {
        "type": "object",
        "required": ["terrain", "timestamp", "u_km", "v_km"],
        "properties": {
            "terrain": {"type": "string"},
            "timestamp": {"type": "string"},
            "u_km": {"type": "number"},
            "v_km": {"type": "number"},
            "w_km": {"type": "number"},
            "altitude_level": {"type": "integer", "minimum": 0},
            "model": {"type": "string"},
            "config": {"type": "string"},
        },
        "description": "Returns current z-layer grids and full u/v/w layer stacks.",
    }


def _point_schema() -> dict:
    return {
        "oneOf": [
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer"},
            },
            {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer"},
            },
        ],
        "description": "Grid coordinate [x, y] or layered coordinate [x, y, z].",
    }
