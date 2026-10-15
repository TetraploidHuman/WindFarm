#!/usr/bin/env python3
"""Download / rebuild large assets that are intentionally not kept in git.

- DEM: SRTM1 ``.hgt`` tiles from AWS Skadi (public, no API key)
- Scenarios: ``training.json`` / ``truth`` via ``python -m windfarm.cli build-scenarios``
  (SRTM crop + Open-Meteo archive wind + local simulator)

Examples::

    # DEM tiles only (fast; ~25 MB each, ~20 tiles)
    python scripts/download_data.py --dem-only

    # DEM + rebuild all curated scenario training/truth datasets
    python scripts/download_data.py --with-scenarios

    # Subset
    python scripts/download_data.py --with-scenarios --only fujian_hills beijing_plain

Optional proxy (Clash etc.)::

    export HTTPS_PROXY=http://127.0.0.1:7897
    export HTTP_PROXY="$HTTPS_PROXY"
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRTM_SIZE = 3601
SRTM_SKADI_BASE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

# Curated scenario tile SW corners (must stay in sync with
# windfarm.data_ingest.DEFAULT_SCENARIOS tile_lat/tile_lon).
SCENARIO_TILES: dict[str, tuple[int, int]] = {
    "fujian_hills": (25, 118),
    "beijing_plain": (39, 116),
    "qinghai_ridge": (36, 100),
    "liaoning_coast": (39, 121),
    "xinjiang_gobi": (42, 93),
    "sichuan_foothills": (30, 103),
    "shanxi_loess": (36, 109),
    "taiwan_hills": (23, 121),
    "gansu_hexi": (39, 98),
    "guizhou_karst": (26, 106),
    "hainan_coast": (18, 109),
    "hubei_jianghan": (30, 114),
    "tibet_lhasa": (29, 91),
    "jilin_forest": (42, 127),
    "neimeng_grass": (43, 116),
    "yunnan_karst": (25, 102),
    "zhejiang_hills": (29, 119),
    "shandong_taishan": (36, 117),
    "chongqing_hills": (29, 106),
    "guangxi_guilin": (25, 110),
}

# Extra 1° tiles historically cached for rtwind / ad-hoc missions.
EXTRA_TILES: tuple[tuple[int, int], ...] = (
    (36, 120),  # Qingdao coast area
)


def _ssl_context() -> ssl.SSLContext:
    candidates: list[str] = []
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(value)
    try:
        import certifi

        candidates.append(certifi.where())
    except Exception:
        pass
    candidates.extend(
        [
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/cert.pem",
        ]
    )
    for path in candidates:
        if path and Path(path).is_file():
            return ssl.create_default_context(cafile=path)
    return ssl.create_default_context()


def _detect_local_clash_proxy() -> str:
    import socket

    hosts = ("172.20.128.142", "127.0.0.1", "172.20.128.1")
    ports = (7897, 7890, 6088)
    for host in hosts:
        for port in ports:
            try:
                with socket.create_connection((host, port), timeout=0.6):
                    return f"http://{host}:{port}"
            except OSError:
                continue
    return ""


def _opener() -> urllib.request.OpenerDirector:
    keys = (
        "WINDFARM_HTTP_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    )
    proxy = next((os.environ[k].strip() for k in keys if os.environ.get(k, "").strip()), "")
    if not proxy:
        proxy = _detect_local_clash_proxy()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=_ssl_context()),
    ]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        print(f"[net] using proxy {proxy}")
    return urllib.request.build_opener(*handlers)


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def skadi_url(lat: int, lon: int) -> str:
    name = tile_name(lat, lon)
    return f"{SRTM_SKADI_BASE}/{name[:3]}/{name}.hgt.gz"


def ensure_srtm_tile(data_dir: Path, lat: int, lon: int, timeout_s: float = 180.0) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{tile_name(lat, lon)}.hgt"
    expected = SRTM_SIZE * SRTM_SIZE * 2
    if path.exists() and path.stat().st_size >= expected:
        return path
    url = skadi_url(lat, lon)
    print(f"[srtm] downloading {path.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "WindFarm/1.0"})
    with _opener().open(request, timeout=timeout_s) as response:
        payload = response.read()
    if url.endswith(".gz") or payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    if len(payload) != expected:
        raise ValueError(f"Unexpected HGT size for {path.name}: {len(payload)}")
    path.write_bytes(payload)
    return path


def _load_scenario_tiles_from_index(scenarios_dir: Path) -> dict[str, tuple[int, int]]:
    """Prefer scenarios/index.json tile hints when present; else use hardcoded map."""
    index_path = scenarios_dir / "index.json"
    if not index_path.exists():
        return dict(SCENARIO_TILES)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(SCENARIO_TILES)
    out = dict(SCENARIO_TILES)
    for row in payload.get("scenarios") or []:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        name = str(row["name"])
        # scenario_meta / index may expose tile as "N25E118" or lat/lon fields.
        tile = row.get("tile") or row.get("srtm_tile")
        if isinstance(tile, str) and len(tile) >= 7 and tile[0] in "NS":
            tlat = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
            tlon = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
            out[name] = (tlat, tlon)
            continue
        meta_path = scenarios_dir / name / "scenario_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                src = meta.get("srtm_tile") or meta.get("tile")
                if isinstance(src, str) and len(src) >= 7:
                    tlat = int(src[1:3]) * (1 if src[0] == "N" else -1)
                    tlon = int(src[4:7]) * (1 if src[3] == "E" else -1)
                    out[name] = (tlat, tlon)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
    return out


def _unique_tiles(scenarios_dir: Path, only: list[str] | None) -> list[tuple[int, int]]:
    catalog = _load_scenario_tiles_from_index(scenarios_dir)
    tiles: dict[tuple[int, int], None] = {}
    for name, pair in catalog.items():
        if only and name not in only:
            continue
        tiles[pair] = None
    if not only:
        for pair in EXTRA_TILES:
            tiles[pair] = None
    return list(tiles.keys())


def download_dem(data_dir: Path, scenarios_dir: Path, only: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    tiles = _unique_tiles(scenarios_dir, only)
    print(f"[dem] ensuring {len(tiles)} SRTM tile(s) under {data_dir}/")
    for lat, lon in tiles:
        name = tile_name(lat, lon)
        try:
            path = ensure_srtm_tile(data_dir, lat, lon)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            print(f"  FAIL {name}.hgt: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ok {name}.hgt ({size_mb:.1f} MB)")
        paths.append(path)
    return paths


def rebuild_scenarios(
    scenarios_dir: Path,
    data_dir: Path,
    only: list[str] | None,
    resolution_m: float,
    base_config: Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "windfarm.cli",
        "build-scenarios",
        "--output-dir",
        str(scenarios_dir),
        "--data-dir",
        str(data_dir),
        "--resolution-m",
        str(resolution_m),
    ]
    if base_config.exists():
        cmd.extend(["--base-config", str(base_config)])
    if only:
        cmd.append("--only")
        cmd.extend(only)
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[scenarios] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SRTM DEM tiles and optionally rebuild scenario training data."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--scenarios-dir", type=Path, default=ROOT / "scenarios")
    parser.add_argument(
        "--dem-only",
        action="store_true",
        help="Only download/ensure .hgt tiles.",
    )
    parser.add_argument(
        "--with-scenarios",
        action="store_true",
        help="Also rebuild scenario training/truth via windfarm.cli build-scenarios.",
    )
    parser.add_argument("--only", nargs="*", help="Optional subset of scenario names")
    parser.add_argument("--resolution-m", type=float, default=30.0)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "config_eval.json",
        help="Optional seed config for scenario rebuild (omitted if missing).",
    )
    args = parser.parse_args()

    only = args.only or None
    download_dem(args.data_dir, args.scenarios_dir, only)

    if args.with_scenarios:
        rebuild_scenarios(
            args.scenarios_dir,
            args.data_dir,
            only,
            args.resolution_m,
            args.base_config,
        )
        print(f"[scenarios] done under {args.scenarios_dir}/")
    elif not args.dem_only:
        print(
            "[hint] DEM ready. To rebuild scenario training/truth as well, re-run with "
            "--with-scenarios (or: python -m windfarm.cli build-scenarios)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
