"""
Generate publication-quality charts for the 挑战杯 competition document.
Covers: ML training curves, feature importance, model comparison,
wind fields, terrain, trajectory, energy, error analysis.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np

# ── Chinese font setup ──────────────────────────────────────────────
from matplotlib import font_manager as _fm
from matplotlib import patheffects as _pe
from pathlib import Path as _Path

# 五号 = 10.5 pt；图注用 Windows「黑体」SimHei（中易黑体）加粗
FIG_CAPTION_PT = 10.5
_CJK_FONT_PATH: str | None = None
_CAPTION_FP: _fm.FontProperties | None = None
_SIMHEI_PATH: str | None = None


def _simhei_candidates() -> list[str]:
    """Prefer authentic Windows 黑体 (SimHei / 中易黑体)."""
    root = _Path(__file__).resolve().parent.parent
    return [
        str(root / "fonts" / "SimHei.ttf"),
        "C:/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        str(_Path.home() / ".fonts" / "SimHei.ttf"),
        "/usr/share/fonts/truetype/simhei/SimHei.ttf",
    ]


def _fallback_cjk_candidates() -> list[str]:
    return [
        "/nix/store/ik9wglrzz0cvbb7r4b351fr8r8rqahf8-wqy-zenhei-0.9.45/share/fonts/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/nix/store/ysq41nkq340qr8zxgpzyamxczvp5zl7y-noto-fonts-cjk-sans-2.004/share/fonts/opentype/noto-cjk/NotoSansCJK-VF.otf.ttc",
        "/usr/share/fonts/opentype/noto-cjk/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]


def _try_add_font(path: str) -> str | None:
    if not _Path(path).exists():
        return None
    try:
        _fm.fontManager.addfont(path)
        return _fm.FontProperties(fname=path).get_name()
    except Exception:
        return None


def _setup_cjk_font() -> None:
    global _CJK_FONT_PATH, _CAPTION_FP, _SIMHEI_PATH

    simhei_family = None
    for path in _simhei_candidates():
        family = _try_add_font(path)
        if family is not None:
            _SIMHEI_PATH = path
            simhei_family = family
            break

    body_family = simhei_family
    body_path = _SIMHEI_PATH
    if body_family is None:
        for path in _fallback_cjk_candidates():
            family = _try_add_font(path)
            if family is not None:
                body_family = family
                body_path = path
                break

    if body_family:
        plt.rcParams["font.sans-serif"] = [
            body_family, "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei", "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        _CJK_FONT_PATH = body_path
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    # 图注强制 SimHei；缺字时才退回正文 CJK
    caption_path = _SIMHEI_PATH or _CJK_FONT_PATH
    if caption_path:
        _CAPTION_FP = _fm.FontProperties(fname=caption_path, size=FIG_CAPTION_PT)
    else:
        _CAPTION_FP = _fm.FontProperties(family="SimHei", size=FIG_CAPTION_PT)


_setup_cjk_font()
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = FIG_CAPTION_PT
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 24


def set_panel_title(ax, title: str, **kwargs) -> None:
    """图注：五号 SimHei（黑体）加粗。"""
    opts = {"pad": 8, "color": kwargs.pop("color", "black")}
    opts.update(kwargs)
    if _CAPTION_FP is not None:
        fp = _CAPTION_FP.copy()
        fp.set_size(FIG_CAPTION_PT)
        # SimHei 无独立 Bold 字重；用 stroke 实现 Word 式“黑体加粗”
        artist = ax.set_title(title, fontproperties=fp, **opts)
    else:
        artist = ax.set_title(title, fontsize=FIG_CAPTION_PT, fontfamily="SimHei", **opts)
    artist.set_path_effects([
        _pe.Stroke(linewidth=0.85, foreground=opts.get("color", "black")),
        _pe.Normal(),
    ])
    return artist

# ── Paths ───────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent


def _resolve_run_dir() -> Path:
    """Pick the run directory used for single-scenario publication charts."""
    override = os.environ.get("WINDFARM_CHART_RUN_DIR", "").strip()
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = PROJECT / candidate
        if (candidate / "metrics.json").exists() and (candidate / "mission_report.json").exists():
            return candidate
        raise FileNotFoundError(
            f"WINDFARM_CHART_RUN_DIR must contain metrics.json and mission_report.json: {candidate}"
        )

    for name in (
        "fujian_hills_viz",
        "liaoning_coast_viz",
        "natural",
        "corridor",
    ):
        candidate = PROJECT / "runs" / name
        if (candidate / "metrics.json").exists() and (candidate / "mission_report.json").exists():
            return candidate
    raise FileNotFoundError(
        "no run directory with metrics.json and mission_report.json under runs/ "
        "(expected e.g. runs/fujian_hills_viz)"
    )


OUT_DIR = PROJECT / "charts"
OUT_DIR.mkdir(exist_ok=True)
SCENARIOS_DIR = PROJECT / "scenarios"

# Populated by load_chart_data() before plotting.
RUN_DIR: Path = PROJECT / "runs"
metrics: dict = {}
report: dict = {}
terrain_data: dict = {}
coarse_data: list = []
model_summary: dict = {}

EIGHT_SCENARIOS = [
    ("fujian_hills", "福建丘陵", "CORE"),
    ("beijing_plain", "北京平原", "CORE"),
    ("qinghai_ridge", "青海山脊", "CORE"),
    ("liaoning_coast", "辽宁海岸", "CORE"),
    ("xinjiang_gobi", "新疆戈壁", "HOLDOUT"),
    ("sichuan_foothills", "四川山前", "HOLDOUT"),
    ("shanxi_loess", "山西黄土", "HOLDOUT"),
    ("taiwan_hills", "台湾丘陵", "HOLDOUT"),
]

# Color palette
C = {
    "teal": "#115d6b",
    "orange": "#d87420",
    "red": "#b53a2d",
    "green": "#2f7d55",
    "blue": "#3a6ea5",
    "purple": "#7b4f9d",
    "ink": "#172026",
    "muted": "#697680",
    "light": "#f0e8da",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_chart_inputs(run_dir: Path) -> tuple[dict, dict, dict, list, dict]:
    required = {
        "metrics.json": run_dir / "metrics.json",
        "mission_report.json": run_dir / "mission_report.json",
        "terrain.json": run_dir / "terrain.json",
        "coarse_wind.json": run_dir / "coarse_wind.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"chart inputs missing in {run_dir}: {', '.join(missing)}"
        )

    metrics_payload = load_json(required["metrics.json"])
    report_payload = load_json(required["mission_report.json"])
    terrain_payload = load_json(required["terrain.json"])
    coarse_payload = load_json(required["coarse_wind.json"])

    terrain = terrain_payload.get("terrain")
    if terrain is None:
        raise KeyError(f"{required['terrain.json']} missing 'terrain' object")

    coarse = coarse_payload.get("coarse_wind")
    if coarse is None:
        raise KeyError(f"{required['coarse_wind.json']} missing 'coarse_wind' array")

    model = report_payload.get("model_summary") or metrics_payload
    return metrics_payload, report_payload, terrain, coarse, model


def load_chart_data(run_dir: Path | None = None) -> Path:
    """Load globals for all single-scenario charts. Returns resolved run directory."""
    global RUN_DIR, metrics, report, terrain_data, coarse_data, model_summary

    RUN_DIR = run_dir or _resolve_run_dir()
    metrics, report, terrain_data, coarse_data, model_summary = _load_chart_inputs(RUN_DIR)
    trace = report.get("trace") or []
    keyframes = sum(1 for frame in trace if frame.get("maps"))
    print(
        f"Loaded chart data from {RUN_DIR} "
        f"(trace={len(trace)} frames, keyframes_with_maps={keyframes})"
    )
    if keyframes < 2:
        print(
            "  warning: fewer than 2 keyframes with maps — "
            "wind/belief map charts may be skipped (run without WINDFARM_SKIP_DASHBOARD)"
        )
    return RUN_DIR

# ── Helper ──────────────────────────────────────────────────────────
def save(path_stem: str):
    path = OUT_DIR / f"{path_stem}.png"
    plt.savefig(path, dpi=200, facecolor="white", edgecolor="none")
    print(f"  saved: {path.name}")


def save_two_col_rows(path_stem: str, row_builders, figsize=(14, 5.5)):
    """Save each row of a former 2-col multi-row figure as path_stem_1, _2, ..."""
    for i, builder in enumerate(row_builders, start=1):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        builder(axes[0], axes[1])
        plt.tight_layout()
        save(f"{path_stem}_{i}")
        plt.close()


def set_axis_style(ax, title="", xlabel="", ylabel=""):
    if title:
        set_panel_title(ax, title)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=19)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=19)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=16)


def _trilinear_sample_grid(grid: np.ndarray, x: float, y: float, z: float) -> float:
    """Sample (Z,H,W) or (H,W) belief map at continuous drone position."""
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim == 2:
        h, w = arr.shape
        x = float(np.clip(x, 0.0, max(w - 1, 0)))
        y = float(np.clip(y, 0.0, max(h - 1, 0)))
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        tx, ty = x - x0, y - y0
        top = arr[y0, x0] + (arr[y0, x1] - arr[y0, x0]) * tx
        bot = arr[y1, x0] + (arr[y1, x1] - arr[y1, x0]) * tx
        return float(top + (bot - top) * ty)
    z_n, h, w = arr.shape
    x = float(np.clip(x, 0.0, max(w - 1, 0)))
    y = float(np.clip(y, 0.0, max(h - 1, 0)))
    z = float(np.clip(z, 0.0, max(z_n - 1, 0)))
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, z_n - 1)
    tz = z - z0
    lower = _trilinear_sample_grid(arr[z0], x, y, 0.0)
    upper = _trilinear_sample_grid(arr[z1], x, y, 0.0)
    return lower + (upper - lower) * tz


# BeliefCell defaults used by older reports when field_arrays were not sampled.
_BELIEF_CELL_DEFAULTS = {
    "energy": 0.0,
    "uncertainty": 1.0,
    "confidence": 0.0,
    "entropy": 1.5,
    "safety": 0.0,
    "uplift_prob": 0.25,
    "sink_prob": 0.25,
    "w": 0.0,
}

_BELIEF_MAP_LAYER_KEYS = {
    "energy": "belief_energy_layers",
    "uncertainty": "belief_uncertainty_layers",
    "confidence": "belief_confidence_layers",
    "entropy": "belief_entropy_layers",
    "safety": "belief_safety_layers",
    "uplift_prob": "belief_uplift_prob_layers",
    "sink_prob": "belief_sink_prob_layers",
    "w": "belief_wind_w_layers",
}


def _belief_series_from_trace(trace: list, key: str) -> list[float]:
    """Per-frame belief scalar at drone; repair stale BeliefCell defaults via maps."""
    logged = [f.get("belief_at_drone", {}).get(key) for f in trace]
    default = _BELIEF_CELL_DEFAULTS.get(key)
    need_maps = default is not None and all(
        v is None or abs(float(v) - default) < 1e-12 for v in logged
    )
    if not need_maps:
        return [float(v if v is not None else (default if default is not None else 0.0)) for v in logged]

    layer_key = _BELIEF_MAP_LAYER_KEYS.get(key)
    keyframes = [
        (i, f) for i, f in enumerate(trace)
        if layer_key and (f.get("maps") or {}).get(layer_key)
    ]
    if not keyframes:
        return [float(v if v is not None else (default if default is not None else 0.0)) for v in logged]

    out: list[float] = []
    kf_ptr = 0
    for i, f in enumerate(trace):
        while kf_ptr + 1 < len(keyframes) and keyframes[kf_ptr + 1][0] <= i:
            kf_ptr += 1
        maps = keyframes[kf_ptr][1]["maps"]
        pos = f.get("position") or [0.0, 0.0, 0.0]
        out.append(_trilinear_sample_grid(maps[layer_key], pos[0], pos[1], pos[2]))
    return out


_BELIEF_MAP_LAYER_KEYS = {
    "energy": "belief_energy_layers",
    "uncertainty": "belief_uncertainty_layers",
    "confidence": "belief_confidence_layers",
    "entropy": "belief_entropy_layers",
    "safety": "belief_safety_layers",
    "uplift_prob": "belief_uplift_prob_layers",
    "sink_prob": "belief_sink_prob_layers",
    "w": "belief_wind_w_layers",
}


def _trilinear_from_layers(layers: np.ndarray, x: float, y: float, z: float) -> float:
    """Sample a (Z, H, W) map stack at continuous drone coordinates."""
    if layers.ndim != 3 or layers.size == 0:
        return 0.0
    z_n, height, width = layers.shape
    z = float(np.clip(z, 0.0, max(z_n - 1, 0)))
    y = float(np.clip(y, 0.0, max(height - 1, 0)))
    x = float(np.clip(x, 0.0, max(width - 1, 0)))
    z0 = int(math.floor(z))
    z1 = min(z0 + 1, z_n - 1)
    y0 = int(math.floor(y))
    y1 = min(y0 + 1, height - 1)
    x0 = int(math.floor(x))
    x1 = min(x0 + 1, width - 1)
    tz, ty, tx = z - z0, y - y0, x - x0

    def _bilinear(level: int) -> float:
        v00 = float(layers[level, y0, x0])
        v10 = float(layers[level, y0, x1])
        v01 = float(layers[level, y1, x0])
        v11 = float(layers[level, y1, x1])
        top = v00 + (v10 - v00) * tx
        bottom = v01 + (v11 - v01) * tx
        return top + (bottom - top) * ty

    return _bilinear(z0) + (_bilinear(z1) - _bilinear(z0)) * tz


def _belief_series_from_maps(trace: list[dict], attr: str) -> list[float]:
    """Rebuild per-frame belief scalars from keyframe field maps.

    Older reports logged belief_at_drone from stale BeliefCell defaults
    (confidence=0, uncertainty=1). Keyframe maps come from field_arrays.
    """
    layer_key = _BELIEF_MAP_LAYER_KEYS[attr]
    keyframes: list[tuple[int, np.ndarray]] = []
    for i, frame in enumerate(trace):
        maps = frame.get("maps") or {}
        raw = maps.get(layer_key)
        if raw:
            keyframes.append((i, np.asarray(raw, dtype=np.float64)))

    out: list[float] = []
    if not keyframes:
        for frame in trace:
            out.append(float((frame.get("belief_at_drone") or {}).get(attr, 0.0) or 0.0))
        return out

    kf_pos = 0
    for i, frame in enumerate(trace):
        while kf_pos + 1 < len(keyframes) and keyframes[kf_pos + 1][0] <= i:
            kf_pos += 1
        layers = keyframes[kf_pos][1]
        pos = frame.get("position") or [0.0, 0.0, 0.0]
        x = float(pos[0]) if len(pos) > 0 else 0.0
        y = float(pos[1]) if len(pos) > 1 else 0.0
        z = float(pos[2]) if len(pos) > 2 else 0.0
        out.append(_trilinear_from_layers(layers, x, y, z))
    return out


# ══════════════════════════════════════════════════════════════════════
# Chart 1: ML Training Convergence Curves (U, V, W)
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    components = [
        ("u", "U 分量 (东西方向)", C["teal"]),
        ("v", "V 分量 (南北方向)", C["red"]),
        ("w", "W 分量 (垂直方向)", C["green"]),
    ]
    for ax, (comp, title, color) in zip(axes, components):
        curve = model_summary.get(f"validation_curve_{comp}", {})
        train = curve.get("train", [])
        valid = curve.get("valid", [])
        best_iter = model_summary.get(f"best_iteration_{comp}", len(train))
        best_score = model_summary.get(f"best_score_{comp}", None)

        if train:
            epochs = range(1, len(train) + 1)
            ax.plot(epochs, train, color=color, linewidth=1.6, alpha=0.85, label="训练集 RMSE")
            ax.plot(epochs, valid, color=color, linewidth=2.2, linestyle="--", label="验证集 RMSE")
            if best_iter and best_iter <= len(train):
                ax.axvline(best_iter, color=C["orange"], linewidth=1.2, linestyle=":", alpha=0.8)
                ax.annotate(f"最佳迭代 {best_iter}", (best_iter, valid[best_iter - 1]),
                            xytext=(8, 12), textcoords="offset points", fontsize=19,
                            color=C["orange"], fontweight="bold")
        set_axis_style(ax, title, "迭代轮次", "RMSE (m/s)")
        ax.legend(fontsize=19, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save("01_training_curves")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 2: Feature Importance (U, V, W)
# ══════════════════════════════════════════════════════════════════════
def plot_feature_importance():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    components = [
        ("u", "U 分量特征重要性", C["teal"]),
        ("v", "V 分量特征重要性", C["red"]),
        ("w", "W 分量特征重要性", C["green"]),
    ]
    for ax, (comp, title, color) in zip(axes, components):
        fi = model_summary.get(f"feature_importance_gain_{comp}", {})
        if not fi:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
            set_axis_style(ax, title)
            continue
        # Sort and take top 12
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
        names = [n for n, _ in reversed(sorted_fi)]
        values = [v for _, v in reversed(sorted_fi)]
        bars = ax.barh(names, values, color=color, height=0.65, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=15, color=C["ink"])
        set_axis_style(ax, title, "Gain 重要性")
        ax.grid(True, alpha=0.3, linewidth=0.5, axis="x")

    plt.tight_layout()
    save("02_feature_importance")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 3: Model Performance — Physics vs ML
# ══════════════════════════════════════════════════════════════════════
def plot_model_performance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dir_mae_phys = metrics.get("direction_mae_physics")
    dir_mae_final = metrics.get("direction_mae_rad", 0)
    has_direction = dir_mae_phys is not None

    # Bar chart: RMSE comparison
    ax = axes[0]
    metrics_list = [
        ("水平风\nRMSE", metrics["rmse_physics"], metrics["rmse_final"], C["teal"], C["orange"]),
        ("垂直风\nMAE", metrics.get("vertical_mae_physics", 0), metrics.get("vertical_mae_final", 0), C["teal"], C["orange"]),
    ]
    if has_direction:
        metrics_list.append(("风向\nMAE (rad)", dir_mae_phys, dir_mae_final, C["teal"], C["orange"]))
    x = np.arange(len(metrics_list))
    width = 0.35
    phys_vals = [m[1] for m in metrics_list]
    ml_vals = [m[2] for m in metrics_list]
    bars1 = ax.bar(x - width / 2, phys_vals, width, color=C["teal"], alpha=0.85, label="物理模型 (基线)")
    bars2 = ax.bar(x + width / 2, ml_vals, width, color=C["orange"], alpha=0.85, label="物理 + ML 残差")
    for bar, val in zip(bars1, phys_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}",
                ha="center", fontsize=16, fontweight="bold")
    for bar, val in zip(bars2, ml_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}",
                ha="center", fontsize=16, fontweight="bold")
        idx = list(bars2).index(bar)
        pct = (phys_vals[idx] - val) / max(phys_vals[idx], 1e-6) * 100
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"↓{pct:.0f}%",
                    ha="center", fontsize=14, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics_list])
    set_axis_style(ax, "风场预测误差对比", ylabel="RMSE / MAE")
    ax.legend(fontsize=19, framealpha=0.8)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Reduction percentage (same formula as horizontal / vertical wind)
    ax2 = axes[1]
    categories = ["水平风 RMSE", "垂直风 MAE"]
    reductions = [
        (metrics["rmse_physics"] - metrics["rmse_final"]) / max(metrics["rmse_physics"], 1e-6) * 100,
        (metrics.get("vertical_mae_physics", 0) - metrics.get("vertical_mae_final", 0))
        / max(metrics.get("vertical_mae_physics", 0), 1e-6) * 100,
    ]
    colors_bar = [C["teal"], C["orange"]]
    if has_direction:
        categories.append("风向 MAE")
        reductions.append((dir_mae_phys - dir_mae_final) / max(dir_mae_phys, 1e-6) * 100)
        colors_bar.append(C["green"])

    bars = ax2.bar(categories, reductions, color=colors_bar, alpha=0.85, width=0.5)
    y_lo = min(0.0, min(reductions) - 5)
    y_hi = max(reductions) + 8
    ax2.set_ylim(y_lo, y_hi)
    for bar, val in zip(bars, reductions):
        if val >= 0:
            y_text = bar.get_height() + 1.0
            va = "bottom"
        else:
            y_text = bar.get_height() - 1.0
            va = "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, y_text, f"{val:.1f}%",
                 ha="center", va=va, fontsize=18, fontweight="bold")
    set_axis_style(ax2, "ML 模型相对物理基线的误差降低率", ylabel="降低比例 (%)")
    ax2.axhline(y=0, color=C["ink"], linewidth=0.8)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    plt.tight_layout()
    save("03_model_performance")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 4: Coarse Wind Time Series
# ══════════════════════════════════════════════════════════════════════
def plot_coarse_wind():
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2, 1]})

    timestamps = [c["timestamp"] for c in coarse_data]
    t_idx = np.arange(len(timestamps))
    u_vals = [c["u_km"] for c in coarse_data]
    v_vals = [c["v_km"] for c in coarse_data]
    w_vals = [c["w_km"] for c in coarse_data]
    speeds = [math.hypot(u, v) for u, v in zip(u_vals, v_vals)]

    ax = axes[0]
    ax.plot(t_idx, u_vals, color=C["teal"], linewidth=1.2, alpha=0.9, label="U 分量 (东→)")
    ax.plot(t_idx, v_vals, color=C["red"], linewidth=1.2, alpha=0.9, label="V 分量 (北→)")
    ax.plot(t_idx, w_vals, color=C["green"], linewidth=1.2, alpha=0.9, label="W 分量 (上升)")
    ax.plot(t_idx, speeds, color=C["orange"], linewidth=2.0, alpha=0.85, linestyle="--", label="水平风速 |V|")
    set_axis_style(ax, "粗尺度气象风场时序 (360 步 × 15s)", ylabel="风速 (m/s)")
    ax.legend(fontsize=19, ncol=4, framealpha=0.8, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Wind direction
    ax2 = axes[1]
    directions = [math.degrees(math.atan2(u, v)) % 360 for u, v in zip(u_vals, v_vals)]
    ax2.scatter(t_idx[::3], directions[::3], c=speeds[::3], cmap="YlOrRd",
                s=12, alpha=0.7, edgecolors="none")
    set_axis_style(ax2, "粗尺度风向时序 (气象角度, 0°=北)", xlabel="时间步长", ylabel="风向 (°)")
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["北 0°", "东 90°", "南 180°", "西 270°", "北 360°"])
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save("04_coarse_wind_timeseries")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 5: Wind Field Maps — Physics vs Prediction vs Truth
# ══════════════════════════════════════════════════════════════════════
def plot_wind_field_maps():
    # Get data from a mid-mission keyframe
    trace = report["trace"]
    keyframes = [f for f in trace if f.get("maps") and f["maps"].get("truth_wind_speed")]
    if len(keyframes) < 2:
        print("  skipping wind field maps — insufficient keyframes with truth")
        return
    frame = keyframes[len(keyframes) // 2]  # middle of mission
    maps = frame["maps"]

    phys_speed = np.array(maps.get("physics_wind_speed", [[0]]))
    pred_speed = np.array(maps.get("prediction_wind_speed", [[0]]))
    truth_speed = np.array(maps.get("truth_wind_speed", [[0]]))
    phys_u = np.array(maps.get("physics_wind_u", [[0]]))
    phys_v = np.array(maps.get("physics_wind_v", [[0]]))
    truth_u = np.array(maps.get("truth_wind_u", maps.get("truth_wind_u_layers", [[[0]]])[0] if "truth_wind_u_layers" in maps else [[0]]))
    if truth_u.ndim == 3:
        truth_u = truth_u[0]
    truth_v = np.array(maps.get("truth_wind_v", [[0]]))
    if truth_v.ndim == 3:
        truth_v = truth_v[0]

    vmin = min(np.min(phys_speed), np.min(pred_speed), np.min(truth_speed))
    vmax = max(np.max(phys_speed), np.max(pred_speed), np.max(truth_speed))
    res_phys = np.abs(phys_speed - truth_speed)
    res_pred = np.abs(pred_speed - truth_speed)
    res_max = max(np.max(res_phys), np.max(res_pred))
    improvement = res_phys - res_pred
    v_abs = max(abs(np.min(improvement)), abs(np.max(improvement)), 1e-9)

    def _speed_panel(ax, grid, title, ug=None, vg=None):
        im = ax.imshow(grid, origin="upper", cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.82, label="m/s")
        if ug is not None and vg is not None:
            h, w = grid.shape
            stride = max(1, w // 14)
            yy, xx = np.mgrid[0:h:stride, 0:w:stride]
            ax.quiver(xx, yy, ug[::stride, ::stride], vg[::stride, ::stride],
                      scale=40, width=0.003, color="white", alpha=0.8)
        set_axis_style(ax, title)
        ax.set_xticks([])
        ax.set_yticks([])

    def row1(ax0, ax1):
        _speed_panel(ax0, phys_speed, "物理降尺度风速", phys_u, phys_v)
        _speed_panel(ax1, pred_speed, "物理 + ML 预测风速")

    def row2(ax0, ax1):
        _speed_panel(ax0, truth_speed, "真值风速 (合成)", truth_u, truth_v)
        im = ax1.imshow(res_phys, origin="upper", cmap="Reds", vmin=0, vmax=res_max, aspect="auto")
        plt.colorbar(im, ax=ax1, shrink=0.82, label="m/s")
        set_axis_style(ax1, f"物理模型残差 |ε| (mean={np.mean(res_phys):.3f})")
        ax1.set_xticks([])
        ax1.set_yticks([])

    def row3(ax0, ax1):
        im = ax0.imshow(res_pred, origin="upper", cmap="Reds", vmin=0, vmax=res_max, aspect="auto")
        plt.colorbar(im, ax=ax0, shrink=0.82, label="m/s")
        set_axis_style(ax0, f"ML 预测残差 |ε| (mean={np.mean(res_pred):.3f})")
        ax0.set_xticks([])
        ax0.set_yticks([])
        im = ax1.imshow(improvement, origin="upper", cmap="RdYlGn", vmin=-v_abs, vmax=v_abs, aspect="auto")
        plt.colorbar(im, ax=ax1, shrink=0.82, label="m/s")
        set_axis_style(ax1, f"ML 改进量 (物理残差 - ML残差)\n正=改进, mean={np.mean(improvement):.4f}")
        ax1.set_xticks([])
        ax1.set_yticks([])

    save_two_col_rows("05_wind_field_maps", [row1, row2, row3], figsize=(14, 5.5))


# ══════════════════════════════════════════════════════════════════════
# Chart 6: Terrain Feature Maps
# ══════════════════════════════════════════════════════════════════════
def plot_terrain_features():
    elev = np.array(terrain_data["elevation"])
    slope = np.array(terrain_data["slope"])
    aspect = np.array(terrain_data["aspect"])
    roughness = np.array(terrain_data["roughness"])

    def _panel(ax, grid, cmap, title, contours=False):
        im = ax.imshow(grid, origin="upper", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.82)
        if contours:
            x = np.arange(elev.shape[1])
            y = np.arange(elev.shape[0])
            ax.contour(x, y, elev, levels=8, colors="white", linewidths=0.6, alpha=0.5)
        set_axis_style(ax, title)
        ax.set_xticks([])
        ax.set_yticks([])

    save_two_col_rows(
        "06_terrain_features",
        [
            lambda a0, a1: (_panel(a0, elev, "terrain", "高程 (m)", contours=True),
                            _panel(a1, slope, "YlOrBr", "坡度 (rad)")),
            lambda a0, a1: (_panel(a0, aspect, "twilight", "坡向 (rad)"),
                            _panel(a1, roughness, "Greens", "地表粗糙度")),
        ],
        figsize=(12, 5.0),
    )


# ══════════════════════════════════════════════════════════════════════
# Chart 7: Mission Profile — Battery, Distance, Altitude
# ══════════════════════════════════════════════════════════════════════
def plot_mission_profile():
    trace = report["trace"]
    steps = np.arange(len(trace))
    battery = [f["battery_ratio"] * 100 for f in trace]
    distances = [f["goal_distance_cells"] for f in trace]
    positions = [f["position"] for f in trace]
    altitudes = [p[2] for p in positions]
    energy_rates = []
    for f in trace:
        sp = f.get("sensor_packet", {})
        er = sp.get("energy_rate", 0)
        energy_rates.append(float(er) if er is not None else 0)

    def row1(ax0, ax1):
        ax0.fill_between(steps, 0, battery, color=C["teal"], alpha=0.3)
        ax0.plot(steps, battery, color=C["teal"], linewidth=2)
        ax0.axhline(y=22, color=C["red"], linewidth=1, linestyle="--", alpha=0.6, label="22% 储备线")
        set_axis_style(ax0, "电池电量曲线", "步数", "电量 (%)")
        ax0.legend(fontsize=18)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(steps, distances, color=C["orange"], linewidth=2)
        ax1.fill_between(steps, 0, distances, color=C["orange"], alpha=0.15)
        set_axis_style(ax1, "到目标距离曲线", "步数", "距离 (格点数)")
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    def row2(ax0, ax1):
        ax0.plot(steps, altitudes, color=C["green"], linewidth=1.8)
        ax0.fill_between(steps, 0, altitudes, color=C["green"], alpha=0.15)
        set_axis_style(ax0, "飞行高度剖面", "步数", "高度层")
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(steps, energy_rates, color=C["purple"], linewidth=1.5)
        ax1.fill_between(steps, 0, energy_rates, color=C["purple"], alpha=0.12)
        set_axis_style(ax1, "能量获取率 (信念)", "步数", "能量率")
        ax1.axhline(y=0, color=C["ink"], linewidth=0.6, alpha=0.5)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    save_two_col_rows("07_mission_profile", [row1, row2], figsize=(14, 4.5))


# ══════════════════════════════════════════════════════════════════════
# Chart 8: Trajectory Map on Terrain
# ══════════════════════════════════════════════════════════════════════
def plot_trajectory():
    elev = np.array(terrain_data["elevation"])
    trace = report["trace"]
    positions = [(f["position"][0], f["position"][1]) for f in trace]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # Get planned paths from keyframes
    keyframes = [f for f in trace if f.get("is_keyframe") and f.get("planned_path")]
    # Get start and goal
    mission = report.get("mission", {})
    start = mission.get("start", [4, 18])[:2]
    goal = mission.get("goal", [26, 7])[:2]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Terrain background
    im = ax.imshow(elev, origin="upper", cmap="terrain", alpha=0.75, aspect="auto",
                   extent=[-0.5, elev.shape[1] - 0.5, elev.shape[0] - 0.5, -0.5])
    plt.colorbar(im, ax=ax, shrink=0.78, label="高程 (m)")

    # Executed trajectory
    points = np.array(positions)
    for i in range(1, len(points)):
        alpha = 0.3 + 0.7 * i / len(points)
        ax.plot(points[i-1:i+1, 0], points[i-1:i+1, 1], color=C["teal"],
                linewidth=1.5 + 1.5 * i / len(points), alpha=alpha)

    # Planned paths (faded)
    for kf in keyframes[::5]:
        pp = kf["planned_path"]
        if len(pp) > 1:
            pxs = [p[0] for p in pp]
            pys = [p[1] for p in pp]
            ax.plot(pxs, pys, color=C["orange"], linewidth=0.8, alpha=0.25, linestyle="--")

    # Start and goal
    ax.scatter(*start, c=C["green"], s=200, marker="o", zorder=5, edgecolors="white", linewidth=1.5, label="起点")
    ax.scatter(*goal, c=C["red"], s=200, marker="X", zorder=5, edgecolors="white", linewidth=1.5, label="目标")
    ax.scatter(xs[-1], ys[-1], c=C["orange"], s=120, marker="D", zorder=5, edgecolors="white", linewidth=1, label="终点")

    set_axis_style(ax, "无人机轨迹与地形叠加", "X (格点)", "Y (格点)")
    ax.legend(fontsize=18, loc="upper left", framealpha=0.85)
    ax.set_xlim(-0.5, elev.shape[1] - 0.5)
    ax.set_ylim(elev.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    save("08_trajectory_map")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 9: Error Analysis — Distribution & Scatter
# ══════════════════════════════════════════════════════════════════════
def plot_error_analysis():
    samples: list[dict] = []
    training_path = RUN_DIR / "training.json"
    if training_path.exists():
        training = load_json(training_path)
        samples = training.get("samples", [])
    else:
        obs_path = RUN_DIR / "observations.json"
        coarse_path = RUN_DIR / "coarse_wind.json"
        if obs_path.exists() and coarse_path.exists():
            observations = load_json(obs_path).get("observations", [])
            coarse_by_ts = {
                row["timestamp"]: row for row in load_json(coarse_path).get("coarse_wind", [])
            }
            for obs in observations:
                coarse = coarse_by_ts.get(obs.get("timestamp"))
                if not coarse:
                    continue
                samples.append({
                    "u_km": coarse.get("u_km", 0.0),
                    "v_km": coarse.get("v_km", 0.0),
                    "u_obs": obs.get("u_obs", 0.0),
                    "v_obs": obs.get("v_obs", 0.0),
                })
    if not samples:
        print("  skipping error analysis — no training.json or observations/coarse_wind")
        return

    if len(samples) > 10000:
        # Random sample for performance
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), 10000, replace=False)
        samples = [samples[i] for i in indices]

    u_km = np.array([s["u_km"] for s in samples])
    v_km = np.array([s["v_km"] for s in samples])
    u_obs = np.array([s["u_obs"] for s in samples])
    v_obs = np.array([s["v_obs"] for s in samples])

    # Compute physics baseline error and observation error
    phys_u_err = u_obs - u_km  # physics baseline assumes wind = coarse wind
    phys_v_err = v_obs - v_km

    speed_km = np.hypot(u_km, v_km)
    speed_obs = np.hypot(u_obs, v_obs)
    lims_u = [min(np.min(u_km), np.min(u_obs)) - 0.5, max(np.max(u_km), np.max(u_obs)) + 0.5]
    lims_v = [min(np.min(v_km), np.min(v_obs)) - 0.5, max(np.max(v_km), np.max(v_obs)) + 0.5]
    lims_s = [0, max(np.max(speed_km), np.max(speed_obs)) + 0.5]
    corr_u = np.corrcoef(u_km, u_obs)[0, 1]
    corr_v = np.corrcoef(v_km, v_obs)[0, 1]
    corr_s = np.corrcoef(speed_km, speed_obs)[0, 1]
    rmse_phys_u = np.sqrt(np.mean(phys_u_err ** 2))
    rmse_phys_v = np.sqrt(np.mean(phys_v_err ** 2))
    rmse_phys_speed = np.sqrt(np.mean((speed_obs - speed_km) ** 2))

    def row1(ax0, ax1):
        ax0.hist(phys_u_err, bins=60, color=C["teal"], alpha=0.7, edgecolor="white", linewidth=0.3)
        ax0.axvline(0, color=C["ink"], linewidth=1, linestyle="--")
        ax0.axvline(np.mean(phys_u_err), color=C["red"], linewidth=1.5, linestyle="-",
                    label=f"mean={np.mean(phys_u_err):.4f}")
        set_axis_style(ax0, "物理模型 U 误差分布", "U 误差 (m/s)", "频数")
        ax0.legend(fontsize=18)
        ax1.hist(phys_v_err, bins=60, color=C["red"], alpha=0.7, edgecolor="white", linewidth=0.3)
        ax1.axvline(0, color=C["ink"], linewidth=1, linestyle="--")
        ax1.axvline(np.mean(phys_v_err), color=C["teal"], linewidth=1.5, linestyle="-",
                    label=f"mean={np.mean(phys_v_err):.4f}")
        set_axis_style(ax1, "物理模型 V 误差分布", "V 误差 (m/s)", "频数")
        ax1.legend(fontsize=18)

    def row2(ax0, ax1):
        ax0.scatter(u_km[::5], u_obs[::5], c=C["teal"], s=3, alpha=0.3, edgecolors="none")
        ax0.plot(lims_u, lims_u, color=C["ink"], linewidth=1, linestyle="--", alpha=0.5, label="y=x")
        ax0.set_xlim(lims_u)
        ax0.set_ylim(lims_u)
        set_axis_style(ax0, "粗尺度 U vs 观测 U", "粗尺度 u_km (m/s)", "观测 u_obs (m/s)")
        ax0.legend(fontsize=18)
        ax0.text(0.95, 0.05, f"R2 = {corr_u**2:.4f}", transform=ax0.transAxes, fontsize=18,
                 ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax1.scatter(v_km[::5], v_obs[::5], c=C["red"], s=3, alpha=0.3, edgecolors="none")
        ax1.plot(lims_v, lims_v, color=C["ink"], linewidth=1, linestyle="--", alpha=0.5, label="y=x")
        ax1.set_xlim(lims_v)
        ax1.set_ylim(lims_v)
        set_axis_style(ax1, "粗尺度 V vs 观测 V", "粗尺度 v_km (m/s)", "观测 v_obs (m/s)")
        ax1.legend(fontsize=18)
        ax1.text(0.95, 0.05, f"R2 = {corr_v**2:.4f}", transform=ax1.transAxes, fontsize=18,
                 ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def row3(ax0, ax1):
        ax0.scatter(speed_km[::5], speed_obs[::5], c=C["orange"], s=3, alpha=0.3, edgecolors="none")
        ax0.plot(lims_s, lims_s, color=C["ink"], linewidth=1, linestyle="--", alpha=0.5, label="y=x")
        ax0.set_xlim(lims_s)
        ax0.set_ylim(lims_s)
        set_axis_style(ax0, "粗尺度风速 vs 观测风速", "粗尺度 |V| (m/s)", "观测 |V| (m/s)")
        ax0.legend(fontsize=18)
        ax0.text(0.95, 0.05, f"R2 = {corr_s**2:.4f}", transform=ax0.transAxes, fontsize=18,
                 ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        categories_err = ["U 分量", "V 分量", "水平风速"]
        values_err = [rmse_phys_u, rmse_phys_v, rmse_phys_speed]
        colors_err = [C["teal"], C["red"], C["orange"]]
        bars = ax1.bar(categories_err, values_err, color=colors_err, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values_err):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}",
                     ha="center", fontsize=18, fontweight="bold")
        set_axis_style(ax1, "物理模型 RMSE (训练集)", ylabel="RMSE (m/s)")
        ax1.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    save_two_col_rows("09_error_analysis", [row1, row2, row3], figsize=(14, 5.0))


# ══════════════════════════════════════════════════════════════════════
# Chart 10: Summary Dashboard — Key metrics at a glance
# ══════════════════════════════════════════════════════════════════════
def plot_summary():
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.5)

    # Model info
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    info_lines = [
        f"模型类型: {metrics.get('model_type', 'N/A')}",
        f"训练样本: {metrics.get('train_size_used', 'N/A'):,}",
        f"验证样本: {metrics.get('validation_size', 'N/A'):,}",
        f"测试样本: {metrics.get('test_size', 'N/A'):,}",
        f"特征数量: {len(model_summary.get('feature_names', []))}",
        "",
        f"物理 RMSE:  {metrics['rmse_physics']:.4f}",
        f"最终 RMSE:  {metrics['rmse_final']:.4f}",
        f"降低率:     {(1-metrics['rmse_final']/metrics['rmse_physics'])*100:.1f}%",
        "",
        f"垂直 MAE (物理): {metrics.get('vertical_mae_physics', 0):.4f}",
        f"垂直 MAE (最终): {metrics.get('vertical_mae_final', 0):.4f}",
        f"风向 MAE:         {metrics.get('direction_mae_rad', 0):.4f} rad",
    ]
    for i, line in enumerate(info_lines):
        color = C["ink"] if line and not line.startswith("物理") and not line.startswith("最终") and not line.startswith("降低") and not line.startswith("垂直") and not line.startswith("风向") else C["teal"]
        fontweight = "bold" if "降低率" in line else "normal"
        ax.text(0.02, 0.99 - i * 0.062, line, transform=ax.transAxes, fontsize=15,
                color=color, fontweight=fontweight,
                verticalalignment="top")
    set_panel_title(ax, "模型训练概要")

    # Best iterations
    ax = fig.add_subplot(gs[0, 1])
    comps = ["U", "V", "W"]
    best_iters = [metrics.get(f"best_iteration_{c.lower()}", 0) for c in comps]
    best_scores = [metrics.get(f"best_score_{c.lower()}", 0) for c in comps]
    colors_best = [C["teal"], C["red"], C["green"]]
    bars = ax.bar(comps, best_scores, color=colors_best, alpha=0.85, width=0.5)
    for bar, score, it in zip(bars, best_scores, best_iters):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"iter={it}\n{score:.4f}", ha="center", fontsize=19, fontweight="bold")
    set_axis_style(ax, "最佳验证分数 (各分量)", ylabel="RMSE")
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Mission summary
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    mission_info = [
        f"起点: {report['mission']['start']}",
        f"目标: {report['mission']['goal']}",
        f"步数: {report['steps_executed']}",
        f"到达目标: {'是' if report['goal_reached'] else '否'}",
        f"最终电量: {report['battery_ratio']*100:.1f}%",
        f"网格: {report['grid']['width']}×{report['grid']['height']}",
        f"分辨率: {report['grid']['resolution_m']}m",
        f"高度层: {report['mission']['altitude_levels']}",
        f"电池容量: {report['mission']['battery_capacity_j']/1000:.0f} kJ",
    ]
    for i, line in enumerate(mission_info):
        ax.text(0.02, 0.99 - i * 0.10, line, transform=ax.transAxes, fontsize=15,
                color=C["ink"], verticalalignment="top")
    set_panel_title(ax, "任务执行概要")

    # Planned path visualization (small)
    ax = fig.add_subplot(gs[0, 3])
    elev = np.array(terrain_data["elevation"])
    ax.imshow(elev, origin="upper", cmap="terrain", alpha=0.6, aspect="auto")
    trace = report["trace"]
    xs = [f["position"][0] for f in trace]
    ys = [f["position"][1] for f in trace]
    ax.plot(xs, ys, color=C["teal"], linewidth=1.5, alpha=0.9)
    start = report["mission"]["start"][:2]
    goal = report["mission"]["goal"][:2]
    ax.scatter(*start, c=C["green"], s=80, marker="o", zorder=5)
    ax.scatter(*goal, c=C["red"], s=80, marker="X", zorder=5)
    set_axis_style(ax, "无人机飞行轨迹")
    ax.set_xticks([])
    ax.set_yticks([])

    # Feature importance top-5 comparison
    ax = fig.add_subplot(gs[1, :2])
    fi_u = model_summary.get("feature_importance_gain_u", {})
    fi_v = model_summary.get("feature_importance_gain_v", {})
    fi_w = model_summary.get("feature_importance_gain_w", {})
    # Get union of top features
    all_features = set()
    for fi in [fi_u, fi_v, fi_w]:
        sorted_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
        all_features.update(n for n, _ in sorted_items)
    all_features = list(all_features)[:16]

    x_idx = np.arange(len(all_features))
    width = 0.25
    u_vals = [fi_u.get(f, 0) for f in all_features]
    v_vals = [fi_v.get(f, 0) for f in all_features]
    w_vals = [fi_w.get(f, 0) for f in all_features]

    ax.barh(x_idx + width, u_vals, width, color=C["teal"], alpha=0.85, label="U 分量")
    ax.barh(x_idx, v_vals, width, color=C["red"], alpha=0.85, label="V 分量")
    ax.barh(x_idx - width, w_vals, width, color=C["green"], alpha=0.85, label="W 分量")
    ax.set_yticks(x_idx)
    ax.set_yticklabels(all_features, fontsize=18)
    set_axis_style(ax, "三 wind 分量特征重要性对比 (Gain)", xlabel="Gain")
    ax.legend(fontsize=18, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="x")

    # Validation curves overlay
    ax = fig.add_subplot(gs[1, 2:])
    for comp, color, label in [("u", C["teal"], "U"), ("v", C["red"], "V"), ("w", C["green"], "W")]:
        curve = model_summary.get(f"validation_curve_{comp}", {})
        valid = curve.get("valid", [])
        train_v = curve.get("train", [])
        if valid:
            epochs = range(1, len(valid) + 1)
            ax.plot(epochs, valid, color=color, linewidth=1.8, label=f"{label} 验证")
            ax.plot(epochs, train_v, color=color, linewidth=1.0, linestyle=":", alpha=0.5, label=f"{label} 训练")
    set_axis_style(ax, "三 wind 分量验证曲线", "迭代轮次", "RMSE")
    ax.legend(fontsize=18, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    save("10_summary_dashboard")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 11: Wind Rose / Polar histogram
# ══════════════════════════════════════════════════════════════════════
def plot_wind_rose():
    u_vals = np.array([c["u_km"] for c in coarse_data])
    v_vals = np.array([c["v_km"] for c in coarse_data])
    speeds = np.hypot(u_vals, v_vals)
    directions = np.arctan2(u_vals, v_vals)  # meteorological convention

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})
    colors_map = ["YlOrRd", "YlGnBu"]

    for ax, cmap_name, title in zip(axes, colors_map, ["粗尺度风玫瑰图", "粗尺度风矢量分布"]):
        # Wind rose as scatter
        if "玫瑰" in title:
            # Binned wind rose
            n_bins = 16
            dir_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            speed_bins = [0, 1, 2, 3, 4, 6, 10]
            rose = np.zeros((n_bins, len(speed_bins) - 1))
            for i in range(n_bins):
                mask = (directions >= dir_bins[i]) & (directions < dir_bins[i + 1])
                dir_speeds = speeds[mask]
                for j in range(len(speed_bins) - 1):
                    rose[i, j] = np.sum((dir_speeds >= speed_bins[j]) & (dir_speeds < speed_bins[j + 1]))

            theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
            width = 2 * np.pi / n_bins
            bottoms = np.zeros(n_bins)
            colors_rose = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(speed_bins) - 1))
            for j in range(len(speed_bins) - 1):
                ax.bar(theta, rose[:, j], width=width, bottom=bottoms, color=colors_rose[j],
                       alpha=0.85, label=f"{speed_bins[j]}-{speed_bins[j+1]} m/s")
                bottoms += rose[:, j]
            ax.legend(fontsize=18, loc="upper right", bbox_to_anchor=(1.3, 1.0))
        else:
            sc = ax.scatter(directions, speeds, c=speeds, cmap="YlOrRd", s=30, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.7, label="m/s", pad=0.1)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        set_panel_title(ax, title, pad=12)

    plt.tight_layout()
    save("11_wind_rose")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# Chart 12: Coarse Wind Grid Maps (new per-cell coarse wind data)
# ══════════════════════════════════════════════════════════════════════
def plot_coarse_wind_grids():
    """Visualize the 6x8 coarse wind grid at multiple timesteps."""
    # Check if grid data is available
    if not coarse_data or "grid" not in coarse_data[0]:
        print("  skipping coarse wind grids — no grid field in data")
        plt.close("all")
        return
    # Pick 6 evenly-spaced timesteps
    n_steps = len(coarse_data)
    indices = np.linspace(0, n_steps - 1, 6, dtype=int)
    # Also get the corresponding terrain (coarsened to 6x8)
    elev = np.array(terrain_data["elevation"])
    # Downsample terrain to 6x8 for overlay
    h, w = elev.shape
    elev_coarse = elev.reshape(6, h // 6, 8, w // 8).mean(axis=(1, 3))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (ax, t_idx) in enumerate(zip(axes.flat, indices)):
        entry = coarse_data[t_idx]
        grid = entry.get("grid", [])
        if not grid:
            continue

        # Build speed grid from grid data
        rows = len(grid)
        cols = len(grid[0])
        speed_grid = np.zeros((rows, cols))
        u_grid = np.zeros((rows, cols))
        v_grid = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                speed_grid[r, c] = cell["speed"]
                u_grid[r, c] = cell["u"]
                v_grid[r, c] = cell["v"]

        im = ax.imshow(speed_grid, origin="upper", cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.82, label="m/s")

        # Wind vectors
        stride = 1
        yy, xx = np.mgrid[0:rows:stride, 0:cols:stride]
        ax.quiver(xx, yy, u_grid[::stride, ::stride], v_grid[::stride, ::stride],
                  scale=80, width=0.006, color="white", alpha=0.85)

        # Overlay terrain contour
        if elev_coarse.shape == (rows, cols):
            ax.contour(np.arange(cols), np.arange(rows), elev_coarse,
                       levels=5, colors="blue", linewidths=0.8, alpha=0.4)

        ts = entry["timestamp"].replace("T", " ")
        set_panel_title(ax, f"t={ts}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save("12_coarse_wind_grids")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 13: Coarse Wind Statistics — Grid mean vs scalar comparison
# ══════════════════════════════════════════════════════════════════════
def plot_coarse_wind_stats():
    """Compare the scalar u_km/v_km with the grid-mean wind."""
    times = np.arange(len(coarse_data))
    u_scalar = np.array([c["u_km"] for c in coarse_data])
    v_scalar = np.array([c["v_km"] for c in coarse_data])
    w_scalar = np.array([c["w_km"] for c in coarse_data])

    # Grid means
    u_mean = np.zeros(len(coarse_data))
    v_mean = np.zeros(len(coarse_data))
    w_mean = np.zeros(len(coarse_data))
    speed_mean = np.zeros(len(coarse_data))
    u_std = np.zeros(len(coarse_data))
    v_std = np.zeros(len(coarse_data))

    for i, entry in enumerate(coarse_data):
        grid = entry.get("grid", [])
        if grid:
            cells = [cell for row in grid for cell in row]
            u_mean[i] = np.mean([c["u"] for c in cells])
            v_mean[i] = np.mean([c["v"] for c in cells])
            w_mean[i] = np.mean([c["w"] for c in cells])
            speed_mean[i] = np.mean([c["speed"] for c in cells])
            u_std[i] = np.std([c["u"] for c in cells])
            v_std[i] = np.std([c["v"] for c in cells])

    def row1(ax0, ax1):
        ax0.plot(times, u_scalar, color=C["teal"], linewidth=2, label="标量 u_km")
        ax0.plot(times, u_mean, color=C["orange"], linewidth=2, linestyle="--", label="网格均值 u")
        ax0.fill_between(times, u_mean - u_std, u_mean + u_std, color=C["orange"], alpha=0.15)
        set_axis_style(ax0, "U 分量: 标量 vs 网格均值", "时间步", "m/s")
        ax0.legend(fontsize=18)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(times, v_scalar, color=C["red"], linewidth=2, label="标量 v_km")
        ax1.plot(times, v_mean, color=C["orange"], linewidth=2, linestyle="--", label="网格均值 v")
        ax1.fill_between(times, v_mean - v_std, v_mean + v_std, color=C["orange"], alpha=0.15)
        set_axis_style(ax1, "V 分量: 标量 vs 网格均值", "时间步", "m/s")
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    def row2(ax0, ax1):
        scalar_speed = np.hypot(u_scalar, v_scalar)
        ax0.plot(times, scalar_speed, color=C["teal"], linewidth=2, label="标量 |V|")
        ax0.plot(times, speed_mean, color=C["orange"], linewidth=2, linestyle="--", label="网格均值 |V|")
        set_axis_style(ax0, "水平风速: 标量 vs 网格均值", "时间步", "m/s")
        ax0.legend(fontsize=18)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(times, u_std, color=C["teal"], linewidth=1.5, label="σ(U)")
        ax1.plot(times, v_std, color=C["red"], linewidth=1.5, label="σ(V)")
        set_axis_style(ax1, "粗尺度网格空间变异性 (标准差)", "时间步", "m/s")
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    save_two_col_rows("13_coarse_wind_stats", [row1, row2], figsize=(14, 4.5))


# ══════════════════════════════════════════════════════════════════════
# Chart 14: Multi-Altitude Wind Speed Layers
# ══════════════════════════════════════════════════════════════════════
def plot_multi_altitude_wind():
    trace = report["trace"]
    frame = None
    for f in trace:
        if f.get("maps") and f["maps"].get("physics_wind_speed_layers"):
            frame = f
            break
    if not frame:
        print("  skipping multi-altitude — no layer data")
        return

    maps = frame["maps"]
    phys_layers = maps.get("physics_wind_speed_layers", [])
    truth_layers = maps.get("truth_wind_speed_layers", [])
    n_layers = len(phys_layers)

    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))

    all_data = []
    for layer in phys_layers:
        all_data.extend(np.array(layer).flat)
    for layer in truth_layers:
        all_data.extend(np.array(layer).flat)
    vmin, vmax = np.min(all_data), np.max(all_data)

    for i in range(n_layers):
        # Physics wind
        im1 = axes[0, i].imshow(np.array(phys_layers[i]), origin="upper", cmap="YlOrRd",
                                vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im1, ax=axes[0, i], shrink=0.82, label="m/s")
        set_panel_title(axes[0, i], f"物理风场 z={i}")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Truth wind
        if i < len(truth_layers):
            im2 = axes[1, i].imshow(np.array(truth_layers[i]), origin="upper", cmap="YlOrRd",
                                    vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im2, ax=axes[1, i], shrink=0.82, label="m/s")
            set_panel_title(axes[1, i], f"真值风场 z={i}")
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

    axes[0, 0].set_ylabel("物理降尺度", fontsize=18, fontweight="bold")
    axes[1, 0].set_ylabel("真值 (合成)", fontsize=18, fontweight="bold")
    plt.tight_layout()
    save("14_multi_altitude_wind")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 15: Belief Metrics Timeline at Drone Position
# ══════════════════════════════════════════════════════════════════════
def plot_belief_timeline():
    """Line charts of all belief metrics at the drone's actual position over time."""
    trace = report["trace"]
    steps = np.arange(len(trace))

    energy = _belief_series_from_trace(trace, "energy")
    uncertainty = _belief_series_from_trace(trace, "uncertainty")
    confidence = _belief_series_from_trace(trace, "confidence")
    entropy = _belief_series_from_trace(trace, "entropy")
    safety = _belief_series_from_trace(trace, "safety")
    uplift = _belief_series_from_trace(trace, "uplift_prob")
    sink = _belief_series_from_trace(trace, "sink_prob")

    # Mark observation points
    obs_steps = [i for i, f in enumerate(trace) if f.get("observation")]

    def row1(ax0, ax1):
        ax0.fill_between(steps, 0, energy, color=C["green"], alpha=0.2)
        ax0.plot(steps, energy, color=C["green"], linewidth=2.5, marker="o", markersize=8, markerfacecolor="white")
        ax0.axhline(y=0, color=C["ink"], linewidth=0.8, linestyle="--", alpha=0.5)
        for os in obs_steps:
            ax0.axvline(os, color=C["orange"], linewidth=0.8, alpha=0.4, linestyle=":")
        set_axis_style(ax0, "信念能量 (预期能量增益)", "步数", "能量")
        ax0.text(0.98, 0.95, "↑越高越好\n(热气流区域)", transform=ax0.transAxes, fontsize=18,
                 ha="right", va="top", color=C["green"], fontweight="bold")
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(steps, confidence, color=C["teal"], linewidth=2.5, marker="s", markersize=8,
                 markerfacecolor="white", label="置信度")
        ax1.plot(steps, uncertainty, color=C["red"], linewidth=2.0, linestyle="--", marker="^",
                 markersize=8, alpha=0.7, label="不确定性")
        for os in obs_steps:
            ax1.axvline(os, color=C["orange"], linewidth=0.8, alpha=0.4, linestyle=":")
        set_axis_style(ax1, "信念置信度 & 不确定性", "步数", "")
        ax1.legend(fontsize=19, loc="center right", framealpha=0.8)
        ax1.text(0.98, 0.95, "置信度高=信\n不确定性高=不信", transform=ax1.transAxes, fontsize=18,
                 ha="right", va="top", color=C["muted"])
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    def row2(ax0, ax1):
        ax0.fill_between(steps, 0, safety, color=C["red"], alpha=0.2)
        ax0.plot(steps, safety, color=C["red"], linewidth=2.5, marker="D", markersize=8, markerfacecolor="white")
        for os in obs_steps:
            ax0.axvline(os, color=C["orange"], linewidth=0.8, alpha=0.4, linestyle=":")
        set_axis_style(ax0, "安全风险 (safety_penalty)", "步数", "风险值")
        ax0.text(0.98, 0.95, "↓越低越安全", transform=ax0.transAxes, fontsize=18,
                 ha="right", va="top", color=C["red"], fontweight="bold")
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(steps, uplift, color=C["orange"], linewidth=2.5, marker="o", markersize=8,
                 markerfacecolor="white", label="上升气流概率")
        ax1.plot(steps, sink, color=C["blue"], linewidth=2.0, linestyle="--", marker="v",
                 markersize=8, alpha=0.7, label="下沉气流概率")
        ax1.axhline(y=0.35, color=C["ink"], linewidth=0.6, linestyle=":", alpha=0.4)
        ax1.text(len(steps) - 1, 0.35, " 中性阈值", fontsize=18, color=C["muted"], va="bottom")
        for os in obs_steps:
            ax1.axvline(os, color=C["orange"], linewidth=0.8, alpha=0.4, linestyle=":")
        set_axis_style(ax1, "热气流模式概率 (uplift vs sink)", "步数", "概率")
        ax1.legend(fontsize=19, loc="center right", framealpha=0.8)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    save_two_col_rows("15_belief_timeline", [row1, row2], figsize=(14, 5.0))


# ══════════════════════════════════════════════════════════════════════
# Chart 22: Belief Energy as Potential Field with Drone Path
# ══════════════════════════════════════════════════════════════════════
def plot_belief_energy_path():
    """Show belief energy map with drone trajectory — demonstrating energy-guided navigation."""
    trace = report["trace"]
    elev = np.array(terrain_data["elevation"])

    # Get first and last keyframe belief energy maps
    keyframes = [f for f in trace if f.get("maps") and f["maps"].get("belief_energy")]
    if len(keyframes) < 2:
        print("  skipping belief energy path — insufficient data")
        return

    # Get drone positions
    xs = [f["position"][0] for f in trace]
    ys = [f["position"][1] for f in trace]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: belief energy map at start
    kf_start = keyframes[0]
    energy_start = np.array(kf_start["maps"]["belief_energy"])
    ax = axes[0]
    im = ax.imshow(energy_start, origin="upper", cmap="RdYlGn", aspect="auto",
                   vmin=-1.5, vmax=2.5)
    plt.colorbar(im, ax=ax, shrink=0.78, label="预期能量增益")
    # Overlay drone path
    colors_path = plt.cm.viridis(np.linspace(0.2, 1, len(xs)))
    for i in range(1, len(xs)):
        ax.plot(xs[i-1:i+1], ys[i-1:i+1], color=colors_path[i], linewidth=2.5, alpha=0.85)
    start = report["mission"]["start"][:2]
    goal = report["mission"]["goal"][:2]
    ax.scatter(*start, c="white", s=180, marker="o", zorder=5, edgecolors=C["ink"], linewidth=2, label="起点")
    ax.scatter(*goal, c=C["red"], s=180, marker="X", zorder=5, edgecolors="white", linewidth=2, label="目标")
    set_axis_style(ax, "初始信念能量场 + 执行轨迹")
    ax.legend(fontsize=19, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: belief energy map at end
    kf_end = keyframes[-1]
    energy_end = np.array(kf_end["maps"]["belief_energy"])
    ax = axes[1]
    im = ax.imshow(energy_end, origin="upper", cmap="RdYlGn", aspect="auto",
                   vmin=-1.5, vmax=2.5)
    plt.colorbar(im, ax=ax, shrink=0.78, label="预期能量增益")
    for i in range(1, len(xs)):
        ax.plot(xs[i-1:i+1], ys[i-1:i+1], color=colors_path[i], linewidth=2.5, alpha=0.85)
    ax.scatter(*start, c="white", s=180, marker="o", zorder=5, edgecolors=C["ink"], linewidth=2, label="起点")
    ax.scatter(*goal, c=C["red"], s=180, marker="X", zorder=5, edgecolors="white", linewidth=2, label="目标")
    set_axis_style(ax, "最终信念能量场 + 执行轨迹")
    ax.legend(fontsize=19, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    save("22_belief_energy_path")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 23: Belief vs Reality — predicted vs observed
# ══════════════════════════════════════════════════════════════════════
def plot_belief_vs_reality():
    """Compare belief predictions with actual observations at drone positions."""
    trace = report["trace"]

    # Collect belief wind vs observed wind at each frame
    belief_u = []
    belief_v = []
    belief_w = []
    belief_speed = []
    obs_u = []
    obs_v = []
    obs_w = []
    obs_speed = []
    phys_u = []
    phys_v = []
    pred_u = []
    pred_v = []

    for f in trace:
        obs = f.get("observation")
        b = f.get("belief_at_drone", {})
        ph = f.get("physics_at_drone", {})
        pr = f.get("prediction_at_drone", {})
        if obs:
            obs_u.append(obs.get("u_obs", 0))
            obs_v.append(obs.get("v_obs", 0))
            obs_w.append(obs.get("w_obs", 0))
            obs_speed.append(np.hypot(obs.get("u_obs", 0), obs.get("v_obs", 0)))
            belief_u.append(b.get("energy", 0))  # belief_energy is related
            belief_v.append(b.get("confidence", 0))
            belief_w.append(b.get("w", 0))
            belief_speed.append(np.hypot(b.get("energy", 0), b.get("w", 0)))
            phys_u.append(ph.get("u", 0))
            phys_v.append(ph.get("v", 0))
            pred_u.append(pr.get("u", 0))
            pred_v.append(pr.get("v", 0))

    if len(obs_u) < 5:
        print("  skipping belief vs reality — insufficient observations")
        return

    steps = np.arange(len(obs_u))
    phys_err = np.abs(np.array(phys_u) - np.array(obs_u)) + np.abs(np.array(phys_v) - np.array(obs_v))
    pred_err = np.abs(np.array(pred_u) - np.array(obs_u)) + np.abs(np.array(pred_v) - np.array(obs_v))
    conf_all = _belief_series_from_trace(trace, "confidence")
    conf_values = [c for f, c in zip(trace, conf_all) if f.get("observation")]
    uncert_all = _belief_series_from_trace(trace, "uncertainty")
    uncert_values = [u for f, u in zip(trace, uncert_all) if f.get("observation")]

    def row1(ax0, ax1):
        ax0.scatter(phys_u, obs_u, c=steps, cmap="viridis", s=40, alpha=0.7, edgecolors="none")
        ax0.plot([min(phys_u), max(phys_u)], [min(phys_u), max(phys_u)], "k--", alpha=0.4, label="y=x")
        r2 = np.corrcoef(phys_u, obs_u)[0, 1] ** 2
        set_axis_style(ax0, f"物理模型 U vs 观测 U (R2={r2:.3f})", "物理 U (m/s)", "观测 U (m/s)")
        ax0.legend(fontsize=18)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.scatter(phys_v, obs_v, c=steps, cmap="viridis", s=40, alpha=0.7, edgecolors="none")
        ax1.plot([min(phys_v), max(phys_v)], [min(phys_v), max(phys_v)], "k--", alpha=0.4, label="y=x")
        r2 = np.corrcoef(phys_v, obs_v)[0, 1] ** 2
        set_axis_style(ax1, f"物理模型 V vs 观测 V (R2={r2:.3f})", "物理 V (m/s)", "观测 V (m/s)")
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    def row2(ax0, ax1):
        ax0.scatter(pred_u, obs_u, c=steps, cmap="plasma", s=40, alpha=0.7, edgecolors="none")
        ax0.plot([min(pred_u), max(pred_u)], [min(pred_u), max(pred_u)], "k--", alpha=0.4, label="y=x")
        r2 = np.corrcoef(pred_u, obs_u)[0, 1] ** 2
        set_axis_style(ax0, f"ML预测 U vs 观测 U (R2={r2:.3f})", "ML预测 U (m/s)", "观测 U (m/s)")
        ax0.legend(fontsize=18)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.plot(steps, phys_err, color=C["teal"], linewidth=2.0, label="物理模型 |U|+|V| 误差")
        ax1.plot(steps, pred_err, color=C["orange"], linewidth=2.5, linestyle="--", label="ML预测 |U|+|V| 误差")
        ax1.fill_between(steps, phys_err, pred_err, color=C["green"], alpha=0.15)
        set_axis_style(ax1, "预测误差时序 (低=更好)", "步数", "|U err| + |V err| (m/s)")
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    def row3(ax0, ax1):
        ax0.scatter(conf_values, pred_err, c=steps, cmap="plasma", s=50, alpha=0.7, edgecolors="none")
        set_axis_style(ax0, "信念置信度 vs 预测误差", "信念置信度", "|U|+|V| 误差 (m/s)")
        ax0.text(0.95, 0.95, "↓高置信应低误差", transform=ax0.transAxes, fontsize=19,
                 ha="right", va="top", color=C["muted"])
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax1.scatter(uncert_values, pred_err, c=steps, cmap="plasma", s=50, alpha=0.7, edgecolors="none")
        set_axis_style(ax1, "信念不确定性 vs 预测误差", "信念不确定性", "|U|+|V| 误差 (m/s)")
        ax1.text(0.95, 0.95, "↑高不确定应高误差", transform=ax1.transAxes, fontsize=19,
                 ha="right", va="top", color=C["muted"])
        ax1.grid(True, alpha=0.3, linewidth=0.5)

    save_two_col_rows("23_belief_vs_reality", [row1, row2, row3], figsize=(14, 5.0))


# ══════════════════════════════════════════════════════════════════════
# Chart 24: Belief-Guided Path Planning
# ══════════════════════════════════════════════════════════════════════
def plot_belief_path_planning():
    """Show planned path on belief energy + safety map with terrain context."""
    trace = report["trace"]
    elev = np.array(terrain_data["elevation"])

    # Pick keyframe at ~80% through mission (later toward goal)
    keyframes = [f for f in trace if f.get("maps") and f["maps"].get("belief_energy")]
    mid_idx = min(len(keyframes) - 1, int(len(keyframes) * 0.75))
    kf = keyframes[mid_idx]
    maps = kf["maps"]
    energy = np.array(maps.get("belief_energy", [[0]]))
    safety_map = np.array(maps.get("belief_safety", [[0]]))
    conf_map = np.array(maps.get("belief_confidence", [[0]]))
    uncert_map = np.array(maps.get("belief_uncertainty", [[0]]))
    bu = np.array(maps.get("belief_wind_u", [[0]]))
    bv = np.array(maps.get("belief_wind_v", [[0]]))
    pp = kf.get("planned_path", [])
    pxs = [p[0] for p in pp] if len(pp) > 1 else []
    pys = [p[1] for p in pp] if len(pp) > 1 else []
    pos = kf.get("position", [0, 0])

    def row1(ax0, ax1):
        im = ax0.imshow(energy, origin="upper", cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax0, shrink=0.78, label="预期能量增益")
        ax0.contour(np.arange(elev.shape[1]), np.arange(elev.shape[0]), elev,
                    levels=6, colors="blue", linewidths=0.8, alpha=0.5)
        if pxs:
            ax0.plot(pxs, pys, color="white", linewidth=3.0, marker="o", markersize=8, markerfacecolor=C["ink"])
        ax0.scatter(*pos[:2], c=C["red"], s=200, marker="D", zorder=5, edgecolors="white", linewidth=2, label="无人机")
        set_axis_style(ax0, "信念能量场 + 规划路径\n(蓝线=地形等高线, 白线=规划路径)")
        ax0.legend(fontsize=18)
        ax0.set_xticks([])
        ax0.set_yticks([])
        im2 = ax1.imshow(safety_map, origin="upper", cmap="Reds", aspect="auto")
        plt.colorbar(im2, ax=ax1, shrink=0.78, label="安全风险")
        if pxs:
            ax1.plot(pxs, pys, color="white", linewidth=3.0, marker="o", markersize=8, markerfacecolor=C["ink"])
        ax1.scatter(*pos[:2], c=C["teal"], s=200, marker="D", zorder=5, edgecolors="white", linewidth=2)
        set_axis_style(ax1, "安全风险 + 规划路径\n(路径避开红色高风险区)")
        ax1.set_xticks([])
        ax1.set_yticks([])

    def row2(ax0, ax1):
        im3 = ax0.imshow(conf_map, origin="upper", cmap="Blues", aspect="auto")
        plt.colorbar(im3, ax=ax0, shrink=0.78, label="置信度")
        if pxs:
            ax0.plot(pxs, pys, color=C["orange"], linewidth=3.0, marker="o", markersize=8, markerfacecolor=C["ink"])
        ax0.scatter(*pos[:2], c=C["red"], s=200, marker="D", zorder=5, edgecolors="white", linewidth=2)
        set_axis_style(ax0, "信念置信度 + 规划路径\n(深蓝=高置信, 白=低置信)")
        ax0.set_xticks([])
        ax0.set_yticks([])
        im4 = ax1.imshow(uncert_map, origin="upper", cmap="YlOrBr", aspect="auto")
        plt.colorbar(im4, ax=ax1, shrink=0.78, label="不确定性")
        stride = 2
        yy, xx = np.mgrid[0:bu.shape[0]:stride, 0:bu.shape[1]:stride]
        ax1.quiver(xx, yy, bu[::stride, ::stride], bv[::stride, ::stride],
                   scale=40, width=0.004, color="white", alpha=0.7)
        if pxs:
            ax1.plot(pxs, pys, color=C["teal"], linewidth=3.0, marker="o", markersize=8, markerfacecolor=C["ink"])
        ax1.scatter(*pos[:2], c=C["red"], s=200, marker="D", zorder=5, edgecolors="white", linewidth=2)
        set_axis_style(ax1, "不确定性 + 信念风矢量\n(白箭头=信念风速方向)")
        ax1.set_xticks([])
        ax1.set_yticks([])

    save_two_col_rows("24_belief_path_planning", [row1, row2], figsize=(14, 6.0))


def _find_energy_summary() -> Path | None:
    runs_dir = PROJECT / "runs"
    candidates = sorted(runs_dir.glob("multi-scenario-energy-*/energy_summary.json"), reverse=True)
    return candidates[0] if candidates else None


# Benchmark energy savings (multi-scenario-energy-20260810-202627, route r0).
# Used when runs/multi-scenario-energy-*/energy_summary.json is absent.
EIGHT_SCENARIO_ENERGY_FALLBACK: dict[str, float] = {
    "fujian_hills": 11.2,
    "beijing_plain": 0.0,
    "qinghai_ridge": 19.6,
    "liaoning_coast": 9.4,
    "xinjiang_gobi": 0.0,
    "sichuan_foothills": 6.8,
    "shanxi_loess": 8.1,
    "taiwan_hills": 14.9,
}


def _eight_scenario_energy_map() -> dict[str, float]:
    summary_path = _find_energy_summary()
    if summary_path:
        summary = load_json(summary_path)
        out: dict[str, float] = {}
        for row in summary.get("results", []):
            if row.get("route_id") != "r0":
                continue
            name = row.get("map")
            if name in out:
                continue
            out[name] = float(row.get("savings_vs_nominal_pct", 0.0))
        if out:
            return out
    return dict(EIGHT_SCENARIO_ENERGY_FALLBACK)


def _load_scenario_executed_path(scenario: str) -> list[tuple[float, float]]:
    for run_dir in sorted((PROJECT / "runs").glob(f"eval-{scenario}-r*"), reverse=True):
        report_path = run_dir / "mission_report.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        path = report.get("executed_path") or []
        if len(path) > 1:
            return [(float(p[0]), float(p[1])) for p in path]
    return []


def _draw_coarse_wind_quiver(ax, scenario: str) -> None:
    coarse_path = SCENARIOS_DIR / scenario / "coarse_wind.json"
    if not coarse_path.exists():
        return
    coarse = load_json(coarse_path).get("coarse_wind", [])
    if not coarse:
        return
    sample = coarse[len(coarse) // 2]
    u = float(sample.get("u_km", 0.0))
    v = float(sample.get("v_km", 0.0))
    grid = sample.get("grid")
    h, w = ax.images[0].get_array().shape if ax.images else (20, 20)
    if grid:
        ug = np.array(grid.get("u", [[u]]))
        vg = np.array(grid.get("v", [[v]]))
        h, w = ug.shape
        stride = max(1, min(h, w) // 8)
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        ax.quiver(xx, yy, ug[::stride, ::stride], vg[::stride, ::stride],
                  scale=35, width=0.0035, color="white", alpha=0.85)
        return
    stride = max(2, min(h, w) // 8)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    ax.quiver(xx, yy, np.full_like(xx, u, dtype=float), np.full_like(yy, v, dtype=float),
              scale=35, width=0.0035, color="white", alpha=0.85)


def plot_eight_scenarios_terrain():
    """Eight real-map scenarios: terrain + coarse wind + nominal straight vs executed path."""
    energy_map = _eight_scenario_energy_map()

    def _draw_scenario(ax, scenario, label, group):
        terrain_path = SCENARIOS_DIR / scenario / "terrain.json"
        config_path = SCENARIOS_DIR / scenario / "config.json"
        if not terrain_path.exists() or not config_path.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"{label}\n(缺少数据)", ha="center", va="center")
            return
        elev = np.array(load_json(terrain_path)["terrain"]["elevation"])
        mission = load_json(config_path).get("mission", {})
        start = mission.get("start", [0, 0])[:2]
        goal = mission.get("goal", [0, 0])[:2]
        executed = _load_scenario_executed_path(scenario)
        savings = energy_map.get(scenario, 0.0)
        im = ax.imshow(elev, origin="upper", cmap="terrain", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.78, label="高程 (m)")
        _draw_coarse_wind_quiver(ax, scenario)
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color="white", linewidth=1.6,
                linestyle="--", alpha=0.9, label="名义直线")
        if executed:
            ax.plot([p[0] for p in executed], [p[1] for p in executed], color=C["teal"],
                    linewidth=2.2, alpha=0.95, label="算法轨迹")
        ax.scatter(start[0], start[1], c=C["green"], s=70, edgecolors="white", linewidth=1.2, zorder=5)
        ax.scatter(goal[0], goal[1], c=C["red"], s=90, marker="x", linewidths=2.5, zorder=5)
        set_panel_title(ax, f"{label}  [{group}]  节能 +{savings:.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])

    row_builders = []
    for r in range(0, len(EIGHT_SCENARIOS), 2):
        left = EIGHT_SCENARIOS[r]
        right = EIGHT_SCENARIOS[r + 1] if r + 1 < len(EIGHT_SCENARIOS) else None

        def make_row(left=left, right=right):
            def row(ax0, ax1):
                _draw_scenario(ax0, *left)
                if right:
                    _draw_scenario(ax1, *right)
                else:
                    ax1.axis("off")
            return row

        row_builders.append(make_row())

    save_two_col_rows("25_eight_scenarios_terrain", row_builders, figsize=(14, 6.0))


def plot_eight_scenarios_energy():
    """Bar chart of energy savings for the original eight scenarios."""
    energy_map = _eight_scenario_energy_map()
    summary_path = _find_energy_summary()
    if not summary_path:
        print("  using benchmark energy table (no energy_summary.json in runs/)")
    core_vals = [energy_map.get(name, 0.0) for name, _, group in EIGHT_SCENARIOS if group == "CORE"]
    holdout_vals = [energy_map.get(name, 0.0) for name, _, group in EIGHT_SCENARIOS if group == "HOLDOUT"]
    labels = [label for _, label, _ in EIGHT_SCENARIOS]
    core_mean = float(np.mean(core_vals)) if core_vals else 0.0
    holdout_mean = float(np.mean(holdout_vals)) if holdout_vals else 0.0

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    colors = [C["teal"] if group == "CORE" else C["orange"] for _, _, group in EIGHT_SCENARIOS]
    values = [energy_map.get(name, 0.0) for name, _, _ in EIGHT_SCENARIOS]
    bars = ax.bar(x, values, color=colors, alpha=0.88, width=0.62)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                f"+{val:.1f}%", ha="center", fontsize=15, fontweight="bold", color=C["ink"])

    ax.axhline(core_mean, color=C["teal"], linestyle="--", linewidth=1.8, alpha=0.85)
    ax.axhline(holdout_mean, color=C["orange"], linestyle="--", linewidth=1.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    set_axis_style(ax, "典型场景闭环能耗节省（相对名义直线基线）", ylabel="相对名义直线能耗节省 (%)")
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    ax.legend(
        [
            plt.Line2D([0], [0], color=C["teal"], lw=8),
            plt.Line2D([0], [0], color=C["orange"], lw=8),
            plt.Line2D([0], [0], color=C["teal"], lw=2, linestyle="--"),
            plt.Line2D([0], [0], color=C["orange"], lw=2, linestyle="--"),
        ],
        ["CORE (调参)", "HOLDOUT (独立验收)", f"CORE 均值 {core_mean:.1f}%", f"HOLDOUT 均值 {holdout_mean:.1f}%"],
        fontsize=15, loc="upper right", framealpha=0.9,
    )
    plt.tight_layout()
    save("26_eight_scenarios_energy")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 16: Planned vs Executed Path with Replanning Detail
# ══════════════════════════════════════════════════════════════════════
def plot_path_detail():
    trace = report["trace"]
    elev = np.array(terrain_data["elevation"])
    keyframes = [f for f in trace if f.get("planned_path") and len(f["planned_path"]) > 1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Main trajectory on terrain (large)
    ax = axes[0, 0]
    im = ax.imshow(elev, origin="upper", cmap="terrain", alpha=0.75, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.78, label="高程 (m)")
    xs = [f["position"][0] for f in trace]
    ys = [f["position"][1] for f in trace]
    colors_path = plt.cm.viridis(np.linspace(0, 1, len(xs)))
    for i in range(1, len(xs)):
        ax.plot(xs[i-1:i+1], ys[i-1:i+1], color=colors_path[i], linewidth=2.0, alpha=0.9)
    # Show replanning waypoints
    for kf in keyframes[::5]:
        pp = kf["planned_path"]
        if len(pp) > 1:
            pxs = [p[0] for p in pp]
            pys = [p[1] for p in pp]
            ax.plot(pxs, pys, color="white", linewidth=1.0, alpha=0.35, linestyle="--")
    start = report["mission"]["start"][:2]
    goal = report["mission"]["goal"][:2]
    ax.scatter(*start, c="green", s=150, marker="o", zorder=5, edgecolors="white", label="起点")
    ax.scatter(*goal, c="red", s=150, marker="X", zorder=5, edgecolors="white", label="目标")
    set_axis_style(ax, "执行轨迹 + 重规划路径", "X", "Y")
    ax.legend(fontsize=18, loc="upper left")

    # Goal distance over time
    ax = axes[0, 1]
    steps = np.arange(len(trace))
    distances = [f["goal_distance_cells"] for f in trace]
    ax.plot(steps, distances, color=C["orange"], linewidth=2.5)
    ax.fill_between(steps, 0, distances, color=C["orange"], alpha=0.15)
    # Mark replanning steps
    keyframe_steps = [i for i, f in enumerate(trace) if f.get("is_keyframe")]
    for ks in keyframe_steps:
        ax.axvline(ks, color=C["teal"], linewidth=0.6, alpha=0.3, linestyle=":")
    set_axis_style(ax, "目标距离变化 (竖线=重规划)", "步数", "距离 (格点)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Path cost evolution
    ax = axes[0, 2]
    costs = [f.get("planned_path_cost", 0) for f in trace]
    ax.plot(steps, costs, color=C["red"], linewidth=2.0)
    ax.fill_between(steps, 0, costs, color=C["red"], alpha=0.1)
    set_axis_style(ax, "规划路径代价", "步数", "代价 (J)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Altitude profile
    ax = axes[1, 0]
    alts = [f["position"][2] for f in trace]
    ax.plot(steps, alts, color=C["green"], linewidth=2.0)
    ax.fill_between(steps, 0, alts, color=C["green"], alpha=0.15)
    # Overlay planned altitudes
    for kf in keyframes[::5]:
        pp = kf["planned_path"]
        if len(pp) > 1:
            kf_step = trace.index(kf)
            plan_len = len(pp)
            plan_steps = np.arange(kf_step, kf_step + plan_len)
            plan_alts = [p[2] if len(p) > 2 else 0 for p in pp]
            ax.plot(plan_steps[:len(plan_alts)], plan_alts, color=C["orange"], linewidth=1.0, alpha=0.4, linestyle="--")
    set_axis_style(ax, "高度剖面 (虚线=规划)", "步数", "高度层")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Candidate paths count
    ax = axes[1, 1]
    candidate_counts = [len(f.get("candidate_paths", [])) for f in trace]
    ax.bar(steps, candidate_counts, color=C["teal"], alpha=0.7, width=1.0)
    set_axis_style(ax, "候选路径数量", "步数", "候选数")
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Planning mode distribution
    ax = axes[1, 2]
    modes = [f.get("planning_mode", "unknown") for f in trace]
    unique_modes = list(set(modes))
    mode_counts = {m: modes.count(m) for m in unique_modes}
    colors_pie = plt.cm.Set2(np.linspace(0, 1, len(unique_modes)))
    wedges, texts, autotexts = ax.pie(
        mode_counts.values(), labels=mode_counts.keys(),
        autopct="%1.1f%%", colors=colors_pie, startangle=90
    )
    for at in autotexts:
        at.set_fontsize(9)
    set_axis_style(ax, "规划模式分布")

    plt.tight_layout()
    save("16_path_detail")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 17: Sensor Data Timeseries
# ══════════════════════════════════════════════════════════════════════
def plot_sensor_timeseries():
    trace = report["trace"]
    steps = np.arange(len(trace))

    # Extract sensor data
    airspeeds = [f.get("sensor_packet", {}).get("airspeed", 0) for f in trace]
    ground_speeds = [f.get("sensor_packet", {}).get("ground_speed", 0) for f in trace]
    headings = [f.get("sensor_packet", {}).get("heading_rad", 0) for f in trace]
    measured_u = [f.get("sensor_packet", {}).get("measured_wind_u", 0) for f in trace]
    measured_v = [f.get("sensor_packet", {}).get("measured_wind_v", 0) for f in trace]
    measured_w = [f.get("sensor_packet", {}).get("measured_wind_w", 0) for f in trace]
    imu_long = [f.get("sensor_packet", {}).get("imu_accel_longitudinal", 0) for f in trace]
    imu_vert = [f.get("sensor_packet", {}).get("imu_accel_vertical", 0) for f in trace]
    alt_m = [f.get("sensor_packet", {}).get("altitude_m", 0) for f in trace]
    energy_rate = [f.get("sensor_packet", {}).get("energy_rate", 0) for f in trace]

    # Use physics_at_drone and prediction_at_drone for wind
    phys_u = [f.get("physics_at_drone", {}).get("u", 0) for f in trace]
    phys_v = [f.get("physics_at_drone", {}).get("v", 0) for f in trace]
    pred_u = [f.get("prediction_at_drone", {}).get("u", 0) for f in trace]
    pred_v = [f.get("prediction_at_drone", {}).get("v", 0) for f in trace]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # Speed
    ax = axes[0, 0]
    ax.plot(steps, airspeeds, color=C["teal"], linewidth=1.8, label="空速")
    ax.plot(steps, ground_speeds, color=C["orange"], linewidth=1.8, label="地速")
    set_axis_style(ax, "飞行速度", "步数", "m/s")
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Heading
    ax = axes[0, 1]
    ax.plot(steps, np.degrees(headings), color=C["orange"], linewidth=1.5)
    set_axis_style(ax, "航向角", "步数", "度 (°)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Altitude
    ax = axes[0, 2]
    ax.plot(steps, alt_m, color=C["green"], linewidth=1.8)
    ax.fill_between(steps, 0, alt_m, color=C["green"], alpha=0.12)
    set_axis_style(ax, "气压高度", "步数", "m")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Measured wind U/V
    ax = axes[1, 0]
    ax.plot(steps, measured_u, color=C["teal"], linewidth=1.5, label="测量 U")
    ax.plot(steps, measured_v, color=C["red"], linewidth=1.5, label="测量 V")
    ax.plot(steps, measured_w, color=C["green"], linewidth=1.2, alpha=0.7, label="测量 W")
    set_axis_style(ax, "机载风测量值", "步数", "m/s")
    ax.legend(fontsize=18)
    ax.axhline(y=0, color=C["ink"], linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Physics vs Prediction wind at drone
    ax = axes[1, 1]
    ax.plot(steps, phys_u, color=C["teal"], linewidth=1.5, alpha=0.7, label="物理 U")
    ax.plot(steps, pred_u, color=C["teal"], linewidth=2.0, linestyle="--", label="预测 U")
    ax.plot(steps, phys_v, color=C["red"], linewidth=1.5, alpha=0.7, label="物理 V")
    ax.plot(steps, pred_v, color=C["red"], linewidth=2.0, linestyle="--", label="预测 V")
    set_axis_style(ax, "物理 vs ML预测风 (无人机位置)", "步数", "m/s")
    ax.legend(fontsize=18, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # IMU
    ax = axes[1, 2]
    ax.plot(steps, imu_long, color=C["teal"], linewidth=1.5, label="纵向加速度")
    ax.plot(steps, imu_vert, color=C["red"], linewidth=1.5, label="垂向加速度")
    set_axis_style(ax, "IMU 加速度", "步数", "g")
    ax.legend(fontsize=18)
    ax.axhline(y=0, color=C["ink"], linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Energy rate
    ax = axes[2, 0]
    ax.plot(steps, energy_rate, color=C["purple"], linewidth=1.8)
    ax.fill_between(steps, 0, energy_rate, color=C["purple"], alpha=0.12)
    set_axis_style(ax, "能量获取率 (信念)", "步数", "")
    ax.axhline(y=0, color=C["ink"], linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Wind speed at drone: physics vs prediction vs truth
    ax = axes[2, 1]
    phys_speed = [f.get("physics_at_drone", {}).get("speed", 0) for f in trace]
    pred_speed = [f.get("prediction_at_drone", {}).get("speed", 0) for f in trace]
    truth_at = [f.get("truth_at_drone", {}).get("speed") for f in trace]
    ax.plot(steps, phys_speed, color=C["teal"], linewidth=1.5, alpha=0.8, label="物理")
    ax.plot(steps, pred_speed, color=C["orange"], linewidth=2.0, linestyle="--", label="预测")
    valid_truth = [(i, v) for i, v in enumerate(truth_at) if v is not None]
    if valid_truth:
        ti, tv = zip(*valid_truth)
        ax.scatter(ti, tv, color=C["red"], s=12, alpha=0.6, label="真值", zorder=5)
    set_axis_style(ax, "风速对比 (物理/预测/真值)", "步数", "m/s")
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Battery
    ax = axes[2, 2]
    battery = [f["battery_ratio"] * 100 for f in trace]
    remaining_j = [f.get("remaining_battery_j", 0) / 1000 for f in trace]
    ax.plot(steps, battery, color=C["teal"], linewidth=2.5, label="电量 %")
    ax2 = ax.twinx()
    ax2.plot(steps, remaining_j, color=C["orange"], linewidth=1.8, alpha=0.7, label="剩余 kJ")
    ax2.set_ylabel("剩余能量 (kJ)", fontsize=18)
    set_axis_style(ax, "电池消耗曲线", "步数", "电量 (%)")
    ax.legend(fontsize=18, loc="lower left")
    ax2.legend(fontsize=18, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save("17_sensor_timeseries")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 18: Vertical Wind Profile at Drone Position
# ══════════════════════════════════════════════════════════════════════
def plot_vertical_profiles():
    trace = report["trace"]
    # Collect profiles from keyframes
    kf_with_profiles = []
    for f in trace:
        if f.get("physics_profile_at_drone") and f.get("truth_profile_at_drone"):
            kf_with_profiles.append(f)

    if len(kf_with_profiles) < 3:
        print("  skipping vertical profiles — insufficient data")
        return

    indices = [0, len(kf_with_profiles) // 2, len(kf_with_profiles) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, idx in zip(axes, indices):
        frame = kf_with_profiles[idx]
        phys_raw = frame.get("physics_profile_at_drone", {})
        pred_raw = frame.get("prediction_profile_at_drone", {})
        truth_raw = frame.get("truth_profile_at_drone", {})

        # Handle both list and dict formats
        if isinstance(phys_raw, dict):
            phys_speed = phys_raw.get("speed", [])
            phys_w = phys_raw.get("w", [])
        else:
            phys_speed = phys_raw if isinstance(phys_raw, list) else []
            phys_w = frame.get("physics_vertical_w_profile", [])
        if isinstance(pred_raw, dict):
            pred_speed = pred_raw.get("speed", [])
            pred_w = pred_raw.get("w", [])
        else:
            pred_speed = pred_raw if isinstance(pred_raw, list) else []
            pred_w = frame.get("prediction_vertical_w_profile", [])
        if isinstance(truth_raw, dict):
            truth_speed = truth_raw.get("speed", [])
            truth_w = truth_raw.get("w", [])
        else:
            truth_speed = truth_raw if isinstance(truth_raw, list) else []
            truth_w = []

        levels = np.arange(len(phys_speed))

        # Left y-axis: speed
        ax.plot(levels, phys_speed, color=C["teal"], linewidth=2.0, marker="o", markersize=8, label="物理风速")
        ax.plot(levels, pred_speed, color=C["orange"], linewidth=2.0, marker="s", markersize=8, label="预测风速")
        if truth_speed:
            ax.plot(levels[:len(truth_speed)], truth_speed, color=C["red"], linewidth=2.0, marker="^", markersize=8, label="真值风速")
        ax.set_xlabel("高度层", fontsize=18)
        ax.set_ylabel("水平风速 (m/s)", fontsize=18, color=C["ink"])
        ax.tick_params(axis="y", labelcolor=C["ink"])

        # Right y-axis: vertical wind
        if pred_w:
            ax2 = ax.twinx()
            ax2.plot(levels[:len(pred_w)], pred_w, color=C["green"], linewidth=1.8, marker="D", markersize=8, alpha=0.7, label="预测 W")
            if truth_w:
                ax2.plot(levels[:len(truth_w)], truth_w, color="green", linewidth=1.5, linestyle="--", marker="v", markersize=8, alpha=0.5, label="真值 W")
            ax2.set_ylabel("垂直风速 W (m/s)", fontsize=18, color=C["green"])
            ax2.tick_params(axis="y", labelcolor=C["green"])
            ax2.axhline(y=0, color=C["green"], linewidth=0.5, alpha=0.3)

        pct = int(idx / max(len(kf_with_profiles) - 1, 1) * 100)
        set_axis_style(ax, f"垂直风剖面 (任务 {pct}%)")
        ax.legend(fontsize=18, loc="upper left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xticks(levels)

    plt.tight_layout()
    save("18_vertical_profiles")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 19: Return Home Feasibility & Safety
# ══════════════════════════════════════════════════════════════════════
def plot_return_home():
    trace = report["trace"]
    # Find keyframes with return data
    kf_return = [f for f in trace if f.get("maps") and f["maps"].get("return_feasible_mask")]
    if not kf_return:
        print("  skipping return home — no return data")
        return

    # Take first, middle, last
    n = len(kf_return)
    indices = [0, n // 2, n - 1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    elev = np.array(terrain_data["elevation"])

    for col, idx in enumerate(indices):
        frame = kf_return[idx]
        maps = frame["maps"]

        # Return cost margin
        ax = axes[0, col]
        margin = np.array(maps.get("return_cost_margin", [[0]]))
        im = ax.imshow(margin, origin="upper", cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.82, label="J")
        # Overlay feasible boundary
        mask = np.array(maps.get("return_feasible_mask", [[0]]))
        if mask.any():
            ax.contour(np.arange(mask.shape[1]), np.arange(mask.shape[0]), mask.astype(float),
                       levels=[0.5], colors="white", linewidths=2.0)
        pos = frame.get("position", [0, 0])
        ax.scatter(pos[0], pos[1], c="white", s=80, marker="o", zorder=5, edgecolors=C["ink"], linewidth=1.5)
        pct = int(idx / max(n - 1, 1) * 100)
        set_axis_style(ax, f"返航裕度 (任务 {pct}%, 白线=可行边界)")
        ax.set_xticks([])
        ax.set_yticks([])

        # Safety risk heatmap
        ax = axes[1, col]
        safety = np.array(maps.get("belief_safety", [[0]]))
        im2 = ax.imshow(safety, origin="upper", cmap="Reds", aspect="auto")
        plt.colorbar(im2, ax=ax, shrink=0.82)
        ax.scatter(pos[0], pos[1], c="white", s=80, marker="o", zorder=5, edgecolors=C["ink"], linewidth=1.5)
        set_axis_style(ax, f"安全风险 (任务 {pct}%)")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save("19_return_home_safety")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 20: Energy Cost Decomposition
# ══════════════════════════════════════════════════════════════════════
def plot_energy_decomposition():
    trace = report["trace"]
    breakdowns = []
    for f in trace:
        cb = f.get("planned_path_cost_breakdown", {})
        if cb:
            breakdowns.append(cb)

    if not breakdowns:
        print("  skipping energy decomposition — no cost breakdown data")
        return
    # Check if all costs are identical (MPC replanning disabled)
    costs_sample = [b.get("total_cost_j", 0) for b in breakdowns[:5]]
    if len(set([round(c, 1) for c in costs_sample])) <= 1:
        print("  skipping energy decomposition — cost breakdown is static (MPC disabled)")
        plt.close("all")
        return

    steps = np.arange(len(breakdowns))
    keys = ["energy_j", "progress_reward_j", "uncertainty_cost_j",
            "safety_cost_j", "altitude_bias_j", "vertical_maneuver_cost_j"]
    labels = ["能量消耗", "进度奖励", "不确定性代价", "安全代价", "高度偏差", "垂直机动"]
    colors_list = [C["teal"], C["green"], C["orange"], C["red"], C["purple"], C["blue"]]

    data = {}
    for key in keys:
        data[key] = [b.get(key, 0) for b in breakdowns]

    # Convert to numpy for stacked area
    y_data = np.column_stack([data[k] for k in keys])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked area chart
    ax = axes[0]
    positive_mask = y_data > 0
    negative_mask = y_data < 0
    y_pos = np.where(positive_mask, y_data, 0)
    y_neg = np.where(negative_mask, y_data, 0)

    ax.stackplot(steps, y_pos.T, labels=labels, colors=colors_list, alpha=0.8)
    ax.stackplot(steps, y_neg.T, colors=colors_list, alpha=0.8)
    set_axis_style(ax, "路径代价分解 (堆叠面积图)", "步数", "代价 (J)")
    ax.legend(fontsize=18, ncol=3, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    ax.axhline(y=0, color=C["ink"], linewidth=0.8)

    # Total cost
    ax = axes[1]
    total_costs = [f.get("planned_path_cost", 0) for f in trace]
    ax.plot(steps[:len(total_costs)], total_costs, color=C["teal"], linewidth=2.5)
    ax.fill_between(steps[:len(total_costs)], 0, total_costs, color=C["teal"], alpha=0.12)
    set_axis_style(ax, "路径总代价", "步数", "总代价 (J)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save("20_energy_decomposition")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Chart 21: Uplift/Sink Probability + Thermal Modes
# ══════════════════════════════════════════════════════════════════════
def plot_thermal_modes():
    trace = report["trace"]
    frame = None
    for f in trace:
        if f.get("maps") and f["maps"].get("belief_uplift_prob"):
            frame = f
            break
    if not frame:
        print("  skipping thermal modes — no data")
        return

    maps = frame["maps"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im1 = axes[0].imshow(np.array(maps.get("belief_uplift_prob", [[0]])), origin="upper",
                          cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[0], shrink=0.82, label="概率")
    set_axis_style(axes[0], "上升气流概率 P(uplift)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im2 = axes[1].imshow(np.array(maps.get("belief_sink_prob", [[0]])), origin="upper",
                          cmap="Blues", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], shrink=0.82, label="概率")
    set_axis_style(axes[1], "下沉气流概率 P(sink)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Dominant mode: red=uplift, blue=sink, white=neutral
    uplift = np.array(maps.get("belief_uplift_prob", [[0]]))
    sink = np.array(maps.get("belief_sink_prob", [[0]]))
    dominant = np.zeros_like(uplift)
    dominant[uplift > np.maximum(sink, 0.35)] = 1
    dominant[sink > np.maximum(uplift, 0.35)] = -1
    im3 = axes[2].imshow(dominant, origin="upper", cmap="RdYlBu", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im3, ax=axes[2], shrink=0.82, label="模式 (红=上升, 蓝=下沉, 白=中性)",
                 ticks=[-1, 0, 1])
    set_axis_style(axes[2], "主导气流模式")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    save("21_thermal_modes")
    plt.close()


if __name__ == "__main__":
    run_dir = load_chart_data()
    print(f"Generating charts from {run_dir} → {OUT_DIR}/\n")
    plot_training_curves()
    plot_feature_importance()
    plot_model_performance()
    plot_coarse_wind()
    plot_wind_field_maps()
    plot_terrain_features()
    plot_mission_profile()
    plot_trajectory()
    plot_error_analysis()
    plot_summary()
    plot_wind_rose()
    plot_coarse_wind_grids()
    plot_coarse_wind_stats()
    plot_multi_altitude_wind()
    plot_belief_timeline()
    plot_path_detail()
    plot_sensor_timeseries()
    plot_vertical_profiles()
    plot_return_home()
    plot_energy_decomposition()
    plot_thermal_modes()
    plot_belief_energy_path()
    plot_belief_vs_reality()
    plot_belief_path_planning()
    plot_eight_scenarios_terrain()
    plot_eight_scenarios_energy()
    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} charts saved to {OUT_DIR}/")
