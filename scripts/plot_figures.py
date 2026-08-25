"""
Generate publication-quality figures for the WindFarm project competition document.

Figure 1: ML wind correction model training effect (learning curves + accuracy comparison)
Figure 2: UAV route map — default (straight-line) vs optimized (executed) path with wind overlay
Figure 3: Altitude profile comparison + height level distribution
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Chinese font setup ─────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "natural"
OUT_DIR = PROJECT_ROOT / "scripts" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = json.loads((RUNS_DIR / "metrics.json").read_text(encoding="utf-8"))
REPORT = json.loads((RUNS_DIR / "mission_report.json").read_text(encoding="utf-8"))
TERRAIN = json.loads((RUNS_DIR / "terrain.json").read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════
# Figure 1: ML Learning Effect
# ═════════════════════════════════════════════════════════════════════

def plot_ml_learning_curves():
    fig = plt.figure(figsize=(16, 10))
    # Use GridSpec for better layout control
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.32,
                          left=0.06, right=0.97, top=0.92, bottom=0.10)

    components = [
        ("u", "U 分量 (东西方向)", "East-West"),
        ("v", "V 分量 (南北方向)", "North-South"),
        ("w", "W 分量 (垂直方向)", "Vertical"),
    ]

    for idx, (comp, title_cn, title_en) in enumerate(components):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        curve = METRICS[f"validation_curve_{comp}"]
        train_rmse = curve["train"]
        valid_rmse = curve["valid"]
        best_iter = METRICS[f"best_iteration_{comp}"]
        best_score = METRICS[f"best_score_{comp}"]

        rounds = np.arange(1, len(train_rmse) + 1)

        # Fill between to show generalization gap
        ax.fill_between(rounds, train_rmse, valid_rmse, alpha=0.12, color="#fdae61")

        ax.plot(rounds, train_rmse, color="#2c7bb6", linewidth=1.0, alpha=0.85, label="训练集 RMSE")
        ax.plot(rounds, valid_rmse, color="#d7191c", linewidth=1.6, alpha=0.92, label="验证集 RMSE")

        # Mark best iteration
        if best_iter and best_iter <= len(valid_rmse):
            ax.plot(best_iter, best_score, marker="o", markersize=8, color="#d7191c",
                    markeredgecolor="white", markeredgewidth=1.5, zorder=10)

            x_offset = -15 if best_iter > len(train_rmse) * 0.7 else 6
            y_offset = 0.018
            ax.annotate(
                f"最佳轮次 = {best_iter}\nRMSE = {best_score:.4f} m/s",
                xy=(best_iter, best_score),
                xytext=(best_iter + x_offset, best_score + y_offset),
                fontsize=7.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffcc",
                          edgecolor="#aaaaaa", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7, connectionstyle="arc3,rad=0.2"),
            )

        ax.set_xlabel("提升轮次 (Boosting Rounds)", fontsize=9)
        ax.set_ylabel("RMSE (m/s)", fontsize=9)
        ax.set_title(f"{title_cn}\n({title_en})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.set_xlim(1, len(train_rmse))
        ax.tick_params(labelsize=8)

    # ── Summary bar chart (spanning row 1, cols 0-2) ──
    ax_bar = fig.add_subplot(gs[1, :])

    categories = ["水平风速 RMSE\n(Wind Speed)", "垂直风速 MAE\n(Vertical W)", "风向 MAE\n(Wind Direction)"]
    physics_vals = [
        METRICS["rmse_physics"],
        METRICS["vertical_mae_physics"],
        METRICS["direction_mae_rad"],
    ]
    final_vals = [
        METRICS["rmse_final"],
        METRICS["vertical_mae_final"],
        METRICS["direction_mae_rad"],
    ]

    x = np.arange(len(categories))
    width = 0.30
    bars1 = ax_bar.bar(x - width / 2, physics_vals, width, color="#fc8d59", edgecolor="white", linewidth=0.8, label="CFD 物理降尺度 (仅物理)")
    bars2 = ax_bar.bar(x + width / 2, final_vals, width, color="#2c7bb6", edgecolor="white", linewidth=0.8, label="物理 + XGBoost ML 修正 (完整模型)")

    for bar, val in zip(bars1, physics_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.4f}", ha="center", fontsize=8.5, fontweight="bold", color="#333333")
    for bar, val in zip(bars2, final_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.4f}", ha="center", fontsize=8.5, fontweight="bold", color="#333333")

    # Improvement annotations
    improvements = [
        (1 - final_vals[0] / physics_vals[0]) * 100,
        (1 - final_vals[1] / physics_vals[1]) * 100,
        float("nan"),
    ]
    for i, imp in enumerate(improvements):
        if not math.isnan(imp) and imp > 0:
            mid_y = (physics_vals[i] + final_vals[i]) / 2
            ax_bar.annotate(
                f"↓ {imp:.1f}%",
                xy=(i, mid_y),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a9641",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1a9641", alpha=0.85),
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=9)
    ax_bar.set_ylabel("误差值", fontsize=10)
    ax_bar.set_title("预测精度对比: 纯物理模型 vs 物理+ML混合模型", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_bar.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Top-level title
    fig.suptitle(
        "XGBoost 风场残差修正模型 — 训练评估",
        fontsize=15, fontweight="bold", y=0.97,
    )

    # Footer info
    fig.text(
        0.5, 0.015,
        f"训练样本: {METRICS['train_size']:,} | 训练集: {METRICS['train_size_fit']} | "
        f"验证集: {METRICS['validation_size']} | 测试集: {METRICS['test_size']:,} | "
        f"模型: {METRICS['model_type']} | 水平风速 RMSE 改善: {improvements[0]:.1f}% | "
        f"垂直风速 MAE 改善: {improvements[1]:.1f}%",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )

    out_path = OUT_DIR / "figure1_ml_training_effect.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Figure 1 -> {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════
# Figure 2: Route Map
# ═════════════════════════════════════════════════════════════════════

def find_straightline_path(start, goal, terrain_width, terrain_height):
    x0, y0, z0 = start
    x1, y1, z1 = goal
    cells = []
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        z = int(round(z0 + (z1 - z0) * t))
        x = max(0, min(terrain_width - 1, x))
        y = max(0, min(terrain_height - 1, y))
        cells.append((x, y, z))
    return cells


def _get_wind_at_keyframe(kf, level=0):
    """Extract wind U, V grids from a keyframe's physics prediction."""
    maps = kf.get("maps", {})
    u_grid = maps.get("physics_wind_u", [])
    v_grid = maps.get("physics_wind_v", [])
    return u_grid, v_grid


def plot_route_map():
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.05, right=0.97, top=0.91, bottom=0.08)

    grid_w = REPORT["grid"]["width"]
    grid_h = REPORT["grid"]["height"]
    elevation = TERRAIN["terrain"]["elevation"]
    mission = REPORT["mission"]
    start = tuple(mission["start"])
    goal = tuple(mission["goal"])

    executed_path = REPORT["executed_path"]
    exec_x = [p[0] for p in executed_path]
    exec_y = [p[1] for p in executed_path]

    keyframes = [f for f in REPORT["trace"] if f.get("is_keyframe")]
    keyframe_positions = [f["position"] for f in keyframes]

    straight = find_straightline_path(start, goal, grid_w, grid_h)
    straight_x = [p[0] for p in straight]
    straight_y = [p[1] for p in straight]

    # Wind field from first keyframe for quiver plot
    first_kf = keyframes[0] if keyframes else None
    wind_u, wind_v = _get_wind_at_keyframe(first_kf) if first_kf else ([], [])

    # Downsample wind grid for quiver arrows
    def downsample_wind(u, v, step=2):
        h, w = len(u), len(u[0]) if u else 0
        xs, ys, us, vs = [], [], [], []
        for yi in range(0, h, step):
            for xi in range(0, w, step):
                xs.append(xi)
                ys.append(yi)
                us.append(u[yi][xi])
                vs.append(v[yi][xi])
        return np.array(xs), np.array(ys), np.array(us), np.array(vs)

    wx, wy, wu, wv = None, None, None, None
    if wind_u and wind_v:
        wx, wy, wu, wv = downsample_wind(wind_u, wind_v, step=2)

    for ax_idx, (ax_label, is_optimized) in enumerate(
        [("A. 默认最短路径 (无风场感知)", False), ("B. 风场感知优化路径 (物理+ML+信念规划)", True)]
    ):
        ax = fig.add_subplot(gs[0, ax_idx])

        # Elevation background
        extent = [-0.5, grid_w - 0.5, grid_h - 0.5, -0.5]
        im = ax.imshow(elevation, cmap="terrain", origin="upper", extent=extent,
                       alpha=0.70, aspect="equal", interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
        cbar.set_label("地形高程 (m)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7)

        # Grid
        ax.set_xticks(range(grid_w))
        ax.set_yticks(range(grid_h))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.12, color="#333333", linewidth=0.25)

        # Contour lines
        if elevation:
            levels = np.linspace(np.min(elevation), np.max(elevation), 10)
            ax.contour(range(grid_w), range(grid_h), elevation, levels=levels,
                       colors="#444444", linewidths=0.35, alpha=0.30)

        # Wind quiver arrows on both maps
        if wx is not None and wu is not None:
            speeds = np.sqrt(wu**2 + wv**2)
            q = ax.quiver(wx, wy, wu, wv, speeds,
                          cmap="coolwarm", alpha=0.55, scale=80, width=0.003,
                          pivot="mid", zorder=8)
            # reference arrow
            ax.quiverkey(q, 0.92, 1.03, 5.0, "5 m/s", labelpos="E",
                         fontproperties={"size": 7}, color="#333333")

        # Start & Goal markers
        ax.scatter(*start[:2], marker="*", s=420, color="#008000", edgecolors="white",
                   linewidths=1.3, zorder=12, label=f"起点 ({start[0]}, {start[1]})")
        ax.scatter(*goal[:2], marker="D", s=170, color="#d7191c", edgecolors="white",
                   linewidths=1.3, zorder=12, label=f"终点 ({goal[0]}, {goal[1]})")

        if not is_optimized:
            # Left panel: Straight-line route only
            ax.plot(straight_x, straight_y, color="#d7191c", linewidth=2.8, linestyle="--",
                    marker="o", markersize=4.5, markerfacecolor="#fc8d59",
                    markeredgewidth=0.5, markeredgecolor="white",
                    zorder=9, label="直线最短路径")
            ax.annotate(
                f"直线距离 ≈ {len(straight)*100:.0f} m\n(不考虑风场影响)",
                xy=(straight_x[len(straight)//2], straight_y[len(straight)//2]),
                fontsize=8.5, color="#d7191c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#d7191c", alpha=0.88),
                ha="center",
            )
        else:
            # Right panel: Optimized route with planning paths
            # Selected planned paths (faded)
            n_kf = len(keyframes)
            plan_indices = [0, n_kf // 4, n_kf // 2, 3 * n_kf // 4, -1]
            plan_colors = ["#a6cee3", "#b2df8a", "#33a02c", "#1f78b4", "#6a3d9a"]
            for ki, (pf_idx, color) in enumerate(zip(plan_indices, plan_colors)):
                if pf_idx >= n_kf:
                    continue
                kf = keyframes[pf_idx]
                pp = kf.get("planned_path", [])
                if len(pp) > 1:
                    px = [float(p[0]) for p in pp]
                    py = [float(p[1]) for p in pp]
                    ax.plot(px, py, color=color, linewidth=0.9, linestyle=":",
                            alpha=0.45, zorder=3)

            # Straight-line reference (thin)
            ax.plot(straight_x, straight_y, color="#d7191c", linewidth=1.0,
                    linestyle="--", alpha=0.35, zorder=2)

            # Executed path
            ax.plot(exec_x, exec_y, color="#004c99", linewidth=2.3, linestyle="-",
                    marker=".", markersize=1.6, markerfacecolor="#2c7bb6",
                    markeredgewidth=0, zorder=7, label="实际飞行路径")

            # Keyframe markers
            kf_x = [p[0] for p in keyframe_positions]
            kf_y = [p[1] for p in keyframe_positions]
            ax.scatter(kf_x, kf_y, marker="s", s=15, color="#fdae61",
                       edgecolors="#333333", linewidths=0.3,
                       zorder=8, alpha=0.65, label="关键帧")

            # Mission summary annotation
            ax.annotate(
                f"执行步数: {REPORT['steps_executed']}\n"
                f"到达终点: {'是' if REPORT['goal_reached'] else '否'}\n"
                f"剩余电量: {REPORT['battery_ratio']*100:.1f}%\n"
                f"飞行距离: {len(executed_path)*100:.0f} m",
                xy=(goal[0], goal[1] - 2.8),
                fontsize=8, color="#004c99", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#004c99", alpha=0.90),
                ha="center",
            )

        ax.set_title(ax_label, fontsize=12.5, fontweight="bold",
                     color="#d7191c" if not is_optimized else "#004c99",
                     pad=8)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
                  ncol=2 if is_optimized else 1, markerscale=0.8)

    fig.suptitle("无人机路径规划: 默认最短路径 vs 风场优化路径", fontsize=15, fontweight="bold", y=0.96)

    # Footer
    fig.text(0.5, 0.015,
             f"网格: {grid_w}×{grid_h} | 分辨率: {REPORT['grid']['resolution_m']} m/格 | "
             f"起点({start[0]},{start[1]},{start[2]}) → 终点({goal[0]},{goal[1]},{goal[2]}) | "
             f"箭头: CFD降尺度风场",
             ha="center", fontsize=7.5, color="#888888", style="italic")

    out_path = OUT_DIR / "figure2_route_comparison.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Figure 2 -> {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════
# Figure 3: Altitude profile
# ═════════════════════════════════════════════════════════════════════

def plot_altitude_profile():
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.97, top=0.90, bottom=0.10)

    grid_w = REPORT["grid"]["width"]
    grid_h = REPORT["grid"]["height"]
    elevation = TERRAIN["terrain"]["elevation"]
    start = tuple(REPORT["mission"]["start"])
    goal = tuple(REPORT["mission"]["goal"])

    executed_path = REPORT["executed_path"]
    exec_x = [p[0] for p in executed_path]
    exec_y = [p[1] for p in executed_path]
    exec_z = [p[2] for p in executed_path]

    straight = find_straightline_path(start, goal, grid_w, grid_h)
    straight_x = [p[0] for p in straight]
    straight_y = [p[1] for p in straight]
    straight_z = [p[2] for p in straight]

    def cum_dist(xs, ys, res_m=100.0):
        dists = [0.0]
        for i in range(1, len(xs)):
            d = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) * res_m
            dists.append(dists[-1] + d)
        return dists

    exec_dist = cum_dist(exec_x, exec_y)
    straight_dist = cum_dist(straight_x, straight_y)

    def ground_elev(path_x, path_y):
        elevs = []
        for x, y in zip(path_x, path_y):
            xi = max(0, min(grid_w - 1, int(round(x))))
            yi = max(0, min(grid_h - 1, int(round(y))))
            elevs.append(elevation[yi][xi])
        return elevs

    exec_elev = ground_elev(exec_x, exec_y)
    straight_elev = ground_elev(straight_x, straight_y)

    alt_step = REPORT["mission"]["altitude_step_m"]

    # ── Top-left: Altitude profile comparison ──
    ax0 = fig.add_subplot(gs[0, :])

    ax0.fill_between(exec_dist, exec_elev, alpha=0.15, color="#2c7bb6")
    ax0.plot(exec_dist, exec_elev, color="#2c7bb6", linewidth=1.0, alpha=0.5)
    ax0.fill_between(straight_dist, straight_elev, alpha=0.15, color="#d7191c")
    ax0.plot(straight_dist, straight_elev, color="#d7191c", linewidth=1.0, alpha=0.5)

    # Altitude paths
    ax0.plot(exec_dist, [z * alt_step for z in exec_z],
             color="#004c99", linewidth=2.0, linestyle="-", marker=".", markersize=1.5,
             label=f"实际飞行高度 (ML优化, {len(executed_path)}步, {exec_dist[-1]:.0f}m)")
    ax0.plot(straight_dist, [z * alt_step for z in straight_z],
             color="#d7191c", linewidth=1.6, linestyle="--", marker=".", markersize=2.5,
             label=f"直线路径高度 (无优化, {len(straight)}步, {straight_dist[-1]:.0f}m)")

    # Ground elevation
    ax0.plot(exec_dist, exec_elev, color="#555555", linewidth=0.8, alpha=0.9, label="地面高程 (实际路径)")
    ax0.plot(straight_dist, straight_elev, color="#666666", linewidth=0.7, alpha=0.7, linestyle=":")

    ax0.set_xlabel("累计水平距离 (m)", fontsize=10)
    ax0.set_ylabel("高度 (m)", fontsize=10)
    ax0.set_title("飞行高度剖面对比", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.85)
    ax0.grid(True, alpha=0.25, linewidth=0.4)
    ax0.tick_params(labelsize=8)

    # ── Bottom-left: Altitude level distribution ──
    ax1 = fig.add_subplot(gs[1, 0])
    levels = list(range(5))
    level_names = [f"L{lv}\n({lv * int(alt_step)}m)" for lv in levels]
    exec_counts = [sum(1 for z in exec_z if int(round(z)) == lv) for lv in levels]
    straight_counts = [sum(1 for z in straight_z if int(round(z)) == lv) for lv in levels]

    x = np.arange(len(levels))
    width = 0.30
    b1 = ax1.bar(x - width / 2, straight_counts, width, color="#fc8d59", edgecolor="white", linewidth=0.5, label="直线路径")
    b2 = ax1.bar(x + width / 2, exec_counts, width, color="#2c7bb6", edgecolor="white", linewidth=0.5, label="优化路径")
    for bars in (b1, b2):
        for bar in bars:
            if bar.get_height() > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         str(int(bar.get_height())), ha="center", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(level_names, fontsize=9)
    ax1.set_xlabel("高度层级", fontsize=10)
    ax1.set_ylabel("步数", fontsize=10)
    ax1.set_title("高度层级分布对比", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.85)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.4)

    # ── Bottom-right: Battery consumption over distance ──
    ax2 = fig.add_subplot(gs[1, 1])

    keyframes = [f for f in REPORT["trace"] if f.get("is_keyframe")]
    kf_positions_raw = [f["position"] for f in keyframes]
    kf_battery = [f["battery_ratio"] for f in keyframes]
    kf_dist = cum_dist([p[0] for p in kf_positions_raw], [p[1] for p in kf_positions_raw])

    # All frames battery
    all_battery = [f["battery_ratio"] for f in REPORT["trace"]]
    # Some reports can have an off-by-one mismatch between path distance and trace length.
    n_batt = min(len(exec_dist), len(all_battery))
    exec_dist_batt = exec_dist[:n_batt]
    all_battery = all_battery[:n_batt]
    ax2.fill_between(exec_dist_batt, all_battery, alpha=0.25, color="#2c7bb6")
    ax2.plot(exec_dist_batt, all_battery, color="#2c7bb6", linewidth=1.0, alpha=0.7, label="实时电量")

    # Keyframe points
    n_kf = min(len(kf_dist), len(kf_battery))
    kf_dist = kf_dist[:n_kf]
    kf_battery = kf_battery[:n_kf]
    ax2.scatter(kf_dist, kf_battery, marker="s", s=25, color="#fdae61",
                edgecolors="#333333", linewidths=0.4, zorder=5, label="关键帧")
    ax2.plot(kf_dist, kf_battery, color="#fdae61", linewidth=1.8, alpha=0.85, marker=".", markersize=2)

    # Reserve line
    reserve = REPORT["mission"]["power_model"]["reserve_energy_ratio"]
    ax2.axhline(y=reserve, color="#d7191c", linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"预留电量线 ({reserve*100:.0f}%)")

    ax2.set_xlabel("累计距离 (m)", fontsize=10)
    ax2.set_ylabel("电量比例", fontsize=10)
    ax2.set_title("电量消耗曲线", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.85, loc="lower left")
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("飞行高度剖面、高度分布与电量消耗分析", fontsize=14, fontweight="bold", y=0.95)

    out_path = OUT_DIR / "figure3_altitude_profile.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Figure 3 -> {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("Generating publication-quality figures...\n")
    plot_ml_learning_curves()
    plot_route_map()
    plot_altitude_profile()
    print(f"\nDone. All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
