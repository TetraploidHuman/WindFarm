from __future__ import annotations

import json
from pathlib import Path

from .dashboard_parts import (
    dashboard_js_data_adapter,
    dashboard_js_map_renderer,
    dashboard_js_path_panel,
    dashboard_js_prelude,
    dashboard_js_training_panel,
    dashboard_styles,
)
from .io import read_json


def _dashboard_template(payload_json: str, live_mode: bool) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WindFarm 实时信念仪表盘</title>
  <style>{dashboard_styles()}</style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel">
        <div id="modeBadge" class="status"></div>
        <h1>风场信念控制台</h1>
        <p class="sub">该仪表盘用于回放或实时展示无人机遥测数据，展示分层风场预测、概率信念更新、安全风险演化、MPC 重规划、残差热力图、电量与能量趋势，以及基于地形的三维概览。</p>
        <div class="stats">
          <div class="stat"><div class="label">任务状态</div><div class="value" id="goalStatus"></div></div>
          <div class="stat"><div class="label">步骤</div><div class="value" id="stepValue"></div></div>
          <div class="stat"><div class="label">电量</div><div class="value" id="batteryValue"></div></div>
          <div class="stat"><div class="label">残差</div><div class="value" id="residualValue"></div></div>
          <div class="stat"><div class="label">网格</div><div class="value" id="gridValue"></div></div>
        </div>
      </div>
      <div class="panel">
        <div class="kv"><div class="label">起点</div><div id="startValue"></div></div>
        <div class="kv"><div class="label">目标</div><div id="goalCellValue"></div></div>
        <div class="kv"><div class="label">当前时间</div><div id="timeValue"></div></div>
        <div class="kv"><div class="label">位置</div><div id="posValue"></div></div>
        <div class="kv"><div class="label">图层</div><div id="layerValue"></div></div>
        <div class="kv"><div class="label">目标距离</div><div id="distValue"></div></div>
        <div class="kv"><div class="label">真值数据</div><div id="truthValue"></div></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <div class="controls">
          <button id="playBtn" type="button">播放</button>
          <input id="stepRange" type="range" min="0" value="0">
          <select id="altitudeSelect"></select>
          <select id="layerSelect">
            <option value="physics_wind_speed">物理风速</option>
            <option value="physics_residual_speed">物理残差</option>
            <option value="belief_energy">信念能量</option>
            <option value="belief_uncertainty">信念不确定性</option>
            <option value="belief_confidence">信念置信度</option>
            <option value="belief_entropy">信念熵</option>
            <option value="belief_safety">安全风险</option>
            <option value="belief_uplift_prob">上升气流概率</option>
            <option value="belief_sink_prob">下沉气流概率</option>
            <option value="belief_wind_speed">信念风速</option>
            <option value="prediction_wind_speed">预测风速</option>
            <option value="truth_wind_speed">真值风速</option>
            <option value="prediction_residual_speed">预测残差</option>
            <option value="return_cost_margin">返航裕度</option>
            <option value="residual_speed">残差热力图</option>
            <option value="terrain_elevation">地形高程</option>
            <option value="terrain_roughness">地表粗糙度</option>
          </select>
        </div>
        <canvas id="mainCanvas" class="main-canvas" width="1440" height="930"></canvas>
        <div class="legend"><span id="legendMin"></span><span id="legendMax"></span></div>
        <div class="mini-grid">
          <div>
            <h3 style="margin-bottom:8px;">电量与能量</h3>
            <canvas id="batteryCanvas" class="chart-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">残差热力图</h3>
            <canvas id="residualCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">风矢量场</h3>
            <canvas id="vectorCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">三维地形</h3>
            <canvas id="terrainCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">物理场 vs 学习场 vs 真值</h3>
            <canvas id="compareCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">训练曲线</h3>
            <canvas id="trainingCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">垂直风剖面</h3>
            <canvas id="profileCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
          <div>
            <h3 style="margin-bottom:8px;">高度轨迹</h3>
            <canvas id="altitudeCanvas" class="mini-canvas" width="640" height="360"></canvas>
          </div>
        </div>
      </div>
      <div class="side-grid">
        <div class="panel">
          <h3 style="margin-bottom:12px;">传感器数据流</h3>
          <div id="sensorGrid" class="sensor-grid"></div>
        </div>
        <div class="panel">
          <h3 style="margin-bottom:12px;">信念与风场</h3>
          <div class="kv"><div class="label">信念能量</div><div id="beliefEnergy"></div></div>
          <div class="kv"><div class="label">信念不确定性</div><div id="beliefUncertainty"></div></div>
          <div class="kv"><div class="label">信念置信度</div><div id="beliefConfidence"></div></div>
          <div class="kv"><div class="label">信念熵</div><div id="beliefEntropy"></div></div>
          <div class="kv"><div class="label">安全风险</div><div id="beliefSafety"></div></div>
          <div class="kv"><div class="label">上升/下沉概率</div><div id="beliefModes"></div></div>
          <div class="kv"><div class="label">预测风</div><div id="predWind"></div></div>
          <div class="kv"><div class="label">观测风</div><div id="obsWind"></div></div>
          <div class="kv"><div class="label">真值风</div><div id="truthWind"></div></div>
        </div>
        <div class="panel">
          <h3 style="margin-bottom:12px;">规划路径</h3>
          <div id="pathBox" class="path-box"></div>
        </div>
        <div class="panel">
          <h3 style="margin-bottom:12px;">MPC 路径代价与返航</h3>
          <div class="kv"><div class="label">预测模式</div><div id="forecastMode"></div></div>
          <div class="kv"><div class="label">当前目标</div><div id="currentGoal"></div></div>
          <div class="kv"><div class="label">路径代价</div><div id="pathCost"></div></div>
          <div class="kv"><div class="label">返航预算</div><div id="returnRadius"></div></div>
          <div class="kv"><div class="label">候选轨迹</div><div id="candidateCount"></div></div>
          <div id="costBreakdownBox" class="path-box"></div>
          <div id="candidateBox" class="path-box"></div>
        </div>
        <div class="panel">
          <h3 style="margin-bottom:12px;">训练摘要</h3>
          <div class="kv"><div class="label">模型</div><div id="modelType"></div></div>
          <div class="kv"><div class="label">最佳迭代 U</div><div id="bestIterU"></div></div>
          <div class="kv"><div class="label">最佳迭代 V</div><div id="bestIterV"></div></div>
          <div class="kv"><div class="label">最佳迭代 W</div><div id="bestIterW"></div></div>
          <div class="kv"><div class="label">最佳分数 U</div><div id="bestScoreU"></div></div>
          <div class="kv"><div class="label">最佳分数 V</div><div id="bestScoreV"></div></div>
          <div class="kv"><div class="label">最佳分数 W</div><div id="bestScoreW"></div></div>
          <div id="importanceBox" class="path-box"></div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const LIVE_MODE = {str(live_mode).lower()};
    const EMBEDDED = {payload_json};
    const state = {{
      report: EMBEDDED,
      summary: EMBEDDED ? {{
        goal_reached: EMBEDDED.goal_reached,
        final_position: EMBEDDED.final_position,
        battery_ratio: EMBEDDED.battery_ratio,
        steps_executed: EMBEDDED.steps_executed
      }} : null,
      trace: (EMBEDDED && EMBEDDED.trace) ? EMBEDDED.trace.slice() : [],
      idx: 0,
      playing: false,
      followLive: LIVE_MODE,
      layer: "belief_energy",
      altitudeLayer: 0,
      interactiveGoal: false,
      streamDone: false,
      latestSeq: -1,
      refreshInFlight: false
    }};
    const el = {{
      modeBadge: document.getElementById("modeBadge"),
      goalStatus: document.getElementById("goalStatus"),
      stepValue: document.getElementById("stepValue"),
      batteryValue: document.getElementById("batteryValue"),
      residualValue: document.getElementById("residualValue"),
      gridValue: document.getElementById("gridValue"),
      startValue: document.getElementById("startValue"),
      goalCellValue: document.getElementById("goalCellValue"),
      timeValue: document.getElementById("timeValue"),
      posValue: document.getElementById("posValue"),
      layerValue: document.getElementById("layerValue"),
      distValue: document.getElementById("distValue"),
      truthValue: document.getElementById("truthValue"),
      beliefEnergy: document.getElementById("beliefEnergy"),
      beliefUncertainty: document.getElementById("beliefUncertainty"),
      beliefConfidence: document.getElementById("beliefConfidence"),
      beliefEntropy: document.getElementById("beliefEntropy"),
      beliefSafety: document.getElementById("beliefSafety"),
      beliefModes: document.getElementById("beliefModes"),
      predWind: document.getElementById("predWind"),
      obsWind: document.getElementById("obsWind"),
      truthWind: document.getElementById("truthWind"),
      pathBox: document.getElementById("pathBox"),
      sensorGrid: document.getElementById("sensorGrid"),
      legendMin: document.getElementById("legendMin"),
      legendMax: document.getElementById("legendMax"),
      stepRange: document.getElementById("stepRange"),
      playBtn: document.getElementById("playBtn"),
      altitudeSelect: document.getElementById("altitudeSelect"),
      layerSelect: document.getElementById("layerSelect"),
      mainCanvas: document.getElementById("mainCanvas"),
      batteryCanvas: document.getElementById("batteryCanvas"),
      residualCanvas: document.getElementById("residualCanvas"),
      vectorCanvas: document.getElementById("vectorCanvas"),
      terrainCanvas: document.getElementById("terrainCanvas"),
      compareCanvas: document.getElementById("compareCanvas"),
      trainingCanvas: document.getElementById("trainingCanvas"),
      profileCanvas: document.getElementById("profileCanvas"),
      altitudeCanvas: document.getElementById("altitudeCanvas"),
      forecastMode: document.getElementById("forecastMode"),
      currentGoal: document.getElementById("currentGoal"),
      pathCost: document.getElementById("pathCost"),
      returnRadius: document.getElementById("returnRadius"),
      candidateCount: document.getElementById("candidateCount"),
      costBreakdownBox: document.getElementById("costBreakdownBox"),
      candidateBox: document.getElementById("candidateBox"),
      modelType: document.getElementById("modelType"),
      bestIterU: document.getElementById("bestIterU"),
      bestIterV: document.getElementById("bestIterV"),
      bestIterW: document.getElementById("bestIterW"),
      bestScoreU: document.getElementById("bestScoreU"),
      bestScoreV: document.getElementById("bestScoreV"),
      bestScoreW: document.getElementById("bestScoreW"),
      importanceBox: document.getElementById("importanceBox")
    }};

    function apiUrl(path) {{
      const prefix = window.location.pathname.replace(/\/index\\.html$/, "").replace(/\/$/, "");
      return `${{prefix}}${{path}}`;
    }}

    {dashboard_js_prelude()}
    {dashboard_js_data_adapter()}
    {dashboard_js_map_renderer()}
    {dashboard_js_training_panel()}
    {dashboard_js_path_panel()}

    function renderStep(name, fn) {{
      try {{
        fn();
      }} catch (error) {{
        console.error(`dashboard render failed at ${{name}}`, error);
        throw new Error(`${{name}}: ${{error.message}}`);
      }}
    }}

    function render() {{
      const frame = currentFrame();
      if (!frame) return;
      try {{
        if (Array.isArray(frame.position) && frame.position.length > 2 && document.activeElement !== el.altitudeSelect) {{
          if (!Number.isFinite(Number(state.altitudeLayer))) state.altitudeLayer = frame.position[2];
          el.altitudeSelect.value = String(state.altitudeLayer);
        }}
        el.stepRange.max = String(Math.max(state.trace.length - 1, 0));
        el.stepRange.value = String(state.idx);
        renderStep("drawMainMap", () => drawMainMap(frame));
        renderStep("drawBatteryChart", () => drawBatteryChart());
        renderStep("drawResidualMap", () => drawResidualMap(frame));
        renderStep("drawVectorField", () => drawVectorField(frame));
        renderStep("drawTerrain3D", () => drawTerrain3D());
        renderStep("drawCompareChart", () => drawCompareChart(frame));
        renderStep("drawTrainingChart", () => drawTrainingChart());
        renderStep("drawVerticalProfile", () => drawVerticalProfile(frame));
        renderStep("drawAltitudeChart", () => drawAltitudeChart());
        renderStep("updateMeta", () => updateMeta(frame));
      }} catch (error) {{
        console.error("dashboard render failed", error);
        el.modeBadge.textContent = `渲染错误: ${{error.message}}`;
        el.modeBadge.classList.add("offline");
      }}
    }}

    function tick() {{
      if (!state.trace.length) return;
      if (LIVE_MODE) {{
        if (!state.followLive) return;
        const latest = Math.max(state.trace.length - 1, 0);
        if (state.idx !== latest) {{
          state.idx = latest;
          render();
        }}
        return;
      }}
      if (!state.playing) return;
      state.idx = (state.idx + 1) % state.trace.length;
      render();
    }}

    async function refreshLive() {{
      if (!LIVE_MODE) return;
      if (state.refreshInFlight) return;
      state.refreshInFlight = true;
      try {{
        const previousSeq = state.latestSeq;
        const response = await fetch(apiUrl(`/api/state?since=${{state.latestSeq}}`), {{ cache: "no-store" }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const data = await response.json();
        let changed = false;
        if (data.report) state.report = data.report;
        if (data.summary) state.summary = data.summary;
        if (data.capabilities) state.interactiveGoal = Boolean(data.capabilities.interactive_goal);
        const nextSeq = data.latest_seq ?? previousSeq;
        if (Array.isArray(data.trace_append) && data.trace_append.length && nextSeq > previousSeq) {{
          const expected = nextSeq - previousSeq;
          const append = expected >= data.trace_append.length ? data.trace_append : data.trace_append.slice(-expected);
          state.trace.push(...append);
          changed = append.length > 0;
          if (state.followLive) state.idx = state.trace.length - 1;
        }}
        state.latestSeq = Math.max(previousSeq, nextSeq);
        state.streamDone = Boolean(data.stream_done);
        el.modeBadge.textContent = data.stream_done ? "实时流已结束" : "实时流进行中";
        if (data.stream_done) el.modeBadge.classList.add("offline");
        if (changed || state.followLive) {{
          render();
        }}
      }} catch (error) {{
        console.error("refreshLive failed", error);
        el.modeBadge.textContent = `连接中断: ${{error.message}}`;
        el.modeBadge.classList.add("offline");
      }} finally {{
        state.refreshInFlight = false;
      }}
    }}

    function applyImmediateFrameUpdate(data) {{
      if (data.report) state.report = data.report;
      if (data.summary) state.summary = data.summary;
      if (data.capabilities) state.interactiveGoal = Boolean(data.capabilities.interactive_goal);
      if (state.report && state.report.mission && Array.isArray(data.goal)) {{
        state.report.mission.goal = data.goal.slice();
      }}
      const nextSeq = data.latest_seq;
      const frame = data.latest_frame;
      if (frame && Number.isFinite(nextSeq)) {{
        if (nextSeq < state.trace.length) {{
          state.trace[nextSeq] = frame;
        }} else if (nextSeq === state.trace.length) {{
          state.trace.push(frame);
        }} else {{
          state.trace.push(frame);
        }}
        state.latestSeq = Math.max(state.latestSeq, nextSeq);
        if (state.followLive) state.idx = state.trace.length - 1;
      }}
      render();
    }}

    async function sendGoal(point) {{
      if (!LIVE_MODE || !state.interactiveGoal) return;
      const response = await fetch(apiUrl("/api/command"), {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ goal: point }})
      }});
      if (!response.ok) return;
      const data = await response.json();
      state.followLive = true;
      state.playing = false;
      el.playBtn.textContent = LIVE_MODE ? "跟随实时" : "播放";
      applyImmediateFrameUpdate(data);
    }}

    async function init() {{
      if (LIVE_MODE) {{
        const response = await fetch(apiUrl("/api/bootstrap"), {{ cache: "no-store" }});
        const data = await response.json();
        state.report = data.report;
        state.summary = data.summary;
        state.trace = data.trace;
        state.interactiveGoal = Boolean(data.capabilities && data.capabilities.interactive_goal);
        state.latestSeq = data.latest_seq ?? (state.trace.length - 1);
        if (state.trace.length) state.idx = state.trace.length - 1;
        el.modeBadge.textContent = state.interactiveGoal ? "交互式导航" : "实时流进行中";
        el.playBtn.textContent = "跟随实时";
        window.setInterval(refreshLive, 100);
      }} else {{
        el.modeBadge.textContent = "回放模式";
        el.playBtn.textContent = "播放";
      }}
      el.playBtn.addEventListener("click", () => {{
        if (LIVE_MODE) {{
          state.followLive = !state.followLive;
          el.playBtn.textContent = state.followLive ? "暂停跟随" : "跟随实时";
          if (state.followLive && state.trace.length) {{
            state.idx = state.trace.length - 1;
            render();
          }}
          return;
        }}
        state.playing = !state.playing;
        el.playBtn.textContent = state.playing ? "暂停" : "播放";
      }});
      const levelMin = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[0] : 0;
      const levelMax = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[1] : 0;
      el.altitudeSelect.innerHTML = Array.from({{ length: levelMax - levelMin + 1 }}, (_, idx) => {{
        const level = levelMin + idx;
        return `<option value="${{level}}">高度层 z=${{level}}</option>`;
      }}).join("");
      el.altitudeSelect.value = String(state.altitudeLayer);
      el.altitudeSelect.addEventListener("change", (event) => {{
        state.altitudeLayer = Number(event.target.value);
        render();
      }});
      el.stepRange.addEventListener("input", (event) => {{
        state.idx = Number(event.target.value);
        if (LIVE_MODE) state.followLive = false;
        const frame = currentFrame();
        if (frame && Array.isArray(frame.position) && frame.position.length > 2) {{
          state.altitudeLayer = frame.position[2];
          el.altitudeSelect.value = String(state.altitudeLayer);
        }}
        render();
      }});
      el.layerSelect.addEventListener("change", (event) => {{
        state.layer = event.target.value;
        render();
      }});
      el.mainCanvas.addEventListener("click", (event) => {{
        if (!LIVE_MODE || !state.report || !state.interactiveGoal) return;
        const point = pointFromCanvas(event);
        sendGoal([point[0], point[1], state.altitudeLayer]);
      }});
      window.setInterval(tick, 700);
      render();
    }}
    init();
  </script>
</body>
</html>"""


def build_dashboard_html(report: dict) -> str:
    return _dashboard_template(json.dumps(report, ensure_ascii=False), live_mode=False)


def build_live_dashboard_html() -> str:
    return _dashboard_template("null", live_mode=True)


def build_dashboard_from_report(report_path: str | Path, output_path: str | Path) -> None:
    report = read_json(report_path)
    Path(output_path).write_text(build_dashboard_html(report), encoding="utf-8")
