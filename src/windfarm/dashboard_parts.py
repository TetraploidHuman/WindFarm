from __future__ import annotations


def dashboard_styles() -> str:
    return """
    :root {
      --bg: #f0e8da;
      --panel: rgba(255,255,255,0.82);
      --ink: #172026;
      --muted: #697680;
      --line: rgba(23,32,38,0.08);
      --accent: #115d6b;
      --accent2: #d87420;
      --danger: #b53a2d;
      --good: #2f7d55;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Microsoft YaHei", serif;
      background:
        radial-gradient(circle at top left, rgba(17,93,107,0.14), transparent 28%),
        radial-gradient(circle at bottom right, rgba(216,116,32,0.14), transparent 28%),
        linear-gradient(135deg, #ede4d6, var(--bg));
    }
    .shell { max-width: 1760px; margin: 0 auto; padding: 24px; }
    .hero, .grid { display: grid; gap: 16px; }
    .hero { grid-template-columns: 1.5fr 1fr; margin-bottom: 16px; }
    .grid { grid-template-columns: minmax(0, 1.65fr) minmax(360px, 0.95fr); }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.6);
      border-radius: 24px;
      box-shadow: 0 18px 50px rgba(26,41,50,0.12);
      padding: 16px;
      backdrop-filter: blur(12px);
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: clamp(34px, 4vw, 58px); line-height: 0.95; letter-spacing: -0.04em; }
    .sub { margin-top: 10px; line-height: 1.55; color: var(--muted); max-width: 60ch; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(17,93,107,0.12);
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 12px;
    }
    .status.offline { background: rgba(181,58,45,0.12); color: var(--danger); }
    .stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
    .stat { padding: 14px; border-radius: 18px; background: rgba(240,232,218,0.8); border: 1px solid rgba(23,32,38,0.06); }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }
    .value { margin-top: 6px; font-size: 24px; font-weight: 700; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
    button, select { border: none; border-radius: 999px; padding: 10px 16px; font: inherit; }
    button { background: #17323d; color: #fff; cursor: pointer; }
    select { background: #fff; color: var(--ink); border: 1px solid rgba(23,32,38,0.12); }
    input[type="range"] { width: min(420px, 100%); accent-color: var(--accent2); }
    canvas {
      width: 100%;
      display: block;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.56), rgba(255,255,255,0.2)),
        repeating-linear-gradient(0deg, transparent 0, transparent 19px, rgba(23,32,38,0.05) 20px),
        repeating-linear-gradient(90deg, transparent 0, transparent 19px, rgba(23,32,38,0.05) 20px);
    }
    .main-canvas {
      aspect-ratio: 1.55 / 1;
      min-height: clamp(720px, 72vh, 980px);
    }
    .chart-canvas, .mini-canvas { aspect-ratio: 16 / 9; }
    .mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
    .side-grid { display: grid; gap: 16px; align-content: start; }
    .kv { display: grid; grid-template-columns: 1fr auto; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .kv:last-child { border-bottom: none; }
    .sensor-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .sensor { padding: 12px; border-radius: 16px; background: rgba(240,232,218,0.8); }
    .sensor .sensor-v { font-size: 20px; font-weight: 700; margin-top: 6px; }
    .path-box { max-height: 220px; overflow: auto; font: 13px Consolas, monospace; white-space: pre-wrap; padding: 12px; border-radius: 16px; background: rgba(240,232,218,0.8); color: var(--muted); }
    .legend { display: flex; justify-content: space-between; margin-top: 8px; color: var(--muted); font-size: 12px; }
    @media (max-width: 1060px) {
      .hero, .grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .mini-grid { grid-template-columns: 1fr; }
    }
    """


def dashboard_js_prelude() -> str:
    return """
    function fmt(value, digits = 2) {
      return value === null || value === undefined || Number.isNaN(Number(value)) ? "--" : Number(value).toFixed(digits);
    }
    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }
    function colorRamp(t) {
      const stops = [[20,37,46], [17,93,107], [216,116,32], [246,226,187]];
      const p = clamp(t, 0, 1) * 3;
      const i = Math.min(2, Math.floor(p));
      const k = p - i;
      const a = stops[i], b = stops[i + 1];
      const mix = (x, y) => Math.round(x + (y - x) * k);
      return `rgb(${mix(a[0], b[0])}, ${mix(a[1], b[1])}, ${mix(a[2], b[2])})`;
    }
    function mergeFrame(frame, previousMaps) {
      if (!frame) return null;
      if (frame.maps) return frame;
      const maps = { ...(previousMaps || {}) };
      if (frame.map_delta && Array.isArray(frame.map_delta.position)) {
        const [x, y] = frame.map_delta.position;
        Object.entries(frame.map_delta).forEach(([key, value]) => {
          if (key === "position") return;
          const grid = maps[key];
          if (Array.isArray(grid) && Array.isArray(grid[y])) {
            const nextGrid = grid.map(row => row.slice());
            nextGrid[y][x] = value;
            maps[key] = nextGrid;
          }
        });
      }
      return { ...frame, maps };
    }
    """


def dashboard_js_data_adapter() -> str:
    return """
    function frameWithMaps(index) {
      let fallbackMaps = null;
      for (let i = 0; i <= index; i += 1) {
        if (state.trace[i] && state.trace[i].maps) fallbackMaps = state.trace[i].maps;
      }
      return mergeFrame(state.trace[index], fallbackMaps);
    }
    function currentFrame() {
      return frameWithMaps(state.idx);
    }
    function residualGrid(frame) {
      if (!frame || !frame.maps || !frame.maps.truth_wind_speed) return null;
      const a = frame.maps.prediction_wind_speed;
      const b = frame.maps.truth_wind_speed;
      if (!a || !b) return null;
      return a.map((row, y) => row.map((value, x) => Math.abs(value - b[y][x])));
    }
    function currentLayer(frame) {
      if (!frame) return null;
      if (state.layer === "terrain_elevation") return state.report.terrain.elevation;
      if (state.layer === "terrain_roughness") return state.report.terrain.roughness;
      if (state.layer === "residual_speed") return residualGrid(frame);
      const stackKey = `${state.layer}_layers`;
      if (frame.maps && Array.isArray(frame.maps[stackKey])) {
        const layers = frame.maps[stackKey];
        const idx = clamp(state.altitudeLayer, 0, layers.length - 1);
        return layers[idx];
      }
      return (frame.maps && frame.maps[state.layer]) || (frame.maps && frame.maps.belief_energy) || null;
    }
    function pointFromCanvas(event) {
      const rect = el.mainCanvas.getBoundingClientRect();
      const scaleX = el.mainCanvas.width / rect.width;
      const scaleY = el.mainCanvas.height / rect.height;
      const x = (event.clientX - rect.left) * scaleX;
      const y = (event.clientY - rect.top) * scaleY;
      const pad = 28;
      const width = state.report.grid.width;
      const height = state.report.grid.height;
      const cellW = (el.mainCanvas.width - pad * 2) / width;
      const cellH = (el.mainCanvas.height - pad * 2) / height;
      const gx = clamp(Math.floor((x - pad) / cellW), 0, width - 1);
      const gy = clamp(Math.floor((y - pad) / cellH), 0, height - 1);
      return [gx, gy];
    }
    """


def dashboard_js_map_renderer() -> str:
    return """
    function asGrid(value) {
      return Array.isArray(value) && value.length > 0 && Array.isArray(value[0]) ? value : null;
    }
    function drawGrid(canvas, grid, options = {}) {
      grid = asGrid(grid);
      if (!grid) return null;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const h = grid.length, w = grid[0].length;
      const pad = options.pad ?? 20;
      const cellW = (canvas.width - pad * 2) / w;
      const cellH = (canvas.height - pad * 2) / h;
      const values = grid.flat().filter(v => v !== null && Number.isFinite(v));
      const min = values.length ? Math.min(...values) : 0;
      const max = values.length ? Math.max(...values) : 1;
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const raw = Number.isFinite(grid[y][x]) ? grid[y][x] : min;
          const t = max === min ? 0.5 : (raw - min) / (max - min);
          ctx.fillStyle = colorRamp(t);
          ctx.fillRect(pad + x * cellW, pad + y * cellH, cellW + 1, cellH + 1);
        }
      }
      return { min, max, pad, cellW, cellH, w, h, ctx };
    }
    function drawMarker(ctx, stats, point, fill, label, ink) {
      if (!Array.isArray(point) || point.length < 2) return;
      const px = stats.pad + (point[0] + 0.5) * stats.cellW;
      const py = stats.pad + (point[1] + 0.5) * stats.cellH;
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(8, Math.min(stats.cellW, stats.cellH) * 0.25), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = ink;
      ctx.font = "bold 14px Georgia";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, px, py + 0.5);
    }
    function layerStroke(level, alpha = 1) {
      const palette = [
        [17, 93, 107],
        [216, 116, 32],
        [181, 58, 45],
        [47, 125, 85],
        [89, 96, 140]
      ];
      const rawLevel = Number.isFinite(Number(level)) ? Number(level) : 0;
      const idx = Math.abs(Math.round(rawLevel)) % palette.length;
      const item = palette[idx];
      return `rgba(${item[0]}, ${item[1]}, ${item[2]}, ${alpha})`;
    }
    function drawLayeredPath(ctx, stats, path, options = {}) {
      if (!Array.isArray(path) || path.length < 2) return;
      const selected = state.altitudeLayer;
      const alphaOther = options.alphaOther ?? 0.2;
      const widthFocus = options.widthFocus ?? 3.2;
      const widthOther = options.widthOther ?? 1.6;
      const showAltitudeLabels = Boolean(options.showAltitudeLabels);
      for (let idx = 1; idx < path.length; idx += 1) {
        const prev = path[idx - 1];
        const curr = path[idx];
        if (!Array.isArray(prev) || !Array.isArray(curr) || prev.length < 2 || curr.length < 2) continue;
        const z = curr[2] ?? prev[2] ?? 0;
        const isFocus = z === selected;
        ctx.strokeStyle = layerStroke(z, isFocus ? 0.95 : alphaOther);
        ctx.lineWidth = isFocus ? widthFocus : widthOther;
        ctx.beginPath();
        ctx.moveTo(stats.pad + (prev[0] + 0.5) * stats.cellW, stats.pad + (prev[1] + 0.5) * stats.cellH);
        ctx.lineTo(stats.pad + (curr[0] + 0.5) * stats.cellW, stats.pad + (curr[1] + 0.5) * stats.cellH);
        ctx.stroke();
        if (showAltitudeLabels && (curr[2] ?? 0) !== (prev[2] ?? 0)) {
          const px = stats.pad + (curr[0] + 0.5) * stats.cellW;
          const py = stats.pad + (curr[1] + 0.5) * stats.cellH;
          ctx.fillStyle = "rgba(255,248,239,0.96)";
          ctx.strokeStyle = layerStroke(z, 0.95);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(px + 8, py - 24, 34, 18, 8);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#172026";
          ctx.font = "bold 11px Georgia";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(`z=${z}`, px + 14, py - 15);
        }
      }
    }
    function drawReturnBoundary(ctx, stats, frame) {
      const mask = asGrid(frame.maps ? frame.maps.return_feasible_mask : null);
      if (!mask) return;
      ctx.save();
      ctx.strokeStyle = "rgba(216,116,32,0.9)";
      ctx.lineWidth = 1.5;
      for (let y = 0; y < mask.length; y += 1) {
        for (let x = 0; x < mask[0].length; x += 1) {
          if (!mask[y][x]) continue;
          const left = x === 0 || !mask[y][x - 1];
          const right = x === mask[0].length - 1 || !mask[y][x + 1];
          const top = y === 0 || !mask[y - 1][x];
          const bottom = y === mask.length - 1 || !mask[y + 1][x];
          const px = stats.pad + x * stats.cellW;
          const py = stats.pad + y * stats.cellH;
          if (left) { ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + stats.cellH); ctx.stroke(); }
          if (right) { ctx.beginPath(); ctx.moveTo(px + stats.cellW, py); ctx.lineTo(px + stats.cellW, py + stats.cellH); ctx.stroke(); }
          if (top) { ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + stats.cellW, py); ctx.stroke(); }
          if (bottom) { ctx.beginPath(); ctx.moveTo(px, py + stats.cellH); ctx.lineTo(px + stats.cellW, py + stats.cellH); ctx.stroke(); }
        }
      }
      ctx.restore();
    }
    function drawMainMap(frame) {
      const grid = currentLayer(frame);
      const stats = drawGrid(el.mainCanvas, grid, { pad: 28 });
      if (!stats) return;
      el.legendMin.textContent = `min ${fmt(stats.min)}`;
      el.legendMax.textContent = `max ${fmt(stats.max)}`;
      const ctx = stats.ctx;
      drawReturnBoundary(ctx, stats, frame);
      const path = state.trace
        .slice(0, state.idx + 1)
        .map(item => (item && Array.isArray(item.position) ? item.position : null))
        .filter(point => Array.isArray(point) && point.length >= 2);
      drawLayeredPath(ctx, stats, path, { alphaOther: 0.24, widthFocus: 4.0, widthOther: 1.8, showAltitudeLabels: true });
      ctx.setLineDash([8, 5]);
      const plannedPath = Array.isArray(frame.planned_path)
        ? frame.planned_path.filter(point => Array.isArray(point) && point.length >= 2)
        : [];
      drawLayeredPath(ctx, stats, plannedPath, { alphaOther: 0.18, widthFocus: 2.8, widthOther: 1.2, showAltitudeLabels: true });
      (frame.candidate_paths || []).slice(1, 3).forEach((candidate, idx) => {
        const candidatePath = Array.isArray(candidate && candidate.path)
          ? candidate.path.filter(point => Array.isArray(point) && point.length >= 2)
          : [];
        drawLayeredPath(ctx, stats, candidatePath, { alphaOther: idx === 0 ? 0.12 : 0.08, widthFocus: 2.0, widthOther: 1.0, showAltitudeLabels: false });
      });
      ctx.setLineDash([]);
      const start = state.report && state.report.mission && Array.isArray(state.report.mission.start) ? state.report.mission.start : null;
      const goal = state.report && state.report.mission && Array.isArray(state.report.mission.goal) ? state.report.mission.goal : null;
      drawMarker(ctx, stats, start, layerStroke((start && start[2]) || 0, 0.95), "S", "#fff");
      drawMarker(ctx, stats, goal, layerStroke((goal && goal[2]) || 0, 0.95), "G", "#fff");
      drawMarker(ctx, stats, frame.position, "#fff8ef", "D", "#172026");
      ctx.fillStyle = "rgba(23,32,38,0.78)";
      ctx.font = "12px Georgia";
      ctx.fillText(`当前查看 z=${state.altitudeLayer}`, stats.pad, stats.pad - 8);
    }
    function drawResidualMap(frame) {
      const grid = (frame.maps && asGrid(frame.maps.prediction_residual_speed)) || residualGrid(frame) || [[0]];
      drawGrid(el.residualCanvas, grid);
    }
    function drawVectorField(frame) {
      const ctx = el.vectorCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.vectorCanvas.width, el.vectorCanvas.height);
      if (!frame || !frame.maps) return;
      const layerIndex = state.altitudeLayer;
      const pickLayer = (stackKey, fallbackKey) => {
        const stack = frame.maps && frame.maps[stackKey];
        if (Array.isArray(stack) && stack.length) {
          return stack[clamp(layerIndex, 0, stack.length - 1)];
        }
        return frame.maps ? frame.maps[fallbackKey] : null;
      };
      const uGrid = pickLayer("belief_wind_u_layers", "belief_wind_u")
        || pickLayer("prediction_wind_u_layers", "prediction_wind_u")
        || pickLayer("physics_wind_u_layers", "physics_wind_u");
      const vGrid = pickLayer("belief_wind_v_layers", "belief_wind_v")
        || pickLayer("prediction_wind_v_layers", "prediction_wind_v")
        || pickLayer("physics_wind_v_layers", "physics_wind_v");
      if (!asGrid(uGrid) || !asGrid(vGrid)) return;
      const pad = 24;
      const width = state.report.grid.width;
      const height = state.report.grid.height;
      const cellW = (el.vectorCanvas.width - pad * 2) / width;
      const cellH = (el.vectorCanvas.height - pad * 2) / height;
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fillRect(0, 0, el.vectorCanvas.width, el.vectorCanvas.height);
      const stride = Math.max(1, Math.floor(width / 12));
      for (let y = 0; y < height; y += stride) {
        for (let x = 0; x < width; x += stride) {
          const u = uGrid[y][x];
          const v = vGrid[y][x];
          const speed = Math.hypot(u, v);
          const angle = Math.atan2(v, u);
          const len = 6 + speed * 2.2;
          const px = pad + (x + 0.5) * cellW;
          const py = pad + (y + 0.5) * cellH;
          const ex = px + Math.cos(angle) * len;
          const ey = py + Math.sin(angle) * len;
          ctx.strokeStyle = colorRamp(speed / 12);
          ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ex, ey); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(ex, ey);
          ctx.lineTo(ex - Math.cos(angle - 0.5) * 4, ey - Math.sin(angle - 0.5) * 4);
          ctx.lineTo(ex - Math.cos(angle + 0.5) * 4, ey - Math.sin(angle + 0.5) * 4);
          ctx.closePath(); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
        }
      }
      ctx.fillStyle = "rgba(23,32,38,0.78)";
      ctx.font = "12px Georgia";
      ctx.fillText(`风矢量层 z=${layerIndex}`, pad, 16);
    }
    function drawTerrain3D() {
      const ctx = el.terrainCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.terrainCanvas.width, el.terrainCanvas.height);
      const elev = asGrid(state.report && state.report.terrain ? state.report.terrain.elevation : null);
      if (!elev) return;
      const height = elev.length, width = elev[0].length;
      const values = elev.flat();
      const min = Math.min(...values), max = Math.max(...values);
      const scaleX = 12, scaleY = 7, scaleZ = 0.5;
      const originX = el.terrainCanvas.width * 0.5;
      const originY = 90;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const z = (elev[y][x] - min) / Math.max(max - min, 1e-6);
          const px = originX + (x - y) * scaleX;
          const py = originY + (x + y) * scaleY - z * 120 * scaleZ;
          ctx.fillStyle = colorRamp(z);
          ctx.fillRect(px, py, 3, 3);
        }
      }
      const frame = currentFrame();
      if (!frame || !Array.isArray(frame.position) || frame.position.length < 2) return;
      const point = frame.position;
      const ix = clamp(Math.round(point[0] || 0), 0, width - 1);
      const iy = clamp(Math.round(point[1] || 0), 0, height - 1);
      const z = (elev[iy][ix] - min) / Math.max(max - min, 1e-6);
      const px = originX + ((point[0] || 0) - (point[1] || 0)) * scaleX;
      const py = originY + ((point[0] || 0) + (point[1] || 0)) * scaleY - z * 120 * scaleZ;
      ctx.fillStyle = "#b53a2d";
      ctx.beginPath(); ctx.arc(px + 2, py + 2, 6, 0, Math.PI * 2); ctx.fill();
    }
    function drawHeatPanel(ctx, grid, x0, width, title) {
      grid = asGrid(grid);
      if (!grid) return;
      const h = grid.length;
      const w = grid[0].length;
      const padY = 36;
      const cellW = width / w;
      const cellH = (ctx.canvas.height - 58) / h;
      const values = grid.flat().filter(v => Number.isFinite(v));
      const min = values.length ? Math.min(...values) : 0;
      const max = values.length ? Math.max(...values) : 1;
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const raw = Number.isFinite(grid[y][x]) ? grid[y][x] : min;
          const t = max === min ? 0.5 : (raw - min) / (max - min);
          ctx.fillStyle = colorRamp(t);
          ctx.fillRect(x0 + x * cellW, padY + y * cellH, cellW + 1, cellH + 1);
        }
      }
      ctx.fillStyle = "#172026";
      ctx.font = "14px Georgia";
      ctx.fillText(title, x0, 20);
    }
    function drawCompareChart(frame) {
      const ctx = el.compareCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.compareCanvas.width, el.compareCanvas.height);
      if (!frame || !frame.maps) return;
      const gap = 18;
      const panelWidth = (el.compareCanvas.width - gap * 4) / 3;
      drawHeatPanel(ctx, asGrid(frame.maps.physics_residual_speed), gap, panelWidth, "物理误差");
      drawHeatPanel(ctx, asGrid(frame.maps.prediction_residual_speed), gap * 2 + panelWidth, panelWidth, "学习误差");
      drawHeatPanel(ctx, asGrid(frame.maps.truth_wind_speed), gap * 3 + panelWidth * 2, panelWidth, "真值风场");
    }
    """


def dashboard_js_training_panel() -> str:
    return """
    function drawBatteryChart() {
      const ctx = el.batteryCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.batteryCanvas.width, el.batteryCanvas.height);
      const frames = state.trace;
      if (!frames.length) return;
      const pad = 28;
      const w = el.batteryCanvas.width - pad * 2;
      const h = el.batteryCanvas.height - pad * 2;
      const energy = frames.map(f => Number(f.sensor_packet.energy_rate || 0));
      const energyMin = Math.min(...energy, -0.1);
      const energyMax = Math.max(...energy, 0.1);
      const drawSeries = (values, color, valueToY) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        values.forEach((value, i) => {
          const x = pad + (i / Math.max(values.length - 1, 1)) * w;
          const y = valueToY(value);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };
      drawSeries(frames.map(f => f.battery_ratio), "#115d6b", value => pad + (1 - value) * h);
      drawSeries(energy, "#d87420", value => pad + (1 - (value - energyMin) / Math.max(energyMax - energyMin, 1e-6)) * h);
      const cursorX = pad + (state.idx / Math.max(frames.length - 1, 1)) * w;
      ctx.strokeStyle = "rgba(181,58,45,0.8)";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cursorX, pad); ctx.lineTo(cursorX, pad + h); ctx.stroke();
    }
    function drawTrainingChart() {
      const ctx = el.trainingCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.trainingCanvas.width, el.trainingCanvas.height);
      const model = state.report.model_summary || {};
      const seriesDefs = [
        ["训练 u", ((model.validation_curve_u || {}).train || []), "#115d6b"],
        ["验证 u", ((model.validation_curve_u || {}).valid || []), "#1f8aa3"],
        ["训练 v", ((model.validation_curve_v || {}).train || []), "#b53a2d"],
        ["验证 v", ((model.validation_curve_v || {}).valid || []), "#dd8d83"],
        ["训练 w", ((model.validation_curve_w || {}).train || []), "#2f7d55"],
        ["验证 w", ((model.validation_curve_w || {}).valid || []), "#7ab493"]
      ].filter(item => item[1].length);
      if (!seriesDefs.length) {
        ctx.fillStyle = "rgba(23,32,38,0.78)";
        ctx.font = "16px Georgia";
        ctx.fillText("当前模型没有训练曲线", 28, 42);
        ctx.font = "13px Georgia";
        ctx.fillText(`model=${model.model_type || '--'}；请使用 xgboost/lightgbm 指标曲线`, 28, 68);
        return;
      }
      const pad = 30;
      const w = el.trainingCanvas.width - pad * 2;
      const h = el.trainingCanvas.height - pad * 2;
      const all = seriesDefs.flatMap(item => item[1]);
      const min = Math.min(...all);
      const max = Math.max(...all);
      seriesDefs.forEach(([label, values, color], index) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        values.forEach((value, idx) => {
          const x = pad + (idx / Math.max(values.length - 1, 1)) * w;
          const y = pad + (1 - (value - min) / Math.max(max - min, 1e-6)) * h;
          if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = "#172026";
        ctx.font = "13px Georgia";
        ctx.fillText(label, pad, 18 + index * 14);
      });
    }
    function drawVerticalProfile(frame) {
      const ctx = el.profileCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.profileCanvas.width, el.profileCanvas.height);
      const pred = frame.prediction_profile_at_drone;
      if (!pred || !Array.isArray(pred.speed) || !pred.speed.length || !Array.isArray(pred.w)) return;
      const truth = frame.truth_profile_at_drone;
      const pad = 34;
      const w = el.profileCanvas.width - pad * 2;
      const h = el.profileCanvas.height - pad * 2;
      const levels = pred.speed.length;
      const values = [...pred.speed, ...(truth ? truth.speed : []), ...pred.w, ...(truth ? truth.w : [])];
      const min = Math.min(...values, -0.1);
      const max = Math.max(...values, 0.1);
      const xAt = idx => pad + (idx / Math.max(levels - 1, 1)) * w;
      const yAt = value => pad + (1 - (value - min) / Math.max(max - min, 1e-6)) * h;
      const drawSeries = (values, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        values.forEach((value, idx) => {
          const x = xAt(idx);
          const y = yAt(value);
          if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };
      drawSeries(pred.speed, "#115d6b");
      drawSeries(pred.w, "#d87420");
      if (truth) {
        drawSeries(truth.speed, "rgba(23,32,38,0.7)");
        drawSeries(truth.w, "rgba(181,58,45,0.7)");
      }
      const currentZ = Array.isArray(frame.position) ? (frame.position[2] || 0) : 0;
      const x = xAt(currentZ);
      ctx.strokeStyle = "rgba(181,58,45,0.8)";
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, pad + h); ctx.stroke();
      ctx.fillStyle = "#172026";
      ctx.font = "13px Georgia";
      ctx.fillText("风速", pad, 16);
      ctx.fillText("垂直风", pad + 72, 16);
    }
    function drawAltitudeChart() {
      const ctx = el.altitudeCanvas.getContext("2d");
      ctx.clearRect(0, 0, el.altitudeCanvas.width, el.altitudeCanvas.height);
      const frames = state.trace;
      if (!frames.length) return;
      const pad = 30;
      const w = el.altitudeCanvas.width - pad * 2;
      const h = el.altitudeCanvas.height - pad * 2;
      const goalPoint = (state.report && state.report.mission && Array.isArray(state.report.mission.goal)) ? state.report.mission.goal : [0, 0, 0];
      const maxAlt = Math.max(...frames.map(f => (f.position[2] || 0)), ...(goalPoint.slice(2)), 1);
      const alts = frames.map(f => f.position[2] || 0);
      const yAt = value => pad + (1 - value / Math.max(maxAlt, 1)) * h;
      ctx.strokeStyle = "#115d6b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      alts.forEach((value, idx) => {
        const x = pad + (idx / Math.max(alts.length - 1, 1)) * w;
        const y = yAt(value);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const frame = currentFrame();
      const plannedPath = frame && Array.isArray(frame.planned_path) ? frame.planned_path : [];
      const plannedAlts = plannedPath.map(point => point[2] || 0);
      if (plannedAlts.length > 1) {
        ctx.strokeStyle = "rgba(216,116,32,0.95)";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 5]);
        ctx.beginPath();
        plannedAlts.forEach((value, idx) => {
          const x = pad + (idx / Math.max(plannedAlts.length - 1, 1)) * w;
          const y = yAt(value);
          if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = "rgba(216,116,32,0.9)";
      const goalAlt = (goalPoint[2] || 0);
      ctx.setLineDash([7, 5]);
      ctx.beginPath(); ctx.moveTo(pad, yAt(goalAlt)); ctx.lineTo(pad + w, yAt(goalAlt)); ctx.stroke();
      ctx.setLineDash([]);
      const cursorX = pad + (state.idx / Math.max(alts.length - 1, 1)) * w;
      ctx.strokeStyle = "rgba(181,58,45,0.8)";
      ctx.beginPath(); ctx.moveTo(cursorX, pad); ctx.lineTo(cursorX, pad + h); ctx.stroke();
      ctx.fillStyle = "#172026";
      ctx.font = "13px Georgia";
      ctx.fillText(`目标高度 z=${goalAlt}`, pad, 16);
      ctx.fillText("实线=已执行高度", pad + 110, 16);
      ctx.fillText("虚线=规划高度", pad + 290, 16);
    }
    """


def dashboard_js_path_panel() -> str:
    return """
    function updateMeta(frame) {
      const prediction = frame.prediction_at_drone || {};
      const truth = frame.truth_at_drone || null;
      const belief = frame.belief_at_drone || {};
      const observation = frame.observation || null;
      const sensor = frame.sensor_packet || {};
      const plannedPath = Array.isArray(frame.planned_path) ? frame.planned_path : [];
      const candidatePaths = Array.isArray(frame.candidate_paths) ? frame.candidate_paths : [];
      const costBreakdown = frame.planned_path_cost_breakdown || {};
      const mission = (state.report && state.report.mission) || { start: [0, 0, 0], goal: [0, 0, 0] };
      const residual = truth ? Math.hypot((prediction.u || 0) - truth.u, (prediction.v || 0) - truth.v, (prediction.w || 0) - truth.w) : null;
      const model = state.report.model_summary || {};
      el.goalStatus.textContent = state.summary && state.summary.goal_reached ? "已到达" : "执行中";
      el.stepValue.textContent = `${state.idx + 1} / ${Math.max(state.trace.length, (state.summary && state.summary.steps_executed) || 0)}`;
      el.batteryValue.textContent = `${fmt(frame.battery_ratio * 100, 1)}%`;
      el.residualValue.textContent = residual === null ? "--" : fmt(residual);
      el.gridValue.textContent = `${state.report.grid.width}x${state.report.grid.height}`;
      el.startValue.textContent = (mission.start || []).join(", ");
      el.goalCellValue.textContent = (mission.goal || []).join(", ");
      el.timeValue.textContent = frame.timestamp;
      el.posValue.textContent = Array.isArray(frame.position) ? frame.position.join(", ") : "--";
      el.layerValue.textContent = el.layerSelect.options[el.layerSelect.selectedIndex].text;
      if (el.altitudeSelect) {
        const label = el.altitudeSelect.options[el.altitudeSelect.selectedIndex]?.text || `高度层 z=${state.altitudeLayer}`;
        el.layerValue.textContent += ` / ${label}`;
      }
      el.distValue.textContent = fmt(frame.goal_distance_cells, 2);
      el.truthValue.textContent = truth ? "可用" : "隐藏";
      el.beliefEnergy.textContent = fmt(belief.energy);
      el.beliefUncertainty.textContent = fmt(belief.uncertainty);
      el.beliefConfidence.textContent = fmt(belief.confidence);
      if (el.beliefEntropy) el.beliefEntropy.textContent = fmt(belief.entropy);
      if (el.beliefSafety) el.beliefSafety.textContent = fmt(belief.safety);
      if (el.beliefModes) el.beliefModes.textContent = `${fmt(belief.uplift_prob)} / ${fmt(belief.sink_prob)}`;
      el.predWind.textContent = `${fmt(prediction.u)} / ${fmt(prediction.v)} / ${fmt(prediction.w)}`;
      el.obsWind.textContent = observation ? `${fmt(observation.u_obs)} / ${fmt(observation.v_obs)} / ${fmt(observation.w_obs)}` : "--";
      el.truthWind.textContent = truth ? `${fmt(truth.u)} / ${fmt(truth.v)} / ${fmt(truth.w)}` : "--";
      const plannedLevels = plannedPath.map(p => p[2] ?? 0);
      const altitudeSummary = plannedLevels.length ? `高度剖面: ${plannedLevels.join(" -> ")}` : "高度剖面: --";
      const safePathLines = plannedPath
        .filter(p => Array.isArray(p) && p.length >= 2)
        .map(p => `[${p[0]}, ${p[1]}, ${p[2] ?? 0}]`);
      el.pathBox.textContent = [altitudeSummary, ...safePathLines].join("\\n");
      el.forecastMode.textContent = frame.forecast_mode || "--";
      el.currentGoal.textContent = (mission.goal || []).join(", ");
      el.pathCost.textContent = `${fmt(frame.planned_path_cost, 1)} (${frame.planning_mode || "mpc_strict_return"})`;
      el.returnRadius.textContent = frame.return_home_budget_j === null || frame.return_home_budget_j === undefined ? "--" : `${fmt(frame.return_home_budget_j, 0)} J`;
      el.candidateCount.textContent = String(candidatePaths.length);
      el.costBreakdownBox.textContent = [
        `total_cost_j ${fmt(costBreakdown.total_cost_j, 1)}`,
        `energy_j ${fmt(costBreakdown.energy_j, 1)}`,
        `progress_reward_j ${fmt(costBreakdown.progress_reward_j, 1)}`,
        `uncertainty_cost_j ${fmt(costBreakdown.uncertainty_cost_j, 1)}`,
        `safety_cost_j ${fmt(costBreakdown.safety_cost_j, 1)}`,
        `altitude_bias_j ${fmt(costBreakdown.altitude_bias_j, 1)}`,
        `vertical_maneuver_cost_j ${fmt(costBreakdown.vertical_maneuver_cost_j, 1)}`
      ].join("\\n");
      el.candidateBox.textContent = candidatePaths.map(item => {
        const path = Array.isArray(item.path) ? item.path : [];
        const levels = path.map(point => point[2] ?? 0).join("->");
        const breakdown = item.path_cost_breakdown || {};
        return `${item.mode || "mpc_strict_return"} w=${fmt(item.heuristic_weight, 2)} h=${item.horizon_steps || "--"} b=${item.beam_width || "--"} cost=${fmt(item.path_cost, 1)} reached=${item.goal_reached} z=${levels} total=${fmt(breakdown.total_cost_j, 1)} energy=${fmt(breakdown.energy_j, 1)} risk=${fmt(breakdown.uncertainty_cost_j, 1)} safety=${fmt(breakdown.safety_cost_j, 1)}`;
      }).join("\\n");
      el.modelType.textContent = model.model_type || "--";
      el.bestIterU.textContent = model.best_iteration_u ?? model.train_size_used ?? model.train_size ?? "--";
      el.bestIterV.textContent = model.best_iteration_v ?? model.validation_size ?? "--";
      el.bestIterW.textContent = model.best_iteration_w ?? model.test_size ?? "--";
      el.bestScoreU.textContent = model.best_score_u === null || model.best_score_u === undefined ? fmt(model.rmse_final, 4) : fmt(model.best_score_u, 4);
      el.bestScoreV.textContent = model.best_score_v === null || model.best_score_v === undefined ? fmt(model.vertical_mae_final, 4) : fmt(model.best_score_v, 4);
      el.bestScoreW.textContent = model.best_score_w === null || model.best_score_w === undefined ? fmt(model.direction_mae_rad, 4) : fmt(model.best_score_w, 4);
      const importanceU = Object.entries(model.feature_importance_gain_u || {}).map(([k, v]) => `u:${k} ${fmt(v, 3)}`);
      const importanceV = Object.entries(model.feature_importance_gain_v || {}).map(([k, v]) => `v:${k} ${fmt(v, 3)}`);
      const importanceW = Object.entries(model.feature_importance_gain_w || {}).map(([k, v]) => `w:${k} ${fmt(v, 3)}`);
      const summaryLines = [];
      if (model.rmse_physics !== undefined) summaryLines.push(`rmse_physics ${fmt(model.rmse_physics, 4)}`);
      if (model.rmse_final !== undefined) summaryLines.push(`rmse_final ${fmt(model.rmse_final, 4)}`);
      if (model.vertical_mae_physics !== undefined) summaryLines.push(`vertical_mae_physics ${fmt(model.vertical_mae_physics, 4)}`);
      if (model.vertical_mae_final !== undefined) summaryLines.push(`vertical_mae_final ${fmt(model.vertical_mae_final, 4)}`);
      if (model.direction_mae_rad !== undefined) summaryLines.push(`direction_mae_rad ${fmt(model.direction_mae_rad, 4)}`);
      if (model.train_size !== undefined) summaryLines.push(`train_size ${model.train_size}`);
      if (model.validation_size !== undefined) summaryLines.push(`validation_size ${model.validation_size}`);
      if (model.test_size !== undefined) summaryLines.push(`test_size ${model.test_size}`);
      const importanceLines = [...importanceU, ...importanceV, ...importanceW];
      el.importanceBox.textContent = [...summaryLines, ...importanceLines].join("\\n") || "当前模型没有特征重要性";
      const sensors = [
        ["空速", `${fmt(sensor.airspeed)} m/s`],
        ["地速", `${fmt(sensor.ground_speed)} m/s`],
        ["航向", `${fmt(sensor.heading_rad)} rad`],
        ["风 U", `${fmt(sensor.measured_wind_u)} m/s`],
        ["风 V", `${fmt(sensor.measured_wind_v)} m/s`],
        ["风 W", `${fmt(sensor.measured_wind_w)} m/s`],
        ["高度", `${fmt(sensor.altitude_m)} m`],
        ["IMU 纵向", `${fmt(sensor.imu_accel_longitudinal)} g`],
        ["IMU 垂向", `${fmt(sensor.imu_accel_vertical)} g`],
        ["能量率", fmt(sensor.energy_rate)]
      ];
      el.sensorGrid.innerHTML = sensors.map(([k, v]) => `<div class="sensor"><div class="label">${k}</div><div class="sensor-v">${v}</div></div>`).join("");
    }
    """
