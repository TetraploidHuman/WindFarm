/* rtwind quad-view: 2×2 panels with selectable content + wireframe 3D attitude */
(() => {
  const PANEL_DEFS = {
    track: { id: "track", label: "运动轨迹" },
    weather: { id: "weather", label: "气象与风场" },
    belief: { id: "belief", label: "信念场" },
    terrain: { id: "terrain", label: "地形" },
    attitude: { id: "attitude", label: "3D姿态" },
    camera: { id: "camera", label: "摄像头" },
  };
  const PANEL_ORDER = ["track", "weather", "belief", "terrain", "attitude", "camera"];
  const DEFAULT_SLOTS = ["track", "weather", "belief", "terrain"];
  const STORAGE_KEY = "rtwind-quad";

  function hasPos(frame) {
    if (!frame) return false;
    const { lat, lon } = frame;
    if (lat == null || lon == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) {
      return false;
    }
    return !(Math.abs(lat) < 1e-6 && Math.abs(lon) < 1e-6);
  }

  function setHudNA(hud) {
    if (!hud) return;
    hud.querySelectorAll("[data-k]").forEach((el) => {
      el.textContent = "暂无";
    });
  }

  // Parametric surface mesh (OpenVSP-style wire grid).
  // Body NED: +X = nose (prop at x≈0.62), −X = tail (x≈−1.02), +Y = right wing, +Z = down.
  function meshSurface(pointFn, nu, nv) {
    const grid = [];
    for (let i = 0; i <= nu; i += 1) {
      const row = [];
      for (let j = 0; j <= nv; j += 1) {
        row.push(pointFn(i / nu, j / nv));
      }
      grid.push(row);
    }
    const segs = [];
    for (let i = 0; i <= nu; i += 1) {
      for (let j = 0; j < nv; j += 1) {
        segs.push([grid[i][j], grid[i][j + 1]]);
      }
    }
    for (let j = 0; j <= nv; j += 1) {
      for (let i = 0; i < nu; i += 1) {
        segs.push([grid[i][j], grid[i + 1][j]]);
      }
    }
    return segs;
  }

  /** High-aspect wing on central pylon. */
  function wingPoint(u, v) {
    const span = 1.12;
    const y = (v - 0.5) * 2 * span;
    const t = Math.min(Math.abs(y) / span, 1);
    const xLE = 0.20 - 0.09 * t;
    const xTE = -0.16 - 0.04 * t;
    const x = xLE + u * (xTE - xLE);
    const z = -0.05 + 0.012 * u;
    return [x, y, z];
  }

  /** Thick central pylon: wing → payload pod. */
  function centerPylonPoint(u, v) {
    const zTop = 0.01;
    const zBot = 0.10;
    const z = zTop + u * (zBot - zTop);
    const th = v * Math.PI * 2;
    const taper = 1 - u * 0.10;
    const rx = (0.08 + u * 0.015) * taper;
    const ry = (0.11 + u * 0.008) * taper;
    const x = 0.05 - u * 0.04;
    return [x, ry * Math.cos(th), z + rx * Math.sin(th) * 0.42];
  }

  /** Belly cylinder (mission pod) — compact, under pylon. */
  function payloadPoint(u, v) {
    const xNose = 0.26;
    const xTail = -0.20;
    const x = xNose + u * (xTail - xNose);
    const th = v * Math.PI * 2;
    const r = 0.08;
    const zCenter = 0.18;
    let rEff = r;
    if (u > 0.90) {
      const t = (u - 0.90) / 0.10;
      rEff = r * (1 - 0.30 * t);
    }
    return [x, rEff * Math.cos(th), zCenter + rEff * Math.sin(th)];
  }

  /** Forward camera gimbal dome. */
  function cameraDomePoint(u, v) {
    const phi = u * (Math.PI / 2);
    const theta = v * Math.PI * 2;
    const r = 0.08;
    const cx = 0.28;
    const cz = 0.18;
    const rim = r * Math.sin(phi);
    return [cx + r * Math.cos(phi), rim * Math.cos(theta), cz + rim * Math.sin(theta)];
  }

  /** Forward boom: wing LE → motor. */
  function motorBoomPoint(u, v) {
    const x = 0.22 + u * (0.50 - 0.22);
    const th = v * Math.PI * 2;
    const r = 0.024;
    const zc = -0.03;
    return [x, r * Math.cos(th), zc + r * Math.sin(th)];
  }

  /** Tractor motor nacelle at front. */
  function motorNacellePoint(u, v) {
    const x = 0.50 + u * (0.58 - 0.50);
    const th = v * Math.PI * 2;
    const r = 0.048;
    const zc = -0.03;
    return [x, r * Math.cos(th), zc + r * Math.sin(th)];
  }

  /** Tractor propeller — two blades. */
  function tractorPropLines() {
    const cx = 0.62;
    const cz = -0.03;
    const r = 0.17;
    return [
      [[cx, 0, cz], [cx, r, cz]],
      [[cx, 0, cz], [cx, -r, cz]],
    ];
  }

  /** Tail boom from wing center aft (same level as wing). */
  const TAIL_X = -1.02;
  const TAIL_Z = 0.02;

  function boomPoint(u, v) {
    const x = -0.14 + u * (TAIL_X - -0.14);
    const th = v * Math.PI * 2;
    const r = 0.018;
    return [x, r * Math.cos(th), TAIL_Z + r * Math.sin(th)];
  }

  /** Horizontal stabilizers — flat, left/right at boom tip. */
  function hstabPoint(u, v) {
    const span = 0.26;
    const y = (v - 0.5) * 2 * span;
    const t = Math.min(Math.abs(y) / span, 1);
    const chord = 0.09 * (1 - 0.25 * t);
    const x = TAIL_X + u * chord;
    const z = TAIL_Z;
    return [x, y, z];
  }

  /** Vertical fin — points up from boom tip (+Z is down). */
  function vfinPoint(u, v) {
    const height = 0.18;
    const rootX = TAIL_X;
    const tipX = TAIL_X - 0.05;
    const x = rootX + u * (tipX - rootX);
    const z = TAIL_Z + v * (-height);
    const y = 0.014 * (1 - 2 * Math.abs(u - 0.5));
    return [x, y, z];
  }

  const AC_MESH_LINES = [
    ...meshSurface(wingPoint, 12, 18),
    ...meshSurface(centerPylonPoint, 10, 12),
    ...meshSurface(payloadPoint, 14, 16),
    ...meshSurface(cameraDomePoint, 10, 14),
    ...meshSurface(motorBoomPoint, 10, 6),
    ...meshSurface(motorNacellePoint, 6, 8),
    ...tractorPropLines(),
    ...meshSurface(boomPoint, 14, 6),
    ...meshSurface(hstabPoint, 6, 10),
    ...meshSurface(vfinPoint, 10, 12),
  ];

  function loadSettings() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { view: "map", slots: DEFAULT_SLOTS.slice() };
      const data = JSON.parse(raw);
      const slots = Array.isArray(data.slots) ? data.slots.slice(0, 4) : DEFAULT_SLOTS.slice();
      while (slots.length < 4) slots.push(DEFAULT_SLOTS[slots.length]);
      for (let i = 0; i < 4; i += 1) {
        if (!PANEL_DEFS[slots[i]]) slots[i] = DEFAULT_SLOTS[i];
      }
      return {
        view: data.view === "quad" ? "quad" : "map",
        slots,
      };
    } catch (_) {
      return { view: "map", slots: DEFAULT_SLOTS.slice() };
    }
  }

  function saveSettings(settings) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        view: settings.view,
        slots: settings.slots,
      }));
    } catch (_) { /* ignore */ }
  }

  function mulMatVec(m, v) {
    return [
      m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
      m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
      m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ];
  }

  function rotX(a) {
    const c = Math.cos(a); const s = Math.sin(a);
    return [[1, 0, 0], [0, c, -s], [0, s, c]];
  }
  function rotY(a) {
    const c = Math.cos(a); const s = Math.sin(a);
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  }
  function rotZ(a) {
    const c = Math.cos(a); const s = Math.sin(a);
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  }
  function mulMat(a, b) {
    const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
      }
    }
    return out;
  }

  const DEG = Math.PI / 180;
  /** Body sky direction (NED +Z = down, so sky = −Z). */
  const BODY_SKY = [0, 0, -1];

  function vecDot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function vecCross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function vecNorm(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  /** Nose (+X) and tail (−X) reference points on the mesh. */
  const AC_NOSE = [0.62, 0, -0.03];
  const AC_TAIL = [-1.02, 0, 0.02];

  function projectOntoPlane(v, normal) {
    const d = vecDot(v, normal);
    return [
      v[0] - normal[0] * d,
      v[1] - normal[1] * d,
      v[2] - normal[2] * d,
    ];
  }

  /**
   * Orthographic camera. eyeDir = body origin → camera position.
   * Sky (−Z) always up on screen.
   */
  function makeViewCamera(eyeDir, upFallback = [1, 0, 0]) {
    const forward = vecNorm(eyeDir.map((c) => -c));
    let upRef = projectOntoPlane(BODY_SKY, forward);
    let upLen = Math.hypot(upRef[0], upRef[1], upRef[2]);
    if (upLen < 1e-6) {
      upRef = projectOntoPlane(upFallback, forward);
      upLen = Math.hypot(upRef[0], upRef[1], upRef[2]) || 1;
    }
    const up0 = [upRef[0] / upLen, upRef[1] / upLen, upRef[2] / upLen];
    const right = vecNorm(vecCross(forward, up0));
    const upFinal = vecNorm(vecCross(right, forward));
    return (v) => ({
      x: vecDot(v, right),
      y: -vecDot(v, upFinal),
      z: -vecDot(v, forward),
    });
  }

  /** Clip rear view at this body +X plane (keep tail / discard nose). */
  const REAR_CLIP_X = 0.10;

  /** Return 0–1 segments kept on the tail (x ≤ maxX) side of a clip plane. */
  function clipSegmentMaxX(a, b, maxX) {
    const ax = a[0];
    const bx = b[0];
    if (ax <= maxX && bx <= maxX) return [[a, b]];
    if (ax > maxX && bx > maxX) return [];
    const t = (maxX - ax) / (bx - ax);
    const p = [
      ax + t * (b[0] - ax),
      a[1] + t * (b[1] - a[1]),
      a[2] + t * (b[2] - a[2]),
    ];
    if (ax > maxX) return [[p, b]];
    return [[a, p]];
  }

  function meshLinesForView(viewId) {
    if (viewId !== "rear") return AC_MESH_LINES;
    const clipped = [];
    AC_MESH_LINES.forEach(([a, b]) => {
      clipSegmentMaxX(a, b, REAR_CLIP_X).forEach((seg) => clipped.push(seg));
    });
    return clipped;
  }

  /** eyeDir: camera position relative to aircraft origin (body NED, +X = nose/prop). */
  const ATTITUDE_VIEWS = {
    // Tail-on with nose clipped; slight elevation to expose H-stab / V-fin.
    rear: { label: "后方", hint: "见尾椎/H翼", project: makeViewCamera([-1, 0.12, -0.18]) },
    front: { label: "前方", hint: "见螺旋桨", project: makeViewCamera([1, 0, 0]) },
    top: { label: "俯视", hint: "机头↑", project: makeViewCamera([0, 0, -1], [1, 0, 0]) },
    right: { label: "右侧", hint: "右翼→", project: makeViewCamera([0, 1, 0]) },
    left: { label: "左侧", hint: "左翼→", project: makeViewCamera([0, -1, 0]) },
    // 45° elevation from above (rear-right); −Z = up in NED.
    iso: { label: "斜视", hint: "后右俯视45°", project: makeViewCamera([-1, 1, -2], [0, 0, -1]) },
  };
  const ATTITUDE_VIEW_ORDER = ["rear", "front", "top", "right", "left", "iso"];
  const DEFAULT_ATTITUDE_VIEW = "rear";

  /**
   * NED body: +X nose, +Y right wing, +Z down.
   * Roll = rot about nose (X); pitch = rot about wing (Y).
   * Sign: right wing down → roll + (matches ImuTracker body frame).
   */
  function buildBodyMatrix(rollDeg, pitchDeg, _viewId) {
    const roll = rollDeg * DEG;
    const pitch = pitchDeg * DEG;
    return mulMat(rotY(pitch), rotX(roll));
  }

  /** Exponential smoothing time constant (ms) for attitude interpolation. */
  const ATT_SMOOTH_MS = 180;

  function readFrameAttitude(frame) {
    if (!frame) return { roll: 0, pitch: 0, yaw: 0 };
    return {
      roll: Number(frame.roll) || 0,
      pitch: Number(frame.pitch) || 0,
      yaw: Number(frame.yaw != null ? frame.yaw : frame.heading) || 0,
    };
  }

  function lerpScalar(from, to, t) {
    return from + (to - from) * t;
  }

  /** Shortest-path lerp for headings / signed angles (degrees). */
  function lerpAngleDeg(from, to, t) {
    let delta = ((to - from + 180) % 360 + 360) % 360 - 180;
    return from + delta * t;
  }

  function smoothAttitudeStep(state, target, dtMs) {
    const k = 1 - Math.exp(-Math.min(dtMs, 50) / ATT_SMOOTH_MS);
    state.roll = lerpScalar(state.roll, target.roll, k);
    state.pitch = lerpScalar(state.pitch, target.pitch, k);
    state.yaw = lerpAngleDeg(state.yaw, target.yaw, k);
  }

  function snapAttitude(state, target) {
    state.roll = target.roll;
    state.pitch = target.pitch;
    state.yaw = target.yaw;
  }

  function projectBodyPoint(bodyMat, p, viewId) {
    const v = mulMatVec(bodyMat, p);
    const view = ATTITUDE_VIEWS[viewId] || ATTITUDE_VIEWS.rear;
    return view.project(v);
  }

  function projectPoint(bodyMat, p, cx, cy, scale, viewId) {
    const pr = projectBodyPoint(bodyMat, p, viewId);
    return {
      x: cx + pr.x * scale,
      y: cy + pr.y * scale,
      z: pr.z,
    };
  }

  function projectMeshLines(bodyMat, w, h, viewId) {
    const scale = Math.min(w, h) * 0.20;
    const cx = w * 0.5;
    const cy = h * 0.54;
    return meshLinesForView(viewId).map(([a, b]) => {
      const pa = projectPoint(bodyMat, a, cx, cy, scale, viewId);
      const pb = projectPoint(bodyMat, b, cx, cy, scale, viewId);
      return {
        x1: pa.x,
        y1: pa.y,
        x2: pb.x,
        y2: pb.y,
        z: (pa.z + pb.z) * 0.5,
      };
    });
  }

  function drawAxisTriad(ctx, bodyMat, w, h, colors, viewId) {
    const ox = 34;
    const oy = h - 34;
    const scale = 20;
    const origin = projectPoint(bodyMat, [0, 0, 0], ox, oy, scale, viewId);
    const axes = [
      { v: [1, 0, 0], color: colors.axisX || "#e05252", label: "X" },
      { v: [0, 1, 0], color: colors.axisY || "#3cb371", label: "Y" },
      { v: [0, 0, 1], color: colors.axisZ || "#3b82f6", label: "Z" },
    ];
    ctx.lineWidth = 1.6;
    ctx.font = "9px ui-monospace, monospace";
    axes.forEach(({ v, color, label }) => {
      const tip = projectPoint(bodyMat, v, ox, oy, scale, viewId);
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(label, tip.x + 3, tip.y + 3);
    });
  }

  function drawEndMarkers(ctx, bodyMat, w, h, viewId, colors) {
    const scale = Math.min(w, h) * 0.20;
    const cx = w * 0.5;
    const cy = h * 0.54;
    [
      { p: AC_NOSE, label: "N", color: colors.nose || "#e05252" },
      { p: AC_TAIL, label: "T", color: colors.tail || "#64748b" },
    ].filter(({ p }) => viewId !== "rear" || p[0] <= REAR_CLIP_X).forEach(({ p, label, color }) => {
      const pt = projectPoint(bodyMat, p, cx, cy, scale, viewId);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "bold 10px ui-monospace, monospace";
      ctx.fillText(label, pt.x + 5, pt.y + 4);
    });
  }

  function drawWireAttitude(canvas, attitude, colors, viewId) {
    const view = ATTITUDE_VIEWS[viewId] || ATTITUDE_VIEWS.rear;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 240;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = colors.bg || "#e9edf3";
    ctx.fillRect(0, 0, w, h);

    const roll = attitude.roll;
    const pitch = attitude.pitch;
    const yaw = attitude.yaw;
    // roll→bank(rotX), pitch→nose(rotY); do not swap, do not apply yaw to mesh
    const bodyMat = buildBodyMatrix(roll, pitch, viewId);
    const lines = projectMeshLines(bodyMat, w, h, viewId).sort((a, b) => a.z - b.z);

    ctx.strokeStyle = colors.wire || "#1e40af";
    ctx.lineWidth = 0.72;
    ctx.lineCap = "round";
    ctx.globalAlpha = 0.95;
    lines.forEach(({ x1, y1, x2, y2 }) => {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    drawEndMarkers(ctx, bodyMat, w, h, viewId, colors);
    drawAxisTriad(ctx, bodyMat, w, h, colors, viewId);

    ctx.fillStyle = colors.muted || "#64748b";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`R ${roll.toFixed(1)}°`, 12, 20);
    ctx.fillText(`P ${pitch.toFixed(1)}°`, 12, 36);
    ctx.fillText(`Y ${yaw.toFixed(1)}°`, 12, 52);
    ctx.fillStyle = colors.wire || "#1e40af";
    const tag = `${view.label} · ${view.hint || ""}`;
    ctx.fillText(tag, w - 12 - ctx.measureText(tag).width, 20);
  }

  function demColor(t) {
    const stops = [
      [0.0, [20, 70, 40]],
      [0.25, [50, 120, 60]],
      [0.5, [140, 120, 70]],
      [0.75, [170, 140, 100]],
      [1.0, [240, 245, 250]],
    ];
    let c = stops[0][1];
    for (let i = 1; i < stops.length; i += 1) {
      if (t <= stops[i][0]) {
        const [t0, c0] = stops[i - 1];
        const [t1, c1] = stops[i];
        const k = (t - t0) / Math.max(t1 - t0, 1e-6);
        c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
        break;
      }
    }
    return c;
  }

  function beliefColor(t) {
    const stops = [
      [0.0, [15, 40, 90]],
      [0.25, [30, 110, 180]],
      [0.5, [40, 180, 140]],
      [0.75, [230, 180, 40]],
      [1.0, [220, 60, 40]],
    ];
    let c = stops[0][1];
    for (let i = 1; i < stops.length; i += 1) {
      if (t <= stops[i][0]) {
        const [t0, c0] = stops[i - 1];
        const [t1, c1] = stops[i];
        const k = (t - t0) / Math.max(t1 - t0, 1e-6);
        c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
        break;
      }
    }
    return c;
  }

  function valuesToDataUrl(values, vmin, vmax, colorFn) {
    const ny = values.length;
    const nx = values[0]?.length || 0;
    if (!ny || !nx) return null;
    const span = Math.max((vmax ?? 1) - (vmin ?? 0), 1e-9);
    const canvas = document.createElement("canvas");
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(nx, ny);
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const v = values[y][x];
        const t = v == null ? 0 : (v - vmin) / span;
        const rgb = colorFn(Math.max(0, Math.min(1, t)));
        const i = (y * nx + x) * 4;
        img.data[i] = rgb[0];
        img.data[i + 1] = rgb[1];
        img.data[i + 2] = rgb[2];
        img.data[i + 3] = v == null ? 0 : 175;
      }
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }

  window.createRtwindQuad = function createRtwindQuad(deps) {
    const {
      api,
      L,
      TILES,
      cartoTileOptions,
      getTheme,
      themeColors,
      getActive,
      getFrame,
      getTrack,
      getBeliefLayer,
      followEnabled,
      onMapModeEnter,
    } = deps;

    const stage = document.getElementById("stage");
    const settingsEl = document.getElementById("quadSettings");
    const btnMap = document.getElementById("btnViewMap");
    const btnQuad = document.getElementById("btnViewQuad");
    const slotSelects = [0, 1, 2, 3].map((i) => document.getElementById(`quadSlot${i}`));
    const panels = [...document.querySelectorAll(".quad-panel")].map((el) => ({
      el,
      title: el.querySelector(".quad-title"),
      body: el.querySelector(".quad-body"),
      type: null,
      map: null,
      basemap: null,
      trackLine: null,
      marker: null,
      windLayer: null,
      overlay: null,
      canvas: null,
      hud: null,
      abort: null,
      busy: false,
      lastKey: "",
      layerTimer: 0,
    }));

    const settings = loadSettings();
    let view = settings.view;
    let slots = settings.slots.slice();
    let lastFollowPan = 0;
    let lastTrackDraw = 0;
    let lastTrackSeq = -1;

    const attTarget = { roll: 0, pitch: 0, yaw: 0 };
    const attDisplay = { roll: 0, pitch: 0, yaw: 0 };
    let attAnimId = 0;
    let attLastTs = 0;

    function hasAttitudePanel() {
      return panels.some((p) => p.type === "attitude");
    }

    function syncAttTarget(frame, snap = false) {
      const next = readFrameAttitude(frame);
      attTarget.roll = next.roll;
      attTarget.pitch = next.pitch;
      attTarget.yaw = next.yaw;
      if (snap) snapAttitude(attDisplay, attTarget);
    }

    function stopAttLoop() {
      if (attAnimId) cancelAnimationFrame(attAnimId);
      attAnimId = 0;
      attLastTs = 0;
    }

    function attLoop(ts) {
      attAnimId = 0;
      if (view !== "quad" || !hasAttitudePanel()) return;
      const dt = attLastTs ? Math.min(ts - attLastTs, 50) : 16;
      attLastTs = ts;
      smoothAttitudeStep(attDisplay, attTarget, dt);
      panels.forEach((p) => {
        if (p.type === "attitude") redrawAttitude(p);
      });
      attAnimId = requestAnimationFrame(attLoop);
    }

    function manageAttLoop(snap = false) {
      syncAttTarget(getFrame(), snap);
      if (view === "quad" && hasAttitudePanel()) {
        if (!attAnimId) attAnimId = requestAnimationFrame(attLoop);
      } else {
        stopAttLoop();
      }
    }

    function fillSlotSelects() {
      slotSelects.forEach((sel, idx) => {
        if (!sel) return;
        sel.innerHTML = PANEL_ORDER.map((id) => {
          const d = PANEL_DEFS[id];
          return `<option value="${d.id}">${d.label}</option>`;
        }).join("");
        sel.value = slots[idx];
      });
    }

    function aircraftIcon() {
      return L.divIcon({
        className: "ac-marker",
        html: '<svg class="ac-icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 2 L19 21 L12 17 L5 21 Z"></path></svg>',
        iconSize: [18, 18],
        iconAnchor: [9, 9],
      });
    }

    function destroyPanel(p) {
      if (p.abort) {
        p.abort.abort();
        p.abort = null;
      }
      if (p.cameraTimer) {
        clearInterval(p.cameraTimer);
        p.cameraTimer = null;
      }
      if (p.cameraImg) {
        p.cameraImg.removeAttribute("src");
        p.cameraImg = null;
      }
      p.cameraMeta = null;
      if (p.map) {
        p.map.remove();
        p.map = null;
      }
      p.basemap = null;
      p.trackLine = null;
      p.marker = null;
      p.windLayer = null;
      p.overlay = null;
      p.canvas = null;
      p.hud = null;
      p.body.innerHTML = "";
      p.type = null;
      p.lastKey = "";
    }

    function makeMap(host) {
      const theme = getTheme();
      const map = L.map(host, {
        zoomControl: false,
        attributionControl: false,
        preferCanvas: true,
      }).setView([27.908, 112.922], 14);
      const basemap = L.tileLayer(TILES[theme] || TILES.dark, cartoTileOptions()).addTo(map);
      return { map, basemap };
    }

    function syncMarkerHeading(marker, heading) {
      const root = marker._icon;
      const el = root && root.querySelector(".ac-icon");
      if (el) el.style.transform = `rotate(${Number(heading) || 0}deg)`;
      if (root) {
        const live = getActive() === "live";
        const icon = root.querySelector(".ac-icon");
        if (icon) icon.classList.toggle("live", live);
      }
    }

    function attachMapAircraft(p, map) {
      const colors = themeColors(getTheme());
      p.trackLine = L.polyline([], {
        color: getActive() === "live" ? colors.live : colors.sim,
        weight: 3,
        opacity: 0.85,
      }).addTo(map);
      p.marker = L.marker([27.908, 112.922], { icon: aircraftIcon(), zIndexOffset: 1000 }).addTo(map);
      const track = getTrack();
      const frame = getFrame();
      if (track.length) {
        p.trackLine.setLatLngs(track.map((pt) => [pt.lat, pt.lon]));
        const last = track[track.length - 1];
        if (hasPos(last)) {
          p.marker.setLatLng([last.lat, last.lon]);
          if (frame?.heading != null) syncMarkerHeading(p.marker, frame.heading);
        }
      } else if (frame && hasPos(frame)) {
        p.marker.setLatLng([frame.lat, frame.lon]);
        syncMarkerHeading(p.marker, frame.heading);
      }
    }

    function buildTrack(p) {
      const host = document.createElement("div");
      host.className = "qmap";
      p.body.appendChild(host);
      const { map, basemap } = makeMap(host);
      p.map = map;
      p.basemap = basemap;
      attachMapAircraft(p, map);
      const track = getTrack();
      if (track.length) {
        const last = track[track.length - 1];
        if (hasPos(last)) map.setView([last.lat, last.lon], 14, { animate: false });
      }
      requestAnimationFrame(() => map.invalidateSize());
    }

    function buildWeather(p) {
      const host = document.createElement("div");
      host.className = "qmap";
      p.body.appendChild(host);
      const hud = document.createElement("div");
      hud.className = "quad-hud";
      hud.innerHTML = "<div><span>风速</span> <strong data-k=\"wind\">—</strong></div>"
        + "<div><span>风向</span> <strong data-k=\"dir\">—</strong></div>"
        + "<div><span>气温</span> <strong data-k=\"temp\">—</strong></div>"
        + "<div><span>地形</span> <strong data-k=\"dem\">—</strong></div>";
      p.body.appendChild(hud);
      const { map, basemap } = makeMap(host);
      p.map = map;
      p.basemap = basemap;
      p.windLayer = L.layerGroup().addTo(map);
      p.hud = hud;
      attachMapAircraft(p, map);
      requestAnimationFrame(() => map.invalidateSize());
    }

    function buildHeat(p, kind) {
      const host = document.createElement("div");
      host.className = "qmap";
      p.body.appendChild(host);
      const { map, basemap } = makeMap(host);
      p.map = map;
      p.basemap = basemap;
      attachMapAircraft(p, map);
      p._heatKind = kind;
      requestAnimationFrame(() => map.invalidateSize());
    }

    function buildAttitude(p) {
      const wrap = document.createElement("div");
      wrap.className = "quad-att-wrap";
      p.attView = p.attView || DEFAULT_ATTITUDE_VIEW;

      const toolbar = document.createElement("div");
      toolbar.className = "quad-att-views";
      ATTITUDE_VIEW_ORDER.forEach((id) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quad-att-view-btn";
        btn.textContent = ATTITUDE_VIEWS[id].label;
        btn.dataset.view = id;
        btn.classList.toggle("active", id === p.attView);
        btn.addEventListener("click", () => {
          p.attView = id;
          toolbar.querySelectorAll(".quad-att-view-btn").forEach((b) => {
            b.classList.toggle("active", b.dataset.view === id);
          });
          redrawAttitude(p);
        });
        toolbar.appendChild(btn);
      });

      const canvas = document.createElement("canvas");
      wrap.appendChild(toolbar);
      wrap.appendChild(canvas);
      p.body.appendChild(wrap);
      p.canvas = canvas;
      p.attToolbar = toolbar;
      manageAttLoop(true);
      redrawAttitude(p);
    }

    function stopCameraPanel(p) {
      if (p.cameraTimer) {
        clearInterval(p.cameraTimer);
        p.cameraTimer = null;
      }
      if (p.cameraImg) p.cameraImg.removeAttribute("src");
    }

    function startCameraPanel(p) {
      stopCameraPanel(p);
      if (!p.cameraImg || !p.cameraMeta) return;
      if (getActive() !== "live") {
        p.cameraMeta.textContent = "真机模式下显示机载摄像头画面";
        return;
      }
      p.cameraMeta.textContent = "连接摄像头流…";
      p.cameraImg.src = `${api("/api/camera/stream")}`;
      p.cameraTimer = setInterval(async () => {
        if (p.type !== "camera") return;
        try {
          const res = await fetch(api("/api/camera/status"), { cache: "no-store" });
          if (!res.ok) return;
          const st = await res.json();
          if (!p.cameraMeta) return;
          p.cameraMeta.textContent = st.has_frame
            ? `帧 #${st.seq} · ${st.updated_at || "—"}`
            : "等待手机上传摄像头画面…";
        } catch (_) { /* ignore */ }
      }, 2000);
    }

    function buildCamera(p) {
      const wrap = document.createElement("div");
      wrap.className = "quad-camera-wrap";
      const img = document.createElement("img");
      img.className = "quad-camera-img";
      img.alt = "机载摄像头";
      const meta = document.createElement("p");
      meta.className = "quad-camera-meta meta";
      meta.textContent = "等待画面…";
      wrap.appendChild(img);
      wrap.appendChild(meta);
      p.body.appendChild(wrap);
      p.cameraImg = img;
      p.cameraMeta = meta;
      startCameraPanel(p);
    }

    function rebuildPanel(index) {
      const p = panels[index];
      const type = slots[index];
      destroyPanel(p);
      p.type = type;
      p.title.textContent = PANEL_DEFS[type]?.label || type;
      if (type === "track") buildTrack(p);
      else if (type === "weather") buildWeather(p);
      else if (type === "belief") buildHeat(p, "belief");
      else if (type === "terrain") buildHeat(p, "terrain");
      else if (type === "attitude") buildAttitude(p);
      else if (type === "camera") buildCamera(p);
      else {
        const empty = document.createElement("div");
        empty.className = "quad-empty";
        empty.textContent = "未选择面板";
        p.body.appendChild(empty);
      }
    }

    function rebuildAll() {
      panels.forEach((_, i) => rebuildPanel(i));
      manageAttLoop(true);
      if (view === "quad") {
        requestAnimationFrame(() => {
          panels.forEach((p) => {
            if (p.map) p.map.invalidateSize();
          });
          refreshLayers(true);
        });
      }
    }

    function setView(next) {
      view = next === "quad" ? "quad" : "map";
      stage.dataset.view = view;
      if (settingsEl) settingsEl.hidden = view !== "quad";
      if (btnMap) btnMap.classList.toggle("active", view === "map");
      if (btnQuad) btnQuad.classList.toggle("active", view === "quad");
      document.getElementById("quadGrid")?.setAttribute("aria-hidden", view === "quad" ? "false" : "true");
      saveSettings({ view, slots });
      if (view === "quad") {
        rebuildAll();
      } else {
        stopAttLoop();
        panels.forEach(destroyPanel);
        requestAnimationFrame(() => {
          if (deps.mainMap) deps.mainMap.invalidateSize();
          if (typeof onMapModeEnter === "function") onMapModeEnter();
        });
      }
    }

    function setSlots(nextSlots) {
      slots = nextSlots.slice(0, 4);
      while (slots.length < 4) slots.push(DEFAULT_SLOTS[slots.length]);
      saveSettings({ view, slots });
      fillSlotSelects();
      if (view === "quad") rebuildAll();
    }

    function redrawAttitude(p) {
      if (!p.canvas) return;
      const light = getTheme() === "light";
      drawWireAttitude(p.canvas, attDisplay, {
        bg: light ? "#e9edf3" : "#141b26",
        wire: light ? "#1e40af" : "#6b9cff",
        muted: light ? "#64748b" : "#8b9bb0",
        axisX: "#e05252",
        axisY: "#3cb371",
        axisZ: "#3b82f6",
      }, p.attView || DEFAULT_ATTITUDE_VIEW);
    }

    function padBounds(lat, lon, padDeg = 0.018) {
      return {
        south: lat - padDeg,
        north: lat + padDeg,
        west: lon - padDeg,
        east: lon + padDeg,
      };
    }

    function windToBearingDeg(u, v) {
      return ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
    }

    function applyWind(p, wind) {
      if (!p.windLayer || !wind?.vectors) return;
      p.windLayer.clearLayers();
      const color = themeColors(getTheme()).wind;
      wind.vectors.forEach((v) => {
        if (v.u == null || v.v == null) return;
        const speed = Math.hypot(v.u, v.v);
        if (speed < 0.05) return;
        const bearing = windToBearingDeg(v.u, v.v);
        const html = `<svg class="wind-arrow" width="18" height="36" viewBox="0 0 40 80" style="transform:rotate(${bearing}deg);transform-origin:50% 100%">
          <line x1="20" y1="72" x2="20" y2="22" stroke="${color}" stroke-width="3" stroke-linecap="round"/>
          <path d="M20 6 L32 28 L20 22 L8 28 Z" fill="${color}"/>
        </svg>`;
        p.windLayer.addLayer(L.marker([v.lat, v.lon], {
          icon: L.divIcon({
            className: "wind-marker",
            html,
            iconSize: [18, 36],
            iconAnchor: [9, 35],
          }),
          interactive: false,
        }));
      });
    }

    function setOverlay(p, url, bounds) {
      if (p.overlay) {
        p.map.removeLayer(p.overlay);
        p.overlay = null;
      }
      if (!url || !bounds) return;
      p.overlay = L.imageOverlay(url, [[bounds.south, bounds.west], [bounds.north, bounds.east]], {
        opacity: 0.65,
        interactive: false,
      }).addTo(p.map);
    }

    async function refreshPanelLayers(p, { force = false } = {}) {
      if (view !== "quad" || !p.map || p.busy) return;
      const frame = getFrame();
      if (!frame) return;
      if (!hasPos(frame)) {
        if (p.hud) setHudNA(p.hud);
        return;
      }
      const b = padBounds(frame.lat, frame.lon, 0.02);
      const key = `${p.type}:${frame.lat.toFixed(3)},${frame.lon.toFixed(3)}:${getBeliefLayer()}`;
      if (!force && key === p.lastKey) return;
      if (p.abort) p.abort.abort();
      p.abort = new AbortController();
      const signal = p.abort.signal;
      p.busy = true;
      try {
        if (p.type === "weather") {
          const meteo = window.RtwindMeteo;
          const [wind, envDem] = await Promise.all([
            meteo
              ? meteo.fetchWindField(b.south, b.west, b.north, b.east, 6, 6, { signal })
              : Promise.reject(new Error("气象客户端未加载")),
            fetch(api(`/api/env/at?lat=${frame.lat}&lon=${frame.lon}&weather=0`), { cache: "no-store", signal })
              .then((r) => (r.ok ? r.json() : null)),
          ]);
          const wx = meteo ? await meteo.fetchWeatherAt(frame.lat, frame.lon, { signal }) : null;
          applyWind(p, wind);
          if (p.hud) {
            const set = (k, v) => {
              const el = p.hud.querySelector(`[data-k="${k}"]`);
              if (el) el.textContent = v;
            };
            set("wind", wx?.wind_speed_mps != null ? Number(wx.wind_speed_mps).toFixed(1) : "暂无");
            set("dir", wx?.wind_dir_deg != null ? `${Math.round(wx.wind_dir_deg)}°` : "暂无");
            set("temp", wx?.temperature_c != null ? `${Number(wx.temperature_c).toFixed(1)}°` : "暂无");
            set("dem", envDem?.dem_msl != null ? `${Number(envDem.dem_msl).toFixed(0)}m` : "暂无");
          }
          p.map.setView([frame.lat, frame.lon], Math.max(p.map.getZoom(), 13), { animate: false });
        } else if (p.type === "terrain") {
          const res = await fetch(
            api(`/api/env/dem?south=${b.south}&west=${b.west}&north=${b.north}&east=${b.east}&nx=20&ny=20`),
            { cache: "no-store", signal },
          );
          if (res.ok) {
            const dem = await res.json();
            const url = valuesToDataUrl(dem.values || [], dem.min, dem.max, demColor);
            setOverlay(p, url, dem.bounds);
          }
          p.map.setView([frame.lat, frame.lon], Math.max(p.map.getZoom(), 13), { animate: false });
        } else if (p.type === "belief") {
          const layer = encodeURIComponent(getBeliefLayer() || "energy");
          const res = await fetch(api(`/api/belief/field?layer=${layer}`), { cache: "no-store", signal });
          if (res.ok) {
            const field = await res.json();
            if (field.ok) {
              const url = valuesToDataUrl(field.values || [], field.min, field.max, beliefColor);
              setOverlay(p, url, field.bounds);
              if (field.center_lat != null) {
                p.map.setView([field.center_lat, field.center_lon], Math.max(p.map.getZoom(), 13), { animate: false });
              }
            }
          }
        }
        p.lastKey = key;
      } catch (err) {
        if (err.name !== "AbortError") {
          /* keep last overlay */
        }
      } finally {
        p.busy = false;
      }
    }

    function refreshLayers(force = false) {
      panels.forEach((p) => {
        if (p.type === "weather" || p.type === "belief" || p.type === "terrain") {
          refreshPanelLayers(p, { force });
        }
      });
    }

    function onFrame(frame) {
      if (view !== "quad" || !frame) return;
      syncAttTarget(frame);
      if (hasAttitudePanel() && !attAnimId) manageAttLoop();
      const colors = themeColors(getTheme());
      const now = performance.now();
      panels.forEach((p) => {
        if (p.type === "attitude") return;
        if (!p.map) return;
        if (hasPos(frame) && p.marker) {
          p.marker.setLatLng([frame.lat, frame.lon]);
          syncMarkerHeading(p.marker, frame.heading);
        }
        if (p.trackLine) {
          const track = getTrack();
          const seq = track.length ? track[track.length - 1].seq : -1;
          if (seq !== lastTrackSeq || now - lastTrackDraw >= 150) {
            lastTrackSeq = seq;
            lastTrackDraw = now;
            p.trackLine.setLatLngs(track.map((pt) => [pt.lat, pt.lon]));
            p.trackLine.setStyle({ color: getActive() === "live" ? colors.live : colors.sim });
          }
        }
        if (p.type === "track" && followEnabled() && hasPos(frame)) {
          if (!lastFollowPan || now - lastFollowPan >= 80) {
            lastFollowPan = now;
            p.map.invalidateSize({ pan: false });
            p.map.setView([frame.lat, frame.lon], p.map.getZoom(), { animate: false });
          }
        }
        const interval = p.type === "belief" ? 8000 : 15000;
        if (now - p.layerTimer > interval) {
          p.layerTimer = now;
          refreshPanelLayers(p, { force: true });
        }
      });
    }

    function onTrack(points) {
      if (view !== "quad") return;
      const pts = points || [];
      panels.forEach((p) => {
        if (p.trackLine) {
          p.trackLine.setLatLngs(pts.map((pt) => [pt.lat, pt.lon]));
          const colors = themeColors(getTheme());
          p.trackLine.setStyle({ color: getActive() === "live" ? colors.live : colors.sim });
        }
      });
    }

    function onBasemapKey() {
      if (view !== "quad") return;
      const theme = getTheme();
      panels.forEach((p) => {
        if (!p.map || !p.basemap) return;
        p.map.removeLayer(p.basemap);
        p.basemap = L.tileLayer(TILES[theme] || TILES.dark, cartoTileOptions()).addTo(p.map);
        p.basemap.bringToBack();
      });
    }

    function onTheme() {
      const theme = getTheme();
      panels.forEach((p) => {
        if (p.map && p.basemap) {
          p.map.removeLayer(p.basemap);
          p.basemap = L.tileLayer(TILES[theme] || TILES.dark, cartoTileOptions()).addTo(p.map);
          p.basemap.bringToBack();
        }
        if (p.type === "attitude") redrawAttitude(p);
      });
      if (view === "quad") refreshLayers(true);
    }

    function onBeliefTick() {
      if (view !== "quad") return;
      panels.forEach((p) => {
        if (p.type === "belief") refreshPanelLayers(p, { force: true });
      });
    }

    function onActiveChange() {
      if (view !== "quad") return;
      panels.forEach((p) => {
        if (p.type === "camera") startCameraPanel(p);
      });
    }

    // Wire UI
    fillSlotSelects();
    slotSelects.forEach((sel, idx) => {
      sel?.addEventListener("change", () => {
        const next = slots.slice();
        next[idx] = sel.value;
        setSlots(next);
      });
    });
    btnMap?.addEventListener("click", () => setView("map"));
    btnQuad?.addEventListener("click", () => setView("quad"));

    // Initial view (without destroying main map)
    stage.dataset.view = view;
    if (settingsEl) settingsEl.hidden = view !== "quad";
    if (btnMap) btnMap.classList.toggle("active", view === "map");
    if (btnQuad) btnQuad.classList.toggle("active", view === "quad");
    if (view === "quad") {
      requestAnimationFrame(() => setView("quad"));
    }

    window.addEventListener("resize", () => {
      if (view !== "quad") return;
      panels.forEach((p) => {
        if (p.map) p.map.invalidateSize();
        if (p.type === "attitude") redrawAttitude(p);
      });
    });

    return {
      get view() { return view; },
      setView,
      setSlots,
      onFrame,
      onTrack,
      onBasemapKey,
      onTheme,
      onBeliefTick,
      onActiveChange,
      refreshLayers,
      isQuad: () => view === "quad",
    };
  };
})();
