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

  // Wireframe glider/UAV in body axes: +X nose, +Y right, +Z down (NED-ish). +X nose, +Y right, +Z down (NED-ish).
  const AC_VERTS = [
    [1.6, 0, 0], // 0 nose
    [-1.2, 0, 0], // 1 tail
    [0.1, 1.4, 0.05], // 2 right wing tip
    [0.1, -1.4, 0.05], // 3 left wing tip
    [-0.15, 0.55, 0.05], // 4 right wing root
    [-0.15, -0.55, 0.05], // 5 left wing root
    [-1.05, 0.45, 0], // 6 h-stab R
    [-1.05, -0.45, 0], // 7 h-stab L
    [-1.15, 0, -0.55], // 8 v-stab top
    [0.35, 0, 0.12], // 9 canopy
    [-0.4, 0, 0.08], // 10 mid fuselage
  ];
  const AC_EDGES = [
    [0, 9], [9, 10], [10, 1], // fuselage
    [3, 5], [5, 4], [4, 2], // wing
    [5, 10], [4, 10],
    [1, 6], [1, 7], [6, 7], // h-stab
    [1, 8], // v-stab
    [0, 2], [0, 3], // leading wing lines
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

  /** Body attitude (deg) then fixed 45° isometric camera. */
  function projectAircraft(rollDeg, pitchDeg, yawDeg, w, h) {
    const roll = (rollDeg * Math.PI) / 180;
    const pitch = (pitchDeg * Math.PI) / 180;
    const yaw = (yawDeg * Math.PI) / 180;
    // Aircraft: yaw (heading) → pitch → roll (ZYX)
    const body = mulMat(rotZ(yaw), mulMat(rotY(pitch), rotX(roll)));
    // Camera: look from NE-ish at 45° elevation (classic isometric tilt)
    const cam = mulMat(rotX((-45 * Math.PI) / 180), rotZ((-45 * Math.PI) / 180));
    const view = mulMat(cam, body);
    const scale = Math.min(w, h) * 0.22;
    const cx = w * 0.5;
    const cy = h * 0.52;
    return AC_VERTS.map((v) => {
      const p = mulMatVec(view, v);
      return { x: cx + p[0] * scale, y: cy + p[1] * scale, z: p[2] };
    });
  }

  function drawWireAttitude(canvas, frame, colors) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 240;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Ground grid hint (45° plane)
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.35;
    for (let i = -4; i <= 4; i += 1) {
      const y0 = h * 0.62 + i * 10;
      ctx.beginPath();
      ctx.moveTo(w * 0.15, y0);
      ctx.lineTo(w * 0.85, y0 - 28);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const roll = frame ? Number(frame.roll) || 0 : 0;
    const pitch = frame ? Number(frame.pitch) || 0 : 0;
    const yaw = frame ? Number(frame.yaw != null ? frame.yaw : frame.heading) || 0 : 0;
    const pts = projectAircraft(roll, pitch, yaw, w, h);

    // Depth-sort edges by average z
    const edges = AC_EDGES.map(([a, b]) => ({
      a, b,
      z: (pts[a].z + pts[b].z) * 0.5,
    })).sort((u, v) => u.z - v.z);

    // Shadow on "ground"
    ctx.strokeStyle = colors.shadow;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.35;
    edges.forEach(({ a, b }) => {
      ctx.beginPath();
      ctx.moveTo(pts[a].x, h * 0.78 + (pts[a].y - h * 0.52) * 0.15);
      ctx.lineTo(pts[b].x, h * 0.78 + (pts[b].y - h * 0.52) * 0.15);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    edges.forEach(({ a, b }, idx) => {
      const mid = idx / Math.max(edges.length - 1, 1);
      ctx.strokeStyle = mid > 0.55 ? colors.accent : colors.line;
      ctx.lineWidth = mid > 0.7 ? 2.2 : 1.5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    });

    // Nose accent dot
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, 3.2, 0, Math.PI * 2);
    ctx.fill();

    // Labels
    ctx.fillStyle = colors.muted;
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`R ${roll.toFixed(1)}°`, 12, 20);
    ctx.fillText(`P ${pitch.toFixed(1)}°`, 12, 36);
    ctx.fillText(`Y ${yaw.toFixed(1)}°`, 12, 52);
    ctx.fillText("45° wireframe", w - 100, 20);
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
      const basemap = L.tileLayer(TILES[theme] || TILES.dark, { maxZoom: 19 }).addTo(map);
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

    function buildTrack(p) {
      const host = document.createElement("div");
      host.className = "qmap";
      p.body.appendChild(host);
      const { map, basemap } = makeMap(host);
      const colors = themeColors(getTheme());
      const trackLine = L.polyline([], {
        color: getActive() === "live" ? colors.live : colors.sim,
        weight: 3,
        opacity: 0.9,
      }).addTo(map);
      const marker = L.marker([27.908, 112.922], { icon: aircraftIcon(), zIndexOffset: 1000 }).addTo(map);
      p.map = map;
      p.basemap = basemap;
      p.trackLine = trackLine;
      p.marker = marker;
      const track = getTrack();
      if (track.length) {
        trackLine.setLatLngs(track.map((pt) => [pt.lat, pt.lon]));
        const last = track[track.length - 1];
        marker.setLatLng([last.lat, last.lon]);
        map.setView([last.lat, last.lon], 14, { animate: false });
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
      p.marker = L.marker([27.908, 112.922], { icon: aircraftIcon(), zIndexOffset: 1000 }).addTo(map);
      requestAnimationFrame(() => map.invalidateSize());
    }

    function buildHeat(p, kind) {
      const host = document.createElement("div");
      host.className = "qmap";
      p.body.appendChild(host);
      const { map, basemap } = makeMap(host);
      p.map = map;
      p.basemap = basemap;
      p.marker = L.marker([27.908, 112.922], { icon: aircraftIcon(), zIndexOffset: 1000 }).addTo(map);
      p._heatKind = kind;
      requestAnimationFrame(() => map.invalidateSize());
    }

    function buildAttitude(p) {
      const wrap = document.createElement("div");
      wrap.className = "quad-att-wrap";
      const canvas = document.createElement("canvas");
      wrap.appendChild(canvas);
      p.body.appendChild(wrap);
      p.canvas = canvas;
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
      const colors = themeColors(getTheme());
      drawWireAttitude(p.canvas, getFrame(), {
        accent: colors.sim,
        line: getTheme() === "light" ? "#334155" : "#c5d0de",
        muted: getTheme() === "light" ? "#64748b" : "#8b9bb0",
        grid: getTheme() === "light" ? "#94a3b8" : "#2a3548",
        shadow: getTheme() === "light" ? "#64748b" : "#0a1018",
      });
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
          const [windRes, envRes] = await Promise.all([
            fetch(api(`/api/env/wind-field?south=${b.south}&west=${b.west}&north=${b.north}&east=${b.east}&nx=6&ny=6`), { cache: "no-store", signal }),
            fetch(api(`/api/env/at?lat=${frame.lat}&lon=${frame.lon}`), { cache: "no-store", signal }),
          ]);
          if (windRes.ok) applyWind(p, await windRes.json());
          if (envRes.ok && p.hud) {
            const env = await envRes.json();
            const set = (k, v) => {
              const el = p.hud.querySelector(`[data-k="${k}"]`);
              if (el) el.textContent = v;
            };
            set("wind", env.wind_speed_mps != null ? Number(env.wind_speed_mps).toFixed(1) : "暂无");
            set("dir", env.wind_dir_deg != null ? `${Math.round(env.wind_dir_deg)}°` : "暂无");
            set("temp", env.temperature_c != null ? `${Number(env.temperature_c).toFixed(1)}°` : "暂无");
            set("dem", env.dem_msl != null ? `${Number(env.dem_msl).toFixed(0)}m` : "暂无");
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
      const colors = themeColors(getTheme());
      const now = performance.now();
      panels.forEach((p) => {
        if (p.type === "attitude") {
          redrawAttitude(p);
          return;
        }
        if (!p.map) return;
        if (hasPos(frame) && p.marker) {
          p.marker.setLatLng([frame.lat, frame.lon]);
          syncMarkerHeading(p.marker, frame.heading);
        }
        if (p.trackLine) {
          const track = getTrack();
          p.trackLine.setLatLngs(track.map((pt) => [pt.lat, pt.lon]));
          p.trackLine.setStyle({ color: getActive() === "live" ? colors.live : colors.sim });
        }
        if (p.type === "track" && followEnabled() && hasPos(frame)) {
          if (!lastFollowPan || now - lastFollowPan >= 80) {
            lastFollowPan = now;
            p.map.invalidateSize({ pan: false });
            p.map.setView([frame.lat, frame.lon], p.map.getZoom(), { animate: false });
          }
        }
        const interval = p.type === "belief" ? 4000 : 10000;
        if (now - p.layerTimer > interval) {
          p.layerTimer = now;
          refreshPanelLayers(p, { force: true });
        }
      });
    }

    function onTrack(points) {
      if (view !== "quad") return;
      panels.forEach((p) => {
        if (p.trackLine) {
          p.trackLine.setLatLngs((points || []).map((pt) => [pt.lat, pt.lon]));
        }
      });
    }

    function onTheme() {
      const theme = getTheme();
      panels.forEach((p) => {
        if (p.map && p.basemap) {
          p.map.removeLayer(p.basemap);
          p.basemap = L.tileLayer(TILES[theme] || TILES.dark, { maxZoom: 19 }).addTo(p.map);
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
      onTheme,
      onBeliefTick,
      onActiveChange,
      refreshLayers,
      isQuad: () => view === "quad",
    };
  };
})();
