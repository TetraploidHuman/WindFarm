(() => {
  const prefix = (() => {
    const basePath = new URL(document.baseURI || window.location.href).pathname
      .replace(/\/index\.html$/i, "")
      .replace(/\/$/, "");
    let p = basePath === "/" ? "" : basePath;
    if (p.endsWith("/static")) p = p.slice(0, -"/static".length);
    return p;
  })();
  const api = (path) => `${prefix}${path}`;

  const els = {
    badge: document.getElementById("sourceBadge"),
    btnSim: document.getElementById("btnSim"),
    btnLive: document.getElementById("btnLive"),
    btnResetSim: document.getElementById("btnResetSim"),
    simLat: document.getElementById("simLat"),
    simLon: document.getElementById("simLon"),
    btnApplySimOrigin: document.getElementById("btnApplySimOrigin"),
    btnGeoSim: document.getElementById("btnGeoSim"),
    pickOrigin: document.getElementById("pickOrigin"),
    simMeta: document.getElementById("simMeta"),
    simPanel: document.getElementById("simPanel"),
    follow: document.getElementById("follow"),
    layerDem: document.getElementById("layerDem"),
    layerWind: document.getElementById("layerWind"),
    layerBelief: document.getElementById("layerBelief"),
    beliefLayer: document.getElementById("beliefLayer"),
    layerMeta: document.getElementById("layerMeta"),
    btnPrefetchRegion: document.getElementById("btnPrefetchRegion"),
    prefetchMeta: document.getElementById("prefetchMeta"),
    prefetchNotice: document.getElementById("prefetchNotice"),
    prefetchNoticeText: document.getElementById("prefetchNoticeText"),
    btnPrefetchAccept: document.getElementById("btnPrefetchAccept"),
    btnPrefetchDismiss: document.getElementById("btnPrefetchDismiss"),
    prefetchProgress: document.getElementById("prefetchProgress"),
    prefetchBarFill: document.getElementById("prefetchBarFill"),
    prefetchProgressText: document.getElementById("prefetchProgressText"),
    mAir: document.getElementById("mAir"),
    mGs: document.getElementById("mGs"),
    mMsl: document.getElementById("mMsl"),
    mAgl: document.getElementById("mAgl"),
    mHdg: document.getElementById("mHdg"),
    mClimb: document.getElementById("mClimb"),
    mRoll: document.getElementById("mRoll"),
    mPitch: document.getElementById("mPitch"),
    mYaw: document.getElementById("mYaw"),
    mLink: document.getElementById("mLink"),
    attBall: document.getElementById("attBall"),
    eDem: document.getElementById("eDem"),
    eWind: document.getElementById("eWind"),
    eDir: document.getElementById("eDir"),
    eTemp: document.getElementById("eTemp"),
    eMeta: document.getElementById("eMeta"),
    statusBar: document.getElementById("statusBar"),
    layerPerception: document.getElementById("layerPerception"),
    layerCognition: document.getElementById("layerCognition"),
    layerPlanning: document.getElementById("layerPlanning"),
    cObs: document.getElementById("cObs"),
    cUnc: document.getElementById("cUnc"),
    cConf: document.getElementById("cConf"),
    pLambda: document.getElementById("pLambda"),
    pMs: document.getElementById("pMs"),
    pMode: document.getElementById("pMode"),
    planMeta: document.getElementById("planMeta"),
    hudPos: document.getElementById("hudPos"),
    hudLambda: document.getElementById("hudLambda"),
    hudSeq: document.getElementById("hudSeq"),
    clock: document.getElementById("clock"),
    altChart: document.getElementById("altChart"),
    spdChart: document.getElementById("spdChart"),
    btnTheme: document.getElementById("btnTheme"),
  };

  const state = {
    active: "sim",
    frame: null,
    track: [],
    follow: true,
    envTimer: 0,
    envReq: 0,
    layerTimer: 0,
    layerFetch: null,
    layerAbort: null,
    layerBusy: false,
    layerBoundsKey: "",
    lastWind: null,
    ignoreMoveEnd: 0,
    lastFollowPanAt: 0,
    altHist: [],
    spdHist: [],
    simOrigin: { lat: 27.908, lon: 112.922, radius_m: 450 },
    missionRegion: null,
    livePrefetchAsked: false,
    livePrefetchPending: null,
    prefetchBusy: false,
    plan: {
      ready: false,
      planned: [],
      baseline: [],
      lambda: 0,
      planning_ms: null,
      planning_mode: null,
      last_error: null,
    },
    systemLayersTimer: 0,
    cartoBasemapKey: "",
    lastDemMsl: null,
  };

  const MISSION_SIZE_KM = 10;

  function isMapView() {
    const stage = document.getElementById("stage");
    return !stage || stage.dataset.view !== "quad";
  }

  function missionCenter() {
    if (state.active === "sim") {
      return { lat: state.simOrigin.lat, lon: state.simOrigin.lon };
    }
    if (state.frame && hasPos(state.frame)) {
      return { lat: state.frame.lat, lon: state.frame.lon };
    }
    const c = map.getCenter();
    return { lat: c.lat, lon: c.lng };
  }

  function hasPos(frame) {
    if (!frame) return false;
    const { lat, lon } = frame;
    if (lat == null || lon == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) {
      return false;
    }
    return !(Math.abs(lat) < 1e-6 && Math.abs(lon) < 1e-6);
  }

  function fmtPos(lat, lon) {
    if (lat == null || lon == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) {
      return "暂无";
    }
    return `${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)}`;
  }

  function snapFollow(force = false) {
    if (!els.follow?.checked || !isMapView() || !state.frame || !hasPos(state.frame)) return;
    const now = performance.now();
    if (!force && state.lastFollowPanAt && now - state.lastFollowPanAt < 80) return;
    state.lastFollowPanAt = now;
    state.ignoreMoveEnd = performance.now() + 400;
    map.invalidateSize({ pan: false });
    map.setView([state.frame.lat, state.frame.lon], map.getZoom(), { animate: false });
  }

  const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([27.908, 112.922], 14);

  function buildCartoTiles(key) {
    const q = key ? `?key=${encodeURIComponent(key)}` : "";
    return {
      dark: `https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png${q}`,
      light: `https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png${q}`,
    };
  }

  function cartoTileOptions() {
    return {
      maxZoom: 20,
      subdomains: "abcd",
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, © <a href="https://carto.com/attributions">CARTO</a>',
    };
  }

  const TILES = buildCartoTiles("");
  let basemap = L.tileLayer(TILES.dark, cartoTileOptions()).addTo(map);
  let basemapTheme = "dark";

  function refreshBasemapLayer(theme = basemapTheme) {
    map.removeLayer(basemap);
    basemap = L.tileLayer(TILES[theme] || TILES.dark, cartoTileOptions()).addTo(map);
    basemap.bringToBack();
    basemapTheme = theme;
  }

  function applyCartoBasemapKey(key) {
    const next = String(key || "").trim();
    if (!next || next === state.cartoBasemapKey) return;
    state.cartoBasemapKey = next;
    Object.assign(TILES, buildCartoTiles(next));
    refreshBasemapLayer(currentTheme());
    if (window.__rtwindQuad) window.__rtwindQuad.onBasemapKey();
  }

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function themeColors(theme) {
    if (theme === "light") {
      return { sim: "#0d9488", live: "#e11d48", wind: "#1d4ed8", alt: "#2563eb" };
    }
    return { sim: "#3dd6c6", live: "#ff6b4a", wind: "#8ec5ff", alt: "#5b8cff" };
  }

  function applyTheme(theme, { persist = true } = {}) {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    if (persist) {
      try { localStorage.setItem("rtwind-theme", next); } catch (_) { /* ignore */ }
    }
    if (basemapTheme !== next) {
      refreshBasemapLayer(next);
    }
    const colors = themeColors(next);
    trackLine.setStyle({ color: state.active === "live" ? colors.live : colors.sim });
    if (typeof originOrbit !== "undefined" && originOrbit) {
      originOrbit.setStyle({ color: colors.sim });
    }
    if (els.btnTheme) {
      els.btnTheme.title = next === "dark" ? "切换到明亮模式" : "切换到暗黑模式";
      els.btnTheme.setAttribute("aria-label", els.btnTheme.title);
    }
    if (window.__rtwindQuad) window.__rtwindQuad.onTheme();
  }

  const trackLine = L.polyline([], { color: "#3dd6c6", weight: 3, opacity: 0.85 }).addTo(map);
  const aircraftIcon = L.divIcon({
    className: "ac-marker",
    html: '<svg class="ac-icon" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M12 2 L19 21 L12 17 L5 21 Z"></path></svg>',
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
  const marker = L.marker([27.908, 112.922], { icon: aircraftIcon, zIndexOffset: 1000 }).addTo(map);

  function aircraftGlyphEl() {
    const root = marker._icon;
    return root ? root.querySelector(".ac-icon") : null;
  }

  function setAircraftHeading(heading) {
    const el = aircraftGlyphEl();
    if (el) el.style.transform = `rotate(${Number(heading) || 0}deg)`;
  }

  function movementBearing(prev, frame) {
    if (!prev || !hasPos(prev) || !hasPos(frame)) return frame.heading;
    const dLat = frame.lat - prev.lat;
    const dLon = frame.lon - prev.lon;
    if (Math.hypot(dLat, dLon) < 1e-10) return frame.heading;
    const rad = (frame.lat * Math.PI) / 180;
    return ((Math.atan2(dLon * Math.cos(rad), dLat) * 180) / Math.PI + 360) % 360;
  }

  let demOverlay = null;
  let beliefOverlay = null;
  const windLayer = L.layerGroup().addTo(map);
  const originIcon = L.divIcon({
    className: "",
    html: '<div class="sim-origin-dot" title="盘旋中心"></div>',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
  const originMarker = L.marker([27.908, 112.922], { icon: originIcon, interactive: false, zIndexOffset: -200 });
  let originOrbit = L.circle([27.908, 112.922], { radius: 450, color: "#3dd6c6", weight: 1, opacity: 0.45, fillOpacity: 0.04 });

  function updateOriginVisual(lat, lon, radiusM) {
    state.simOrigin = { lat, lon, radius_m: radiusM };
    els.simLat.value = Number(lat).toFixed(5);
    els.simLon.value = Number(lon).toFixed(5);
    els.simMeta.textContent = `盘旋中心 ${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)} · r=${Math.round(radiusM)}m`;
    originMarker.setLatLng([lat, lon]);
    originOrbit.setLatLng([lat, lon]);
    originOrbit.setRadius(radiusM);
    if (state.active === "sim") {
      if (!map.hasLayer(originMarker)) originMarker.addTo(map);
      if (!map.hasLayer(originOrbit)) originOrbit.addTo(map);
    }
  }

  function setPickOriginMode(on) {
    els.pickOrigin.checked = on;
    document.getElementById("map").classList.toggle("pick-origin", on && state.active === "sim");
  }

  async function setSimOrigin(lat, lon, { pan = true, clearTrack = true } = {}) {
    const res = await fetch(api("/api/sim/control"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "reset", lat, lon }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "模拟重置失败");
    updateOriginVisual(data.origin_lat, data.origin_lon, data.radius_m);
    if (clearTrack) {
      applyTrack([]);
      state.altHist = [];
      state.spdHist = [];
    }
    if (pan) map.setView([lat, lon], Math.max(map.getZoom(), 14));
    refreshEnv(lat, lon, null);
    scheduleLayerRefresh(300, { force: true });
    return data;
  }

  function fmt(n, d = 1) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "暂无";
    return Number(n).toFixed(d);
  }

  function linkLabel(link) {
    const map = {
      ok: "正常",
      waiting: "等待中",
      stale: "超时",
      offline: "离线",
      connecting: "连接中",
      reconnecting: "重连中",
    };
    return map[link] || link || "—";
  }

  function sourceLabel(active) {
    return active === "live" ? "真机" : "模拟";
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
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function beliefColor(t) {
    // Cool → hot: low belief energy / prob to high.
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
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function mapBoundsQuery({ pad = 0.2 } = {}) {
    const b = map.getBounds();
    let south = b.getSouth();
    let west = b.getWest();
    let north = b.getNorth();
    let east = b.getEast();
    // Expand beyond visible edges so layers cover the full viewport.
    const dLat = (north - south) * pad;
    const dLon = (east - west) * pad;
    south -= dLat;
    north += dLat;
    west -= dLon;
    east += dLon;
    return { south, west, north, east };
  }

  function layerGridSize() {
    // More arrows when zoomed out; fewer/sharper when zoomed in.
    const z = map.getZoom();
    if (z <= 10) return { windNx: 10, windNy: 10, demNx: 28, demNy: 28 };
    if (z <= 12) return { windNx: 8, windNy: 8, demNx: 24, demNy: 24 };
    if (z <= 14) return { windNx: 7, windNy: 7, demNx: 20, demNy: 20 };
    if (z <= 16) return { windNx: 6, windNy: 6, demNx: 16, demNy: 16 };
    return { windNx: 5, windNy: 5, demNx: 14, demNy: 14 };
  }

  function anyMapLayerOn() {
    return (
      els.layerDem.checked
      || els.layerWind.checked
      || (els.layerBelief && els.layerBelief.checked)
    );
  }

  function scheduleLayerRefresh(delayMs = 600, { force = false } = {}) {
    if (state.layerFetch) clearTimeout(state.layerFetch);
    state.layerFetch = setTimeout(() => refreshMapLayers({ force }), delayMs);
  }

  function boundsKey(b) {
    // Coarser key: refresh when view moves ~1–2% of span, not every pan pixel.
    const latSpan = Math.max(b.north - b.south, 1e-6);
    const lonSpan = Math.max(b.east - b.west, 1e-6);
    const qLat = Math.round(b.south / (latSpan * 0.15));
    const qLon = Math.round(b.west / (lonSpan * 0.15));
    const qZoom = Math.round(map.getZoom());
    return `${qZoom}:${qLat},${qLon}:${latSpan.toFixed(4)}x${lonSpan.toFixed(4)}`;
  }

  async function refreshMapLayers({ force = false } = {}) {
    const wantDem = els.layerDem.checked;
    const wantWind = els.layerWind.checked;
    const wantBelief = els.layerBelief && els.layerBelief.checked;
    if (!wantDem && !wantWind && !wantBelief) {
      if (demOverlay) {
        map.removeLayer(demOverlay);
        demOverlay = null;
      }
      if (beliefOverlay) {
        map.removeLayer(beliefOverlay);
        beliefOverlay = null;
      }
      windLayer.clearLayers();
      els.layerMeta.textContent = "图层已关闭";
      return;
    }
    const bounds = mapBoundsQuery({ pad: 0.22 });
    const key = boundsKey(bounds) + (wantBelief ? `:b:${els.beliefLayer?.value || "energy"}` : "");
    if (!force && key === state.layerBoundsKey && (demOverlay || beliefOverlay || windLayer.getLayers().length)) {
      return;
    }
    if (state.layerBusy && !force) {
      scheduleLayerRefresh(1500, { force: true });
      return;
    }
    if (state.layerAbort) state.layerAbort.abort();
    state.layerAbort = new AbortController();
    const signal = state.layerAbort.signal;
    const { south, west, north, east } = bounds;
    const grid = layerGridSize();
    state.layerBusy = true;
    els.layerMeta.textContent = "图层加载中…";
    const bits = [];
    try {
      if (wantWind) {
        if (!window.RtwindMeteo) throw new Error("气象客户端未加载");
        const wind = await window.RtwindMeteo.fetchWindField(
          south, west, north, east, grid.windNx, grid.windNy, { signal },
        );
        applyWindLayer(wind);
        bits.push(wind.missing > 0 ? `风场(缺${wind.missing})` : "风场");
        els.layerMeta.textContent = `${bits.join(" + ")} 已更新…`;
      } else {
        windLayer.clearLayers();
      }
      if (wantDem) {
        const res = await fetch(
          api(`/api/env/dem?south=${south}&west=${west}&north=${north}&east=${east}&nx=${grid.demNx}&ny=${grid.demNy}`),
          { cache: "no-store", signal },
        );
        if (!res.ok) throw new Error(`地形 HTTP ${res.status}`);
        const dem = await res.json();
        applyDemLayer(dem);
        bits.push("地形");
        if (dem.missing > 0) {
          bits.push(`缺${dem.missing}点`);
          if (els.prefetchMeta) {
            els.prefetchMeta.textContent = `任务区内仍有 ${dem.missing} 个地形点缺失，请「预下载任务区」`;
          }
        }
      } else if (demOverlay) {
        map.removeLayer(demOverlay);
        demOverlay = null;
      }
      if (wantBelief) {
        const layer = encodeURIComponent(els.beliefLayer?.value || "energy");
        const res = await fetch(api(`/api/belief/field?layer=${layer}`), { cache: "no-store", signal });
        if (!res.ok) throw new Error(`信念 HTTP ${res.status}`);
        const data = await res.json();
        if (data.ok) {
          applyBeliefLayer(data);
          bits.push(`信念(${data.layer}${data.obs_count != null ? `·${data.obs_count}` : ""})`);
        } else {
          if (beliefOverlay) {
            map.removeLayer(beliefOverlay);
            beliefOverlay = null;
          }
          bits.push("信念(待锚定)");
        }
      } else if (beliefOverlay) {
        map.removeLayer(beliefOverlay);
        beliefOverlay = null;
      }
      state.layerBoundsKey = key;
      const dirHint = wantWind && state.lastWind?.vectors?.[0]?.dir_deg != null
        ? ` · 来向${Math.round(state.lastWind.vectors[0].dir_deg)}°/箭头=吹向`
        : "";
      els.layerMeta.textContent = bits.length ? `${bits.join(" + ")} 已更新${dirHint}` : "图层待机";
    } catch (err) {
      if (err.name === "AbortError") return;
      els.layerMeta.textContent = `图层错误: ${err.message}`;
    } finally {
      state.layerBusy = false;
    }
  }

  function applyDemLayer(dem) {
    const b = dem.bounds;
    const ny = dem.values.length;
    const nx = dem.values[0]?.length || 0;
    if (!ny || !nx) return;
    const vmin = dem.min ?? 0;
    const vmax = dem.max ?? (vmin + 1);
    const span = Math.max(vmax - vmin, 1);
    const canvas = document.createElement("canvas");
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(nx, ny);
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const v = dem.values[y][x];
        const t = v == null ? 0 : (v - vmin) / span;
        const rgb = demColor(t).match(/\d+/g).map(Number);
        const i = (y * nx + x) * 4;
        img.data[i] = rgb[0];
        img.data[i + 1] = rgb[1];
        img.data[i + 2] = rgb[2];
        img.data[i + 3] = v == null ? 0 : 170;
      }
    }
    ctx.putImageData(img, 0, 0);
    const bounds = [[b.south, b.west], [b.north, b.east]];
    if (demOverlay) map.removeLayer(demOverlay);
    demOverlay = L.imageOverlay(canvas.toDataURL(), bounds, { opacity: 0.55, interactive: false, pane: "overlayPane", zIndex: 200 });
    demOverlay.addTo(map);
    if (typeof demOverlay.bringToFront === "function") demOverlay.bringToFront();
  }

  function applyBeliefLayer(field) {
    const b = field.bounds;
    const values = field.values || [];
    const ny = values.length;
    const nx = values[0]?.length || 0;
    if (!b || !ny || !nx) return;
    const vmin = field.min ?? 0;
    const vmax = field.max ?? (vmin + 1);
    const span = Math.max(vmax - vmin, 1e-9);
    const canvas = document.createElement("canvas");
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(nx, ny);
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const v = values[y][x];
        const t = v == null ? 0 : (v - vmin) / span;
        const rgb = beliefColor(t).match(/\d+/g).map(Number);
        const i = (y * nx + x) * 4;
        img.data[i] = rgb[0];
        img.data[i + 1] = rgb[1];
        img.data[i + 2] = rgb[2];
        // Fade near-zero cells so DEM can show through.
        const alpha = v == null ? 0 : Math.round(40 + 150 * Math.min(1, Math.abs(t - 0.5) * 2 + 0.25));
        img.data[i + 3] = alpha;
      }
    }
    ctx.putImageData(img, 0, 0);
    const bounds = [[b.south, b.west], [b.north, b.east]];
    if (beliefOverlay) map.removeLayer(beliefOverlay);
    beliefOverlay = L.imageOverlay(canvas.toDataURL(), bounds, {
      opacity: 0.72,
      interactive: false,
      pane: "overlayPane",
      zIndex: 350,
    });
    beliefOverlay.addTo(map);
    if (typeof beliefOverlay.bringToFront === "function") beliefOverlay.bringToFront();
  }

  function windToBearingDeg(u, v) {
    // Aviation/map bearing: 0=north, 90=east — direction wind blows TO.
    return ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
  }

  function applyWindLayer(wind) {
    state.lastWind = wind;
    windLayer.clearLayers();
    if (!wind || !wind.vectors) return;
    const color = themeColors(currentTheme()).wind;
    const view = map.getBounds().pad(0.08);
    const nx = Math.max(wind.nx || 2, 2);
    const ny = Math.max(wind.ny || 2, 2);
    const b = wind.bounds || {};
    const cellLat = Math.max(((b.north ?? 0) - (b.south ?? 0)) / (ny - 1), 1e-6);

    // Whole-arrow size in screen pixels (scales with zoom).
    const mid = map.getCenter();
    const p0 = map.latLngToLayerPoint(mid);
    const cellPx = Math.abs(
      map.latLngToLayerPoint([mid.lat + cellLat, mid.lng]).y - p0.y,
    );
    const arrowPx = Math.round(Math.min(Math.max(cellPx * 0.7, 22), 56));
    const w = Math.round(arrowPx * 0.45);
    const h = arrowPx;
    // Anchor at shaft base (bottom center); SVG points north before rotate.
    const anchorX = w / 2;
    const anchorY = h - 1;

    wind.vectors.forEach((v) => {
      if (v.u == null || v.v == null) return;
      if (!view.contains([v.lat, v.lon])) return;
      const speed = Math.hypot(v.u, v.v);
      if (speed < 0.05) return;
      const bearing = windToBearingDeg(v.u, v.v);
      const stroke = Math.max(1.5, arrowPx / 18);
      const html = `<svg class="wind-arrow" width="${w}" height="${h}" viewBox="0 0 40 80" style="transform:rotate(${bearing}deg);transform-origin:50% 100%" aria-hidden="true">
        <line x1="20" y1="72" x2="20" y2="22" stroke="${color}" stroke-width="${stroke * 2.2}" stroke-linecap="round"/>
        <path d="M20 6 L32 28 L20 22 L8 28 Z" fill="${color}"/>
      </svg>`;
      const mark = L.marker([v.lat, v.lon], {
        icon: L.divIcon({
          className: "wind-marker",
          html,
          iconSize: [w, h],
          iconAnchor: [anchorX, anchorY],
        }),
        interactive: false,
      });
      windLayer.addLayer(mark);
    });
  }

  function redrawWindForZoom() {
    if (state.lastWind && els.layerWind.checked) applyWindLayer(state.lastWind);
  }

  function setActiveUi(active) {
    const prev = state.active;
    state.active = active;
    els.btnSim.classList.toggle("active", active === "sim");
    els.btnLive.classList.toggle("active", active === "live");
    els.simPanel.style.display = active === "sim" ? "" : "none";
    const glyph = aircraftGlyphEl();
    if (glyph) glyph.classList.toggle("live", active === "live");
    const colors = themeColors(currentTheme());
    trackLine.setStyle({ color: active === "live" ? colors.live : colors.sim });
    if (active === "sim") {
      updateOriginVisual(state.simOrigin.lat, state.simOrigin.lon, state.simOrigin.radius_m);
      dismissLivePrefetchNotice();
    } else {
      map.removeLayer(originMarker);
      map.removeLayer(originOrbit);
      setPickOriginMode(false);
      if (prev !== "live") {
        state.livePrefetchAsked = false;
      }
    }
    document.getElementById("map").classList.toggle("pick-origin", els.pickOrigin.checked && active === "sim");
    if (window.__rtwindQuad) window.__rtwindQuad.onActiveChange();
  }

  function setBadge(frame) {
    const link = frame?.link || "waiting";
    els.badge.className = `badge ${state.active}${link === "stale" ? " stale" : ""}`;
    els.badge.textContent = `${sourceLabel(state.active)} · ${linkLabel(link)}`;
  }

  function pushHist(arr, value, max = 80) {
    arr.push(value);
    if (arr.length > max) arr.shift();
  }

  function drawSpark(canvas, values, color) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (values.length < 2) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 1e-3);
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * (w - 4) + 2;
      const y = h - 4 - ((v - min) / span) * (h - 8);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function setPrefetchProgress(pct, message) {
    if (els.prefetchProgress) els.prefetchProgress.hidden = false;
    if (els.prefetchNotice) els.prefetchNotice.hidden = true;
    if (els.prefetchBarFill) els.prefetchBarFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (els.prefetchProgressText) els.prefetchProgressText.textContent = message || "下载中…";
  }

  function hidePrefetchProgress() {
    if (els.prefetchProgress) els.prefetchProgress.hidden = true;
  }

  function showLivePrefetchNotice(lat, lon) {
    state.livePrefetchPending = { lat, lon };
    if (els.prefetchNoticeText) {
      els.prefetchNoticeText.textContent =
        `已定位 ${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)}。是否预下载以飞机为中心的 ${MISSION_SIZE_KM}×${MISSION_SIZE_KM} km 地形与气象？`;
    }
    if (els.prefetchNotice) {
      els.prefetchNotice.hidden = false;
      try {
        els.prefetchNotice.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } catch (_) { /* ignore */ }
    }
    hidePrefetchProgress();
  }

  function dismissLivePrefetchNotice() {
    if (els.prefetchNotice) els.prefetchNotice.hidden = true;
    state.livePrefetchPending = null;
  }

  function maybeAskLivePrefetch(frame) {
    if (!frame || frame.source !== "live") return;
    if (state.livePrefetchAsked || state.prefetchBusy) return;
    if (!hasPos(frame)) return;
    // Already have a nearby prefetched region — skip prompt.
    if (state.missionRegion?.center_lat != null) {
      const dLat = Math.abs(state.missionRegion.center_lat - frame.lat);
      const dLon = Math.abs(state.missionRegion.center_lon - frame.lon);
      if (dLat < 0.02 && dLon < 0.02) {
        state.livePrefetchAsked = true;
        return;
      }
    }
    state.livePrefetchAsked = true;
    showLivePrefetchNotice(frame.lat, frame.lon);
  }

  async function runPrefetchStream(center, { fromNotice = false } = {}) {
    if (state.prefetchBusy) return null;
    state.prefetchBusy = true;
    if (els.btnPrefetchRegion) els.btnPrefetchRegion.disabled = true;
    if (fromNotice) dismissLivePrefetchNotice();
    setPrefetchProgress(0, `开始预下载 ${MISSION_SIZE_KM}×${MISSION_SIZE_KM} km…`);
    if (els.prefetchMeta) {
      els.prefetchMeta.textContent =
        `正在预下载（中心 ${center.lat.toFixed(5)}, ${center.lon.toFixed(5)}）…`;
    }
    let finalResult = null;
    try {
      const res = await fetch(api("/api/env/prefetch/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
        body: JSON.stringify({
          center_lat: center.lat,
          center_lon: center.lon,
          size_km: MISSION_SIZE_KM,
          weather_nx: 11,
          weather_ny: 11,
          dem: true,
          weather: false,
        }),
      });
      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          let ev;
          try {
            ev = JSON.parse(line);
          } catch (_) {
            continue;
          }
          const pct = Number(ev.pct) || 0;
          setPrefetchProgress(pct, ev.message || "下载中…");
          if (ev.type === "done") {
            finalResult = ev;
            if (ev.error) throw new Error(ev.error);
          }
        }
      }
      if (!finalResult) throw new Error("未收到完成事件");
      state.missionRegion = finalResult;
      const demOk = finalResult.dem?.ok ?? 0;
      const demFail = finalResult.dem?.failed ?? 0;
      const wOk = finalResult.weather?.ok ?? 0;
      const wFail = finalResult.weather?.failed ?? 0;
      const tip = `${MISSION_SIZE_KM}×${MISSION_SIZE_KM} km · 地形 ${demOk}/${demOk + demFail} 瓦片 · 气象 ${wOk}/${wOk + wFail} 点`
        + (finalResult.ok ? " · 完成" : (finalResult.partial ? " · 部分完成" : " · 部分失败"));
      setPrefetchProgress(100, tip);
      if (els.prefetchMeta) els.prefetchMeta.textContent = tip;
      scheduleLayerRefresh(80, { force: true });
      refreshSystemLayers(true);
      if (window.__rtwindQuad && window.__rtwindQuad.isQuad()) {
        window.__rtwindQuad.refreshLayers(true);
      }
      setTimeout(() => hidePrefetchProgress(), 2500);
      return finalResult;
    } catch (err) {
      setPrefetchProgress(100, `预下载失败: ${err.message}`);
      if (els.prefetchMeta) els.prefetchMeta.textContent = `预下载失败: ${err.message}`;
      throw err;
    } finally {
      state.prefetchBusy = false;
      if (els.btnPrefetchRegion) els.btnPrefetchRegion.disabled = false;
    }
  }

  function updateAglDisplay(frame) {
    if (!frame) return;
    if (frame.alt_agl != null) {
      els.mAgl.textContent = fmt(frame.alt_agl, 1);
      return;
    }
    const msl = frame.alt_msl;
    if (msl != null && state.lastDemMsl != null) {
      els.mAgl.textContent = fmt(msl - state.lastDemMsl, 1);
      return;
    }
    els.mAgl.textContent = "暂无";
  }

  function applyFrame(frame) {
    if (!frame) return;
    const prev = state.frame;
    state.frame = frame;
    setBadge(frame);
    els.mAir.textContent = fmt(frame.airspeed, 1);
    els.mGs.textContent = fmt(frame.groundspeed, 1);
    els.mMsl.textContent = fmt(frame.alt_msl, 1);
    updateAglDisplay(frame);
    els.mHdg.textContent = fmt(frame.heading, 0);
    els.mClimb.textContent = fmt(frame.climb_rate, 2);
    els.mRoll.textContent = fmt(frame.roll, 1);
    els.mPitch.textContent = fmt(frame.pitch, 1);
    els.mYaw.textContent = fmt(frame.yaw, 0);
    els.mLink.textContent = linkLabel(frame.link);
    els.attBall.style.transform = `translateY(${(Number(frame.pitch) || 0) * 1.2}px) rotate(${Number(frame.roll) || 0}deg)`;
    els.hudPos.textContent = fmtPos(frame.lat, frame.lon);
    els.hudSeq.textContent = frame.seq != null ? `序号 ${frame.seq}` : "序号 暂无";
    els.clock.textContent = frame.t || "暂无";

    maybeAskLivePrefetch(frame);
    if (hasPos(frame)) {
      marker.setLatLng([frame.lat, frame.lon]);
      setAircraftHeading(movementBearing(prev, frame));
    } else {
      setAircraftHeading(frame.heading);
    }

    if (frame.alt_msl != null) pushHist(state.altHist, frame.alt_msl);
    if (frame.airspeed != null) pushHist(state.spdHist, frame.airspeed);
    const colors = themeColors(currentTheme());
    drawSpark(els.altChart, state.altHist, colors.alt);
    drawSpark(els.spdChart, state.spdHist, colors.sim);

    if (els.follow.checked) {
      snapFollow();
      if (isMapView()) {
        const viewKey = boundsKey(mapBoundsQuery({ pad: 0.22 }));
        if (viewKey !== state.layerBoundsKey && anyMapLayerOn()) {
          scheduleLayerRefresh(2500);
        }
      }
    }

    const now = performance.now();
    if (hasPos(frame) && now - state.envTimer > 20000) {
      state.envTimer = now;
      refreshEnv(frame.lat, frame.lon, frame.alt_msl);
    }
    if (now - state.layerTimer > (els.layerBelief && els.layerBelief.checked ? 5000 : 20000) && anyMapLayerOn() && isMapView()) {
      state.layerTimer = now;
      scheduleLayerRefresh(200, { force: true });
    }
    if (window.__rtwindQuad) window.__rtwindQuad.onFrame(frame);
    refreshSystemLayers();
  }

  function setStatusBar(msg) {
    if (!els.statusBar) return;
    const text = (msg || "").trim();
    if (!text) {
      els.statusBar.hidden = true;
      els.statusBar.textContent = "";
      return;
    }
    els.statusBar.hidden = false;
    els.statusBar.textContent = text;
  }

  function fmtPct(v) {
    if (v == null || !Number.isFinite(Number(v))) return "—";
    return `${Math.round(Number(v) * 100)}%`;
  }

  function updatePlanUi() {
    const lam = state.plan.lambda;
    if (els.pLambda) els.pLambda.textContent = fmtPct(lam);
    if (els.hudLambda) els.hudLambda.textContent = `λ ${fmtPct(lam)}`;
    if (els.pMs) {
      els.pMs.textContent = state.plan.planning_ms != null ? fmt(state.plan.planning_ms, 0) : "—";
    }
    if (els.pMode) els.pMode.textContent = state.plan.planning_mode || "—";
    if (els.planMeta) {
      if (state.plan.last_error) els.planMeta.textContent = state.plan.last_error;
      else if (state.plan.ready) els.planMeta.textContent = "规划运行中（路径线已隐藏）";
      else els.planMeta.textContent = "信念锚定后每 5s 自动重规划";
    }
  }

  function applyPlanPath(data) {
    if (!data) return;
    state.plan = {
      ready: !!data.ok,
      planned: data.planned || [],
      baseline: data.baseline || [],
      lambda: data.lambda != null ? data.lambda : state.plan.lambda,
      planning_ms: data.planning_ms != null ? data.planning_ms : state.plan.planning_ms,
      planning_mode: data.planning_mode != null ? data.planning_mode : state.plan.planning_mode,
      last_error: data.last_error != null ? data.last_error : state.plan.last_error,
    };
    updatePlanUi();
  }

  function clearPlanPath() {
    applyPlanPath({
      ok: false,
      planned: [],
      baseline: [],
      lambda: 0,
      planning_ms: null,
      planning_mode: null,
      last_error: null,
    });
  }

  function applySystemLayers(data) {
    if (!data) return;
    const perc = data.perception || {};
    const cog = data.cognition || {};
    const planning = data.planning || {};

    const setLayer = (row, cls, detail) => {
      if (!row) return;
      row.classList.remove("ready", "partial", "wait");
      row.classList.add(cls);
      if (detail) {
        const el = row.querySelector(".layer-detail");
        if (el) el.textContent = detail;
      }
    };

    let percCls = "wait";
    let percMeta = "DEM / 风场 · 未预下载";
    if (perc.ready || perc.dem_ok) {
      percCls = perc.ready ? "ready" : "partial";
      const km = perc.size_km || MISSION_SIZE_KM;
      percMeta = perc.dem_ok
        ? `DEM 就绪 · 气象由浏览器直连 · ${km}×${km} km`
        : `地形部分就绪 · ${km}×${km} km`;
    } else if (perc.prefetched || perc.partial) {
      percCls = "partial";
      percMeta = `部分就绪 · DEM ${perc.dem_ok ? "✓" : "—"} · 气象走客户端`;
    } else {
      percCls = "wait";
      percMeta = "DEM 未预下载 · 气象走客户端 IP";
    }
    setLayer(els.layerPerception, percCls, percMeta);

    let cogCls = "wait";
    let cogMeta = "信念场 · 未锚定";
    if (cog.anchored && (cog.obs_count || 0) > 0) {
      cogCls = "ready";
      cogMeta = `已锚定 · 观测 ${cog.obs_count}`;
    } else if (cog.anchored) {
      cogCls = "partial";
      cogMeta = "已锚定 · 等待观测";
    }
    setLayer(els.layerCognition, cogCls, cogMeta);

    let planCls = "wait";
    let planMeta = "路径 · 等待信念";
    if (planning.ready) {
      planCls = "ready";
      planMeta = `路径就绪 · λ ${fmtPct(planning.lambda)} · ${planning.planning_ms != null ? `${Math.round(planning.planning_ms)} ms` : "—"}`;
    } else if (planning.last_error) {
      planCls = "partial";
      planMeta = planning.last_error;
    } else if (cog.anchored) {
      planCls = "partial";
      planMeta = "信念就绪 · 规划中";
    }
    setLayer(els.layerPlanning, planCls, planMeta);

    if (els.cObs) els.cObs.textContent = cog.obs_count != null ? String(cog.obs_count) : "—";
    if (els.cUnc) els.cUnc.textContent = cog.mean_uncertainty != null ? fmt(cog.mean_uncertainty, 3) : "—";
    if (els.cConf) els.cConf.textContent = cog.mean_confidence != null ? fmt(cog.mean_confidence, 3) : "—";
    if (planning.lambda != null) state.plan.lambda = planning.lambda;
    updatePlanUi();
  }

  async function refreshSystemLayers(force = false) {
    const now = performance.now();
    if (!force && now - state.systemLayersTimer < 4000) return;
    state.systemLayersTimer = now;
    try {
      const res = await fetch(api("/api/system/layers"), { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      applySystemLayers(data);
    } catch (_) {
      /* ignore */
    }
  }

  function applyTrack(points) {
    state.track = points || [];
    trackLine.setLatLngs(state.track.map((p) => [p.lat, p.lon]));
    if (window.__rtwindQuad) window.__rtwindQuad.onTrack(state.track);
  }

  async function refreshEnv(lat, lon, altMsl) {
    const reqId = ++state.envReq;
    const hadValues = els.eWind.textContent !== "暂无" && els.eWind.textContent !== "—";
    try {
      if (!Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) {
        els.eDem.textContent = "暂无";
        els.eWind.textContent = "暂无";
        els.eDir.textContent = "暂无";
        els.eTemp.textContent = "暂无";
        els.eMeta.textContent = "暂无 GPS，环境数据不可用";
        setStatusBar("暂无 GPS，环境数据不可用");
        return;
      }
      if (!hadValues) els.eMeta.textContent = "环境加载中…";
      const demPromise = fetch(api(`/api/env/at?lat=${lat}&lon=${lon}&weather=0`), { cache: "no-store" })
        .then(async (res) => {
          if (!res.ok) throw new Error(`地形 HTTP ${res.status}`);
          return res.json();
        });
      const wxPromise = window.RtwindMeteo
        ? window.RtwindMeteo.fetchWeatherAt(lat, lon)
        : Promise.reject(new Error("气象客户端未加载"));
      const [demPart, wxPart] = await Promise.allSettled([demPromise, wxPromise]);
      if (reqId !== state.envReq) return;

      let demErr = "";
      if (demPart.status === "fulfilled") {
        const data = demPart.value;
        els.eDem.textContent = fmt(data.dem_msl, 1);
        if (data.dem_msl != null) state.lastDemMsl = Number(data.dem_msl);
        if (state.frame) updateAglDisplay(state.frame);
        if (data.error) demErr = data.error;
      } else {
        els.eDem.textContent = "暂无";
        demErr = demPart.reason?.message || "地形失败";
      }

      let wxErr = "";
      if (wxPart.status === "fulfilled") {
        const wx = wxPart.value;
        els.eWind.textContent = fmt(wx.wind_speed_mps, 1);
        els.eDir.textContent = fmt(wx.wind_dir_deg, 0);
        els.eTemp.textContent = fmt(wx.temperature_c, 1);
        const stale = wx._cached ? " · 缓存" : "";
        els.eMeta.textContent = `${wx.source || "open-meteo-client"}${stale}`;
        wxErr = wx._soft_error || "";
      } else {
        els.eWind.textContent = "暂无";
        els.eDir.textContent = "暂无";
        els.eTemp.textContent = "暂无";
        els.eMeta.textContent = "气象客户端";
        wxErr = wxPart.reason?.message || "气象失败";
      }
      setStatusBar([demErr, wxErr].filter(Boolean).join(" · "));
    } catch (err) {
      if (reqId !== state.envReq) return;
      els.eMeta.textContent = "环境请求失败";
      setStatusBar(`环境错误: ${err.message}`);
    }
  }

  async function switchSource(next) {
    if (next === state.active) return;
    const ok = window.confirm(`切换到「${sourceLabel(next)}」模式？当前航迹将被清空。`);
    if (!ok) return;
    const res = await fetch(api("/api/source"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active: next }),
    });
    const data = await res.json();
    setActiveUi(data.active);
    applyTrack([]);
    clearPlanPath();
    refreshSystemLayers(true);
    state.altHist = [];
    state.spdHist = [];
    setBadge({ link: next === "live" ? "waiting" : "ok" });
  }

  els.btnSim.addEventListener("click", () => switchSource("sim"));
  els.btnLive.addEventListener("click", () => switchSource("live"));
  els.btnResetSim.addEventListener("click", async () => {
    await fetch(api("/api/sim/control"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "reset" }),
    });
    applyTrack([]);
    clearPlanPath();
    state.altHist = [];
    state.spdHist = [];
  });
  els.btnApplySimOrigin.addEventListener("click", async () => {
    const lat = Number(els.simLat.value);
    const lon = Number(els.simLon.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      els.simMeta.textContent = "经纬度无效";
      return;
    }
    try {
      await setSimOrigin(lat, lon);
    } catch (err) {
      els.simMeta.textContent = `错误: ${err.message}`;
    }
  });
  els.btnGeoSim.addEventListener("click", () => {
    if (!navigator.geolocation) {
      els.simMeta.textContent = "定位不可用";
      return;
    }
    els.simMeta.textContent = "定位中…";
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          await setSimOrigin(pos.coords.latitude, pos.coords.longitude);
        } catch (err) {
          els.simMeta.textContent = `错误: ${err.message}`;
        }
      },
      (err) => {
        els.simMeta.textContent = `定位失败: ${err.message}`;
      },
      { enableHighAccuracy: true, timeout: 12000 },
    );
  });
  els.pickOrigin.addEventListener("change", () => setPickOriginMode(els.pickOrigin.checked));
  els.follow.addEventListener("change", () => {
    if (els.follow.checked) snapFollow(true);
  });
  map.on("click", async (ev) => {
    if (!els.pickOrigin.checked || state.active !== "sim") return;
    try {
      await setSimOrigin(ev.latlng.lat, ev.latlng.lng);
    } catch (err) {
      els.simMeta.textContent = `错误: ${err.message}`;
    }
  });
  els.layerDem.addEventListener("change", () => scheduleLayerRefresh(50, { force: true }));
  els.layerWind.addEventListener("change", () => scheduleLayerRefresh(50, { force: true }));
  if (els.btnPrefetchRegion) {
    els.btnPrefetchRegion.addEventListener("click", () => {
      const center = missionCenter();
      runPrefetchStream(center).catch(() => {});
    });
  }
  if (els.btnPrefetchAccept) {
    els.btnPrefetchAccept.addEventListener("click", () => {
      const pending = state.livePrefetchPending || missionCenter();
      runPrefetchStream(pending, { fromNotice: true }).catch(() => {});
    });
  }
  if (els.btnPrefetchDismiss) {
    els.btnPrefetchDismiss.addEventListener("click", () => {
      dismissLivePrefetchNotice();
      if (els.prefetchMeta) els.prefetchMeta.textContent = "已跳过本次预下载，可稍后手动点击按钮";
    });
  }
  if (els.layerBelief) {
    els.layerBelief.addEventListener("change", () => scheduleLayerRefresh(50, { force: true }));
  }
  if (els.beliefLayer) {
    els.beliefLayer.addEventListener("change", () => {
      scheduleLayerRefresh(50, { force: true });
      if (window.__rtwindQuad) window.__rtwindQuad.onBeliefTick();
    });
  }
  map.on("moveend", () => {
    if (performance.now() < state.ignoreMoveEnd) {
      redrawWindForZoom();
      return;
    }
    if (anyMapLayerOn()) scheduleLayerRefresh(800);
  });
  map.on("zoomend", () => {
    redrawWindForZoom();
    if (anyMapLayerOn()) scheduleLayerRefresh(500, { force: true });
  });
  map.on("zoom", () => {
    // Live rescale arrows while pinching/scrolling.
    redrawWindForZoom();
  });
  els.btnTheme.addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
    if (anyMapLayerOn()) scheduleLayerRefresh(50, { force: true });
    if (state.altHist.length || state.spdHist.length) {
      const colors = themeColors(currentTheme());
      drawSpark(els.altChart, state.altHist, colors.alt);
      drawSpark(els.spdChart, state.spdHist, colors.sim);
    }
  });
  applyTheme(currentTheme(), { persist: false });

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}${prefix}/api/ws`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "hello" || msg.type === "source_changed") {
        if (msg.active) setActiveUi(msg.active);
      }
      if (msg.type === "telemetry" && msg.frame) {
        applyFrame(msg.frame);
        if (hasPos(msg.frame) && (state.track.length === 0 || state.track[state.track.length - 1].seq !== msg.frame.seq)) {
          state.track.push({
            lat: msg.frame.lat,
            lon: msg.frame.lon,
            seq: msg.frame.seq,
          });
          if (state.track.length > 1200) state.track.shift();
          trackLine.setLatLngs(state.track.map((p) => [p.lat, p.lon]));
          if (window.__rtwindQuad) window.__rtwindQuad.onTrack(state.track);
        }
      }
      if (msg.type === "track_snapshot") applyTrack(msg.track);
      if (msg.type === "belief_tick" && els.layerBelief && els.layerBelief.checked) {
        scheduleLayerRefresh(400, { force: true });
      }
      if (msg.type === "belief_tick") {
        refreshSystemLayers(true);
      }
      if (msg.type === "belief_tick" && window.__rtwindQuad) {
        window.__rtwindQuad.onBeliefTick();
      }
      if (msg.type === "plan_update") {
        applyPlanPath(msg);
        refreshSystemLayers(true);
      }
      if (msg.type === "plan_progress" && msg.lambda != null) {
        state.plan.lambda = msg.lambda;
        updatePlanUi();
        if (els.layerPlanning) {
          const detail = els.layerPlanning.querySelector(".layer-detail");
          if (detail && state.plan.ready) {
            detail.textContent = `路径就绪 · λ ${fmtPct(msg.lambda)} · ${state.plan.planning_ms != null ? `${Math.round(state.plan.planning_ms)} ms` : "—"}`;
          }
        }
      }
    };
    ws.onclose = () => {
      els.badge.textContent = `${sourceLabel(state.active)} · 重连中`;
      setTimeout(connectWs, 1500);
    };
  }

  window.__rtwindQuad = window.createRtwindQuad({
    api,
    L,
    TILES,
    cartoTileOptions,
    mainMap: map,
    getTheme: currentTheme,
    themeColors,
    getActive: () => state.active,
    getFrame: () => state.frame,
    getTrack: () => state.track,
    getBeliefLayer: () => (els.beliefLayer && els.beliefLayer.value) || "energy",
    followEnabled: () => !!(els.follow && els.follow.checked),
    onMapModeEnter: () => snapFollow(true),
  });

  fetch(api("/api/config/public"))
    .then((r) => r.json())
    .then((d) => applyCartoBasemapKey(d.carto_basemap_key))
    .catch(() => {});

  fetch(api("/api/plan/path"))
    .then((r) => r.json())
    .then((d) => applyPlanPath(d))
    .catch(() => {});

  fetch(api("/api/system/layers"))
    .then((r) => r.json())
    .then((d) => applySystemLayers(d))
    .catch(() => {});

  fetch(api("/api/env/region"))
    .then((r) => r.json())
    .then((d) => {
      if (d.prefetched && els.prefetchMeta) {
        state.missionRegion = d;
        const km = d.size_km || MISSION_SIZE_KM;
        els.prefetchMeta.textContent = `已预下载 ${km}×${km} km 任务区 · ${d.fetched_at || ""}`;
      }
    })
    .catch(() => {});

  fetch(api("/api/sim"))
    .then((r) => r.json())
    .then((d) => {
      if (d.origin_lat != null && d.origin_lon != null) {
        updateOriginVisual(d.origin_lat, d.origin_lon, d.radius_m || 450);
      }
    })
    .catch(() => {});

  fetch(api("/api/source"))
    .then((r) => r.json())
    .then((d) => setActiveUi(d.active || "sim"))
    .finally(() => {
      connectWs();
      refreshEnv(state.simOrigin.lat, state.simOrigin.lon, null);
      if (anyMapLayerOn()) scheduleLayerRefresh(300, { force: true });
    });
})();
