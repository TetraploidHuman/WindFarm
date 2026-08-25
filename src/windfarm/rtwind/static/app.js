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
    layerMeta: document.getElementById("layerMeta"),
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
    hudPos: document.getElementById("hudPos"),
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
  };

  const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([27.908, 112.922], 14);
  const TILES = {
    dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
  };
  let basemap = L.tileLayer(TILES.dark, {
    maxZoom: 19,
    attribution: "&copy; OSM &copy; CARTO",
  }).addTo(map);
  let basemapTheme = "dark";

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
      map.removeLayer(basemap);
      basemap = L.tileLayer(TILES[next], {
        maxZoom: 19,
        attribution: "&copy; OSM &copy; CARTO",
      }).addTo(map);
      basemap.bringToBack();
      basemapTheme = next;
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
    if (!prev) return frame.heading;
    const dLat = frame.lat - prev.lat;
    const dLon = frame.lon - prev.lon;
    if (Math.hypot(dLat, dLon) < 1e-10) return frame.heading;
    const rad = (frame.lat * Math.PI) / 180;
    return ((Math.atan2(dLon * Math.cos(rad), dLat) * 180) / Math.PI + 360) % 360;
  }

  let demOverlay = null;
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
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
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
    if (!wantDem && !wantWind) {
      if (demOverlay) {
        map.removeLayer(demOverlay);
        demOverlay = null;
      }
      windLayer.clearLayers();
      els.layerMeta.textContent = "图层已关闭";
      return;
    }
    const bounds = mapBoundsQuery({ pad: 0.22 });
    const key = boundsKey(bounds);
    if (!force && key === state.layerBoundsKey && (demOverlay || windLayer.getLayers().length)) {
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
        const res = await fetch(
          api(`/api/env/wind-field?south=${south}&west=${west}&north=${north}&east=${east}&nx=${grid.windNx}&ny=${grid.windNy}`),
          { cache: "no-store", signal },
        );
        if (!res.ok) throw new Error(`风场 HTTP ${res.status}`);
        applyWindLayer(await res.json());
        bits.push("风场");
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
        applyDemLayer(await res.json());
        bits.push("地形");
      } else if (demOverlay) {
        map.removeLayer(demOverlay);
        demOverlay = null;
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
    } else {
      map.removeLayer(originMarker);
      map.removeLayer(originOrbit);
      setPickOriginMode(false);
    }
    document.getElementById("map").classList.toggle("pick-origin", els.pickOrigin.checked && active === "sim");
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

  function applyFrame(frame) {
    if (!frame) return;
    const prev = state.frame;
    state.frame = frame;
    setBadge(frame);
    els.mAir.textContent = fmt(frame.airspeed, 1);
    els.mGs.textContent = fmt(frame.groundspeed, 1);
    els.mMsl.textContent = fmt(frame.alt_msl, 1);
    els.mAgl.textContent = frame.alt_agl != null ? fmt(frame.alt_agl, 1) : els.mAgl.textContent;
    els.mHdg.textContent = fmt(frame.heading, 0);
    els.mClimb.textContent = fmt(frame.climb_rate, 2);
    els.mRoll.textContent = fmt(frame.roll, 1);
    els.mPitch.textContent = fmt(frame.pitch, 1);
    els.mYaw.textContent = fmt(frame.yaw, 0);
    els.mLink.textContent = linkLabel(frame.link);
    els.attBall.style.transform = `translateY(${(-frame.pitch) * 1.2}px) rotate(${frame.roll}deg)`;
    els.hudPos.textContent = `${frame.lat.toFixed(5)}, ${frame.lon.toFixed(5)}`;
    els.hudSeq.textContent = `序号 ${frame.seq}`;
    els.clock.textContent = frame.t;

    marker.setLatLng([frame.lat, frame.lon]);
    setAircraftHeading(movementBearing(prev, frame));

    pushHist(state.altHist, frame.alt_msl);
    pushHist(state.spdHist, frame.airspeed);
    const colors = themeColors(currentTheme());
    drawSpark(els.altChart, state.altHist, colors.alt);
    drawSpark(els.spdChart, state.spdHist, colors.sim);

    if (els.follow.checked) {
      // 无动画跟随：动画 panTo(0.25s) 在高频率遥测下会堆积，看起来像巨大延迟
      state.ignoreMoveEnd = performance.now() + 400;
      const nowPan = performance.now();
      if (!state.lastFollowPanAt || nowPan - state.lastFollowPanAt >= 80) {
        state.lastFollowPanAt = nowPan;
        map.setView([frame.lat, frame.lon], map.getZoom(), { animate: false });
      }
      const viewKey = boundsKey(mapBoundsQuery({ pad: 0.22 }));
      if (viewKey !== state.layerBoundsKey && (els.layerDem.checked || els.layerWind.checked)) {
        scheduleLayerRefresh(2500);
      }
    }

    const now = performance.now();
    if (now - state.envTimer > 8000) {
      state.envTimer = now;
      refreshEnv(frame.lat, frame.lon, frame.alt_msl);
    }
    if (now - state.layerTimer > 20000 && (els.layerDem.checked || els.layerWind.checked)) {
      state.layerTimer = now;
      scheduleLayerRefresh(200, { force: true });
    }
  }

  function applyTrack(points) {
    state.track = points || [];
    trackLine.setLatLngs(state.track.map((p) => [p.lat, p.lon]));
  }

  async function refreshEnv(lat, lon, altMsl) {
    const reqId = ++state.envReq;
    const hadValues = els.eWind.textContent !== "—";
    try {
      if (!hadValues) els.eMeta.textContent = "环境加载中…";
      const res = await fetch(api(`/api/env/at?lat=${lat}&lon=${lon}`), { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (reqId !== state.envReq) return;
      els.eDem.textContent = fmt(data.dem_msl, 1);
      els.eWind.textContent = fmt(data.wind_speed_mps, 1);
      els.eDir.textContent = fmt(data.wind_dir_deg, 0);
      els.eTemp.textContent = fmt(data.temperature_c, 1);
      if (data.dem_msl != null && altMsl != null) {
        els.mAgl.textContent = fmt(altMsl - data.dem_msl, 1);
      }
      const stale = data.stale ? " · 缓存" : "";
      els.eMeta.textContent = `${data.weather_source || "open-meteo-forecast"}${stale}${data.error ? " · " + data.error : ""}`;
    } catch (err) {
      if (reqId !== state.envReq) return;
      els.eMeta.textContent = `环境错误: ${err.message}`;
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
  map.on("moveend", () => {
    if (performance.now() < state.ignoreMoveEnd) {
      redrawWindForZoom();
      return;
    }
    if (els.layerDem.checked || els.layerWind.checked) scheduleLayerRefresh(800);
  });
  map.on("zoomend", () => {
    redrawWindForZoom();
    if (els.layerDem.checked || els.layerWind.checked) scheduleLayerRefresh(500, { force: true });
  });
  map.on("zoom", () => {
    // Live rescale arrows while pinching/scrolling.
    redrawWindForZoom();
  });
  els.btnTheme.addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
    if (els.layerWind.checked || els.layerDem.checked) scheduleLayerRefresh(50, { force: true });
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
        if (state.track.length === 0 || state.track[state.track.length - 1].seq !== msg.frame.seq) {
          state.track.push({
            lat: msg.frame.lat,
            lon: msg.frame.lon,
            seq: msg.frame.seq,
          });
          if (state.track.length > 1200) state.track.shift();
          trackLine.setLatLngs(state.track.map((p) => [p.lat, p.lon]));
        }
      }
      if (msg.type === "track_snapshot") applyTrack(msg.track);
    };
    ws.onclose = () => {
      els.badge.textContent = `${sourceLabel(state.active)} · 重连中`;
      setTimeout(connectWs, 1500);
    };
  }

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
      if (els.layerDem.checked || els.layerWind.checked) scheduleLayerRefresh(300, { force: true });
    });
})();
