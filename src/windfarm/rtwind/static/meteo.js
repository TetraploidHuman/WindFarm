(() => {
  const OPEN_METEO = "https://api.open-meteo.com/v1/forecast";
  const cache = new Map();
  const TTL_MS = 10 * 60 * 1000;

  function cacheKey(lat, lon) {
    return `${Number(lat).toFixed(3)},${Number(lon).toFixed(3)}`;
  }

  function fromCurrent(cur) {
    const spd = cur?.wind_speed_10m;
    const deg = cur?.wind_direction_10m;
    const temp = cur?.temperature_2m;
    let u = null;
    let v = null;
    if (spd != null && deg != null) {
      const rad = ((Number(deg) + 180) * Math.PI) / 180;
      u = Number(spd) * Math.sin(rad);
      v = Number(spd) * Math.cos(rad);
    }
    return {
      wind_speed_mps: spd != null ? Number(spd) : null,
      wind_dir_deg: deg != null ? Number(deg) : null,
      wind_u: u,
      wind_v: v,
      temperature_c: temp != null ? Number(temp) : null,
    };
  }

  function explainHttp(status, body) {
    if (status === 429) {
      if (body && /daily/i.test(body)) return "气象今日额度已用尽（按你的网络 IP）";
      return "气象限流，请稍后再试（按你的网络 IP）";
    }
    return `气象 HTTP ${status}`;
  }

  async function fetchWeatherAt(lat, lon, { signal } = {}) {
    const key = cacheKey(lat, lon);
    const hit = cache.get(key);
    const now = Date.now();
    if (hit && now - hit.t < TTL_MS) {
      return { ...hit.data, _cached: true, source: "open-meteo-client" };
    }
    const q = new URLSearchParams({
      latitude: String(Number(lat).toFixed(3)),
      longitude: String(Number(lon).toFixed(3)),
      current: "temperature_2m,wind_speed_10m,wind_direction_10m",
      wind_speed_unit: "ms",
      timezone: "UTC",
    });
    const res = await fetch(`${OPEN_METEO}?${q}`, { cache: "no-store", signal });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      if (hit) {
        return {
          ...hit.data,
          _cached: true,
          _soft_error: explainHttp(res.status, body),
          source: "open-meteo-client",
        };
      }
      throw new Error(explainHttp(res.status, body));
    }
    const payload = await res.json();
    const data = fromCurrent(payload.current || {});
    cache.set(key, { t: now, data });
    return { ...data, _cached: false, source: "open-meteo-client" };
  }

  async function fetchWindField(south, west, north, east, nx = 6, ny = 6, { signal } = {}) {
    nx = Math.max(2, Math.min(12, Number(nx) || 6));
    ny = Math.max(2, Math.min(12, Number(ny) || 6));
    const s = Math.min(Number(south), Number(north));
    const n = Math.max(Number(south), Number(north));
    const w = Math.min(Number(west), Number(east));
    const e = Math.max(Number(west), Number(east));
    const lats = [];
    const lons = [];
    for (let i = 0; i < ny; i++) lats.push(s + ((n - s) * i) / (ny - 1));
    for (let j = 0; j < nx; j++) lons.push(w + ((e - w) * j) / (nx - 1));
    const points = [];
    for (const lat of lats) for (const lon of lons) points.push([lat, lon]);

    const need = [];
    const weatherMap = new Map();
    let staleAny = false;
    for (const [lat, lon] of points) {
      const key = cacheKey(lat, lon);
      const hit = cache.get(key);
      if (hit && Date.now() - hit.t < TTL_MS) {
        weatherMap.set(key, { ...hit.data, _cached: true });
        staleAny = true;
      } else {
        need.push([lat, lon]);
      }
    }

    const chunkSize = 40;
    for (let i = 0; i < need.length; i += chunkSize) {
      const chunk = need.slice(i, i + chunkSize);
      const q = new URLSearchParams({
        latitude: chunk.map((p) => p[0].toFixed(3)).join(","),
        longitude: chunk.map((p) => p[1].toFixed(3)).join(","),
        current: "temperature_2m,wind_speed_10m,wind_direction_10m",
        wind_speed_unit: "ms",
        timezone: "UTC",
      });
      const res = await fetch(`${OPEN_METEO}?${q}`, { cache: "no-store", signal });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(explainHttp(res.status, body));
      }
      const payload = await res.json();
      const rows = Array.isArray(payload) ? payload : null;
      const now = Date.now();
      if (rows) {
        rows.forEach((row, idx) => {
          const [lat, lon] = chunk[idx];
          const data = fromCurrent(row.current || {});
          const key = cacheKey(lat, lon);
          cache.set(key, { t: now, data });
          weatherMap.set(key, data);
        });
      } else {
        const cur = payload.current || {};
        chunk.forEach(([lat, lon], idx) => {
          const rowCur = {};
          for (const k of ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]) {
            const val = cur[k];
            rowCur[k] = Array.isArray(val) ? val[idx] : val;
          }
          const data = fromCurrent(rowCur);
          const key = cacheKey(lat, lon);
          cache.set(key, { t: now, data });
          weatherMap.set(key, data);
        });
      }
    }

    let missing = 0;
    const vectors = points.map(([lat, lon]) => {
      const wx = weatherMap.get(cacheKey(lat, lon)) || {};
      if (wx.wind_u == null && wx.wind_v == null) missing += 1;
      return {
        lat,
        lon,
        u: wx.wind_u,
        v: wx.wind_v,
        speed_mps: wx.wind_speed_mps,
        dir_deg: wx.wind_dir_deg,
      };
    });

    return {
      bounds: { south: s, west: w, north: n, east: e },
      nx,
      ny,
      vectors,
      stale: staleAny,
      missing,
      source: "open-meteo-client",
      fetched_at: new Date().toISOString(),
    };
  }

  window.RtwindMeteo = {
    fetchWeatherAt,
    fetchWindField,
  };
})();
