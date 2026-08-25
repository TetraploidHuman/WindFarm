# tetraploid.cn/rtwind 实时真机仪表盘计划书

> 目标：在现有 WindFarm 仿真回放仪表盘（`/wind`）之外，新增一条 **真机 Live + 模拟机 Sim** 的实时监视链路（统一遥测帧，环境用真实 DEM/气象），对外路径为 `http://tetraploid.cn:11024/rtwind`。
>
> 本文只定技术选型与功能范围，不涉及具体实现代码。

---

## 1. 背景与差距

| 能力 | 现有 `/wind` | 目标 `/rtwind` |
|------|-------------|----------------|
| 坐标 | 网格索引 `(x,y,z)` | WGS84 GPS（纬度/经度/海拔） |
| 风场 | 场景内历史 Open-Meteo archive + 合成真值 | 机位点 **实时/近实时** 气象 |
| 地形 | 离线 SRTM 场景切片 | 机位周边 **按需裁切/缓存** 的真实 DEM |
| 机态 | 仿真 `DroneState` / 合成 `sensor_packet` | 无人机回传：航速、高度、方位、姿态等 |
| 链路 | 无飞控协议 | 需遥测接入（MAVLink 或等价推送） |
| 前端 | Python 生成 HTML+Canvas 网格热力页 | **主参照 SkyDash、辅参照 MavDeck**；真地图态势台，不沿用 `/wind` |

结论：`/rtwind` 不是给现有 `serve-dashboard` 换皮肤，而是一条 **“遥测接入 → 地理解算 → 环境查询 → 实时推送 → 地图展示”** 的新服务。后端可复用 DEM/风场与 Caddy 挂载经验；**UI 风格与现有 Wind 控制台脱钩，单独设计。**

---

## 2. 产品目标（第一期）

**一句话**：打开 `tetraploid.cn/rtwind`，可在 **真实无人机** 与 **模拟无人机** 之间切换（**同时只激活一个源**）；机态经统一遥测协议进前端，并叠加机位处真实地形与近实时风场。

### 2.1 必须做到（MVP）

1. **双数据源（一期硬需求）**
   - **Live（真机）**：MAVLink / GCS 转发 / HTTP ingest，接真实飞控或真机链路。
   - **Sim（模拟）**：进程内模拟器持续产出 GPS/姿态/航速等（数据源为模拟，非真飞），字段与 Live **同一套 `TelemetryFrame`**，前端无感或仅徽章区分。
   - **互斥激活**：任意时刻 Hub 只服务一个活跃源（`live` **或** `sim`）。切换时停写旧源、清/归档旧航迹缓冲，再启新源；禁止双轨并行推送。
   - UI / API 切换 `source=live|sim`；默认 `sim`，保证无硬件也能演示。
2. **遥测展示**：位置（lat/lon/alt）、对地/空速、航向、姿态（roll/pitch/yaw）、相对高度（AGL，由 DEM 推算）、时间戳与链路状态（在线/超时）；并标明当前源 `live` / `sim`。
3. **环境叠加**：以机位为中心的 DEM 阴影/等高；机位处或邻域的风速/风向（水平 + 可选垂直估计）。环境层 **不区分** 机态来源——真机与模拟机都查同一套真实 DEM / Open-Meteo。
4. **航迹回放窗口**：最近 N 分钟（或 N 点）GPS 轨迹；可暂停跟随、拖动查看历史帧。
5. **公网路径**：`/rtwind` 页面 + `/rtwind/api/*`，挂在本机 Caddy `11024`。

### 2.2 第一期明确不做

- 不接闭环 MPC 在线规划下发飞控（可预留接口）。
- 不做多机编队作战台（Sim 可先单机；可选仿 SkyDash 少机演示，非重点）。
- 不做用户登录体系。
- 不替换现有 `/wind` 仿真仪表盘（`/wind` 仍是网格信念回放；`/rtwind` Sim 是 **WGS84 真地图上的假飞机**）。
- 不做亚米级地图商用底图付费依赖（优先开源瓦片 + 自有 DEM）。

### 2.3 第二期可选增强

- 机载风速估计 vs 气象预报对比层。
- 信念场 / 规划走廊叠加（复用现有 `belief`/`planner`，需把 GPS→网格映射打通）。
- 移动端适配、任务目标点下发。
- Sim 轨迹脚本化（按航点/风场扰动重放），或与 `/wind` 任务报告互导。

---

## 3. 技术选型建议

### 3.1 总原则

- **后端继续 Python**：与现有 `data_ingest`（SRTM）、Open-Meteo、风场下推同一语言栈，方便复用。
- **UI 与 `/wind` 脱钩**：不复用现有仪表盘的配色、字体、面板与布局；`/rtwind` 视觉以 **SkyDash 为主、MavDeck 为辅**（见 §5.4）。
- **前端地图层**：真 GPS 用地图库（Leaflet / 可选 Deck.gl），不做旧页 Canvas 网格热力。
- **实时通道用推送**：遥测 5–20 Hz 时，长轮询会吃紧；推荐 WebSocket，HTTP 轮询仅作兜底。
- **反向代理沿用 Caddy**：与 `/wind` 相同模式，独立端口，避免撞 `/api`（DeepSeek）。

### 3.2 推荐栈（MVP）

| 层级 | 选型 | 理由 |
|------|------|------|
| 服务框架 | **FastAPI + uvicorn**（或 Starlette） | 原生 WebSocket、OpenAPI、异步拉取气象；比 stdlib `http.server` 更适合真实时 |
| 遥测接入 | **统一 `TelemetryHub` + 可插拔 Source**：`LiveMavlinkSource` / `HttpIngestSource` / **`SimDroneSource`** | 真机与模拟共用下游；Sim 参考 SkyDash `simulation.py` 思路，但输出 WGS84 帧 |
| 地形 | 复用现有 **SRTM1 Skadi** 下载/裁切逻辑（`data_ingest.py`）+ 本地瓦片缓存 | 已验证；按机位 bbox 懒加载 |
| 气象 | **Open-Meteo Forecast API**（实时预报）+ 可选 **Archive** 补历史 | 项目已用 archive；forecast 补“此刻” |
| 前端 | **主参照 [SkyDash](https://github.com/mangod12/skydash)** 的地图态势台风格；辅参照 [MavDeck](https://github.com/DanWilson00/MavDeck) 的遥测查看交互。实现可用 React+Vite+Tailwind+Leaflet（与 SkyDash 同族，易对齐），或同等视觉的轻量静态页 | 见 §5.4 |
| 页面托管 | FastAPI 静态目录或 Jinja 模板 | 单进程可同时出页与 API |
| 进程部署 | 本机绑定 `127.0.0.1:8877`，Caddy `handle_path /rtwind*` → 反代 | 与 `/wind→8876` 对称 |
| 持久化（可选） | SQLite / 环形内存缓冲 | 先内存 ring buffer；需要回放再落盘 |

### 3.3 备选与为何不优先

| 备选 | 不优先原因 |
|------|------------|
| 继续 stdlib `ThreadingHTTPServer` | 无原生 WS，扩展成本高 |
| React/Vue 完整 SPA（与 SkyDash 同级功能面） | 不做 OSINT/情报全站；**允许** SkyDash 同族的轻量 React 监视页 |
| 自建 WRF/本地测风站 | 成本与运维远超竞赛/演示期 |
| Google Maps / 高德商用 SDK | Key、配额、合规；开源底图足够 |

### 3.4 坐标与单位约定

- 地理：**WGS84**，纬度/经度（度），高度区分 **MSL / AGL / 相对起飞点**。
- 姿态：欧拉角 **度**（前端显示），内部可用弧度。
- 风速：m/s；风向：**来向气象角**（度，0=北，顺时针）并同时给 `u/v`（东/北分量）。
- 时间：UTC ISO-8601，前端转本地时区。

---

## 4. 系统架构

```
无人机 / GCS / 模拟器              本机 WindFarm 主机                     公网
─────────────────────            ─────────────────────                 ────
MAVLink (Live) ──────┐
HTTP ingest (Live) ──┼──► SourceAdapter ──► TelemetryHub (:8877)
SimDroneSource ──────┘         │              ├ latest + track ring
                               │              ├ source: live | sim
                               ├─ GeoContext（DEM / Open-Meteo，与 source 无关）
                               ├─ REST  /rtwind/api/...
                               └─ WS    /rtwind/api/ws
                                         │
                               Caddy :11024  strip /rtwind → :8877
                                         │
                               浏览器 ◄── http://tetraploid.cn:11024/rtwind
                                         （UI 切换 Live / Sim）
```

数据流要点：

1. 任意 Source 产出同一 `TelemetryFrame`（含 `source`、`vehicle_id`、姿态、GPS、速度等）→ Hub；**非活跃源不向 Hub 写帧**（Sim 停表 / Live 停收或丢弃）。
2. 前端只订阅读活跃源；切换 `live`↔`sim` 时发 `source_changed`，徽章更新，航迹按策略重置。
3. 机位变化超过阈值或定时触发 DEM/气象刷新（**环境始终用真实数据**）。
4. 默认活跃源为 `sim`；Live 有有效包时可 **提示** 切换，但不自动抢占（避免无操作覆盖模拟演示）。

---

## 5. 功能清单（按模块）

### 5.1 遥测接入（`rtwind.telemetry`）— Live + Sim

**统一帧（两种源必须填齐）**

`TelemetryFrame`：`source`（`live`|`sim`）、`vehicle_id`、`t`、`lat`/`lon`/`alt_msl`、`heading`、`roll`/`pitch`/`yaw`、`airspeed`/`groundspeed`、`climb_rate`、可选 `battery`、`link`。

**Live（真机）**

- [ ] MAVLink 监听：`GLOBAL_POSITION_INT`、`ATTITUDE`、`VFR_HUD`、`SYS_STATUS`（可选）。
- [ ] HTTP `POST /rtwind/api/ingest`：GCS/手机/第三方推 JSON（同样进 Hub，`source=live`）。
- [ ] 链路看门狗：超过 T 秒无包 → `link=stale`。

**Sim（模拟，数据源为假）**

- [ ] 进程内 `SimDroneSource`：按固定/可配航线生成 WGS84 轨迹与姿态噪声（可参考 SkyDash 多机模拟节奏，一期默认单机）。
- [ ] 可选：从 JSONL / 录制日志回放（仍标 `source=sim`）。
- [ ] 启动参数：`--sim on|off`、`--sim-origin lat,lon`、`--active-source live|sim`（默认 `sim`）。

**Hub / 互斥切换（已拍板：同时只激活一个源）**

- [ ] `TelemetryHub` 仅保留 **当前活跃源** 的 latest + 环形航迹。
- [ ] `GET/POST /rtwind/api/source`：查询/切换；`POST` 时停旧源、启新源，可选清空 track。
- [ ] WS 推送 `source_changed` + 后续仅活跃源 `telemetry`。
- [ ] 前端：Live / Sim 互斥切换（单选）+ 源状态徽章；切换需二次确认或明确按钮，避免误触。
- [ ] Live 未连接时选 Live → 徽章 `LIVE` + `link=waiting/stale`，不回退静默灌 Sim。

### 5.2 地理与环境（`rtwind.geo`）

- [ ] GPS → 本地 ENU / 网格索引（复用场景 `resolution_m` 与原点）。
- [ ] 按机位 bbox 拉取/缓存 DEM，计算 AGL = MSL − DEM(lat,lon)。
- [ ] Open-Meteo forecast：机位点 10 m/80 m/120 m 风；短缓存（如 5–15 min）。
- [ ] （可选）调用现有物理下推，生成机位附近粗风场栅格供热力图。

### 5.3 API 设计（建议）

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/rtwind/` | 仪表盘页面 |
| GET | `/rtwind/api/health` | 进程健康；各 source 是否在线 |
| GET | `/rtwind/api/source` | 当前活跃源（唯一）+ 各源可用性 |
| POST | `/rtwind/api/source` | 互斥切换 `{ "active": "live"|"sim" }`（停旧启新） |
| GET | `/rtwind/api/telemetry/latest` | 活跃源最新一帧（无 `?source=` 旁路读非活跃） |
| GET | `/rtwind/api/telemetry/track?since=&limit=` | 航迹点 |
| GET | `/rtwind/api/env/at?lat=&lon=` | 该点 DEM 高程 + 风 |
| GET | `/rtwind/api/env/dem?bbox=` | DEM 栅格/GeoJSON 等高摘要 |
| GET | `/rtwind/api/env/wind-field?bbox=` | 风场栅格摘要 |
| WS | `/rtwind/api/ws` | 推送 `telemetry` / `env_update` / `heartbeat` / `source_changed` |
| POST | `/rtwind/api/ingest` | Live 侧非 MAVLink 推送（演示桥/第三方） |
| POST | `/rtwind/api/sim/control` | （可选）启停/重置模拟机、改原点 |

> Caddy 使用 `handle_path /rtwind*` 时，上游看到的是去掉前缀后的路径；实现时与 `/wind` 一样做好 **挂载前缀兼容**（`root_path` 或双路径匹配）。

### 5.4 前端页面（`/rtwind`）

**UI 参照（已拍板）**

| 优先级 | 项目 | 链接 | 参照什么 | 不参照什么 |
|--------|------|------|----------|------------|
| **主** | **SkyDash** | [mangod12/skydash](https://github.com/mangod12/skydash) · [在线演示](https://wonderful-cliff-0325f3800.7.azurestaticapps.net) | 整体气质与信息架构：深色态势台、**地图为主操作面**、侧栏/浮层遥测、图层与坐标工具感、WS 实时监视节奏；栈可对齐 React + Vite + Tailwind + Zustand + Leaflet（+ 可选 Deck.gl / Recharts） | OSINT/实体图谱、Shodan、多任务情报工作流、多机编队产品定位；不整仓 fork 业务 |
| **辅** | **MavDeck** | [DanWilson00/MavDeck](https://github.com/DanWilson00/MavDeck) · [在线演示](https://mavdeck.netlify.app) | 遥测查看体验：拖拽式时序图网格、消息监视、航迹地图、单位制切换、暗/亮模式、PWA 式「打开即用」 | 不做完整参数调参/方言调试工作台；一期仍以监视为主 |

实现原则：

- **像 SkyDash 的「空间监视台」**，不是民航整页 PFD，也不是游戏战斗机绿线 HUD。
- **不参考、不移植** 现有 Wind / `/wind` 控制台（`dashboard.py` / `dashboard_parts.py`）。
- 可借鉴 SkyDash 截图与组件分区，业务只保留：单机（或少机）真 GPS、机态、航迹、风场/DEM 环境层。
- 许可：二者均为 **MIT**；参照视觉与交互合法，复制代码时保留许可证声明。

**信息结构（一屏主任务，对齐 SkyDash 监视面）**

1. **主视图（近全屏地图）**：底图 + 机位（朝向=航向）+ 航迹 + 风矢/风速半透明层。
2. **机态 HUD / 侧栏**：空速、地速、MSL、AGL、航向、roll/pitch/yaw、链路状态、更新时间。
3. **环境摘要**：当地风速/风向、气温（若有）、DEM 高程、数据源与刷新时刻。
4. **辅面板（可自 MavDeck 借鉴）**：可选 1～2 个时序小图（高度/空速）、链路/消息状态条。
5. **控制**：跟随/自由漫游、轨迹时长、**Live / Sim 源切换**、演示重置。

### 5.5 运维与挂载

- [ ] 进程：`python -m windfarm.rtwind serve --host 127.0.0.1 --port 8877`
- [ ] Caddy（NixOS `configuration.nix` 持久化，避免仅 admin API 临时写入）：

```caddy
handle_path /rtwind* {
  reverse_proxy 127.0.0.1:8877 {
    header_up Host {host}
  }
}
```

- [ ] （建议）用户级 systemd：开机拉起 rtwind；与 `/wind` 进程分离。
- [ ] 安全：公网默认 **只读展示**；`/ingest` 与 MAVLink 端口仅绑局域网或加 token。

---

## 6. 与现有代码的复用关系

| 现有模块 | 复用方式 |
|----------|----------|
| `data_ingest.py` / SRTM | 抽“按 lat/lon/半径取 DEM”函数给 rtwind |
| Open-Meteo 调用 | 新增 forecast 客户端；archive 仅用于历史对比 |
| `dashboard.py` / `dashboard_parts.py` | **仅作反例**：不复用页面、样式、布局与交互骨架；`/rtwind` 全新前端 |
| `live_server.py` `/api/state` | 协议思想可参考；实现换 WS + 地理 JSON |
| `types.Observation` / `sensor_packet` | 字段对齐，二期接信念与规划 |
| Caddy `/wind` 经验 | 同模式挂 `/rtwind`，独立端口 |

---

## 7. 分期实施计划

### Phase 0 — 约定与骨架（0.5–1 天）

- 敲定端口 `8877`、路径前缀、`TelemetryFrame` schema。
- FastAPI 空服务 + 静态页 Hello + Caddy `/rtwind` 打通。

### Phase 1 — 双源机态展示（2–3 天）

- `SimDroneSource` 默认跑通 → Hub → WS/REST；前端 Live/Sim 切换与源徽章。
- `LiveMavlinkSource` + `/ingest` 接入同一 Hub。
- Leaflet 地图（SkyDash 式机位/航迹）；机态侧栏。
- （可选）MavDeck 式高度/空速时序小图。
- 无硬件时仅靠 Sim 即可完整演示。

### Phase 2 — 真环境叠加（2–4 天）

- 机位 DEM 懒加载与 AGL。
- Open-Meteo forecast 缓存与风矢图层。
- 环境 REST + 偶发 `env_update` 推送。

### Phase 3 — 硬化（1–2 天）

- 超时/断线 UI、限流、token、systemd、写入 NixOS Caddy。
- 简单录制/回放文件格式（JSONL）。

### Phase 4 —（可选）接 WindFarm 大脑

- GPS↔网格、粗风下推、信念热力、只读展示规划建议（不下发飞控）。

---

## 8. 风险与对策

| 风险 | 对策 |
|------|------|
| 真机暂不可用 | Sim 为默认可用源；Live 离线时 UI 明确提示，不假装有真机 |
| 真假数据混淆 | 每帧带 `source`；UI 强制徽章；录制文件头写入源类型 |
| Open-Meteo 限流/离线 | 本地短缓存；失败时保留上一有效值并标“陈旧” |
| SRTM 首次下载慢 | 预缓存飞行区域瓦片；异步加载时地图先显 GPS |
| 公网暴露遥测 | 只读；敏感接口鉴权；MAVLink 不直接暴露公网 |
| Caddy 重启丢路由 | 必须写入 `configuration.nix`，勿只依赖 admin API |
| 坐标系混乱（AGL/MSL/相对高） | UI 明确标注；内部 schema 分字段 |

---

## 9. 成功标准（验收）

1. 打开 `/rtwind`，默认 **Sim** 可见连续航迹与机态；徽章为 `SIM`。
2. 切换到 **Live**（或真机上线后），徽章为 `LIVE`；机态与注入/飞控一致。
3. 同一套 DEM/风场面在两种源下均可按机位刷新，并标注环境数据时间。
4. 真机接入后姿态/航向误差在可接受范围（姿态角级、位置与 GCS 一致）。
5. `/wind` 与站点其它路径（`/love`、`/ds`）不受影响。

---

## 10. 待你拍板的问题

实现前建议确认这几项（影响接口与工期）：

1. **飞控协议（Live）**：PX4 / ArduPilot MAVLink，还是仅 GCS/手机 HTTP 推送？
2. **Sim 默认空域**：模拟机起飞原点 lat/lon（及巡航半径），便于预缓存 DEM。
3. **刷新期望**：界面 1 Hz 是否足够，还是必须 ≥5 Hz 姿态动画？
4. **公网是否需要鉴权**：`/rtwind` 是否像 `/ds` 一样 basic_auth？
5. **第一期是否要信念/规划图层**，还是严格只做“真机/模拟机 + 真环境监视”？

> 已拍板：Live / Sim **同时只激活一个源**（互斥切换，不双轨缓冲）。

---

## 11. 推荐结论（摘要）

- **技术栈**：Python FastAPI + WebSocket + pymavlink + 复用 SRTM/Open-Meteo；前端视觉 **主参照 SkyDash、辅参照 MavDeck**（推荐 React+Vite+Tailwind+Leaflet 便于对齐）+ Caddy `/rtwind` → `:8877`。
- **功能核心**：**Live 真机 + Sim 模拟机** 互斥双源（同时只激活一个）；统一进 Hub；真 GPS 机态、航迹、真实 DEM/近实时风场；UI 切换并强制标清源类型。
- **UI**：主参照 SkyDash、辅参照 MavDeck；独立于 `/wind`。
- **策略**：与 `/wind` 并行；先监视后规划；无硬件时靠 Sim 演示。

确认上述问题后，可按 Phase 0→3 开工实现。
