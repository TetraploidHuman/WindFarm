# WindFarm

WindFarm 是一个面向不确定风环境下自主飞行的 Python 工程原型，覆盖了分层风场预测、实时信念更新、风险感知与在线路径优化的完整闭环。

当前版本已经实现：

- 基于地形的粗尺度风场下推
- 物理场之上的残差学习修正
- 在线信念预测与观测融合
- 显式的上升气流 / 中性 / 下沉气流模式概率
- 持续更新的安全风险场
- 带返航能量约束的有限时域 MPC 风格重规划
- 任务执行、回放仪表盘、实时仪表盘与 HTTP API

## 快速开始 (安装与执行)

1. **初始化环境与安装** (解决模块找不到的问题):
```bash
# 在根目录下执行，使 Python 能识别 src 目录下的包
.venv\Scripts\python.exe -m pip install -e .
```

2. **运行演示脚本** (推荐，全流程自动化展示):
```bash
.venv\Scripts\python.exe -m windfarm.cli generate-config --output config.json
.venv\Scripts\python.exe -m windfarm.cli run-demo --config config.json --runs-dir runs --run-name latest
# 最后启动仪表盘查看结果 (根据run-demo实际输出的路径调整)
.venv\Scripts\python.exe -m windfarm.cli build-dashboard --report runs/latest/mission_report.json --output dashboard.html
```

3. **分步执行典型任务流程**:
```bash
.venv\Scripts\python.exe -m windfarm.cli generate-config --output config.json
.venv\Scripts\python.exe -m windfarm.cli simulate --config config.json --output-dir examples
.venv\Scripts\python.exe -m windfarm.cli train --config config.json --terrain examples/terrain.json --training examples/training.json --model-out examples/model.json --metrics-out examples/metrics.json
.venv\Scripts\python.exe -m windfarm.cli run-mission --config config.json --terrain examples/terrain.json --model examples/model.json --coarse-wind examples/coarse_wind.json --observations examples/observations.json --truth examples/truth.json --output examples/mission_report.json
.venv\Scripts\python.exe -m windfarm.cli build-dashboard --report examples/mission_report.json --output examples/dashboard.html
```

## 核心模块

- `src/windfarm/physics.py`：地形感知的风场下推。
- `src/windfarm/ml.py`：物理风场上的残差学习模型。
- `src/windfarm/belief.py`：信念预测、观测融合、模式概率、信念熵与安全风险更新。
- `src/windfarm/planner.py`：有限时域 MPC 风格束搜索规划器，综合风险、安全、高度和返航预算。
- `src/windfarm/execution.py`：闭环导航执行、在线重规划与任务报告生成。
- `src/windfarm/simulator.py`：地形、真值风场、训练数据与观测数据模拟。
- `src/windfarm/dashboard.py`：静态回放与实时仪表盘页面生成。
- `src/windfarm/live_server.py`：实时回放服务与交互式导航服务。
- `src/windfarm/api_server.py`：训练、预测、导航任务的 HTTP API。

## 线程数 / 多核

`config.json` 里 `model.n_jobs` 控制 XGBoost / LightGBM 训练与推理线程：

- `0`（默认）：按本机 CPU 自动适配（当前 32 核机器会使用 28 线程，预留 4 核给系统和规划器）
- `>0`：手动指定线程数
- `<0`：使用全部逻辑核

注意：路径规划器本身仍是单线程束搜索；多核主要加速风场残差模型训练。

已做的运行加速：

- 训练先按 `max_training_samples` 子采样，再提特征（避免百万级无效特征计算）
- u/v/w 三个残差模型并行训练（每模型均分线程）
- 网格预测改为批量 XGBoost/LightGBM 推理
- 任务循环尊重 `mission.max_steps`（不再被粗风时序拖到 360 步）

## 信念与规划模型

当前实现中的每个信念单元会维护：

- 期望能量收益
- 风场均值与方差
- 上升 / 中性 / 下沉三类模式概率
- 信念熵
- 置信度与不确定性
- 危险概率与安全惩罚

规划器在有限预测时域内评估候选控制序列，综合考虑：

- 能量代价
- 目标推进收益
- 不确定性惩罚
- 安全风险惩罚
- 高度机动代价
- 终端推进奖励
- 返航能量可行性

每次只执行第一步动作，下一时刻再基于最新观测重新更新信念并重规划。

## CLI 命令

- `generate-config`：生成默认配置文件。
- `simulate`：生成地形、粗风、观测、训练集和真值场。
- `train`：训练残差风场模型。
- `predict`：生成分层风场预测结果。
- `plan`：基于粗风与观测执行在线规划。
- `run-mission`：执行完整闭环任务并生成报告。
- `build-dashboard`：生成静态回放仪表盘。
- `serve-dashboard`：启动实时回放仪表盘服务。
- `serve-navigate`：启动交互式实时导航仪表盘。
- `serve-api`：启动 HTTP API 服务。
- `run-demo`：在 `runs/` 下生成完整演示任务。

## 仪表盘内容

仪表盘当前支持展示：

- 预测风场、真值风场与残差热力图
- 信念能量、不确定性、置信度、熵和安全风险图层
- 上升 / 下沉概率图层
- 已执行与规划中的三维路径
- 电量、能量率与高度变化历史
- 垂直风剖面
- MPC 候选轨迹与代价分解

## 验证

运行回归测试：

```bash
python -m unittest tests.test_pipeline
```

### 真实场景数据（防过拟合）

数据来源：SRTM1 DEM（`.hgt`）+ Open-Meteo 历史风场。正式评测清单已收敛为 **8** 张代表性地图（磁盘上可能仍留有旧场景目录，评测会忽略）。

| 分组 | 场景 | 地形类型 |
|------|------|----------|
| Core（调参常用） | fujian_hills / beijing_plain / qinghai_ridge / liaoning_coast | 东南丘陵、华北平原、高原脊、辽东海岸 |
| Holdout（独立验收） | xinjiang_gobi / sichuan_foothills / shanxi_loess / taiwan_hills | 戈壁、四川弱风边缘、黄土沟壑、台湾迎风丘陵 |

评测并行：正式 **8** 场景默认 **`--workers 8`**（吃满多核）；场景数远大于 8 时再降。曾经 18 路并行会刷交换区，与现在无关。

构建 / 补下载：

```bash
# 若 Windows 上 Clash Verge 已开「允许局域网」，在 NixOS 开发机走代理（更快下 SRTM）：
export HTTPS_PROXY=http://172.20.128.142:7897   # 以实际探测到的主机/端口为准
export HTTP_PROXY="$HTTPS_PROXY"

.venv-linux/bin/python -m windfarm.cli build-scenarios \
  --output-dir scenarios --data-dir data --base-config config_eval.json \
  --resolution-m 30
```

能量评测（只跑上述 8 张，并分别汇总 CORE / HOLDOUT）：

```bash
.venv-linux/bin/python scripts/eval_multi_scenario_energy.py --workers 8
.venv-linux/bin/python scripts/eval_multi_scenario_energy.py --only sichuan_foothills taiwan_hills
```

**规则**：不要只对着某一张 holdout 调参；改动必须以 CORE+HOLDOUT 均值与「无单场景大亏」为准。

## 已知问题诊断：风场利用表现不佳（2026-08-08）

多场景能量评测（`scripts/eval_multi_scenario_energy.py`）是防过拟合的主指标：必须同时看 core 与 holdout 真实地图，禁止为单图加特例阈值。

### 已落地的通用修复

- 走廊/MPC 终选放宽；不再把获胜 MPC 强制 snap 成直线
- 抬升气动信用加强；规划与执行共用 `uplift_energy_scale`
- 安全项只罚下沉侧 `max(-w,0)`；`expected_energy_gain` 不再把无向水平风速当收益
- 巡航高度：平原默认 ≥5% 证据才爬升；沿途地形抬升大时放宽到 ≥3%；未赚到的 sticky 会降回 clearance
- 热盘旋不再因「水平前进少」被反向惩罚抬升
- 走廊：只生成不差于直线的候选，终选需模型能量明确更省  
  - 硬地板 `CORRIDOR_HARD_FLOOR_MPS=1.20`（默认禁弱水平风乱绕）  
  - 软地板 `CORRIDOR_MIN_WIND_MPS=1.80`；介于两者之间**仅当存在真实侧向 |风速| 边**才解锁  
  - **同带 commit**：走廊按自身巡航高度带判 ambient/edge/Joules（不再用 clearance sticky 误杀）  
  - 剪切带有真边时 Joules≈1%、commit 风险税 0.02  
  - **硬地板以下几何解锁**：无风模型能量仍比同带直线省 ≥3%（或两段 via DEM 减爬 ≥10 m）  
  - **硬地板以下微风垂直解锁**：水平风 ≥0.40 且路径平均 `w` 明显高于直线；无风 Er0 有上限；轻绕行惩罚。用于抬升走廊，不是地图特例
  - **硬地板以下微风水平速解锁**：路径 p75 `|wind|` 明显高于直线（且 Er0 有界）；与垂直解锁并列，覆盖「有剪切但平均 `w` 仍近 0」的微风叶
  - 微风走廊偏置加密到 450/550 m、绕行上限 380 m（仍在 clearance 带生成；曾试多巡航带强制爬升，台湾 live 大亏已撤回）
- **顶风 speed-to-fly（执行层）**：水平顶风 ≥0.55 m/s 时按 `0.85×head` 提高空速计入 `path_model`；静风/顺风不加速。真值上盘旋/爬升抬升走廊对闽等图反而更费电，故优先 MacCready 而非热盘旋
- 信念预测改为与先验滤波融合（不再整场覆盖），近期观测与侧向风差可保留
- 强风且 climb_earned、相对 clearance ≥15% 节能时，高度带一次对齐；执行层巡航窗跟踪 `preferred`（可下调，不再只 `max`）
- 强风已 earned 的高 sticky：降高需 ≥8%（防青海爬上又冲回 1.3）；高于 clearance+1.2 时**不跳过 MPC**（greedy 高带路径更差）
- **已 earned 高巡航带（>clearance+1）**：任务后半程冻结 cruise AGL，末端按 eval 同款 baseline 折线下降（避免工程进场阶梯降高 + 近终点 MPC 乱动）；直线 guide 不再用 preferred 覆盖折线高度
- **earned 后微爬**：强风已 earned 且仍低于 `clearance+1.67` 时，仅允许一步跳到邻近更优细带（≥0.3% 更便宜）；禁止宽先验把更高带一起抬热（曾导致窜到 z=3）
- **防过拟合**：不按地图写特例。曾试「沿航迹顺风边」与「微风高带强制爬升」，live 变差已撤回；现用通用地形/几何 + 微风垂直/水平速门控

### 最新 8 场景结果（`multi-scenario-energy-20260809-230107`）

| 分组 | vs 名义直线 | vs 最佳高度带 |
|------|-------------|---------------|
| CORE（闽/京/青/辽） | **+9.2%** | **+4.3%** |
| HOLDOUT（疆/川/晋/台） | **+4.3%** | **+4.3%** |
| 全部 8 场景 | **+6.7%** | **+4.3%** |

| 场景 | save_agl% | save_best% | 备注 |
|------|-----------|------------|------|
| fujian_hills | **+9.6** | **+9.6** | 顶风 speed-to-fly（原 +0.5） |
| beijing_plain | 0.0 | 0.0 | 平地对照 |
| qinghai_ridge | **+19.6** | **+0.0** | 微爬对齐 2.667；zmax≈2.67 |
| liaoning_coast | **+7.4** | **+7.4** | 走廊正样本（规划侧空速改动曾误伤，已撤回） |
| xinjiang_gobi | 0.0 | 0.0 | 干旱开阔 |
| sichuan_foothills | **+1.8** | **+1.8** | 微风垂直抬升走廊 |
| shanxi_loess | **+6.7** | **+6.7** | 走廊正样本 |
| taiwan_hills | **+8.5** | **+8.5** | 几何走廊 + 轻顶风加速 |

对比：speed-to-fly 后全场 **+5.5% → +6.7%**；福建 **+0.5 → +9.6**；台湾 **+8.2 → +8.5**；辽/青/晋/川/京/疆无倒退。

### 负收益 / 权衡（当前）

| 类型 | 状态 | 说明 |
|------|------|------|
| A. 弱风误走廊 | **已收敛** | 硬地板 + 几何/微风垂直/水平速门；四川转正，北京/新疆仍 0 |
| B. 高度带未对齐 oracle | **已收敛** | 青海贴合 `agl_2.667`（`save_best≈0`） |
| C. 台湾弱风走廊 | **已收敛（几何）** | 主因是 AGL/DEM 捷径而非 |风速| 边 |

### 仍未解决（按优先级）

1. **信念边质量/校准**（微风剪切、台湾高带机会）  
2. **开环上界评测**（量清剩余 headroom 再投 ML）  
3. 局部热核盘旋：当前能量模型下对 8 图真值多为负收益；仅在上界显示有利时再开  

### 评测怎么读

| 结果类型 | 含义 |
|---------|------|
| `save_agl% ≈ 0` 且 `z_max≈1.0` | 贴合名义直线，无额外风场收益 |
| `save_agl%` 高但 `save_best%≈0` | 主要是选对了巡航带，不是动态寻风 |
| `save_*` 大幅为负且 `z_max≈3` | 高度策略失控（应视为回归失败） |

### 下一步（仍需跨 8 场景验证）

1. 抬高青海 `save_best%`（只用强风证据，避免台湾式爬高）  
2. 提升弱风真剪切的信念边质量 / Joules 一致性  
3. 评测增加消融基线（仅水平风 / 禁抬升 / 最优离线开环）  

## 后续方向

- 继续把 `save_best%` 推向 8 场景均值稳定为正，且无单场景大亏
- 接入真实 DEM、地表覆盖与机载遥测日志
- 从单机信念更新扩展到多机协同信念共享
