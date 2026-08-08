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

## 已知问题诊断：风场利用表现不佳（2026-08-08）

多场景能量评测（`scripts/eval_multi_scenario_energy.py`）是防过拟合的主指标：改参数必须同时看福建 / 北京 / 青海 / 青岛四场景，禁止为单图加特例阈值。

### 已落地的通用修复

- 走廊/MPC 终选放宽；不再把获胜 MPC 强制 snap 成直线
- 抬升气动信用加强；规划与执行共用 `uplift_energy_scale`
- 安全项只罚下沉侧 `max(-w,0)`；`expected_energy_gain` 不再把无向水平风速当收益
- 巡航高度：平原默认 ≥5% 证据才爬升；沿途地形抬升大时放宽到 ≥3%；未赚到的 sticky 会降回 clearance
- 热盘旋不再因「水平前进少」被反向惩罚抬升

### 最新四场景结果（`multi-scenario-energy-20260808-202201`）

| 场景 | vs 名义直线 | vs 最佳高度带 | z_max | 说明 |
|------|-------------|---------------|-------|------|
| fujian_hills | 0% | 0% | 1.00 | 贴名义直线（风场差异不足） |
| beijing_plain | 0% | 0% | 1.00 | 同上 |
| qinghai_ridge | **+18.3%** | -1.7% | 2.70 | 主要靠选对更高巡航带 |
| qingdao_coast | **0%** | 0% | 1.07 | 已消除此前错误爬高（曾 -4%～-58%） |
| 均值 | +4.6% | -0.4% | — | 无单场景大亏；相对最佳带仍接近持平 |

对比：修复前糟糕 run `...-104637` 均值 **-29%**（错误高层）。

### 仍未解决

1. **几乎仍是直线导引** — 走廊/MPC/盘旋很少进入终选，动态绕行利用风场仍弱  
2. **`save_best%` 未稳定为正** — 青海略逊于最优常数 AGL；平原无额外增益  
3. **信念每步预测仍整场覆盖观测** — 在线感知积累有限  

### 评测怎么读

| 结果类型 | 含义 |
|---------|------|
| `save_agl% ≈ 0` 且 `z_max≈1.0` | 贴合名义直线，无额外风场收益 |
| `save_agl%` 高但 `save_best%≈0` | 主要是选对了巡航带，不是动态寻风 |
| `save_*` 大幅为负且 `z_max≈3` | 高度策略失控（应视为回归失败） |

### 下一步（仍需跨四场景验证）

1. 让走廊/MPC 在模型能量更优时稳定胜出，而不只是「不比直线差太多」  
2. 信念风场改为滤波融合，减少高度层误判  
3. 评测增加消融基线（仅水平风 / 禁抬升 / 最优离线开环）  

## 后续方向

- 继续把 `save_best%` 推向四场景均值稳定为正，且无单场景大亏
- 接入真实 DEM、地表覆盖与机载遥测日志
- 从单机信念更新扩展到多机协同信念共享
