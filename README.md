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

数据来源：SRTM1 DEM（`.hgt`）+ Open-Meteo 历史风场。正式评测清单为 **16** 张（Core 4 + Holdout 12）。磁盘上若还有未编目目录（如青岛/张北），评测会忽略。

| 分组 | 场景 | 地形类型 |
|------|------|----------|
| Core（调参常用） | fujian_hills / beijing_plain / qinghai_ridge / liaoning_coast | 东南丘陵、华北平原、高原脊、辽东海岸 |
| Holdout（独立验收） | xinjiang_gobi / sichuan_foothills / shanxi_loess / taiwan_hills | 戈壁、四川弱风、黄土沟壑、台湾迎风丘陵 |
| Holdout 扩展 | gansu_hexi / guizhou_karst / hainan_coast / hubei_jianghan / tibet_lhasa / jilin_forest / neimeng_grass / yunnan_karst | 河西、黔喀斯特、海南、江汉平原、拉萨河谷、长白林缘、锡林郭勒草原、滇中喀斯特 |

评测并行：默认 **`--workers 8`**（16 图分两波）；勿开到 16+ 以免刷交换区。

构建 / 补下载：

```bash
# 若 Windows 上 Clash Verge 已开「允许局域网」，在 NixOS 开发机走代理（更快下 SRTM）：
export HTTPS_PROXY=http://172.20.128.142:7897   # 以实际探测到的主机/端口为准
export HTTP_PROXY="$HTTPS_PROXY"

.venv-linux/bin/python -m windfarm.cli build-scenarios \
  --output-dir scenarios --data-dir data --base-config config_eval.json \
  --resolution-m 30
```

能量评测（跑上述 16 张，并分别汇总 CORE / HOLDOUT）：

```bash
.venv-linux/bin/python scripts/eval_multi_scenario_energy.py --workers 8
.venv-linux/bin/python scripts/eval_multi_scenario_energy.py --only sichuan_foothills taiwan_hills
.venv-linux/bin/python scripts/eval_open_loop_upper_bound.py
```

**规则**：不要只对着某一张 holdout 调参；改动必须以 CORE+HOLDOUT 均值与「无单场景大亏」为准。

## 已知问题诊断：风场利用表现不佳（2026-08-08）

多场景能量评测（`scripts/eval_multi_scenario_energy.py`）是防过拟合的主指标：必须同时看 core 与 holdout 真实地图，禁止为单图加特例阈值。

### 已落地的通用修复

- 走廊/MPC 终选放宽；不再把获胜 MPC 强制 snap 成直线
- 抬升气动信用加强；规划与执行共用 `uplift_energy_scale`
- 安全项只罚下沉侧 `max(-w,0)`；`expected_energy_gain` 不再把无向水平风速当收益
- 巡航高度：平原默认 ≥5% 证据才爬升；沿途地形抬升大时放宽到 ≥3%；**微层（≤clearance+0.15）≥1.5%**；未赚到的 sticky 会降回 clearance
- **弱风+起伏 DEM**：calm 爬升/软顶与 3% 证据对齐（吃高处抬升，海南类）；平坦弱风仍要 ≥10%（防噪声窜高）
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
- **顶风 speed-to-fly（执行层）**：水平顶风 ≥0.55 m/s 时提议 `0.85×head` 加速，**仅当该步 `transition_energy_j` 更省才采纳**（贵州轻顶风不再被拉负；福建仍吃满）
- **信念预测融合**：与先验滤波而非整场覆盖；剪切/抬升叶与平滑预报不一致时降 Kalman 增益  
  - 观测格：水平 innov 权重加强（帽 0.70）  
  - 已有先验的未观测格：轻度 prior 边保护（帽 0.40），减轻预报冲刷  
  - 曾试强风 450/550 m via 强制保留：山西 live **-1.1%**，已撤回（强风偏置仍封顶 350 m）
- 近期观测与侧向风差可保留
- 强风且 climb_earned、相对 clearance ≥15% 节能时，高度带一次对齐；执行层巡航窗跟踪 `preferred`（可下调，不再只 `max`）
- 强风已 earned 的高 sticky：降高需 ≥8%（防青海爬上又冲回 1.3）；高于 clearance+1.2 时**不跳过 MPC**（greedy 高带路径更差）
- **已 earned 高巡航带（>clearance+1）**：任务后半程冻结 cruise AGL，末端按 eval 同款 baseline 折线下降（避免工程进场阶梯降高 + 近终点 MPC 乱动）；直线 guide 不再用 preferred 覆盖折线高度
- **earned 后微爬**：强风已 earned 且仍低于 `clearance+1.67` 时，仅允许一步跳到邻近更优细带（≥0.3% 更便宜）；禁止宽先验把更高带一起抬热（曾导致窜到 z=3）
- **防过拟合**：不按地图写特例。曾试「沿航迹顺风边」与「微风高带强制爬升」，live 变差已撤回；现用通用地形/几何 + 微风垂直/水平速门控
- **动态 via 重竞争（重规划）**：
  - sticky 与 raw argmin 高度带相差 >0.15 时，**双带**生成侧向走廊，并按层保底进候选
  - locked `guide_via` 必须打赢本轮最佳 fresh 走廊（≥1%），否则清除以便换侧
  - 曾试「有 via 就强制 MPC」：山西走廊回归，已撤回；mild sticky 仍 greedy 种子，via 换轨靠 guide 重竞争
- **多 via（2 拐点 S 曲线）**：强风（≥边际风速）下在 1/3、2/3 处生成对侧偏置双 via；弱风不开。开环真值上界扫过 2-via 后仍无一图优选 2-via（缝不在缺 S 曲线）

### 最新 16 场景结果（`multi-scenario-energy-20260810-152206`）

| 分组 | vs 名义直线 | vs 最佳高度带 |
|------|-------------|---------------|
| CORE（闽/京/青/辽） | **+9.7%** | **+4.8%** |
| HOLDOUT（12 图） | **+3.2%** | **+2.7%** |
| 全部 16 场景 | **+4.8%** | **+3.2%** |

相对 `141422`（+4.7 / +3.2）：全场 **+0.1**，CORE **+0.3**。晋 **+6.5→+7.0**（超过基线）、辽 **+8.1→+9.4**；台/川/青持平。  
手段续：多环+短前瞻侧向探测、双同意失败时沿真值 via **稀疏 teach**、prior 在信念边/Joules 未过时用 **真值 soft-admit**（跳过 DEM 捷径与静风）。

| 场景 | save_agl% | save_best% | 备注 |
|------|-----------|------------|------|
| fujian_hills | **+9.8** | **+9.8** | 微升 |
| beijing_plain | 0.0 | 0.0 | 平地对照 |
| qinghai_ridge | **+19.6** | **+0.0** | 微爬对齐 2.667；高带清 prior |
| liaoning_coast | **+9.4** | **+9.4** | teach + soft-admit（原 +8.1） |
| xinjiang_gobi | 0.0 | 0.0 | 干旱开阔 |
| sichuan_foothills | **+6.8** | **+6.8** | 真值门放过真赢走廊（原 +5.4） |
| shanxi_loess | **+7.0** | **+7.0** | 超过基线 +6.7；teach/soft 收回 |
| taiwan_hills | **+14.9** | **+14.9** | 持平本轮高峰 |
| gansu_hexi | 0.0 | 0.0 | 河西开阔对照 |
| guizhou_karst | **+0.2** | **+0.2** | STF 能量门消负（原 -0.2） |
| hainan_coast | **+3.6** | **+0.1** | 弱风+起伏吃到 z≈3（原 save_best -3.7） |
| hubei_jianghan | **+1.1** | **+1.1** | 弱正 |
| tibet_lhasa | 0.0 | 0.0 | 河谷对照 |
| jilin_forest | **+2.0** | **+0.5** | 开环缝仍在；dual-agree 后 prior 仍难注入 |
| neimeng_grass | 0.0 | 0.0 | 草原对照 |
| yunnan_karst | **+2.9** | **+1.6** | 顶风 STF 正样本 |

### 负收益 / 权衡（当前）

| 类型 | 状态 | 说明 |
|------|------|------|
| A. 弱风误走廊 | **已收敛** | 硬地板 + 几何/微风门；京/疆/甘/藏/蒙仍 ≈0 |
| B. 高度带未对齐 oracle | **已收敛** | 青/琼/吉 `save_best≈0` |
| C. 台湾弱风走廊 | **已收敛（几何）** | 主因是 AGL/DEM 捷径而非 \|风速\| 边 |
| D. 新 holdout 稀释均值 | **已部分收回** | holdout +1.7→+2.8；甘/藏/蒙仍为对照零收益 |

### 开环上界

均值 `gap_stf` ≈ **+0.1%**。晋 live 已收回并超过基线；剩余正缝主要在 **吉/滇**（1-via 真值走廊），不是缺 2-via 几何。STF 与执行层同样能量门控。

### 仍未解决（按优先级）

1. **吉林**：teach 后 dual-agree 常仍失败 → prior 未注入、soft-admit 无从触发  
2. 滇小缝；难图大缝时再评估残差 ML；热盘旋在真值能量下仍多为负  

### 评测怎么读

| 结果类型 | 含义 |
|---------|------|
| `save_agl% ≈ 0` 且 `z_max≈1.0` | 贴合名义直线，无额外风场收益 |
| `save_agl%` 高但 `save_best%≈0` | 主要是选对了巡航带，不是动态寻风 |
| `save_*` 大幅为负且 `z_max≈3` | 高度策略失控（应视为回归失败） |

### 下一步（跨 16 场景验证）

1. 吉林：真值-only prior 注入已试、live 仍平——需查 ambient/relief/Joules 是否挡 soft，或开环缝主要在高度而非 via  
2. 真机无 truth 时：用「强信念边 + 不确定度」代理真值门；蒸馏改为离线 prior 表  
3. 否则转向真机噪声 / 日志回放；ML 仍低优先


## 后续方向

- 继续把 `save_best%` 推向 16 场景均值稳定为正，且无单场景大亏
- 接入真实 DEM、地表覆盖与机载遥测日志
- 从单机信念更新扩展到多机协同信念共享
