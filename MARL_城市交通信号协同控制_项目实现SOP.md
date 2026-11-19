# 基于 MARL 的城市交通信号协同控制：项目实现 SOP

> 面向课程项目的研究与工程执行文档。目标是交付可复现的多路口交通控制实验、可交互演示和完整技术报告。研究状态核对日期：2026-11-19。GitHub 项目活跃度与依赖会变化，正式开工时应记录所用 commit、SUMO 版本和 Python 锁文件。相关情况实时更新到 https://github.com/AnXMuy/MARL-XJTU 中，新建分支。

## 1. 项目定义与边界

### 1.1 研究问题

在给定路网和随机/时变交通需求下，将信号灯控制器建模为多个协作智能体，比较分散式规则与多智能体强化学习策略能否降低系统拥堵，同时不以牺牲公平性、稳定性和安全约束为代价。主要研究问题：

1. 局部观测下的多路口协作，是否优于单路口独立控制？
2. 在训练流量与测试流量、随机种子、需求偏移不同的情况下，策略是否仍有效？
3. 交通需求的时空不均衡是否支持潮汐车道/动态引导带来额外收益？

### 1.2 课程项目可交付范围

采用“核心可完成、扩展可降级”的边界：

- **核心**：微观仿真；多个信号交叉口；固定时制、感应控制、Max-Pressure、独立 DQN 和 MAPPO；可调各入口流量；策略切换；实时/回放可视化；统计对比；技术报告。
- **扩展 A**：协调式绿波（offset）基线、IPPO；路面诱导信息面板（只提供路线建议，不改车辆动力学）。
- **扩展 B**：潮汐车道（可逆车道）规则控制及其收益/副作用分析。
- 不把车辆级自动驾驶、行人/公交优先、事故响应、真实城市大规模路网同时纳入核心。若实现时间不足，潮汐车道只做单独实验，不并入 MARL 动作空间。

**重要建模约束**：信号灯切相必须经过黄灯和全红清空时段，禁止策略直接从冲突相位跳转。潮汐车道切换要有封闭、清空、缓冲时间与方向互斥约束；演示中的路线引导要说明其假设（车辆遵从率），不得让“推荐路线”凭空增加路网容量。

## 2. 推荐技术路线

### 2.1 仿真器与框架

| 组件 | 推荐 | 理由与限制 |
|---|---|---|
| 微观交通仿真 | [Eclipse SUMO](https://github.com/eclipse-sumo/sumo) | 成熟开源交通仿真器，支持信号灯、路由、车道、检测器、TraCI；适合作为课程项目的可信主环境。主要是 CPU 仿真，48GB GPU 不会自动让单个 SUMO episode 更快。 |
| Python 接口/环境 | [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) | 将 SUMO 封装为 Gymnasium/PettingZoo 风格环境，降低多信号智能体接入成本。开始前核对当前版本的 Gymnasium、PettingZoo、TraCI API 和多智能体接口。 |
| MARL 训练 | 首选自建小型 MAPPO（PyTorch）或 [RLlib](https://github.com/ray-project/ray)；实验框架也可参考 [EPyMARL](https://github.com/uoe-agents/epymarl) | MAPPO 易讲清集中训练、分散执行和参数共享；RLlib 功能全面但依赖/配置面较大。EPyMARL 适合参考算法实现，不应假设它可直接接 SUMO。课程交付优先简单、可复现训练循环。 |
| 可选快速仿真 | [CityFlow](https://github.com/cityflow-project/CityFlow) | 面向多路口控制且仿真吞吐较高，适合大量训练；需自行核对现有 Python/编译环境及信号控制接口。可作为扩展或替代，不建议首周同时维护两套仿真器。 |
| 结果追踪 | TensorBoard 或 CSV + Matplotlib | 简单可靠；每个 run 记录配置、种子、代码版本、依赖版本和原始 episode 指标。 |
| Demo | Streamlit + SUMO GUI/自绘路网回放 | Streamlit 快速交付参数面板和指标；GUI 可以实时看车辆，但远程/无显示服务器应支持无头运行与轨迹回放。 |

相关 GitHub 仓库不是同一层级：SUMO 是仿真器，SUMO-RL 是环境封装，RLlib/EPyMARL 是算法训练工具。建议本项目以 SUMO + SUMO-RL 的接口模式 + 自己可解释的 MAPPO 训练代码为主，避免把一个大型通用框架当成开箱即用的交通控制研究代码。固定依赖到具体 tag/commit，保留上游许可证和引用。

### 2.2 CUDA 13.2 与硬件策略

SUMO 仿真主要受 CPU、TraCI 步进和环境并行度影响；48GB GPU 对 MAPPO 的小型 MLP 不构成瓶颈。CUDA Toolkit 13.2 与 PyTorch 轮子是否匹配取决于 PyTorch 当期发行，不要根据本机 toolkit 版本硬编码 `cu132`。安装前检查 NVIDIA 驱动、`nvidia-smi` 和 PyTorch 官方安装矩阵；优先使用 PyTorch 官方当前支持的 CUDA runtime wheel，或 CPU 版先完成正确性验证。通常无需本机完整 CUDA Toolkit，除非编译自定义 CUDA 扩展。SUMO/TraCI 单独安装并做 GUI 与无头 smoke test。

建议 Python 3.10/3.11（以所选 SUMO-RL 和算法库兼容矩阵为准），在锁定文件中记录精确版本。首日先确认 `sumo --version`、Python 能启动仿真、Gymnasium reset/step、PyTorch device 检查和一条 episode 成功；再开始训练。

## 3. 城市环境设计

### 3.1 环境层级

1. **单路口单元测试**：四进口、直行/左转/右转车道，验证相位安全、黄灯转换、检测和奖励符号。
2. **核心训练网格**：3×3 信号交叉口，路口间距 250–400 m；每进口至少 1 条直行/右转混合车道，按需要设置专用左转道；边界接入/流出边。优先使用 SUMO net/route XML 或 SUMO 工具生成，不手工伪造车辆轨迹。
3. **泛化测试网格**：4×4 或不同路口间距/拓扑的网格。它测试空间规模迁移，不与训练网格共享所有需求种子。
4. **结构化案例**：主干道走廊、十字主干道+支路、早晚高峰方向不均衡。至少一个场景要体现上游放行造成下游回堵。

核心路网固定拓扑，随机化 OD 需求和出发时间；这样可以把策略差异归因于控制而不是地图变化。所有 baselines 用完全相同网络、需求 realization、仿真时长、车辆类型、warm-up 和随机种子进行配对比较。

### 3.2 交通需求生成

- 定义 OD 路径/路线和 4/8 个边界入口；每个入口有可独立调节的车辆到达率（veh/h）或流量倍率。
- 生成均匀需求、单方向潮汐峰值、交叉流量、脉冲事件（短时流量突增）和未见过的需求组合。
- 使用 SUMO route 文件和可复现随机种子；写脚本验证车辆出发数、成功到达数、未出发/teleport 数，避免把过载造成的排队误认为较低旅行时间。
- 配置至少低/中/高三个需求强度。高负荷要包括接近容量和超容量情形，但报告须区分“网络过饱和下仍有积压”与可稳定运行区间。
- 每回合仿真建议 3600–5400 s，前 300–600 s 作为 warm-up（根据网络尺寸和需求验证）；最终时间、需求规模以实际运行能力调整并在报告声明。

### 3.3 SUMO 仿真配置

- 固定仿真步长（例如 1 s），信号策略每 5–10 s 决策一次；使用动作持续时间时，所有方法按相同决策节奏/约束比较。
- 决策只允许合法相位/延长绿灯或请求下一相位；执行层插入最小绿灯、黄灯（典型 3–5 s，需场景核对）及全红（典型 1–2 s，按交叉口设计确定）。记录真实执行动作而非只记录策略请求。
- 固定车辆跟驰/换道模型和参数；至少记录 SUMO 版本、网络生成命令、车辆类别与参数。
- 禁用或单独统计 teleport。出现 teleport 的 episode 不能静默丢弃；建议标记为失败/严重拥堵，并报告计数。
- 设定 spillback/下游占有率状态，避免 Max-Pressure 或 RL 奖励鼓励向已堵死下游继续放行。

### 3.4 潮汐车道及路面引导扩展

**阶段安排**：先完成信号策略主线和独立路线诱导，再实现潮汐车道，最后评估组合策略。潮汐车道不要与 MAPPO 一开始就做联合混合动作（信号离散、车道方向切换慢时标、路由连续/组合动作），这会显著扩大状态与信用分配难度。

- 潮汐车道限定在选定走廊，提供双向需求统计；控制器每 5–15 min 决策一次，经过“停止新车进入→清空→安全缓冲→切换方向”的状态机。最小保持时间和切换成本必须明确。
- Baseline 为固定方向、基于方向需求比阈值的规则、（可选）滚动预测规则。若做 RL，单独使用高层 options/离散方向动作，并与规则潮汐车道比较。
- 路面引导实现为可控的动态路线推荐：入口车辆按遵从概率选择推荐路径，候选路径有合理绕行上限；记录推荐采纳率、绕行距离、各边流量和公平性。基线为不引导、最短时间静态引导、按拥堵估计的动态引导。
- 核心主结果先不包括扩展，扩展表单独列出，避免把多个机制的贡献混为一谈。

## 4. 多智能体建模

### 4.1 Agent、观测、动作

- 每个受控信号交叉口为一个 agent；边界无信号节点不是 agent。
- **局部观测**（每个进口/车道/相位）：队列长度或 halting vehicle count、占有率、等待时间统计、当前相位 one-hot、当前相位已持续时间、可选下游出口占有率/剩余储存空间。统一归一化，缺失车道用 mask。不得把未来到达量泄漏给策略。
- **通信/邻域扩展**：先用本地观测完成 MAPPO；再消融相邻路口的队列/相位摘要。集中 critic 可在训练时使用全局状态，但执行 actor 仍只能读部署时可得的局部信息。
- **离散动作**：保持当前绿灯一个控制周期，或请求下一合法相位。输出 mask 屏蔽不合法切换；最小绿时段、黄灯和全红由环境执行。不要允许每秒任意切相。
- **参数共享**：同拓扑相同相位定义的路口共享 actor 参数，输入附加路口/进口特征（可选）；集中 critic 用所有 agent 的拼接状态或 pooled embedding。该方案降低样本需求并利于同构路网扩展。

### 4.2 奖励

主奖励先选一种稳定、可解释定义，再做 reward ablation：

`r_t = - (全网累计 halting queue 或累计 delay 的增量) / N_agents`

更稳妥地用累计等待时间增量（避免回合越长负奖励无界）；也可定义局部负队列/负等待时间，团队共享奖励作为协作主设定。对极端队列加入裁剪但同时记录未裁剪原始值。不要将“平均速度”单独作为奖励，它可能鼓励少数车辆高速而大量车辆滞留。

辅助项只在有明确目标时加入：超限等待惩罚、下游堵塞惩罚、相位切换惩罚。每项都做权重和消融，报告 reward 与真实交通指标的关系，避免只优化训练回报。

### 4.3 训练算法

主算法：**MAPPO + 参数共享 actor + centralized value critic + GAE**。记录 PPO clip、学习率、熵系数、value loss、GAE λ、γ、batch/minibatch、更新轮数、梯度裁剪和 action mask。使用规范 MAPPO 实现并做小环境 sanity check；不以论文默认超参代替调试。

对照：

- **Independent DQN (IDQN)**：每路口独立 Q 网络（可参数共享）；用于显示独立学习的非平稳性/协作局限。
- **IPPO**：共享参数但每智能体本地 critic，区分集中 critic 的贡献。
- （时间允许）QMIX：离散协作动作的值分解对照；保持可比训练预算。

每个算法都需训练曲线、独立评估脚本和 checkpoint metadata。不要只挑最好 seed 报告；训练 seed、评估 seed 分离。

## 5. 基线与评价

至少实现以下规则并按相同仿真预算评估：

1. **Fixed-time**：预设周期、相位绿信比分配；计划用饱和流率或场景需求估算，提供合理工程基线，不故意设弱。
2. **Actuated**：基于检测器占有率/车头时距的绿灯延长、gap-out、min/max green；信号安全约束与 MARL 一致。
3. **Max-Pressure**：按相位的上游队列与下游队列差（压力）选择相位，带最小绿/黄灯安全机制。它是强且有理论动机的在线基线；需要在具备 lane movement 数据的相位层实现，不要用简化错误公式冒称标准 Max-Pressure。
4. **协调定时/绿波**（可选但推荐）：走廊 offset + 周期/绿信比，体现传统协调控制。
5. **IDQN、IPPO、MAPPO**：学习型对比。

### 5.1 必报指标

- **平均旅行时间**：对成功到达车辆统计；另报完成率/未完成车辆数。避免只看 completed vehicles 的选择偏差。
- **平均延误**：相对自由流基准时间，明确计算公式；或 SUMO waiting time 作为单独指标，名称不要混用。
- **累计等待时间/总等待时间**、平均 halting queue、最大队列/95 分位队列。
- **吞吐量**：单位时间完成车辆数与回合总到达量；分开报告出发、完成、网络内车辆。
- **速度/停车次数**：平均速度和每车停车次数。
- **公平性**：旅行时间或等待时间 95th percentile、Gini/Jain 指数，避免平均数掩盖长尾。
- **拥堵与稳定性**：teleport 数、spillback 时长/次数、最大排队、需求变化后的恢复时间。
- **计算成本**：环境 steps/s、训练 wall-clock、总环境交互步数、GPU/CPU 资源；训练墙钟时间与策略质量分开。
- **扩展专属**：潮汐车道切换次数/空车道时间/切换期间受阻；诱导采纳率、额外行驶距离、路线公平性。

### 5.2 统计规范

- 最终评估至少 10 个训练 seed（课程资源不足时至少 5 个并解释）× 每 seed 20 个固定测试需求种子；对所有策略采用配对 seeds。
- 训练中 checkpoint 选择只看 validation seeds；test seeds 只在最终评估用一次或固定公开。
- 报均值±标准差、95% bootstrap CI 或置信区间；给出相对 fixed-time 和 Max-Pressure 的改善百分比。必要时对配对差值做 bootstrap/非参数检验，不只报 p 值。
- 记录失败、超时和所有 episode，不因结果差而删 seed。明确训练环境数量、总仿真车辆量、每个算法的交互预算。

## 6. 实验矩阵与消融

### 6.1 主实验

| ID | 实验 | 目的 |
|---|---|---|
| E0 | 单路口 smoke test + 安全检查 | 验证接口、动作约束、奖励/指标 |
| E1 | 3×3，低/中/高静态需求：所有 baselines | 核心效能比较 |
| E2 | 早高峰/晚高峰潮汐流量 + 未见需求强度 | 鲁棒性与分布偏移 |
| E3 | 4×4 或另一拓扑 zero-shot/少量微调 | 空间泛化；说明参数共享/观测尺度条件 |
| E4 | 突发需求/单入口流量翻倍 | 响应时间、稳定性、最差路口 |
| E5 | 潮汐车道规则控制与静态方向对照 | 评估扩展机制自身，不先归功于 MARL |
| E6 | 路面引导 no-guidance/static/dynamic | 评估路线诱导和遵从率敏感性 |
| E7 | 信号 MARL +（可选）路面引导/潮汐扩展 | 组合收益和交互效应 |

### 6.2 消融（最低要求）

1. MAPPO vs IPPO：集中 critic 的贡献。
2. 参数共享 vs 每路口独立 actor：样本效率与泛化。
3. 全局团队奖励 vs 局部奖励（奖励尺度相同）：协作信用分配。
4. 有/无下游拥堵观测或 spillback 惩罚：是否减少把车放入堵塞路段。
5. 观测只含队列 vs 队列+等待/占有率：状态信息贡献。
6. 有/无邻居信息：显式通信信息价值；若没有实现通信，不能宣称测试了通信。
7. 固定需求训练 vs demand randomization：鲁棒性。
8. 去除黄灯安全层仅可在受控仿真单元测试对比，不能把不安全策略作为真实建议；主实验始终启用安全层。

扩展消融：潮汐车道开/关及需求不平衡阈值；路线推荐遵从率 0/0.5/1；奖励中是否含 fairness 或切换成本。每个消融只改变一个因素，使用相同训练预算和种子集。

## 7. Demo 规格

### 7.1 用户流程

1. 选择地图/场景：3×3 网格、走廊、早高峰/晚高峰。
2. 通过入口列表或路网热区图调节每个入口流量（veh/h 或倍率）；设置仿真时长、随机种子。
3. 选择控制策略：固定时制、感应、Max-Pressure、已训练 MARL checkpoint；可选择单策略或同一需求下并排对照。
4. 点击开始/暂停/单步/重置；展示当前信号相位、车辆、队列、入口需求和策略名。车辆颜色区分类别/转向，信号灯相位有明确颜色图例。
5. 结束后展示 KPI 对比和时间序列：网络总队列、完成车辆、平均等待时间、旅行时间分布；允许导出 CSV/JSON 和回放。

### 7.2 架构要求

- UI 参数通过 schema 校验，禁止任意路径/命令注入；run 配置序列化并保存。
- 仿真进程和 UI 解耦：后台 worker 管理 SUMO，按固定间隔发出车辆/信号/指标快照；提供停止、超时、进程清理。
- 优先支持确定性回放：保存需求、种子、SUMO 配置、策略 checkpoint hash 和版本信息。
- 同一批次并行策略比较必须使用相同 seed/需求，并限制 CPU 进程数，避免浏览器演示抢占所有 CPU。
- 失败状态可见（SUMO 不可用、策略 checkpoint 缺失、仿真超时），不能静默显示空地图。
- 提供无头模式用于测试/服务器，GUI 可选；若实时动画性能不足，可预运行轨迹后回放，清楚标注“回放”。

### 7.3 Demo 验收

- 任一入口流量从 0 调高后，生成车辆数/网络状态确实发生改变。
- 同一种子和参数重复运行可复现需求；不同策略可在同一需求上逐帧/逐指标比较。
- 可视化信号相位切换符合安全序列；暂停/停止会结束后端进程。
- 高流量场景显示拥堵与未完成车辆，不以截断数据伪造“清空”。
- 页面能清晰展示训练策略 checkpoint 来源和评价指标；未经训练的随机策略不能标为 MARL 结果。

## 8. 仓库结构建议

```text
traffic-marl/
  configs/{network,demand,training,eval}/
  sumo/{networks,scenarios,templates}/
  src/traffic_marl/
    env/          # SUMO-RL adapter、multi-agent API、安全 action wrapper
    agents/       # MAPPO、IPPO、IDQN、inference
    baselines/    # fixed-time、actuated、max-pressure、green-wave
    metrics/      # 单一权威指标实现
    demo/         # UI、worker、live snapshot/replay
    utils/        # seeds、config、logging、checkpoint
  scripts/{generate_demand,train,evaluate,make_figures}.py
  tests/{unit,integration}/
  runs/           # gitignore；配置和汇总结果另存
  reports/figures/
  docs/           # SOP、实验记录、报告草稿
  requirements.lock / pyproject.toml
```

统一接口建议：`reset(seed, demand_config) -> observations, info`、`step(actions) -> observations, rewards, terminated, truncated, info`。保留每车、每路口、全网三个粒度的指标，并确保 `evaluate.py` 和 demo 使用同一指标实现。

## 9. 按阶段执行的 SOP

### 阶段 0：环境与复现（第 1–2 天）

1. 安装/确认 SUMO、Python、PyTorch、TraCI；保存版本快照与机器信息。
2. 单路口生成路线，运行 GUI 和无头模式；用 Python 读取检测器、信号相位、车辆 waiting time。
3. 写最小 smoke test：reset、运行 1000 步、结束进程、指标非 NaN、种子可复现。
4. 失败先解决 API/版本问题，不进入 MARL 实现。

### 阶段 1：路网和规则基线（第 3–5 天）

1. 建 3×3 网格与场景生成器；给网络/路线输出做 schema 和容量 sanity checks。
2. 实现固定时制、感应、Max-Pressure；单测最小绿、转换阶段、下游堵塞行为。
3. 运行需求强度扫描，确定低/中/高场景及稳定/过饱和边界。
4. 做统一 evaluator，导出逐回合 CSV；先得到基线表和队列图。

### 阶段 2：MARL 环境接口与算法（第 6–10 天）

1. 将路口观测、动作 mask、奖励、episode 截断写进环境 adapter，先人工动作验证。
2. 跑小型随机策略和 single-intersection DQN/MAPPO sanity test；检查 reward、value、entropy、KL、动作分布和信号转换记录。
3. 实现参数共享 MAPPO + centralized critic；先单环境训练正确，再逐步增加 SUMO 并行环境数。
4. 实现 IDQN、IPPO 对照；相同环境交互预算和评估程序。
5. 对训练曲线做多 seed；checkpoint 按验证集选取，冻结后只跑 test seeds。

### 阶段 3：实验、消融、扩展（第 11–15 天）

1. 执行 E1–E4，自动生成 KPI 表、CI、需求/队列时间序列和失效记录。
2. 执行最低要求消融；检查每次只改变一个变量。
3. 核心结果稳定后才实现路线推荐扩展；完成后才尝试潮汐车道状态机与独立规则实验。
4. 对每个扩展比较单项收益和组合交互，不把变化归因于 RL 而没有对应消融。

### 阶段 4：Demo 与报告（第 16–20 天）

1. 用冻结 checkpoint 构建可选策略列表；不在用户点击 demo 时临时训练。
2. 实现流量调节、并排对比、实时/回放、KPI 和导出；执行 demo 验收用例。
3. 从原始结果脚本生成所有图表，禁止手动改动数据点；图上标样本数、误差线和单位。
4. 写报告、复跑关键实验、归档配置/checkpoint/hash/依赖版本和 README 运行命令。

上述天数是参考排期，若课程周期更短，保留阶段 0–2、E1–E4、最低消融和 demo；删去潮汐车道 RL，而不是删掉基线或统计规范。

## 10. 技术报告大纲与分析要求

建议标题：**基于多智能体强化学习的城市交通信号协同控制：面向需求波动的仿真评估**。报告章节：

1. **摘要**：问题、仿真环境、MAPPO/基线、主要量化结果、限制；结果未完成前不预写结论。
2. **Introduction**：城市路口耦合、固定时制局限、局部控制与网络拥堵传播；明确研究问题、贡献和边界。
3. **相关工作**：传统 fixed-time/actuated/Max-Pressure；单路口 RL（如 PressLight）；多路口协调 MARL（如 CoLight、MPLight）；交通仿真基准与可复现性。说明本文不是宣称首创 MAPPO，而是做系统化比较与需求泛化分析。
4. **问题定义**：路网图、agent、观测、动作、奖励、转移、训练/执行信息边界和安全层。
5. **方法**：SUMO 网络与需求生成；MAPPO 结构/参数共享/集中 critic；基线具体算法；潮汐/引导扩展单独描述。
6. **实验设计**：硬件软件、版本、路网、交通需求、训练预算、种子、超参搜索、指标公式和统计方法。
7. **结果**：总体 KPI 表、低中高需求曲线、训练曲线、箱线/分布图、拥堵热力图、泛化、消融、效率/稳定性。
8. **讨论**：在哪些需求区间有效；平均收益是否以长尾恶化为代价；下游 spillback、过饱和与 teleport；算力、样本成本、仿真到现实差距；为何某些方法失败。
9. **总结与未来工作**：只总结实验支持的结论；未来包括真实路网校准、公交/行人优先、通信失效、鲁棒/安全 RL。
10. **复现附录**：依赖锁定、命令、参数、随机种子、数据生成和结果文件索引。

必须把“仿真结果”限定为仿真结果，不宣称真实城市部署效果。图表结论要能从原始 episode 数据复算。报告至少给出一张网络图、一张方法流程/CTDE 图、一张主要 KPI 表、两张需求/队列趋势图、一张消融图和一张失败/限制分析。

## 11. 风险与验收门槛

### 风险控制

- **接口不兼容**：SUMO-RL、PettingZoo、Gymnasium 版本变化；锁版本，写 smoke test。
- **吞吐量瓶颈**：SUMO CPU/TraCI 慢；先测 steps/s，再决定并发。MAPPO 本身不是瓶颈时不要为了用满 GPU 而增大网络。
- **指标偏差**：只看完成车辆会遗漏卡死车辆；同时报告完成率、残留车、teleport。
- **规则基线太弱**：参数应调到合理周期/阈值；Max-Pressure 实现经 movement queues 验证。
- **训练数据泄漏**：训练/验证/测试需求种子拆分并固定。
- **多机制归因混乱**：信号、路由、可逆车道分阶段并分别消融。
- **安全动作**：策略意图与执行器最终相位均记录；非法/冲突相位请求计数必须为零。

### 最终验收清单

- [ ] 从全新环境按 README 安装，并可运行一个基线 episode 和一个 MARL checkpoint episode。
- [ ] 固定 seed 可复现需求；测试 seed 与训练 seed 不重叠。
- [ ] 固定时制、感应、Max-Pressure、IDQN、IPPO、MAPPO 指标统一且相同需求对照。
- [ ] 所有安全相位约束通过单元测试；主实验非法相位请求为零。
- [ ] 关键结果有多 seed 统计、配对评估、置信区间和原始数据。
- [ ] Demo 用户能调各入口需求、选策略、开始/暂停/重置，并看到路网、相位和 KPI。
- [ ] Demo 同场景对比可导出；高负荷积压和失败状态如实呈现。
- [ ] 技术报告包含结果解释、负面/不确定结果、限制和复现信息。
- [ ] 归档代码 commit、依赖锁、网络和需求配置、训练配置、checkpoint、评估 CSV、生成图表脚本。

## 12. 代表性论文与官方资料

### 仿真与工具

- SUMO 官方文档与下载：[Eclipse SUMO](https://sumo.dlr.de/docs/)、[TraCI](https://sumo.dlr.de/docs/TraCI.html)
- SUMO 仓库：[eclipse-sumo/sumo](https://github.com/eclipse-sumo/sumo)
- SUMO-RL：[LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl)
- CityFlow 论文：Zhang et al., “CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario,” WWW 2019, pp. 3620–3624, [DOI: 10.1145/3308558.3314139](https://doi.org/10.1145/3308558.3314139)。项目：[cityflow-project/CityFlow](https://github.com/cityflow-project/CityFlow)
- RLlib：[ray-project/ray](https://github.com/ray-project/ray)
- EPyMARL：[uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)

### 交通信号控制与 MARL

- Wei et al., “PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network,” KDD 2019, [DOI: 10.1145/3292500.3330949](https://doi.org/10.1145/3292500.3330949)。将压力式交通控制与深度 RL 结合，适合作为奖励/压力基线的相关工作。
- Wei et al., “CoLight: Learning Network-level Cooperation for Traffic Signal Control,” CIKM 2019, [DOI: 10.1145/3357384.3357902](https://doi.org/10.1145/3357384.3357902)。通过图注意力建模邻居路口协作。
- Zheng et al., “Learning Phase Competition for Traffic Signal Control,” CIKM 2019（MPLight），[ACM DL 检索](https://dl.acm.org/action/doSearch?AllField=Learning+Phase+Competition+for+Traffic+Signal+Control)。基于相位竞争的交通信号 RL 方法；引用卷页和 DOI 时请核对出版社元数据。
- Schulman et al., “Proximal Policy Optimization Algorithms,” 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。PPO 基础。
- Yu et al., “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” NeurIPS 2022, [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)。MAPPO 的主要方法参考。
- Lowe et al., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” NeurIPS 2017, [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)。MADDPG 与 centralized training / decentralized execution 背景。
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.，策略梯度、价值函数和实验基础。
- Press et al., “Max Pressure Control of a Network of Signalized Intersections,” Transportation Research Record 2011（基于压力的网络控制理论）；报告引用时补全出版社元数据与页码。

### 研究与代码核对说明

写作前逐篇通过 DOI/出版社页面核验作者、年份、卷页与 DOI；不要从二手博客复制引文。GitHub stars 不代表科学质量或维护承诺。先查看仓库 license、最近 release/commit、issue、Python 兼容性、示例运行方式和依赖约束，再决定是否直接依赖。推荐本项目自行维护薄适配层，避免深度 fork 上游仓库。

## 13. 执行智能体首轮任务模板

将以下文字作为后续 Codex 等编码智能体的启动指令：

> 阅读本 SOP 并在仓库中逐项实现。先调查本机 SUMO/Python/PyTorch/CUDA 驱动实际版本，锁定兼容依赖；先完成单路口 smoke test 和安全相位单测，再做 3×3 SUMO 环境、规则基线、统一指标 evaluator、MAPPO、IPPO/IDQN、多 seed 实验、交互 demo 和报告。每阶段提交可运行结果与复现命令，不得伪造训练结果或把随机策略称为 MARL。核心实验先完成后再扩展路线引导和潮汐车道。所有策略必须采用同一个需求种子进行配对测试，统计完成率、旅行时间、等待、队列、吞吐、长尾和 teleport。结果从原始 CSV 生成图表，报告说明未完成工作、仿真假设和局限。若运行库 API 与文档不同，以安装版本的官方文档/源码为准，并记录版本及改动理由。
