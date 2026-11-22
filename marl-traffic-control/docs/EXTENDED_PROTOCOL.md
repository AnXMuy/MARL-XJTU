# 扩展研究 v2：独立需求复验与机制消融

## 从 v1 得到的研究问题

v1 的 24 轮训练中，局部奖励 IPPO/MAPPO 在若干样例上复现了固定 10 秒的轨迹，且网络奖励在部分初始化下出现相位冻结。v2 不把原来较慢的固定时制当成唯一对照，分别问：固定时制选得更强后，学习策略是否还有优势？集中 critic、奖励范围和 actor 交通传感器分别有什么作用？训练预算增加和需求形态变化会改变结论吗？

## 冻结矩阵

| 环节 | 样本和总数 |
|---|---|
| CUDA 训练 | 4 配置 × 5 初始化 × 60 轮 = 1,200；每轮 16 路口、120 个决策点，共 2,304,000 路口决策 |
| 固定时制验证 | 6 周期 × 3 档 × 5 需求 = 90 |
| 基线主测试 | 6 固定 + Max Pressure + hold，共 8 × 3 档 × 10 需求 = 240 |
| 学习策略主测试 | 4 配置 × 5 初始化 × 3 档 × 10 需求 = 600 |
| 24 轮预算切片 | IPPO/MAPPO × 5 初始化 × 3 档 × 5 需求 = 150，与主测试 60 轮行配对 |
| 分布外压力 | 2 基线 + 2 学习方法 × 5 初始化，共 12 × 3 工况 × 5 需求 = 180 |
| 合计 | 1,260 评估；其中 90 验证，840 主测试 |

需求：C1/C2/C3 每 1/0.5/0.25 秒尝试生成一辆车，计划 300/600/1,200 辆；300 秒注入，600 秒截止。沿用 [容量校准](CALIBRATION.md)，C3 指过载压力，不是稳定运行点。训练每 15 轮循环 5 个需求种子 × 3 档。训练需求 101–105；验证 1500–1504；主测试 2000–2009；分布外 3000–3004；初始化 11/22/33/44/55。主测试种子与 v1 分离。不同方法在相同工况/需求上用同一文件和仿真种子。

## 对照与消融

- **固定时制**：10/15/20/30/45/60 秒，按每档验证集主指标均值选取；周期是请求切相的间隔（含切换后的黄灯），不是全程绿灯长度。测试时六档均报告，不能再据测试表选一个当作已验证基线。`hold` 只作退化诊断。
- **IPPO → MAPPO**：相同共享局部 actor 和奖励，只改变 critic 的输入和结构；不能把差异解释为执行期邻居通信。
- **MAPPO → MAPPO-network**：保留结构和训练协议，将局部负停车车辆数改为全网每路口平均的负停车车辆数；保留同一缩放，联合改变了信用分配和奖励异质性。
- **MAPPO → MAPPO-phase**：actor 前三项（相位 one-hot、是否满足换相时间）保留，8 个密度/队列字段置零；网络参数维度、critic 输入与奖励不变。不是所有路口完全开环：actor 仍看相位状态。
- **24 → 60**：同一训练轨迹同一初始化，固定切片；不选最佳测试 checkpoint。v1/v2 同时改变了需求池和训练预算，不是纯预算对照。

## 分布外工况

在独立 C2 规模配对路线基础上，`burst` 保留每辆车路线，分段重映射出发时间，把 70% 车辆压到前 120 秒；`hotspot` 保留车辆数和出发时刻，对包含 C2 相邻边的完整路线给 8 倍重采样权重，因此同时改变起讫和路径结构，不是独立的转向比例消融。`sustained` 每 0.5 秒注入，需求持续 600 秒、计划 1,200 辆、截止 900 秒；车辆绝对规模和观察窗均变化。固定时制使用 C2 验证选择，不针对压力场景再调。仅使用最终 IPPO/MAPPO 权重，不微调。

## 测量与推断

主指标是观察窗内逐秒 SUMO 停车数之和除以计划车辆数；它既包含已完成也包含仍在路网内的车辆，但不包含入网前等待，也不外推截断后的代价。共同报告完成率、未入网与未完成数、待插入车辆秒、旅行时间、实际切相、局部队列积分的最坏值/中位/P95 和 Jain 指数。Jain 越接近 1 只代表队列积分更均匀，不意味着整体队列更短。SUMO-RL 1.4.5 的默认 `time_to_teleport=-1` 关闭堵塞超时传送；只做行程标记和车辆差额审计，不能据差额推断其他传送事件。该版本未执行最大绿灯封顶；仿真为 5 秒决策、3 秒黄灯、5 秒最短绿灯、无全红段。

主比较每档五项：MAPPO 对验证选时制、MAPPO 对 Max Pressure、IPPO 对 MAPPO、网络奖励对 MAPPO、仅相位 actor 对 MAPPO。报告 10,000 次成对需求 bootstrap 和初始化×需求交叉 bootstrap；同一需求/初始化标签内配对。15 个主比较采用条件需求差值精确符号翻转、Holm 校正；预算和 OOD 区间作探索性描述。只有 5 个初始化，不能把 50 个格子当成独立样本；未做等效性检验、未把验证选时制的不确定性纳入区间。统计口径在 v2 评估汇总前固定，研究设计仍受 v1 发现启发。

机制审计将每个最终学习策略的主测试 episode 与同需求固定 10 秒逐秒轨迹及指标对比；再用保存的原始观测离线回放 greedy actor，测算换相请求、合法时点请求、实际切相、交通特征清零和跨路口替换造成的动作翻转。扰动有可能离开训练分布，不能当作封闭环部署干预的因果证据。网络奖励和局部奖励的训练数值不是同一个物理绩效度量。

## 文献与实现对照

- Schulman 等，*Proximal Policy Optimization Algorithms*，[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)，2017。这里复用概率比裁剪和 GAE，不把近似 KL、裁剪率本身当成性能保证。
- Yu 等，*The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*，[NeurIPS 2022 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)，[arXiv:2103.01955](https://arxiv.org/abs/2103.01955)。原论文强调实现和超参数；本项目的展平全局观测 critic 是受其启发的版本，未实现 value normalization、value loss clipping、orthogonal initialization 或 death masking，也未复制论文任务和训练预算。
- Varaiya，*Max pressure control of a network of signalized intersections*，Transportation Research Part C 36 (2013), 177–195，[DOI:10.1016/j.trc.2013.08.014](https://doi.org/10.1016/j.trc.2013.08.014)。工程基线按绿灯连接累加入口车道与出口车道的在道车辆数差；没有估计转向比例，也不引用理论控制器的定理保证本实验稳定性。
- Henderson 等，*Deep Reinforcement Learning That Matters*，[AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694)，[arXiv:1709.06560](https://arxiv.org/abs/1709.06560)。相同预算、公开训练种子、明确指标和实现细节比展示最好一次更重要。
- Agarwal 等，*Deep Reinforcement Learning at the Edge of the Statistical Precipice*，[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)，[arXiv:2108.13264](https://arxiv.org/abs/2108.13264)。本项目采纳报告区间的原则，但不是直接套用其跨任务分层方法：这里需求与初始化有交叉重复结构。
- Wei 等，*PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*，KDD 2019，1290–1298，[DOI:10.1145/3292500.3330949](https://doi.org/10.1145/3292500.3330949)。可作为压力奖励的后续独立对照；本项目的负停车数不是 PressLight 的压力奖励。
- Wei 等，*CoLight: Learning Network-level Cooperation for Traffic Signal Control*，CIKM 2019，1913–1922，[DOI:10.1145/3357384.3357902](https://doi.org/10.1145/3357384.3357902)。图注意力解决的是执行期邻居信息利用；本项目没有复现 CoLight，集中 critic 与此方向不能互相替代。两项 DOI 与 Max Pressure 的注册元数据保存在 `results/published/literature/`。

## 复现顺序

在项目目录启用已安装 `sumo-rl==1.4.5` 的 `.venv`，原冻结目录不重复生成。新实验先复制 `configs/extended.json` 并改 `output`，然后给各阶段传同一 `--protocol` 参数。每个阶段单独记日志，训练自动从 `checkpoint.pt` 恢复，已完成评估会检查输入与权重哈希后跳过。

```bash
.venv/bin/python scripts/extended.py --stage generate
.venv/bin/python scripts/extended.py --stage train --workers 8
.venv/bin/python -m pip install libsumo==1.28.0
LIBSUMO_AS_TRACI=quiet .venv/bin/python scripts/extended.py --stage baselines --workers 8
.venv/bin/python scripts/extended.py --stage select
LIBSUMO_AS_TRACI=quiet .venv/bin/python scripts/extended.py --stage evaluate --workers 8
LIBSUMO_AS_TRACI=quiet .venv/bin/python scripts/extended.py --stage ood --workers 8
.venv/bin/python scripts/analyze_extended.py --output results/study-v2
.venv/bin/python -m pytest -q
```

训练始终使用 TraCI，8 个 CUDA 进程共享一张 48 GiB 级显卡。最初 2 worker TraCI 基线停止后保留已生成的原始结果于 `evaluation-traci-pilot/`（不纳入正式矩阵）；正式评估统一使用同版本 SUMO 的进程内 libsumo，并检查缓存后端不混用。后端配对小样本（每档固定 10 秒、Max Pressure、24 轮 MAPPO 各一个，需求 2000）9/9 的逐秒轨迹、流量指标、可用动作逐字节/逐值一致；中位单集计时比 13.74×，并非整套任务加速比。样本只用于后端校验，不用于选策略或时制。公开证据见[后端对照记录](../results/published/backend-pilot/comparison.json)，用 `.venv/bin/python scripts/compare_backends.py` 重跑。CPU 配额为 10 核；GPU 只承担 PPO 和网络计算。
