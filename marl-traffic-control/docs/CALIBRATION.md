# 容量预试与资料核对

在独立需求 71、72 上试跑 `scripts/calibrate.py`：固定路网，前 300 s 注入、600 s 停止；两个基线 × 五档注入周期 × 两个需求，共 20 个 episode。单元格分别列出两个种子的结果。

| 注入周期 (s) | 计划车数 | 固定时制完成数 | 最大压力完成数 | 固定时制等待 (s/车) | 最大压力等待 (s/车) |
|---:|---:|---:|---:|---:|---:|
| 3 | 100 | 100 / 100 | 100 / 100 | 21.75 / 26.23 | 4.13 / 4.09 |
| 1 | 300 | 300 / 300 | 300 / 300 | 23.99 / 29.54 | 10.84 / 12.96 |
| 0.5 | 600 | 600 / 600 | 600 / 600 | 51.85 / 50.54 | 29.17 / 28.34 |
| 0.25 | 1200 | 820 / 778 | 927 / 907 | 152.68 / 158.37 | 104.38 / 100.24 |
| 0.125 | 2400 | 150 / 137 | 187 / 172 | 328.55 / 330.98 | 304.97 / 309.56 |

1 s 与 0.5 s 都排空，但后者等待增加，选为 C1 与 C2。0.25 s 时全部计划车可插入，却在控制窗口内明显积压，选为 C3 过载压力测试。0.125 s 时未插入达 134–194 辆且完成率仅 5.7%–7.8%，不作为主实验档位。此结果是有限时间脉冲注入证据，不是稳态容量估计；正式测试不使用预试种子。完整数据见[容量预试记录](../results/published/calibration/rows.json)；预试前半段尚未保存 `pending_vehicle_seconds` 字段，本文表格不依赖它。

## 文献与实现版本

- Lopez et al. (2018), *Microscopic Traffic Simulation using SUMO*, IEEE ITSC，[DOI 10.1109/ITSC.2018.8569938](https://doi.org/10.1109/ITSC.2018.8569938)。题名与年份经 Crossref 核对。
- Pravin Varaiya (2013), *Max pressure control of a network of signalized intersections*, Transportation Research Part C 36, 177–195，[DOI 10.1016/j.trc.2013.08.014](https://doi.org/10.1016/j.trc.2013.08.014)。经 Crossref 核对；项目基线采用简化等权车辆数压力。
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*，[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。
- Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*，[NeurIPS 2022 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)。
- [SUMO-RL 1.4.5 PyPI 发布](https://pypi.org/project/sumo-rl/1.4.5/)（MIT），[上游仓库](https://github.com/LucasAlegre/sumo-rl)。SUMO 1.28.0 来自 `eclipse-sumo` 与 `sumo-data` PyPI wheel，发布许可为 EPL-2.0 或 GPL-2.0-or-later。环境版本见[实验环境记录](../results/published/experiment/environment.json)。
