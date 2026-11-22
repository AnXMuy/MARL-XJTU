# MARL-XJTU：城市交通信号控制

本分支发布基于 SUMO-RL 1.4.5 的 4×4 网格信号控制基准，比较固定时制、Max Pressure、共享参数 IPPO 和 MAPPO，并检验 critic、奖励范围和 actor 交通观测的作用。

- [任务、方法、主要结果与安装](marl-traffic-control/README.md)
- [实验设定和复现命令](marl-traffic-control/docs/PROTOCOL.md)
- [完整报告](marl-traffic-control/results/published/experiment/REPORT.md) · [数据索引](marl-traffic-control/results/published/README.md)

## 主要结果

- **显著差异**：高需求下，局部奖励 MAPPO 的道路内停车指标为 **80.99 秒/车**，低于本项目 Max Pressure 的 116.41 秒/车，降幅 **30.4%**；改用网络奖励后升至 **189.80 秒/车**。两项对比的 Holm 校正 p 值均为 0.0293。
- **未见显著差异**：高需求下，MAPPO 未显著优于独立验证选出的 10 秒固定时制（80.99 对 80.98 秒/车）；MAPPO 对比 IPPO、使用交通观测的 MAPPO 对比仅用相位的 MAPPO，也未检出显著差异。这不代表这些方法等效，也不能把相对 Max Pressure 的优势归因于路口协同。

本分支只包含交通信号项目；其他项目仍可在其原分支或历史提交中访问。第三方依赖按其自身许可分发。本项目尚未指定独立的再分发许可证；公开可见不等于授予开源许可。
