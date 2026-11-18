# 研究依据与框架决策

调研日期：2026-11-18。已读取以下原始论文摘要和官方仓库资料。

| 来源 | 判断与本项目采用方式 |
|---|---|
| [BenchMARL](https://github.com/facebookresearch/BenchMARL) | 主训练框架，实际使用 Experiment、MAPPO/IPPO、TorchRL collector、GAE 和 PPO loss；并非自写 PPO 改名。 |
| [TorchRL](https://pytorch.org/rl/) | TensorDict 批量环境接口，物理仿真与采样张量留在 GPU，避免逐房间 Python 环境调用。 |
| [MAPPO，Yu et al., NeurIPS 2022](https://arxiv.org/abs/2103.01955) | 合作任务的可靠候选；采用共享 actor、集中 critic。原论文并未证明其在智能家居一定最优。 |
| [官方 MAPPO 实现](https://github.com/marlbenchmark/on-policy) | 对照 PPO、GAE、参数共享等实现，项目实际执行的是 BenchMARL 实现。 |
| [QMIX，Rashid et al., ICML 2018](https://arxiv.org/abs/1803.11485) | 单调价值混合有表示约束，但不能据此断言旧项目效果差都是算法责任。按用户要求不再使用。 |
| [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) | 借鉴需求响应与统一天气、价格的控制器比较方法。本项目不是 CityLearn benchmark。 |
| [Sinergym](https://github.com/ugr-sail/sinergym) | EnergyPlus 高保真迁移候选；当前 RC 仿真尚不能证明真实住宅节能。 |
| [home-llm](https://github.com/acon96/home-llm) | 本地语言模型控制集成参考；本项目语言层只修改偏好，动作交给控制策略。无“首个语音+MARL”主张。 |
| [CleanMARL](https://github.com/AmineAndam04/cleanmarl) | 算法可读性参考，未作为当前训练框架。 |

## 为什么采用 BenchMARL + MAPPO

BenchMARL 提供统一的多算法实验接口，TorchRL 支持 GPU 批量采样，适合本项目的四区域协作和共享奖励。选 MAPPO 是可检验的工程起点；IPPO 在完全相同的 actor 信息、模型大小、采样预算下作为对照。没有证据时不宣称集中 critic 必然有优势。

更大模型不等同于更好控制。当前 actor 为三层 256、critic 为三层 512；先通过 1024 个并行环境、四次独立训练提高 GPU 吞吐和实验可信度。保留 hidden/layers/envs 参数用于后续容量消融。正式主线未使用自写试验中的 PopArt，不混淆两个实现。

## 结论边界

新旧环境的结果不做百分比前后比较。测试日与模型选择验证日分离。日间 bootstrap 与训练种子离散度分别报告。成本、舒适偏离、照明不匹配和超功率同时考察，不能只挑低费用。框架迁移不能替代奖励、观测、单位与基线设计。
