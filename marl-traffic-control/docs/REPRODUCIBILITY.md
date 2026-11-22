# 复现验证记录

## 运行环境

- Linux x86_64，Python 3.13.13，RTX 4090（49140 MiB，驱动 595.71.05）；PyTorch `2.14.0+cu130`，CUDA 13.0。机器可见 128 个 CPU，cgroup 实际配额为 10 CPU。观察过三并行训练进程合计使用约 1.8 GiB GPU 显存；不是峰值或完整性能分析。仿真通过 TraCI 的独立 SUMO 进程推进，未安装 Python `libsumo`，也未用 GPU 加速交通仿真。
- 从空 venv 经腾讯源安装 `requirements.lock`，`pip check` 通过；独立运行 `scripts/sumo_smoke.py` 通过 3 个 PettingZoo API cycles，`scripts/evaluate_baseline.py --controller max-pressure` 跑通。该新环境未重新下载大体积 CUDA Torch；CUDA 训练及权重复现实验使用本地项目 venv 复用宿主的 Torch。PyTorch CUDA 索引已核对存在 Python 3.13 x86_64 wheel。
- 上述是 v1 的环境和验收记录；原始安装与测试日志未收入 Git。SUMO-RL wrapper 的 `render_mode` 提示是上游警告。

## 从冒烟到冻结矩阵

| 验证步骤 | 结果 |
|---|---|
| RESCO 4×4 API 冒烟 | 真实 SUMO，PettingZoo parallel API 检查通过 |
| 空路网 / 单车 / 同进口 100 车 | 完成、未完成、未插入守恒；验证 null、旅行时间、排队和插入积压 |
| 单路口和 2×2 | 最小绿内请求不切相，切相前三秒为黄灯，观测和动作在空间内 |
| 60 s CUDA 冒烟训练 | IPPO、MAPPO 各一轮，写 checkpoint，loss 有限 |
| 断点训练重放 | CPU 两轮完整训练和第一轮后中断续训权重逐张量相等；不重复记录未提交 episode |
| 600 s 同路线重复评估 | C1 / 1000 / Max Pressure 两次独立运行，指标相等、逐秒 JSON 完全相同 |
| 正式训练 | 9 模型 × 24 轮，216 条日志，首条完成至末条完成 11 分 16 秒；不含训练前安装、场景生成或首轮耗时 |

公开结果与原始记录的范围见[数据索引](../results/published/README.md)。早期冒烟结果不混入正式性能比较。扩展实验的后端对照和复现命令见 [v2 协议](EXTENDED_PROTOCOL.md)。
