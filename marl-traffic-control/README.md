# MARL 城市交通信号控制

在 4×4 合成信号网格上比较固定时制、Max Pressure、共享参数 IPPO 和 MAPPO。独立验证集用于选固定时制；独立测试集用于比较。另有集中 critic、奖励范围、actor 交通观测消融，以及训练预算和分布外需求实验。

## 主要结果

v2 完成 1,200 个训练 episode 和 1,260 次评估。三个需求档位均从六个候选中选中 10 秒固定时制。主指标是观察窗内停车车辆秒除以计划车辆数（秒/车，越低越好）：

| 控制器 | C1 | C2 | C3 |
| --- | ---: | ---: | ---: |
| 固定 10 秒 | 10.05 | 24.82 | 80.98 |
| Max Pressure | 12.43 | 32.11 | 116.41 |
| IPPO | 10.05 | 24.82 | 81.06 |
| MAPPO | 10.05 | 24.82 | 80.99 |
| MAPPO，仅相位 actor | 10.05 | 24.82 | 80.98 |
| MAPPO，网络奖励 | 74.60 | 106.28 | 189.80 |

C1/C2 的 IPPO 和 MAPPO 各有 50/50 次与固定 10 秒的完整记录指标和逐秒轨迹一致。C3 的 MAPPO 相对固定 10 秒差值的交叉 bootstrap 95% 区间为 [−0.41, +0.51]，不支持优越性，也不是等效性检验。实验只覆盖单一网格和两种允许绿灯相位，不能据此推断现实路网部署效果。

[v2 报告](results/published/study-v2/REPORT.md) · [数据索引](results/published/README.md) · [扩展协议与文献](docs/EXTENDED_PROTOCOL.md) · [v1 报告](docs/REPORT.md) · [容量预试](docs/CALIBRATION.md)

## 安装与冒烟验证

在 `marl-traffic-control/` 下执行。原研究使用 Linux x86_64、Python 3.13.13、SUMO-RL 1.4.5、SUMO 1.28.0、PyTorch 2.14.0+cu130。CUDA 训练需兼容驱动；只运行 SUMO API 或规则基线不需要 GPU。已有匹配的宿主 PyTorch 时，可创建 `--system-site-packages` venv 并跳过 Torch 安装。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
# PPO、完整测试套件和 v2 矩阵使用 Torch。按硬件选择对应的 PyTorch wheel：
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip check
python scripts/check_environment.py
python scripts/sumo_smoke.py
python scripts/evaluate_baseline.py --controller max-pressure
python -m pytest -q
```

普通 PyPI 依赖可使用中科大镜像 `--index-url https://pypi.mirrors.ustc.edu.cn/simple/`；如镜像跳转失败则使用默认 PyPI 或可用镜像。SUMO 二进制由 Python wheel 提供；正式 v2 评估额外安装 `libsumo==1.28.0`，训练仍走 TraCI。

## 重建实验

- [v2 复现顺序](docs/EXTENDED_PROTOCOL.md#复现顺序)：生成需求 → CUDA 训练 → 规则基线与验证选时制 → 主测试、预算、OOD → 原始记录审计与汇总。完成后的输出写入 `results/study-v2/`，不影响已发布汇总。
- v1：依次执行 `python scripts/study.py --stage generate`、`--stage train`、`--stage evaluate`（每次均使用同一脚本），再执行 `python scripts/analyze.py`。v1 固定 30 秒的主对照较弱，与 v2 的验证集选时制不能混用。
- 不要让多个研究进程同时写同一个输出目录。不要加载不可信的 `.pt` 文件：PyTorch checkpoint 包含可执行反序列化数据。

## 目录

`configs/` 存放实验配置；`scenarios/` 存放路网和 v1 需求；`src/traffic_signal_marl/` 存放控制器、算法及指标；`scripts/` 存放实验和诊断入口；`tests/` 存放单测与真实 SUMO 集成测试；`results/published/` 存放版本化报告与汇总。
