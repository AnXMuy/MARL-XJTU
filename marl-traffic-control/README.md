# MARL 城市交通信号控制

**任务**：控制 SUMO 4×4 路网中的 16 个信号路口。每 5 秒，各路口决定保持当前绿灯还是请求切换到另一绿灯；所有方法共用 5 秒最小绿和 3 秒黄灯。低、中、高需求（C1/C2/C3）分别在前 300 秒计划发车 300、600、1,200 辆，统一在第 600 秒结束观测。C3 是过载压力测试。

**重点对比**：MAPPO 的各路口根据局部观测选相，训练时使用集中 critic。本文重点展示两项对比：与按进出车道车辆数差选相的 Max Pressure 比较，以及用网络奖励替换局部奖励的消融。

## 主要结果

四个学习配置各训练 5 个初始化、每个 60 轮，共 1,200 个训练回合；另有 1,260 次评估。主指标是观察窗内道路停车车辆秒除以计划车辆数（秒/车，越低越好；不含入网前排队）。

| 控制器 | C1 | C2 | C3 |
| --- | ---: | ---: | ---: |
| Max Pressure | 12.43 | 32.11 | 116.41 |
| MAPPO，局部奖励 | **10.05** | **24.82** | **80.99** |
| MAPPO，网络奖励 | 74.60 | 106.28 | 189.80 |

- **高需求下，比本项目的 Max Pressure 少 30.4%**：低、中、高需求分别少 2.38、7.29、35.41 秒/车（三项 Holm 校正 p = 0.0293）。高需求的 95% 差值区间为 [−42.83, −29.59] 秒/车。
- **奖励消融：高需求下，局部奖励优于网络奖励**。改用网络奖励后，指标增加 108.80 秒/车（Holm 校正 p = 0.0293，95% 差值区间 [17.12, 210.09]）。差值由未四舍五入的数据计算。

### 未见显著差异的对照

下表列出高需求（C3）的配对结果。差值为左减右，单位为秒/车；95% 区间同时考虑需求和训练初始化。

| 对比 | 差值 | 95% 区间 | Holm 校正 p 值 |
| --- | ---: | ---: | ---: |
| 局部奖励 MAPPO − 验证选出的 10 秒固定时制 | +0.01 | [−0.41, +0.51] | 1.0000 |
| IPPO − 局部奖励 MAPPO | +0.07 | [−0.27, +0.42] | 1.0000 |
| 仅相位 actor 的 MAPPO − 局部奖励 MAPPO | −0.01 | [−0.51, +0.41] | 1.0000 |

低、中需求下，这三项对照也未检出显著差异。未检出差异不代表等效；结果也不足以把相对 Max Pressure 的收益归因于集中 critic 或路口协同。p 值以实际训练的初始化为条件，完整对照见实验报告附录。

[完整报告](results/published/experiment/REPORT.md) · [数据索引](results/published/README.md) · [实验协议与文献](docs/PROTOCOL.md) · [容量预试](docs/CALIBRATION.md)

## 安装与冒烟验证

在 `marl-traffic-control/` 下执行。原研究使用 Linux x86_64、Python 3.13.13、SUMO-RL 1.4.5、SUMO 1.28.0、PyTorch 2.14.0+cu130。CUDA 训练需兼容驱动；只运行 SUMO API 或规则基线不需要 GPU。已有匹配的宿主 PyTorch 时，可创建 `--system-site-packages` venv 并跳过 Torch 安装。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
# PPO、完整测试套件和正式实验需要 Torch。按硬件选择对应的 PyTorch wheel：
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip check
python scripts/check_environment.py
python scripts/sumo_smoke.py
python scripts/evaluate_baseline.py --controller max-pressure
python -m pytest -q
```

普通 PyPI 依赖可使用中科大镜像 `--index-url https://pypi.mirrors.ustc.edu.cn/simple/`；如镜像跳转失败则使用默认 PyPI 或可用镜像。SUMO 二进制由 Python wheel 提供；正式评估额外安装 `libsumo==1.28.0`，训练仍走 TraCI。

## 重建实验

- [复现顺序](docs/PROTOCOL.md#复现顺序)：生成需求 → CUDA 训练 → 规则基线与验证选时制 → 主测试、预算、分布外测试 → 原始记录审计与汇总。新输出写入 `results/experiment/`，不影响已发布汇总。
- 不要让多个研究进程同时写同一个输出目录。不要加载不可信的 `.pt` 文件：PyTorch checkpoint 包含可执行反序列化数据。

## 目录

`configs/` 存放实验配置；`scenarios/` 存放路网及冒烟和容量校准输入；`src/traffic_signal_marl/` 存放控制器、算法及指标；`scripts/` 存放实验和诊断入口；`tests/` 存放单测与真实 SUMO 集成测试；`results/published/` 存放报告与汇总。
