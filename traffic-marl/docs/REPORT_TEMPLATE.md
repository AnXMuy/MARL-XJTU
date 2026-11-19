# 基于多智能体强化学习的城市交通信号协同控制：面向需求波动的仿真评估

## 摘要
仅基于 `runs/` 原始 episode CSV 填写结果。没有真实仿真结果时不得声称 MARL 优于规则策略。

## 研究问题与贡献
局部可观测路口协作、需求偏移泛化、安全和效率权衡。

## 仿真环境与边界
真实 SUMO 版本、1 s step；首版仅直行车流、每进口单车道、相位 0/1 放行互不冲突的方向组；明确无转向/行人/公交。网络为 1x1/3x3/4x4，需求 pre-generated。

## 方法
状态为进口车道 halting count、占有率、等待、下游占有、当前相位与持续时间；MAPPO 参数共享 actor、mean pooled centralized critic；IPPO 本地 critic；IDQN 独立 Q；请求下一合法相位，由安全状态机完成黄灯和全红。写明 reward 的实现形式、裁剪和团队/局部设定。

## 实验设计
SUMO/Python/PyTorch/驱动、CPU/GPU、配置 hash；fixed-time、actuated、movement queue pressure 和可选 offset。相同需求 XML SHA 对齐。训练 seeds 与 demand seeds 分离，记录总 environment/agent steps。

## 指标定义
车辆完成 cohort 的旅行时间；平均延误=旅行时间减 route free-flow；车辆停驶秒数不称作 travel delay；完成率分母为 scheduled vehicles；teleport 禁止并按 episode 记录；队列以每秒 halt count 计算；公平性报告 p95 与 Gini/Jain。

## 结果
均值、标准差、bootstrap CI、配对差值；正面、无显著差异和失败 episode 全部报告。附原始数据生成图。

## 消融与泛化
MAPPO/IPPO、共享/独立 actor、团队/局部奖励、下游观测、队列/丰富观测、邻居观测、需求随机化；每次只改一个因素。4x4 zero-shot 需要共享 actor 和尺寸无关 critic。

## 讨论、限制和复现附录
注明 SUMO 仿真假设不代表真实城市部署。附依赖锁、commit、SUMO 版本、命令、种子表、checkpoint hash、失败列表。
