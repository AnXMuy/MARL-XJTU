# 实验数据字典

所有CSV从仓库JSON导出：`python -m training.publication`。UTF-8 BOM编码，可直接用Excel打开。脚本逐项检验汇总值等于20日平均，并核对最终实验权重SHA256。

|文件|粒度|用途|
|---|---|---|
|summary.csv|study × scene × controller|108行，七组研究的完整八指标|
|daily.csv|再加seed|2160行，重算配对统计；跨study存在重复样本，不应合并当作独立样本|

`scene`：default默认、eco省电、sleep睡眠、changes连续改口。初始偏好及定时改口定义见 `experiments/v2/evaluate.py`；所有方法共享相同外生序列。

`study`对应 `assets/<study>.json`：metrics为初始多种子；architecture_study为约束及1h MPC；lookahead_study为4h策略前瞻；mpc4h_study为4h MPC；capacity_control为256宽30M；capacity_metrics为512宽30M两种子；final_study为冻结8h方案新日复验。

控制器名称：thermostat简单占用恒温器；coordinated_rule带预热与容量分配的协调规则；mappo42/43/44与ippo42为训练种子；mappo_original为20M种子42原策略；mappo_feasible为同权重联合可行执行；mappo512在最终研究中特指30M种子43加可行执行；mappo_lookahead在探索中为4h、最终为8h；mpc4h为有限束宽32的4h模型规划。不同study的相似名称不保证架构完全相同。

八个字段的单位与计算式见技术报告11.1节。百分率在CSV存0–1实数。`n`为汇总的日数量，`seed`是外生条件种子。不要将team_return误作元或将舒适偏离误作百分率。

JSON包含96步轨迹时，可进一步检查温度、目标、占用、动作与成本；architecture_study仅存逐日汇总。最终配对bootstrap区间保存在final_study.json各控制器的paired_vs_rule。分析时以相同seed配对，区别测试日不确定性与训练种子不确定性。
