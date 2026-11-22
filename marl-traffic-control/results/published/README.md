# 实验数据索引

| 目录 | 内容 |
| --- | --- |
| [`experiment/`](experiment/REPORT.md) | 报告、冻结协议、固定时制验证选择、需求哈希、评估与路口级数据、训练诊断、配对对比和图 |
| [`calibration/`](calibration/rows.json) | 容量预试记录 |
| [`backend-pilot/`](backend-pilot/comparison.json) | 9 组 TraCI/libsumo 对照结果 |
| `literature/` | PressLight、CoLight、Max Pressure 的 DOI 注册元数据 |

`experiment/tables/episodes.csv` 包含全部 1,260 次评估，`intersections.csv` 包含 20,160 条局部指标，`actions.csv` 包含 600 次主测试动作回放摘要，`training.csv` 包含 1,200 条训练诊断。`contrasts.csv` 保留所有比较，不只保留显著项。在本目录执行 `sha256sum -c sha256.txt` 校验发布文件。

公开版不包含逐秒和逐动作原始轨迹、SUMO tripinfo、模型 checkpoint、本机日志及早期源码快照。按[实验协议](../../docs/PROTOCOL.md#复现顺序)可重新生成，再使用分析脚本核验完整矩阵、重放动作和生成报告。重新训练不保证跨硬件产生逐字节相同的权重。

训练表中的路径相对于实验输出目录，数值不变；路网只改生成器 XML 注释中的本机路径，不改仿真元素。`network-provenance.json` 同时记录原始和发布路网哈希。`experiment/inputs.json` 保留原实验哈希；重新生成的 `network_sha256` 与 `published_sha256` 比较，需求哈希可直接比较。已有原始实验包时，可将原始路网传给分析脚本的 `--recorded-network`，仍执行原始字节哈希检查。

`experiment/protocol.json` 与 `configs/experiment.json` 相同；与原始运行时的配置相比，只更新了输出目录和说明文字，实验参数不变。
