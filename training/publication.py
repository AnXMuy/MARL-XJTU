"""Export measured tables and verify archived model/data integrity (no training)."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['cost', 'energy_kwh', 'occupied_degree_hours', 'occupied_violation_rate',
           'peak_kw', 'overload_kwh', 'team_return', 'light_mismatch']
STUDIES = ['metrics', 'architecture_study', 'lookahead_study', 'mpc4h_study',
           'capacity_control', 'capacity_metrics', 'final_study']


def main():
    target = ROOT / 'assets/tables'
    target.mkdir(exist_ok=True)
    summaries, days = [], []
    sections = ['## 11. 完整指标附件与数据溯源', '',
        '以下表格由 `python -m training.publication` 从已保存JSON直接生成，没有补造或手填实验数字。每行是20个测试日的算术均值；峰值列为每日峰值的均值，不是跨日最大值。违规率为各日“违规有人房间小时 / 有人房间小时”的均值，不是任一房间违规的整屋步数比例。', '',
        '### 11.1 指标定义', '',
        '|字段|定义与单位|方向|', '|---|---|---|',
        '|cost|Σ price(t) × power(t) × 0.25，元/日|低|',
        '|energy_kwh|Σ power(t) × 0.25，kWh/日|低，须结合舒适|',
        '|occupied_degree_hours|Σ 有人指示 × max(绝对温差−1,0) × 0.25，℃·h/日|低|',
        '|occupied_violation_rate|Σ 有人且温差>1的房间小时 / Σ有人房间小时，无占用时为0|低|',
        '|peak_kw|一天总功率最大值，kW|结合4kW容量|',
        '|overload_kwh|Σ max(power−4,0) × 0.25，kWh/日|低|',
        '|team_return|96步原始团队奖励之和；不含训练用0.2缩放|高|',
        '|light_mismatch|96步有效占用房间照明不符比例的均值|低|', '',
        '舒适度报告将房间度时求和，奖励则在每步对有效占用房间取平均，二者不是同一归一化。能量和电费相关但不能互换；低价时供暖增加可能改善舒适，却增加用电。', '',
        '### 11.2 数据分层', '',
        '|阶段|样本与用途|选择约束|', '|---|---|---|',
        '|训练|1024并行家庭，每回合随机物理参数、日程、偏好|训练分布内采样|',
        '|权重验证|seed900000起128家庭，随机中途偏好|按验证回报选权重|',
        '|时域验证|910000–910002，四场景|比较1/2/4小时，再扩展8小时|',
        '|探索评估|100000–100019，四场景|用于发现问题，不作为最终独立确认|',
        '|冻结后复验|100020–100039，四场景，六控制器|480次整日回放，同条件配对|', '',
        '这些是同一模拟器分布的新种子，不能称跨建筑、跨季节外部分布验证。冻结方案仍源于此前探索，不声称预注册或反复试验后的无偏全流程统计检验。', '',
        '### 11.3 冻结方案完整八指标', '']
    for study in STUDIES:
        data = json.loads((ROOT / f'assets/{study}.json').read_text())
        results = data.get('results', data)
        for scene, controllers in results.items():
            if 'summary' in controllers:
                controllers = {'mpc4h': controllers}
            for controller, record in controllers.items():
                summary = record['summary']
                rows = record['days']
                assert len(rows) == 20, (study, scene, controller)
                for metric in METRICS:
                    avg = sum(row['metrics'][metric] for row in rows) / len(rows)
                    assert abs(avg - summary[metric]) < 1e-6, (study, metric)
                summaries.append(dict(study=study, scene=scene, controller=controller, n=len(rows), **summary))
                for row in rows:
                    days.append(dict(study=study, scene=scene, controller=controller, seed=row['seed'], **row['metrics']))
    for name, rows, extra in [('summary', summaries, 'n'), ('daily', days, 'seed')]:
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['study', 'scene', 'controller', extra] + METRICS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    for title, metrics, headers in [
        ('费用、能量与舒适', METRICS[:4], ['电费/元', '电量/kWh', '偏离/℃·h', '违规比例']),
        ('容量、回报与照明', METRICS[4:], ['日峰均值/kW', '超容量/kWh', '团队回报', '照明不符'])]:
        sections += ['#### ' + title, '', '|场景|控制器|' + '|'.join(headers) + '|', '|---|---|' + '---:|' * 4]
        for row in summaries:
            if row['study'] != 'final_study':
                continue
            values = [f"{row[k]:.2%}" if k == 'occupied_violation_rate' else f"{row[k]:.4f}" for k in metrics]
            sections.append('|' + '|'.join([row['scene'], row['controller']] + values) + '|')
        sections.append('')
    sections += ['### 11.4 所有探索的机器可读附件', '',
        f'完整CSV包含 **{len(summaries)}行场景/控制器汇总、{len(days)}行逐日记录**。不同研究复用相同测试日与基线，所以这些行不能当作独立样本数。', '',
        '- [summary.csv](../assets/tables/summary.csv)：七组研究全部八指标。',
        '- [daily.csv](../assets/tables/daily.csv)：逐日种子与原始指标，可重算配对差。',
        '- [数据字典](../assets/tables/README.md)：文件、控制器名、统计口径。',
        '- [证据清单](../assets/evidence/manifest.json)：数据及七个actor的SHA256。', '',
        '发布权重是验证选中的actor，用于评估与演示；不含critic、优化器和采样器状态，因此不是可无缝续训的完整训练快照。完整训练可按配置从头复现，硬件/库版本变化可能产生数值差异。', '']
    report = ROOT / 'docs/技术报告.md'
    text = report.read_text()
    block = '<!-- PUBLICATION_TABLES -->\n' + '\n'.join(sections) + '\n<!-- END_PUBLICATION_TABLES -->\n\n'
    if '<!-- PUBLICATION_TABLES -->' in text:
        text = re.sub(r'<!-- PUBLICATION_TABLES -->.*?<!-- END_PUBLICATION_TABLES -->\n\n', lambda _: block, text, flags=re.S)
    else:
        text = text.replace('## 参考文献', block + '## 参考文献')
    report.write_text(text)
    files = sorted(set((ROOT / 'assets').glob('*.json')) | set((ROOT / 'results').glob('*/*/best_actor.pt')) |
                   set((ROOT / 'results').glob('*/*/run_manifest.json')) | set((ROOT / 'results').glob('*/*/validation.json')) |
                   set((ROOT / 'results').glob('*/*/throughput.json')) | set(target.glob('*.csv')))
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    final = json.loads((ROOT / 'assets/final_study.json').read_text())
    assert manifest[final['checkpoint']]['sha256'] == final['sha256']
    assert manifest[final['larger_policy_checkpoint']]['sha256'] == final['larger_policy_sha256']
    (ROOT / 'assets/evidence/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'summary_rows': len(summaries), 'daily_rows': len(days), 'files_hashed': len(manifest), 'aggregate_and_checkpoint_checks': 'passed'}))


if __name__ == '__main__':
    main()
