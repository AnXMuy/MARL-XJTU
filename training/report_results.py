"""Insert measured tables and verification records in the technical report."""
import json,re
from pathlib import Path

def main():
 d=json.loads(Path('assets/metrics.json').read_text());parts=['每个单元为20个固定测试日平均；所有测试均为当前v3物理模型。','', '| 场景 | 控制器 | 电费/元 | 舒适偏离/℃·h | 违规时间比例 | 超容量/kWh | 照明不符 |','|---|---|---:|---:|---:|---:|---:|']
 for scene,runs in d['results'].items():
  for label,run in runs.items():
   m=run['summary'];parts.append(f"| {scene} | {label} | {m['cost']:.3f} | {m['occupied_degree_hours']:.3f} | {m['occupied_violation_rate']:.2%} | {m['overload_kwh']:.4f} | {m['light_mismatch']:.3f} |")
 parts+=['','### MAPPO训练种子差异','', '| 场景 | 日电费：三种子均值±标准差 | 舒适偏离：均值±标准差 |','|---|---:|---:|']
 ss=json.loads(Path('assets/report_figs/seed_summary.json').read_text())
 for scene,m in ss.items():
  c=m['cost'];v=m['occupied_degree_hours'];parts.append(f"| {scene} | {c['training_seed_mean']:.3f} ± {c['training_seed_std']:.3f} | {v['training_seed_mean']:.3f} ± {v['training_seed_std']:.3f} |")
 parts+=['','### 与协调规则的成对差异','', '以下为预先指定种子42，负数表示MAPPO更低；区间是测试日bootstrap，不代表训练种子置信区间。','', '| 场景 | 电费差 [95% CI] | 舒适偏离差 [95% CI] |','|---|---:|---:|']
 for scene,entries in d['paired_vs_coordinated_rule'].items():
  row=[]
  for k in ['cost','occupied_degree_hours']:
   m=entries['mappo42'][k];lo,hi=m['paired_bootstrap_95ci'];row.append(f"{m['mean_difference']:.3f} [{lo:.3f}, {hi:.3f}]")
  parts.append(f'| {scene} | '+ ' | '.join(row)+' |')
 parts+=['','### 结果解释','']
 for scene,runs in d['results'].items():
  ref=runs['coordinated_rule']['summary'];avg=ss[scene]
  saving=(1-avg['cost']['training_seed_mean']/ref['cost'])*100
  change=(1-avg['occupied_degree_hours']['training_seed_mean']/ref['occupied_degree_hours'])*100
  parts.append(f"- {scene}：MAPPO三种子平均费用相对协调规则{'降低' if saving>=0 else '增加'}{abs(saving):.2f}%，舒适偏离{'降低' if change>=0 else '增加'}{abs(change):.2f}%。")
 parts+=['','必须结合超容量和照明指标判断，不能将上述平均变化写成所有测试日保证。IPPO仅一个种子，不足以支持算法显著性排名。','', '![总体对照](../assets/report_figs/comparison.png)','', '![固定测试日时序](../assets/report_figs/day_trace.png)','', '### 训练资源','', '| 运行 | 环境转移 | 用时/秒 | 峰值torch分配/MiB |','|---|---:|---:|---:|']
 for name in ['mappo42','mappo43','mappo44','ippo42']:
  m=json.loads((Path('results/layout')/name/'throughput.json').read_text());parts.append(f"| {name} | {m['frames']} | {m['seconds']:.1f} | {m['gpu_peak_mib']:.1f} |")
 p=Path('docs/技术报告.md');text=p.read_text();text=re.sub(r'<!-- RESULTS -->.*?<!-- END_RESULTS -->','<!-- RESULTS -->\n'+'\n'.join(parts)+'\n<!-- END_RESULTS -->',text,flags=re.S)
 checks=['发布验收：24项单元与集成测试通过（8条第三方弃用/未来兼容警告），见 `assets/evidence/tests.txt`。','实际框架导出动作一致性：见 `assets/evidence/export_check.txt`。','四条指令与一天回放：见 `assets/v2/scripted_trace.json`。']
 bp=Path('assets/demo/browser_check.json')
 if bp.exists():checks.append('浏览器测试：'+bp.read_text())
 else:checks.append('浏览器检查尚未完成；不能记作通过。')
 text=re.sub(r'<!-- VALIDATION -->.*?<!-- END_VALIDATION -->','<!-- VALIDATION -->\n'+'\n\n'.join(checks)+'\n<!-- END_VALIDATION -->',text,flags=re.S);p.write_text(text)
if __name__=='__main__':main()
