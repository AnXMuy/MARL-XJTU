"""Generate architecture comparison figure and factual report from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 d=json.loads(Path('assets/final_study.json').read_text());scenes=list(d['results'])
 labels=['coordinated_rule','mappo_original','mappo_feasible','mappo512','mappo_lookahead','mpc4h']
 names=['Coordinated rule','MAPPO 256 / 20M','MAPPO + constraint','MAPPO 512 / 30M + constraint','MAPPO + lookahead','Beam MPC 4h']
 fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
 for j,(metric,title) in enumerate([('cost','Daily cost / CNY'),('occupied_degree_hours','Occupied discomfort / degree-hours')]):
  for i,label in enumerate(labels):
   vals=[d['results'][s][label]['summary'][metric] for s in scenes]
   axes[j].bar(np.arange(4)+(i-2.5)*.135,vals,.125,label=names[i])
  axes[j].set_xticks(np.arange(4),scenes);axes[j].set_title(title);axes[j].spines[['top','right']].set_visible(False)
 axes[0].legend(fontsize=8);fig.savefig('assets/report_figs/architecture_comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 for ax,s in zip(axes.flat,scenes):
  for label,name in zip(labels,names):
   m=d['results'][s][label]['summary'];ax.scatter(m['occupied_degree_hours'],m['cost'],s=65,label=name)
  ax.set_title(s);ax.set_xlabel('Occupied discomfort / degree-hours');ax.set_ylabel('Daily cost / CNY');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.savefig('assets/report_figs/cost_comfort.png',dpi=180);plt.close(fig)
 lines=['# 算法与架构优化复验','',
 '本轮没有把扩大网络当作效果保证。完成了三组相同 30M 预算训练（256宽/seed42，512宽/seed42、43），以及联合约束执行、短时域 MPC、学习策略前瞻三类实验。主训练框架仍为实际 BenchMARL/TorchRL MAPPO。',
 '', '## 冻结方案', '',
 '四房共享 actor 提出动作；集中执行层枚举满足总功率的首步供暖组合，照明沿用学习策略；用同一 RC 模型向前模拟32步（8小时），后续动作由 MAPPO 决定，按原团队奖励累计值选择首步，然后下一时刻重新规划。不是单纯换规则，也不是纯分散执行 MAPPO，准确名称是“MAPPO + 基于模型的策略前瞻”。',
 '', '容量可行性包括灯光功率。约束只保证本仿真当前动作的4kW可行性，不保证真实设备安全。未来天气/日程已知是假设；未来语音指令不进入前瞻。预测误差鲁棒性尚未验证。',
 '', '## 实验协议','',
 '512宽复验展示的是两组扩容训练中验证回报较好的seed43，不能代表512宽的种子均值。先在验证日910000–910002比较1/2/4小时，再扩展8小时；按原团队回报选择。前期探索使用100000–100019。为避免反复看测试集，冻结8小时架构后，最终复验改用未参与探索的100020–100039，四种偏好场景，共80个场景日/控制器。不能称为预注册研究；仍需更多训练种子与外部环境验证。',
 '', '下表为新的20日均值。温度偏离是有人区域超出目标±1℃后的度时，不等于不达标步数；低度时可能仍有频繁但轻微的越界。',
 '', '|场景|控制器|电费/元|温度偏离/°C·h|不达标比例|超功率/kWh|','|---|---|---:|---:|---:|---:|']
 for s in scenes:
  for label in labels:
   m=d['results'][s][label]['summary'];lines.append(f"|{s}|{label}|{m['cost']:.3f}|{m['occupied_degree_hours']:.3f}|{m['occupied_violation_rate']:.1%}|{m['overload_kwh']:.4f}|")
 lines+=['','## 探索记录与发现','','|问题|实验|观察|取舍|','|---|---|---|---|',
 '|集中critic是否必需|同actor观测的MAPPO/IPPO，20M步|IPPO有竞争力，未证明MAPPO普遍占优|保留IPPO对照，不宣称集中critic优势|',
 '|动作是否满足容量|原策略与联合可行分配，同模型同日|消除测试超功率，节费变化小|独立保留约束执行层|',
 '|单用物理规划是否更好|1h/4h有限束宽MPC|部分场景费用低，但温度偏离更大|作为模型基线，不当作最优下界|',
 '|规划多远合适|MAPPO续接的1/2/4h前瞻验证|1h回报−72.98，2h−44.18，4h−32.59|短时域拖延供暖，扩大时域|',
 '|8h是否值得|额外8h验证，独立新日复验|验证回报−31.96优于4h；新日结果见上表|冻结8h，不用测试日选时域|',
 '|更大网络是否更好|同30M预算256/512宽，512另一个种子|seed42扩容未胜出；不同种子有变化|不能把容量当作收益因果|',
 '|效果如何呈现|真实策略与规则同步自动运行|同屏数值差值、功率、电费、舒适度可同时观察|避免只展示动作动画|',
 '', '探索原始数字（测试日100000–100019，非最终确认集）：', '', '|场景|控制器|电费/元|温度偏离/°C·h|', '|---|---|---:|---:|']
 for file,models in [('assets/architecture_study.json',['coordinated_rule','mappo','mappo_feasible','mpc_1h']),('assets/lookahead_study.json',['mappo_lookahead'])]:
  exp=json.loads(Path(file).read_text())
  for scene,controllers in exp['results'].items():
   for label in models:
    m=controllers[label]['summary'];lines.append(f"|{scene}|{label}|{m['cost']:.3f}|{m['occupied_degree_hours']:.3f}|")
 lines+=['','## 新方案相对协调规则','', '|场景|电费变化|温度偏离变化|电费差95%配对区间/元|','|---|---:|---:|---|']
 for s in scenes:
  new=d['results'][s]['mappo_lookahead'];m=new['summary'];b=d['results'][s]['coordinated_rule']['summary'];ci=new['paired_vs_rule']['cost']['ci95']
  lines.append(f"|{s}|{(m['cost']/b['cost']-1)*100:+.2f}%|{(m['occupied_degree_hours']/b['occupied_degree_hours']-1)*100:+.2f}%|[{ci[0]:.3f}, {ci[1]:.3f}]|")
 lines+=['','与原MAPPO比较（同权重，仅增加前瞻与可行执行）：','','|场景|电费变化|温度偏离变化|','|---|---:|---:|']
 for scene in scenes:
  new=d['results'][scene]['mappo_lookahead']['summary'];old=d['results'][scene]['mappo_original']['summary']
  lines.append(f"|{scene}|{(new['cost']/old['cost']-1)*100:+.2f}%|{(new['occupied_degree_hours']/old['occupied_degree_hours']-1)*100:+.2f}%|")
 lines+=['','结论：前瞻不是原策略的全面替代。它主要改善默认/省电/改口的舒适控制，睡眠费用降低；默认/省电费用增加，睡眠温度偏离可能略增。保留raw/feasible/lookahead三种模式，不将权衡包装成单指标全面提升。']
 lines+=['','区间来自测试日配对bootstrap，不代表训练种子不确定性。MPC为有限束宽启发式，不是最优控制下界；部分场景省钱但舒适更差，不能只按电费排优劣。费用下降也不等于节电：峰谷移位可降低费用，却因供暖增加而提高总耗电。当前尚未完成严格的同舒适阈值成本曲线。','', '## 扩容消融（相同训练预算）','', '|训练|步数|最佳验证回报|选中步数|用时/秒|峰值torch显存/MiB|','|---|---:|---:|---:|---:|---:|']
 for run in ['mappo256_42','mappo512_42','mappo512_43']:
  p=Path('results/capacity')/run;v=max(json.loads((p/'validation.json').read_text()),key=lambda x:x['validation_return']);t=json.loads((p/'throughput.json').read_text())
  lines.append(f"|{run}|{t['frames']}|{v['validation_return']:.3f}|{v['frames']}|{t['seconds']:.1f}|{t['gpu_peak_mib']:.0f}|")
 lines+=['','同预算模型的探索日测试指标：','','|场景|模型|电费/元|耗电/kWh|温度偏离/°C·h|','|---|---|---:|---:|---:|']
 for file in ['assets/capacity_control.json','assets/capacity_metrics.json']:
  exp=json.loads(Path(file).read_text())
  for scene,controllers in exp['results'].items():
   for label,entry in controllers.items():
    if label.startswith('mappo'):
     m=entry['summary'];lines.append(f"|{scene}|{label}|{m['cost']:.3f}|{m['energy_kwh']:.3f}|{m['occupied_degree_hours']:.3f}|")
 lines+=['','同seed42时512宽的最佳验证回报没有胜过256宽；seed43较好并不足以证明容量因果收益。采用原256策略进行前瞻研究，是为了隔离执行架构变化。多任务并发训练已充分利用GPU计算，显存占用不是实验效果指标。',
 '', '## 展示与边界','',
 '当前前端采用四房数据表与四项定量时序，默认自动推进，支持暂停/倍速、持续麦克风、温度与照明偏好、面积与热连通编辑。主表并列模型、规则、绝对差值与变化比例；房间明细可展开。固定Canvas双缓冲绘制四项曲线，避免逐步替换图片引起闪烁。可切换简单恒温器或协调规则并从同种子重跑，未人为调差规则参数；独立实验表仍采用协调规则。两者同步接收用户目标。累计电费与舒适偏离直接来自运行轨迹。一天结束循环测试日并保留最新用户偏好；这是演示循环，不是连续多日物理预测。',
 '', '仍是四个固定功能区、16–36㎡/区的冬季供暖仿真；未接入真实住房。浏览器声学识别、本地语音服务部署及任意CAD导入未验收。',
 '', '## 复现','', '```bash', 'python -m training.final_study', 'python -m training.optimization_report', 'python -m training.video', 'python demo/app.py --checkpoint results/layout/mappo42/best_actor.pt --controller lookahead --horizon 32 --port 7861','```',
 '', '原始证据：`assets/final_study.json`（模型hash、逐日指标、逐步动作）；扩容对照：`assets/capacity_metrics.json`与`assets/capacity_control.json`；验证过程：`assets/lookahead_selection.json`与`assets/lookahead32_validation.json`。',
 '', '![架构对照](../assets/report_figs/architecture_comparison.png)','', '![费用与舒适](../assets/report_figs/cost_comfort.png)']
 if Path('assets/controller_latency.json').exists():
  latency=json.loads(Path('assets/controller_latency.json').read_text())
  lines+=['','## 执行开销','','单住宅CPU、torch单线程，验证日连续24次决策去掉首步。本机测量，不是通用硬件承诺。','','|控制器|中位/ms|P95/ms|','|---|---:|---:|']
  for name,m in latency['controllers'].items():lines.append(f"|{name}|{m['median_ms']:.2f}|{m['p95_ms']:.2f}|")
 Path('docs/架构优化复验.md').write_text('\n'.join(lines)+'\n')
 # Keep the full research record inside the main technical report too.
 report=Path('docs/技术报告.md');text=report.read_text()
 section='\n\n## 10. 算法与架构优化：探索、消融与复验\n\n'+ '\n'.join(lines[2:]).replace('## ', '### ').replace('../assets/','../assets/')+'\n'
 begin='<!-- BEGIN_ARCHITECTURE_STUDY -->';end='<!-- END_ARCHITECTURE_STUDY -->'
 block=begin+section+end+'\n\n'
 if begin in text:
  first,tail=text.split(begin,1);text=first+block+tail.split(end,1)[1].lstrip()
 else:text=text.replace('## 参考文献',block+'## 参考文献')
 report.write_text(text)
if __name__=='__main__':main()
