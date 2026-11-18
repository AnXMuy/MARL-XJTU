"""Render only measured results; no prescribed savings or hand-edited values."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--metrics',default='assets/metrics.json');p.add_argument('--output',default='assets/report_figs');a=p.parse_args()
 d=json.loads(Path(a.metrics).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 scenes=list(d['results']);policies=list(d['results'][scenes[0]])
 colors=['#a8aaa1','#bd865f','#235f54','#468779','#84ab9d','#5a6b94']
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 for ax,metric,title in zip(axs.flat,['cost','occupied_degree_hours','overload_kwh','light_mismatch'],['Daily cost / CNY','Occupied discomfort / degree-hours','Capacity exceedance / kWh','Lighting mismatch / fraction']):
  for i,policy in enumerate(policies):
   ax.bar(np.arange(4)+(i-(len(policies)-1)/2)*.13,[d['results'][s][policy]['summary'][metric] for s in scenes],width=.12,label=policy,color=colors[i%len(colors)])
  ax.set_xticks(np.arange(4),scenes);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 axs[0,0].legend(fontsize=8);fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
 for policy,color in [('coordinated_rule','#bd865f'),('mappo42','#235f54'),('ippo42','#5a6b94')]:
  rows=d['results']['changes'][policy]['days'][0]['trajectory'];x=np.arange(96)/4
  axs[0].plot(x,[r['power_kw'] for r in rows],label=policy,color=color)
  axs[1].plot(x,[r['temps'][0] for r in rows],label=policy,color=color)
  axs[2].plot(x,np.cumsum([r['cost'] for r in rows]),label=policy,color=color)
 axs[0].axhline(4,color='red',ls='--',label='4 kW soft capacity');axs[0].legend()
 for ax,label in zip(axs,['Power / kW','Living room / C','Cumulative cost / CNY']):
  ax.set_ylabel(label)
  for hour in [7,17,21]:ax.axvline(hour,color='#aaa',ls=':')
 axs[-1].set_xlabel('Hour; dotted lines = preference changes');fig.savefig(out/'day_trace.png',dpi=170);plt.close(fig)
 summary={s:{} for s in scenes}
 for s in scenes:
  for metric in ['cost','occupied_degree_hours','overload_kwh']:
   values=[d['results'][s][f'mappo{k}']['summary'][metric] for k in [42,43,44]]
   summary[s][metric]={'training_seed_mean':float(np.mean(values)),'training_seed_std':float(np.std(values,ddof=1)),'seeds':[42,43,44]}
 (out/'seed_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
