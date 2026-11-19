"""Matched evaluation. Every strategy receives byte-identical generated demand files."""
import argparse
import itertools
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from traffic_marl.config import Config
from traffic_marl.utils.provenance import save_json
from evaluate import run

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='runs/matrix')
    p.add_argument('--seeds',type=int,default=5); p.add_argument('--demand-seeds',type=int,default=20)
    p.add_argument('--duration',type=int,default=3600); p.add_argument('--size',type=int,default=3)
    p.add_argument('--checkpoint-dir',action='append',default=[]); p.add_argument('--limit',type=int)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    for entry in args.checkpoint_dir:
        name,path=entry.split('=',1)
        if ':' not in name: raise ValueError('learned checkpoint names must be algorithm:training_seed=path')
        algorithm,seed=name.rsplit(':',1); checkpoints[(algorithm,int(seed))]=path
    policies=['fixed','actuated','max_pressure','green_wave']+sorted({algorithm for algorithm,_ in checkpoints})
    conditions=list(itertools.product(range(args.demand_seeds),[150,300,600],['uniform','morning','evening','pulse']))
    jobs=[(0,demand_seed,rate,scenario,policy)
          for demand_seed,rate,scenario in conditions
          for policy in ('fixed','actuated','max_pressure','green_wave')]
    jobs += [(train_seed,demand_seed,rate,scenario,policy)
          for train_seed in range(args.seeds)
          for demand_seed,rate,scenario in conditions
          for policy in sorted({alg for alg,_ in checkpoints})
          if (policy,train_seed) in checkpoints]
    if args.limit: jobs=jobs[:args.limit]
    episodes_path=out/'episodes.csv'
    rows=pd.read_csv(episodes_path).to_dict('records') if episodes_path.exists() else []
    completed={(str(r.get('policy')),int(r.get('train_seed',0)),int(r.get('demand_seed',-1)),
                int(r.get('rate',-1)),str(r.get('scenario'))) for r in rows
               if pd.isna(r.get('failure')) or str(r.get('failure',''))==''}
    for train_seed,demand_seed,rate,scenario,policy in jobs:
        key=(policy,train_seed,demand_seed,rate,scenario)
        if key in completed: continue
        cfg=Config(size=args.size,duration=args.duration,warmup=min(300,args.duration//5),seed=10000+demand_seed,rate=rate,scenario=scenario)
        # Training seed is an explicit block in the schedule for learned policies, not a confounded demand seed.
        folder=out/f'train_{train_seed}'/f'demand_{demand_seed}_{rate}_{scenario}'/policy
        try:
            checkpoint=checkpoints.get((policy,train_seed))
            summary=run(cfg,policy,str(folder),checkpoint)
            row={**summary,'train_seed':train_seed,'demand_seed':demand_seed,'failure':''}
        except Exception as e:
            row={'policy':policy,'train_seed':train_seed,'demand_seed':demand_seed,'rate':rate,'scenario':scenario,
                 'seed':cfg.seed,'failure':f'{type(e).__name__}: {e}'}
            save_json(folder/'failure.json',{'error':row['failure'],'traceback':traceback.format_exc(),'config':cfg.dict()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(episodes_path,index=False)
        print(json.dumps(row),flush=True)
    df=pd.DataFrame(rows)
    if len(df):
        df['pair']=df['demand_seed'].astype(str)+'_'+df['rate'].astype(str)+'_'+df['scenario'].astype(str)
        metrics=['travel_time_mean','travel_time_p95','delay_mean','waiting_mean','waiting_p95','waiting_gini','waiting_jain',
                 'queue_mean','queue_p95','max_lane_queue','throughput_per_hour','completion_rate','teleports','spillback_lane_seconds','stops_mean','speed_mean']
        tables=[]
        for keys,g in df[df.failure.fillna('')==''].groupby(['policy','rate','scenario']):
            rec=dict(zip(['policy','rate','scenario'],keys)); rec['n']=len(g)
            for metric in metrics:
                values=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy()
                if not len(values): continue
                rng=np.random.default_rng(42); boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
                rec[metric+'_mean']=values.mean(); rec[metric+'_std']=values.std(ddof=1) if len(values)>1 else 0
                rec[metric+'_ci_low'],rec[metric+'_ci_high']=np.percentile(boot,[2.5,97.5])
            tables.append(rec)
        pd.DataFrame(tables).to_csv(out/'summary.csv',index=False)
        # Paired difference table over exact demand keys vs fixed and max pressure.
        diffs=[]
        valid=df[df.failure.fillna('')=='']
        demand_keys=['demand_seed','rate','scenario']
        for metric in metrics:
            base=valid[valid.policy.isin(['fixed','max_pressure'])][demand_keys+['policy',metric]]
            for policy in sorted({alg for alg,_ in checkpoints}):
                learned=valid[valid.policy==policy]
                for baseline in ('fixed','max_pressure'):
                    reference=base[base.policy==baseline][demand_keys+[metric]]
                    paired=learned.merge(reference,on=demand_keys,suffixes=('_learned','_baseline'))
                    x=(paired[f'{metric}_learned']-paired[f'{metric}_baseline']).dropna().to_numpy()
                    if not len(x): continue
                    rng=np.random.default_rng(17); boot=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)])
                    diffs.append({'metric':metric,'policy':policy,'baseline':baseline,'n_pairs':len(x),
                        'mean_difference':x.mean(),'ci_low':np.percentile(boot,2.5),'ci_high':np.percentile(boot,97.5)})
        pd.DataFrame(diffs).to_csv(out/'paired_differences.csv',index=False)
if __name__=='__main__': main()
