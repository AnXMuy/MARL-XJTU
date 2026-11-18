"""Held-out paired evaluation. Test seeds never used for checkpoint selection."""
import argparse,json,itertools,hashlib
from pathlib import Path
import numpy as np
import torch
from .env import HomeBatch,VERSION

class Learned:
    def __init__(self,path,device='cpu'):
        data=torch.load(path,map_location=device,weights_only=False)
        self.sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest();self.path=str(path);self.device=device;self.framework='BenchMARL' if 'policy' in data else 'pilot'
        if self.framework=='BenchMARL':
            if data['environment']!=VERSION:raise ValueError('Environment version mismatch')
            self.algorithm=data['args']['algorithm'];self.weights=[]
            state=data['policy']
            keys=sorted([k for k in state if k.endswith('.weight')],key=lambda k:int(k.split('params.')[1].split('.')[0]))
            for key in keys:self.weights.append((state[key],state[key[:-6]+'bias']))
            self.manifest=data
        else:raise ValueError('Only versioned BenchMARL checkpoints are supported')
    def act(self,env):
        with torch.no_grad():
            if self.framework=='BenchMARL':
                x=env.observe(True)
                for i,(w,b) in enumerate(self.weights):
                    x=torch.nn.functional.linear(x,w,b)
                    if i<len(self.weights)-1:x=x.tanh()
                return x.argmax(-1)


class Rule:
    def __init__(self,coordinated=False):self.coordinated=coordinated
    def act(self,e):
        target,lights,active=e.targets();gap=target-e.temps
        # Same physical frost/overheat bounds for every controller, no hidden clamp.
        active=active|(e.temps<12.5)
        low=target-1;high=target+1
        if self.coordinated:
            arriving=e.targets(min(e.t+4,95))[2]
            # Preheat only for an imminent occupancy or imminent price increase.
            low=torch.where(arriving&~active,target-.5,low)
            active=active|arriving
            if e.prices[min(e.t+4,95)]>e.prices[min(e.t,95)]:low=torch.where(active,target+.2,low)
        hvac=torch.where(active&(e.temps<low),2,torch.where(active&(e.temps<high),e.hvac,0))
        hvac=torch.where(e.temps<12.5,1,hvac)
        if self.coordinated:
            # Select a feasible joint allocation closest to requested heating,
            # weighting rooms by current comfort urgency; no policy-only safeguard.
            candidates=torch.tensor(list(itertools.product(range(3),repeat=4)),device=e.device)
            powers=e.kw[candidates].sum(-1)[None,:]+e.lkw[lights].sum(-1)[:,None]
            urgency=(gap+1).clamp_min(.1)*active+0.1
            error=((candidates[None]-hvac[:,None]).float().square()*urgency[:,None]).sum(-1)
            error=error+1000*(powers-e.config.capacity).clamp_min(0)
            hvac=candidates[error.argmin(-1)]
        return hvac*3+lights

def preference(scenario):
    p=[0,0,0,0,1,.5,1,1,.4,1,0,0,0,0]
    if scenario=='eco':p[8]=.9
    if scenario=='sleep':p[1]=-2;p[9:]=[0,1,0,0,0]
    return p

def rollout(controller,seed,scenario,capacity=4.):
    e=HomeBatch();e.config.capacity=capacity;e.reset(seed=seed,preference=preference(scenario),switches=False)
    rows=[]
    for t in range(96):
        if scenario=='changes':
            if t==28:e.set_preference([2,0,0,0,1,.5,1,1,.4,0,0,0,0,1])
            if t==68:
                p=e.preference.clone();p[:,8]=.9;e.set_preference(p)
            if t==84:e.set_preference(preference('sleep'))
        _,_,_,info=e.step(controller.act(e))
        rows.append({k:v[0].cpu().tolist() for k,v in info.items()})
    occ=sum(r['occupied_hours'] for r in rows)
    metrics={'cost':sum(r['cost'] for r in rows),'energy_kwh':sum(r['energy_kwh'] for r in rows),
        'occupied_degree_hours':sum(r['degree_hours'] for r in rows),
        'occupied_violation_rate':sum(r['violation_hours'] for r in rows)/max(occ,1e-9),
        'peak_kw':max(r['power_kw'] for r in rows),'overload_kwh':sum(r['overload_kw']*.25 for r in rows),
        'team_return':sum(r['reward'] for r in rows),'light_mismatch':np.mean([r['light_mismatch'] for r in rows])}
    return {'seed':seed,'metrics':metrics,'trajectory':rows}

def bootstrap_ci(x,seed=1):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(2000,len(x)))].mean(-1)
    return [float(v) for v in np.quantile(means,[.025,.975])]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='append',default=[]);p.add_argument('--days',type=int,default=20)
    p.add_argument('--output',default='assets/v2/metrics.json');p.add_argument('--seed-start',type=int,default=100000)
    args=p.parse_args();torch.set_num_threads(1)
    policies={'thermostat':Rule(),'coordinated_rule':Rule(True)}
    for spec in args.checkpoint:
        label,path=spec.split('=',1);policies[label]=Learned(path)
    results={};paired={}
    for scene in ('default','eco','sleep','changes'):
        results[scene]={};paired[scene]={}
        for label,controller in policies.items():
            days=[rollout(controller,s,scene) for s in range(args.seed_start,args.seed_start+args.days)]
            summary={k:float(np.mean([d['metrics'][k] for d in days])) for k in days[0]['metrics']}
            results[scene][label]={'summary':summary,'days':days}
            print(scene,label,json.dumps(summary),flush=True)
        ref=results[scene]['coordinated_rule']['days']
        for label,entry in results[scene].items():
            paired[scene][label]={}
            for metric in ('cost','occupied_degree_hours','overload_kwh','team_return'):
                diffs=[d['metrics'][metric]-b['metrics'][metric] for d,b in zip(entry['days'],ref)]
                paired[scene][label][metric]={'mean_difference':float(np.mean(diffs)),'paired_bootstrap_95ci':bootstrap_ci(diffs)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'environment':VERSION,'test_seed_start':args.seed_start,'days':args.days,
        'checkpoints':{k:v.path for k,v in policies.items() if isinstance(v,Learned)},'checkpoint_sha256':{k:v.sha256 for k,v in policies.items() if isinstance(v,Learned)},'results':results,'paired_vs_coordinated_rule':paired,
        'limits':'Simulation only. CIs reflect day variability, not training-seed uncertainty. No assumed saving threshold.'},separators=(',',':')))
if __name__=='__main__':main()
