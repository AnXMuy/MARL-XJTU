"""Predeclared controller comparison, no test-set controller selection."""
import argparse,json
from pathlib import Path
import torch
from experiments.v2.evaluate import Learned,Rule,rollout,bootstrap_ci
from experiments.v2.controllers import FeasibleLearned,Predictive

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/layout/mappo42/best_actor.pt');p.add_argument('--days',type=int,default=20);a=p.parse_args();torch.set_num_threads(1)
 policies={'coordinated_rule':Rule(True),'mappo':Learned(a.checkpoint),'mappo_feasible':FeasibleLearned(a.checkpoint),'mpc_1h':Predictive(4,12),'mpc_4h':Predictive(16,32)}
 out={'protocol':'Same v3 physics, seeds100000+, scenarios predeclared. MPC uses known physics; allocator is centralized, not retraining.','results':{}}
 for scene in ['default','eco','sleep','changes']:
  out['results'][scene]={}
  for label,controller in policies.items():
   runs=[rollout(controller,s,scene) for s in range(100000,100000+a.days)]
   metrics={k:sum(r['metrics'][k] for r in runs)/len(runs) for k in runs[0]['metrics']}
   out['results'][scene][label]={'summary':metrics,'days':[{'seed':r['seed'],'metrics':r['metrics']} for r in runs]}
   print(scene,label,json.dumps(metrics),flush=True)
  ref=out['results'][scene]['coordinated_rule']['days']
  for label,entry in out['results'][scene].items():
   entry['paired_vs_rule']={}
   for metric in ['cost','occupied_degree_hours','overload_kwh']:
    diff=[r['metrics'][metric]-b['metrics'][metric] for r,b in zip(entry['days'],ref)]
    entry['paired_vs_rule'][metric]={'difference':sum(diff)/len(diff),'ci95':bootstrap_ci(diff)}
 Path('assets/architecture_study.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
