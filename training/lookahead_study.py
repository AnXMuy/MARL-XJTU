"""Exploratory horizon study. Final fresh-day confirmation: training.final_study."""
import argparse,json,time
from pathlib import Path
import torch
from experiments.v2.controllers import PolicyLookahead,FeasibleLearned
from experiments.v2.evaluate import Rule,rollout,bootstrap_ci

def evaluate(controller,seeds,scenarios):
    out={}
    for scene in scenarios:
        runs=[rollout(controller,s,scene) for s in seeds]
        out[scene]={'summary':{k:sum(r['metrics'][k] for r in runs)/len(runs) for k in runs[0]['metrics']},'days':runs}
        print(type(controller).__name__,getattr(controller,'horizon',None),scene,out[scene]['summary'],flush=True)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/layout/mappo42/best_actor.pt');a=p.parse_args()
    torch.set_num_threads(1);start=time.time();scenes=['default','eco','sleep','changes'];validation={}
    for h in (4,8,16,32):
        data=evaluate(PolicyLookahead(a.checkpoint,h),range(910000,910003),scenes)
        validation[h]={'return':sum(v['summary']['team_return'] for v in data.values())/4,'scenarios':{k:v['summary'] for k,v in data.items()}}
    horizon=max(validation,key=lambda h:validation[h]['return'])
    out={'checkpoint':a.checkpoint,'sha256':FeasibleLearned(a.checkpoint).sha256,'validation':validation,'selected_horizon':horizon,
         'protocol':'Horizon selected by mean original team return on validation seeds910000:910003 only. Test seeds100000:100020. Perfect known weather and occupancy forecast; future voice commands not visible. Lighting from learned actor.','results':{}}
    Path('assets/lookahead_selection.json').write_text(json.dumps(out,indent=2))
    for name,controller in [('coordinated_rule',Rule(True)),('mappo_feasible',FeasibleLearned(a.checkpoint)),('mappo_lookahead',PolicyLookahead(a.checkpoint,horizon))]:
        data=evaluate(controller,range(100000,100020),scenes)
        for scene,entry in data.items():out['results'].setdefault(scene,{})[name]=entry
    for scene,policies in out['results'].items():
        for name,entry in policies.items():
            entry['paired_vs_rule']={}
            for metric in ('cost','occupied_degree_hours','overload_kwh','team_return'):
                diff=[d['metrics'][metric]-b['metrics'][metric] for d,b in zip(entry['days'],policies['coordinated_rule']['days'])]
                entry['paired_vs_rule'][metric]={'difference':sum(diff)/len(diff),'ci95':bootstrap_ci(diff)}
    out['seconds']=time.time()-start
    Path('assets/lookahead_study.json').write_text(json.dumps(out,separators=(',',':')))
if __name__=='__main__':main()
