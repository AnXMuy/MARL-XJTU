"""Frozen architecture validation on fresh days after exploratory comparisons."""
import json,time
from pathlib import Path
import torch
from training.lookahead_study import evaluate
from experiments.v2.evaluate import Learned,Rule,bootstrap_ci
from experiments.v2.controllers import FeasibleLearned,PolicyLookahead,Predictive

def main():
 torch.set_num_threads(1);start=time.time();path='results/layout/mappo42/best_actor.pt'
 out={'protocol':'Fresh seeds100020:100040 after exploratory studies. Architecture frozen: policy rollout horizon32, original reward, joint feasible first action, learned lighting. Perfect forecast; no future voice commands.','checkpoint':path,'sha256':Learned(path).sha256,'horizon':32,'larger_policy_checkpoint':'results/capacity/mappo512_43/best_actor.pt','larger_policy_sha256':Learned('results/capacity/mappo512_43/best_actor.pt').sha256,'larger_policy_selection':'Better original validation return of two 512-width seeds; not selected using final test days.','results':{}}
 policies={'coordinated_rule':Rule(True),'mappo_original':Learned(path),'mappo_feasible':FeasibleLearned(path),'mappo512':FeasibleLearned('results/capacity/mappo512_43/best_actor.pt'),'mappo_lookahead':PolicyLookahead(path,32),'mpc4h':Predictive(16,32)}
 for name,controller in policies.items():
  data=evaluate(controller,range(100020,100040),['default','eco','sleep','changes'])
  for scene,entry in data.items():out['results'].setdefault(scene,{})[name]=entry
  Path('assets/final_study.partial.json').write_text(json.dumps(out,separators=(',',':')))
 for scene,policies in out['results'].items():
  for name,entry in policies.items():
   entry['paired_vs_rule']={}
   for metric in ('cost','occupied_degree_hours','overload_kwh','team_return'):
    diff=[d['metrics'][metric]-b['metrics'][metric] for d,b in zip(entry['days'],policies['coordinated_rule']['days'])]
    entry['paired_vs_rule'][metric]={'difference':sum(diff)/len(diff),'ci95':bootstrap_ci(diff)}
 out['seconds']=time.time()-start
 Path('assets/final_study.json').write_text(json.dumps(out,separators=(',',':')))
 Path('assets/final_study.partial.json').unlink()
if __name__=='__main__':main()
