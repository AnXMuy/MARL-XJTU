"""Local single-household CPU latency, including planning; not GPU throughput."""
import json,time
from pathlib import Path
import torch
from experiments.v2.env import HomeBatch
from experiments.v2.evaluate import Learned,Rule
from experiments.v2.controllers import FeasibleLearned,PolicyLookahead,Predictive

def main():
 torch.set_num_threads(1);path='results/layout/mappo42/best_actor.pt';out={}
 for name,policy in [('rule',Rule(True)),('mappo',Learned(path)),('feasible',FeasibleLearned(path)),('lookahead8h',PolicyLookahead(path,32)),('mpc4h',Predictive(16,32))]:
  e=HomeBatch();e.reset(seed=910030,switches=False);times=[]
  for i in range(24):
   start=time.perf_counter();action=policy.act(e);times.append((time.perf_counter()-start)*1000);e.step(action)
  x=torch.tensor(times[1:]);out[name]={'median_ms':float(x.median()),'p95_ms':float(torch.quantile(x,.95)),'n':len(x)}
 Path('assets/controller_latency.json').write_text(json.dumps({'protocol':'CPU, torch threads1,24 sequential decisions on validation seed910030; first warmup excluded. Measured on this server; not a general hardware guarantee.','controllers':out},indent=2))
 print(json.dumps(out))
if __name__=='__main__':main()
