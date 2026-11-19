import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from traffic_marl.config import Config
from traffic_marl.env.sumo_env import SumoEnv
from traffic_marl.agents.ppo import PPO,gae,update
from traffic_marl.agents.dqn import IDQN
from traffic_marl.utils.provenance import metadata,save_json,sha256
from evaluate import run

TRAIN_SEEDS=list(range(1000,1100))
VALID_SEEDS=[5000,5001]
TEST_SEEDS=list(range(10000,10020))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=['mappo','ippo','idqn'],default='mappo')
    p.add_argument('--config',default='configs/train.json')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--output',default='runs/mappo0'); p.add_argument('--independent-actors',action='store_true')
    p.add_argument('--randomize-demand',action='store_true'); p.add_argument('--validate-every',type=int,default=10)
    p.add_argument('--device',default='cuda',choices=['cuda'])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.episodes<1 or args.validate_every<1: p.error('positive episode/validation count required')
    if not torch.cuda.is_available(): p.error('CUDA is required for training; CPU training is disabled')
    device=torch.device('cuda')
    torch.set_num_threads(1)
    if device.type=='cuda': torch.cuda.set_device(0)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg=Config(**json.loads(Path(args.config).read_text()))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    save_json(out/'training_config.json',{**vars(args),'device':str(device)})
    save_json(out/'provenance.json',{**metadata(),'training_device':str(device),
        'cuda_device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
    save_json(out/'splits.json',{'train':TRAIN_SEEDS,'validation':VALID_SEEDS,'test':TEST_SEEDS})
    model=None; steps=0; best=float('inf'); rows=[]; start=time.monotonic(); first=None; start_ep=0
    latest_path=out/'latest.pt'
    if args.resume:
        if not latest_path.exists(): p.error('--resume requested but latest.pt is missing')
        first=torch.load(latest_path,map_location='cpu',weights_only=True)
        steps=first['steps']; rows=first['rows']; best=first['best']; start_ep=first['metadata']['episode']+1
    for ep in range(start_ep,args.episodes):
        cfg.seed=TRAIN_SEEDS[ep%len(TRAIN_SEEDS)]
        base=Config(**cfg.dict())
        if args.randomize_demand:
            base.rate=float(np.random.choice([150,300,600]))
            base.scenario=str(np.random.choice(['uniform','morning','evening','pulse']))
        with SumoEnv(base,out/f'train_{ep:04d}') as env:
            obs,info=env.reset()
            if model is None:
                model=IDQN(obs.shape[1],len(obs)) if args.algorithm=='idqn' else PPO(obs.shape[1],args.algorithm,not args.independent_actors,len(obs))
                model=model.to(device)
                optimizer=torch.optim.Adam(model.parameters(),lr=3e-4) if args.algorithm!='idqn' else None
                if first is not None:
                    model.load_state_dict(first['model'])
                    if optimizer is not None: optimizer.load_state_dict(first['optimizer'])
                    else:
                        for opt,state in zip(model.optimizers,first['optimizers']): opt.load_state_dict(state)
            done=False; total=0; buf=[]; stats={}
            while not done:
                mask=info['action_mask']
                if args.algorithm=='idqn':
                    a=model.act(obs,mask,max(.05,1-steps/10000))
                else: a,lp,val=model.act(obs,mask)
                nxt,r,_,done,info=env.step(a)
                steps+=1; total+=float(r.mean())
                if args.algorithm=='idqn':
                    # Time-limit truncation is bootstrapped, not treated as an absorbing state.
                    model.store(obs,a,r,nxt,info['action_mask'],False)
                    if steps%4==0: stats=model.update()
                else: buf.append((obs,mask,a,lp,val,r))
                obs=nxt
                if args.algorithm!='idqn' and (len(buf)>=128 or done):
                    bo,bm,ba,bl,bv,br=map(np.array,zip(*buf))
                    bootstrap=model.act(obs,info['action_mask'],True)[2]
                    adv,ret=gae(br,bv,bootstrap)
                    stats=update(model,optimizer,(bo,bm,ba,bl,ret,adv))
                    buf=[]
            summary=env.finish()
        ckmeta={'algorithm':args.algorithm,'obs_dim':obs.shape[1],'n_agents':len(obs),
                'shared':not args.independent_actors,'training_steps':steps,'agent_steps':steps*len(obs),
                'train_seed':args.seed,'episode':ep,'config':cfg.dict(),'train_demand_seeds':TRAIN_SEEDS,
                'validation_seeds':VALID_SEEDS,'test_seeds':TEST_SEEDS,'hyperparameters':{
                    'lr':3e-4,'gamma':.99,'gae_lambda':.95,'clip':.2,'entropy':.01,
                    'epochs':4,'rollout':128,'minibatch':64,'grad_clip':.5,'hidden':[64,64]},
                'wall_seconds':time.monotonic()-start,'device':str(device),
                'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
        resume_state={'model':model.state_dict(),'metadata':ckmeta,'steps':steps,'rows':rows+[{'episode':ep}],
                      'best':best,'optimizer':optimizer.state_dict() if optimizer is not None else None,
                      'optimizers':[opt.state_dict() for opt in model.optimizers] if args.algorithm=='idqn' else None}
        torch.save(resume_state,latest_path)
        row={'episode':ep,'steps':steps,'return':total,'queue_mean':summary['queue_mean'],**stats}
        if (ep+1)%args.validate_every==0 or ep==args.episodes-1:
            vals=[]
            for seed in VALID_SEEDS:
                vc=Config(**cfg.dict()); vc.seed=seed
                vals.append(run(vc,args.algorithm,out/f'validation_{ep}_{seed}',out/'latest.pt')['queue_mean'])
            row['validation_queue']=float(np.mean(vals))
            if row['validation_queue']<best:
                best=row['validation_queue']
                torch.save({'model':model.state_dict(),'metadata':{**ckmeta,'selection':'minimum mean validation queue','validation_queue':best}},out/'best.pt')
        rows.append(row)
        resume_state['rows']=rows
        resume_state['best']=best
        torch.save(resume_state,latest_path)
        with open(out/'curve.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            w.writeheader(); w.writerows(rows)
        print(json.dumps(row),flush=True)
    save_json(out/'manifest.json',{'best_sha256':sha256(out/'best.pt'),'steps':steps,'episodes':args.episodes,'wall_seconds':time.monotonic()-start,'status':'trained; performance must be assessed against baselines'})
if __name__=='__main__': main()
