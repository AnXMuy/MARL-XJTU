import argparse
import json
from pathlib import Path
import traceback
from traffic_marl.config import Config
from traffic_marl.env.sumo_env import SumoEnv
from traffic_marl.baselines.controllers import Controller
from traffic_marl.agents.checkpoint import load
from traffic_marl.utils.provenance import save_json

def run(config,policy,directory,checkpoint=None,replay=False):
    model=meta=None
    if checkpoint:
        model,meta=load(checkpoint)
        if policy!=meta['algorithm']: raise ValueError('policy/checkpoint mismatch')
        train=meta['config']
        for key in ('downstream_obs','queue_only','neighbors','min_green','yellow','all_red','decision'):
            if getattr(config,key)!=train[key]: raise ValueError(f'checkpoint requires {key}={train[key]}')
    elif policy not in ('fixed','actuated','max_pressure','green_wave'):
        raise ValueError('learned policy requires a trained checkpoint')
    control=Controller(policy) if model is None else None
    try:
        with SumoEnv(config,directory,replay) as env:
            obs,info=env.reset()
            done=False
            while not done:
                if model is None: actions=control.actions(env)
                elif policy=='idqn': actions=model.act(obs,info['action_mask'])
                else: actions=model.act(obs,info['action_mask'],True)[0]
                obs,_,_,done,info=env.step(actions)
            summary=env.finish()
            summary.update(policy=policy,seed=config.seed,scenario=config.scenario,rate=config.rate,size=config.size,
                           demand_sha256=env.manifest['demand_sha256'],checkpoint=meta)
            save_json(Path(directory)/'summary.json',summary)
            return summary
    except BaseException as e:
        save_json(Path(directory)/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'config':config.dict(),'policy':policy})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config'); p.add_argument('--policy',default='max_pressure')
    p.add_argument('--checkpoint'); p.add_argument('--output',default='runs/eval')
    p.add_argument('--replay',action='store_true')
    args=p.parse_args()
    config=Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    print(json.dumps(run(config,args.policy,args.output,args.checkpoint,args.replay),indent=2))
if __name__=='__main__': main()
