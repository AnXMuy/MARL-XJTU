import torch
from traffic_marl.agents.ppo import PPO
from traffic_marl.agents.dqn import IDQN
from traffic_marl.utils.provenance import sha256

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    meta=ck['metadata']
    if meta['training_steps']<=0: raise ValueError('untrained checkpoint')
    if meta['algorithm']=='idqn': model=IDQN(meta['obs_dim'],meta['n_agents'])
    else: model=PPO(meta['obs_dim'],meta['algorithm'],meta['shared'],meta['n_agents'])
    model.load_state_dict(ck['model']); model.eval()
    return model,{**meta,'sha256':sha256(path)}
