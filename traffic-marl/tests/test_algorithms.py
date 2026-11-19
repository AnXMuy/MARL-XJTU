import numpy as np
import torch
from traffic_marl.agents.ppo import PPO,gae

def test_action_mask_and_centralized_value():
    m=PPO(12,'mappo',True,2)
    obs=torch.randn(2,12)
    mask=torch.tensor([[True,False],[True,True]])
    acts,_,_=m.act(obs.numpy(),mask.numpy())
    assert acts[0]==0
    assert m.values(obs).shape==(2,)

def test_gae_bootstraps_time_limit():
    adv,ret=gae(np.array([[1.]]),np.array([[2.]]),np.array([3.]),gamma=.5,lam=.9)
    assert ret.item()==2.5
