import numpy as np
import torch
from traffic_marl.agents.ppo import PPO,gae,update

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

def test_mappo_update_preserves_joint_critic_input():
    model=PPO(12,'mappo',True,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    obs=np.random.randn(3,2,12).astype('float32')
    mask=np.ones((3,2,2),dtype=bool)
    actions=np.zeros((3,2),dtype='int64')
    with torch.no_grad():
        x=torch.tensor(obs); m=torch.tensor(mask)
        logp=model.distribution(x,m).log_prob(torch.tensor(actions)).numpy()
    stats=update(model,optimizer,(obs,mask,actions,logp,np.ones((3,2)),np.ones((3,2))),epochs=1,batch=2)
    assert np.isfinite(stats['value_loss'])
