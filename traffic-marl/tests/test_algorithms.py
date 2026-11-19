import numpy as np
import torch
import pytest
from traffic_marl.agents.ppo import PPO,gae,update
from traffic_marl.agents.dqn import IDQN

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

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_mappo_and_idqn_update_on_cuda():
    ppo=PPO(12,'mappo',True,2).to('cuda')
    optimizer=torch.optim.Adam(ppo.parameters(),lr=1e-3)
    obs=np.random.randn(2,2,12).astype('float32'); mask=np.ones((2,2,2),dtype=bool)
    actions=np.zeros((2,2),dtype='int64')
    with torch.no_grad():
        x=torch.tensor(obs,device='cuda'); m=torch.tensor(mask,device='cuda')
        logp=ppo.distribution(x,m).log_prob(torch.tensor(actions,device='cuda')).cpu().numpy()
    assert update(ppo,optimizer,(obs,mask,actions,logp,np.ones((2,2)),np.ones((2,2))),epochs=1)['value_loss']>=0
    dqn=IDQN(12,2).to('cuda')
    for _ in range(64):
        dqn.store(obs[0],np.zeros(2,dtype='int64'),np.ones(2),obs[1],mask[1],False)
    assert dqn.update()['q_loss']>=0
    assert next(ppo.parameters()).device.type=='cuda'
    assert next(dqn.parameters()).device.type=='cuda'
