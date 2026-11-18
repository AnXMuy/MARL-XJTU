import torch
import pytest
from experiments.v2.env import HomeBatch,OBS
from experiments.v2.evaluate import Rule

def test_physical_units():
    e=HomeBatch(2);e.reset(seed=4,switches=False)
    e.set_layout([25]*4,torch.ones(4,4)-torch.eye(4))
    e.temps[1]=e.temps[0];e.outside[1]=e.outside[0]
    e.step([[0]*4,[6]*4])
    assert torch.allclose(e.temps[1]-e.temps[0],torch.full((4,),.25/1.6*3*2),atol=1e-5)

def test_preferences():
    e=HomeBatch();p=[0,-2,0,0,1,.5,1,1,.4,0,1,0,0,0];e.set_preference(p)
    assert e.targets()[2].tolist()==[[False,True,False,False]]
    before=e.observe().clone();p[1]=0;e.set_preference(p)
    assert e.targets()[0][0,1]==22 and not torch.equal(before,e.observe())
    with pytest.raises(ValueError):e.set_preference([0]*14)

def test_full_day_and_feasible_rule():
    e=HomeBatch(2);e.reset(seed=6,switches=False)
    for _ in range(96):
        obs,r,done,info=e.step(Rule(True).act(e));assert torch.isfinite(obs).all() and torch.isfinite(r).all()
        assert info['overload_kw'].max()<1e-5
    assert done and obs.shape==(2,4,OBS)

def test_seed_reproducibility():
    a,b=HomeBatch(),HomeBatch();a.reset(seed=100);b.reset(seed=100)
    for key in ('schedule','outside','temps','area','connections'):assert torch.equal(getattr(a,key),getattr(b,key))

def test_layout_affects_dynamics_and_observation():
    e=HomeBatch(2);e.reset(seed=1,switches=False)
    e.set_layout([[16]*4,[36]*4],torch.zeros(4,4));e.temps[:]=22;e.outside[:]=22
    before=e.observe().clone();e.step([[6]*4]*2)
    assert (e.temps[0]>e.temps[1]).all()
    e.set_layout([25]*4,torch.ones(4,4)-torch.eye(4))
    assert not torch.equal(before,e.observe())
    with pytest.raises(ValueError):e.set_layout([2]*4,torch.eye(4))

def test_internal_exchange_conserves_energy():
    a,b=HomeBatch(),HomeBatch();a.reset(seed=2);b.reset(seed=2)
    a.set_layout([20,25,30,35],torch.zeros(4,4));b.set_layout([20,25,30,35],torch.ones(4,4)-torch.eye(4))
    a.temps[:]=torch.tensor([18,20,24,26]);b.temps[:]=a.temps
    a.step([0]*4);b.step([0]*4)
    assert abs(((b.temps-a.temps)*a.capacitance).sum())<1e-5

def test_torchrl_contract():
    from experiments.v2.torchrl_env import TorchRLHome
    from torchrl.envs.utils import check_env_specs
    e=TorchRLHome(2,'cpu');check_env_specs(e)
    td=e.rollout(96);assert td.batch_size==torch.Size([2,96])
