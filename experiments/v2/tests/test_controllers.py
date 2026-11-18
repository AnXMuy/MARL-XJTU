import copy
from pathlib import Path
import pytest
import torch
from experiments.v2.env import HomeBatch
from experiments.v2.controllers import FeasibleLearned,PolicyLookahead,Predictive

CHECKPOINT=Path('results/layout/mappo42/best_actor.pt')

def test_mpc_valid_without_state_mutation():
    e=HomeBatch();before=e.temps.clone()
    action=Predictive().act(e)
    assert action.shape==(1,4) and torch.equal(before,e.temps) and e.t==0
    _,_,_,info=e.step(action)
    assert info['overload_kw'].max()<1e-5

@pytest.mark.skipif(not CHECKPOINT.exists(),reason='Requires trained model artifact')
def test_feasible_allocator_and_lookahead_full_day():
    torch.set_num_threads(1)
    for controller in (FeasibleLearned(CHECKPOINT),PolicyLookahead(CHECKPOINT,horizon=4)):
        e=HomeBatch();e.reset(seed=910020,switches=False)
        for _ in range(96):
            before=copy.deepcopy((e.t,e.temps,e.preference))
            action=controller.act(e)
            assert e.t==before[0] and torch.equal(e.temps,before[1]) and torch.equal(e.preference,before[2])
            obs,r,_,info=e.step(action)
            assert info['overload_kw'].max()<1e-5 and torch.isfinite(r).all()
