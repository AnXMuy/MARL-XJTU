"""Explicit constrained execution and short-horizon physical planning baselines.
No controller here may be labelled a newly trained MARL algorithm.
"""
import itertools
import torch
from .evaluate import Learned

class FeasibleLearned(Learned):
    """Maximize actor joint log probability over capacity-feasible HVAC allocations.
    Both HVAC and lighting can change. Shared allocator is centralized.
    """
    def act(self,e):
        with torch.no_grad():
            x=e.observe(True)
            for i,(w,b) in enumerate(self.weights):
                x=torch.nn.functional.linear(x,w,b)
                if i<len(self.weights)-1:x=x.tanh()
            # Optimize the full lighting/HVAC joint choice under total load.
            # 9^4=6561; tractable for a four-zone deployment.
            greedy=x.argmax(-1)
            bad=(e.kw[greedy//3]+e.lkw[greedy%3]).sum(-1)>e.config.capacity+1e-6
            if not bad.any():return greedy
            if not hasattr(self,"candidates"):
                self.candidates=torch.tensor(list(itertools.product(range(9),repeat=4)),device=e.device)
            candidates=self.candidates
            power=(e.kw[candidates//3]+e.lkw[candidates%3]).sum(-1)
            feasible=candidates[power<=e.config.capacity+1e-6]
            score=sum(x[bad,i][:,feasible[:,i]] for i in range(4))
            greedy[bad]=feasible[score.argmax(-1)]
            return greedy

class Predictive:
    """Receding-horizon beam-search MPC using the same known RC dynamics.
    Strong model-based baseline, not a learned multi-agent policy.
    """
    def __init__(self,horizon=4,beam=12,comfort=2.):self.horizon=horizon;self.beam=beam;self.comfort=comfort
    def act(self,e):
        assert e.n==1,'Evaluation-only MPC currently supports one household'
        candidates=torch.tensor(list(itertools.product(range(3),repeat=4)),device=e.device)
        temp=e.temps.clone();score=torch.zeros(1,device=e.device);first=None
        c=e.config
        for depth in range(min(self.horizon,96-e.t)):
            k=e.t+depth;target,lights,active=e.targets(k)
            power=e.kw[candidates].sum(-1)+e.lkw[lights].sum(-1)
            choices=candidates[power<=c.capacity+1e-6]
            power=e.kw[choices].sum(-1)+e.lkw[lights].sum(-1)
            t=temp[:,None,:].expand(-1,len(choices),-1)
            exchange=c.coupling*(e.connections[:,None,:,:]*(t[:,:,None,:]-t[:,:,:,None])).sum(-1)
            new=t+c.dt/e.capacitance[:,None,:]*(e.conductance[:,None,:]*(e.outside[:,k,None,None]-t)+c.cop*e.kw[choices][None]+exchange)
            excess=((new-target[:,None]).abs()-c.tolerance).clamp_min(0)*active[:,None]
            comfort=excess.sum(-1)/active.sum(-1).clamp_min(1)
            safety=((12-new).clamp_min(0)+(new-30).clamp_min(0)).mean(-1)
            loss=self.comfort*comfort+(.5+1.5*e.preference[:,8,None])*power[None]*c.dt*e.prices[k]+2*safety
            values=score[:,None]+loss
            take=values.flatten().topk(min(self.beam,values.numel()),largest=False).indices
            parent=take//len(choices);action=take%len(choices)
            first=choices[action] if first is None else first[parent]
            temp=new.reshape(-1,4)[take];score=values.flatten()[take]
        return first[score.argmin()][None]*3+e.targets()[1]


class PolicyLookahead(FeasibleLearned):
    """Model-based rollout improvement of a learned joint policy.

    Enumerate feasible first HVAC actions, retain learned lighting, then follow
    MAPPO for H-1 simulated steps. Choose lowest cumulative *original* loss.
    Forecasts assume the simulator's known weather/schedule, no future commands.
    This is centralized planning plus MARL, not decentralized MAPPO alone.
    """
    def __init__(self,path,horizon=8,device='cpu'):
        super().__init__(path,device);self.horizon=horizon
        self.hvac_candidates=torch.tensor(list(itertools.product(range(3),repeat=4)),device=device)

    @torch.no_grad()
    def act(self,e):
        from .env import HomeBatch
        assert e.n==1, 'Online lookahead supports one residence'
        base=super().act(e)
        actions=self.hvac_candidates*3+base[0]%3
        power=(e.kw[actions//3]+e.lkw[actions%3]).sum(-1)
        actions=actions[power<=e.config.capacity+1e-6]
        # All branches start from the exact same physical and exogenous state.
        sim=HomeBatch.__new__(HomeBatch)
        sim.n=len(actions);sim.device=e.device;sim.config=e.config;sim.t=e.t;sim.switches=False
        for name in ('kw','lkw','prices'):setattr(sim,name,getattr(e,name))
        for name in ('ids','area','connections','capacitance','conductance','temps','hvac','lights','outside','schedule','preference'):
            value=getattr(e,name)
            setattr(sim,name,value.expand(sim.n,*value.shape[1:]).clone())
        loss=torch.zeros(sim.n,device=e.device)
        for depth in range(min(self.horizon,96-e.t)):
            action=actions if depth==0 else super().act(sim)
            _,reward,_,_=sim.step(action)
            loss-=reward
        return actions[loss.argmin()][None]
