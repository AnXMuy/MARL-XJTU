from collections import deque
import random
import numpy as np
import torch
from torch import nn
from traffic_marl.agents.ppo import mlp

class IDQN(nn.Module):
    """Independent per-junction Q networks and replay buffers, target network + Huber."""
    def __init__(self,obs_dim,n_agents):
        super().__init__()
        self.nets=nn.ModuleList([mlp(obs_dim,2) for _ in range(n_agents)])
        self.targets=nn.ModuleList([mlp(obs_dim,2) for _ in range(n_agents)])
        self.targets.load_state_dict(self.nets.state_dict())
        self.buffers=[deque(maxlen=20000) for _ in range(n_agents)]
        self.optimizers=[torch.optim.Adam(net.parameters(),lr=3e-4) for net in self.nets]
        self.updates=0
    @torch.no_grad()
    def act(self,obs,mask,epsilon=0):
        if len(obs)!=len(self.nets): raise ValueError('IDQN requires training grid size')
        actions=[]
        for net,x,m in zip(self.nets,obs,mask):
            if random.random()<epsilon: a=int(np.random.choice(np.flatnonzero(m)))
            else: a=int(net(torch.tensor(x)).masked_fill(~torch.tensor(m),-1e9).argmax())
            actions.append(a)
        return np.array(actions)
    def store(self,obs,actions,rewards,next_obs,next_mask,terminal):
        for i,b in enumerate(self.buffers): b.append((obs[i],actions[i],rewards[i],next_obs[i],next_mask[i],terminal))
    def update(self):
        losses=[]
        for net,target,opt,b in zip(self.nets,self.targets,self.optimizers,self.buffers):
            if len(b)<64: continue
            o,a,r,no,nm,done=map(np.array,zip(*random.sample(b,64)))
            q=net(torch.tensor(o)).gather(1,torch.tensor(a)[:,None]).squeeze(1)
            with torch.no_grad():
                nq=target(torch.tensor(no)).masked_fill(~torch.tensor(nm),-1e9).max(1).values
                y=torch.tensor(r,dtype=torch.float32)+.99*nq*(1-torch.tensor(done,dtype=torch.float32))
            loss=nn.functional.smooth_l1_loss(q,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),.5); opt.step()
            losses.append(loss.item())
        self.updates+=1
        if self.updates%100==0: self.targets.load_state_dict(self.nets.state_dict())
        return {'q_loss':float(np.mean(losses)) if losses else 0.0}
