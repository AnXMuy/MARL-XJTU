"""Batched torch thermal simulator. All controllers see the same exogenous data.

RC approximation: degrees C, kW, kWh, hour; 15-minute zero-order hold.
Power capacity is a soft demand-response constraint, never a hidden action filter.
"""
from dataclasses import dataclass, asdict
import torch

VERSION = 'coordinated-home-v3.0'
OBS = 38

@dataclass
class Config:
    dt: float = .25
    steps: int = 96
    capacitance: float = 1.6  # kWh / degree C per zone
    conductance: float = .12  # kW / degree C to outdoors
    coupling: float = .05  # kW / degree C across internal doors
    cop: float = 3.0
    capacity: float = 4.0  # contracted controllable load kW
    tolerance: float = 1.0
    comfort_weight: float = 2.0
    overload_weight: float = 2.0

class HomeBatch:
    def __init__(self, n=1, device='cpu', config=None):
        self.n, self.device = n, torch.device(device)
        self.config = config or Config()
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.kw = torch.tensor([0., .8, 2.], device=self.device)
        self.lkw = torch.tensor([0., .02, .06], device=self.device)
        self.ids = torch.eye(4, device=self.device).expand(n,-1,-1)
        self.reset()

    def random(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def reset(self, seed=None, preference=None, switches=True):
        if seed is not None: self.gen.manual_seed(seed)
        n,d=self.n,self.device
        self.t=0
        self.area=16+20*self.random(n,4)
        edge=self.random(n,4,4)>.45
        self.connections=(edge|edge.transpose(1,2)).float()
        self.connections.diagonal(dim1=1,dim2=2).zero_()
        self.capacitance=self.config.capacitance*self.area/25
        self.conductance=self.config.conductance*(self.area/25).sqrt()
        self.temps=20.5+2*self.random(n,4)
        self.hvac=torch.zeros(n,4,dtype=torch.long,device=d)
        self.lights=self.hvac.clone()
        h=torch.arange(96,device=d)[None,:,None]/4
        # Each episode has independent weather and schedule; evaluation seeds held out.
        self.outside=9+4*torch.sin((h[:,:,0]-9)*torch.pi/12)+2*(self.random(n,1)-.5)
        jitter=(self.random(n,1,4)-.5)*1.0
        h=(h+jitter)%24
        windows=[((7,9),(18,23)),((0,7),(23,24)),((7,8),(18,19)),((6.5,7.5),(21,22))]
        self.schedule=torch.stack([sum(((h[:,:,i]>=a)&(h[:,:,i]<b)).int() for a,b in w)>0 for i,w in enumerate(windows)],-1)
        # Some days someone works from home; seed-matched across all controllers.
        self.schedule[:,:,0] |= (h[:,:,0]>=11)&(h[:,:,0]<17)&(self.random(n,1)>.75)
        hour=torch.arange(96,device=d)/4
        self.prices=torch.where(((hour>=8)&(hour<12))|((hour>=18)&(hour<22)),1.2,
                         torch.where((hour<7)|(hour>=23),.4,.8))
        self.switches=switches
        self.preference=self.sample_preference() if preference is None else torch.as_tensor(preference,device=d,dtype=torch.float32).reshape(-1,14).expand(n,-1).clone()
        return self.observe()

    def set_layout(self, areas, connections):
        area=torch.as_tensor(areas,device=self.device,dtype=torch.float32).reshape(-1,4).expand(self.n,-1)
        graph=torch.as_tensor(connections,device=self.device,dtype=torch.float32).reshape(-1,4,4).expand(self.n,-1,-1)
        if not torch.isfinite(area).all() or (area<16).any() or (area>36).any():raise ValueError('Room areas must be 16–36 square metres')
        if not torch.isfinite(graph).all() or not torch.equal(graph,graph.transpose(1,2)) or ((graph<0)|(graph>1)).any() or graph.diagonal(dim1=1,dim2=2).any():raise ValueError('Connections must be symmetric, zero diagonal, within [0,1]')
        self.area=area.clone();self.connections=graph.clone()
        self.capacitance=self.config.capacitance*self.area/25
        self.conductance=self.config.conductance*(self.area/25).sqrt()
        return self.observe()

    def sample_preference(self):
        p=torch.zeros(self.n,14,device=self.device)
        p[:,:4]=(self.random(self.n,4)-.5)*4
        p[:,4:8]=torch.round(self.random(self.n,4)*2)/2
        p[:,8]=self.random(self.n)
        scene=torch.multinomial(torch.tensor([.60,.15,.10,.10,.05],device=self.device),self.n,replacement=True,generator=self.gen)
        p[:,9:]=torch.nn.functional.one_hot(scene,5)
        return p

    def set_preference(self, p):
        value=torch.as_tensor(p,device=self.device,dtype=torch.float32).reshape(-1,14).expand(self.n,-1)
        if not torch.isfinite(value).all() or (value[:,:4].abs()>3).any() or ((value[:,8]<0)|(value[:,8]>1)).any() or not torch.isin(value[:,4:8],torch.tensor([0.,.5,1.],device=self.device)).all() or not ((value[:,9:]==0)|(value[:,9:]==1)).all() or not (value[:,9:].sum(-1)==1).all():
            raise ValueError('Invalid preference vector')
        self.preference=value.clone()
        return self.observe()

    def targets(self, slot=None):
        slot=min(self.t,95) if slot is None else min(slot,95)
        p=self.preference;scene=p[:,9:].argmax(-1)
        target=22+p[:,:4]
        light=(2*p[:,4:8]).long()
        active=self.schedule[:,slot].clone()
        sleeping=scene==1;away=scene==2;movie=scene==3;home=scene==4
        active[sleeping]=False;active[sleeping,1]=True
        # Scene sets scope, explicit temperature preferences still remain editable.
        active[away]=False
        active[movie,0]=True;light[movie,0]=1
        active[home,0]=True
        light[~active]=0
        return target,light,active

    def observe(self, shared=True):
        k=min(self.t,95);target,light,active=self.targets(k)
        next_active=self.targets(min(k+4,95))[2]
        b=lambda x:x[:,None,None].expand(-1,4,-1)
        # Local physical state + effective targets + calendar forecast + room identity.
        obs=torch.cat([(self.temps-22).unsqueeze(-1)/8,(target-self.temps).unsqueeze(-1)/5,
            ((target-22)/3).unsqueeze(-1),active.float().unsqueeze(-1),next_active.float().unsqueeze(-1),
            torch.nn.functional.one_hot(self.hvac,3),torch.nn.functional.one_hot(self.lights,3),
            (light.float()/2).unsqueeze(-1),b(self.outside[:,k]/20),
            torch.full((self.n,4,1),float(self.prices[k]),device=self.device),
            torch.full((self.n,4,1),float(self.prices[min(k+4,95)]),device=self.device),
            torch.full((self.n,4,1),__import__('math').sin(k*__import__('math').pi/48),device=self.device),
            torch.full((self.n,4,1),__import__('math').cos(k*__import__('math').pi/48),device=self.device),
            b(self.preference[:,8]),self.preference[:,None,9:].expand(-1,4,-1),self.ids,
            b((self.kw[self.hvac].sum(-1)/self.config.capacity)),
            b((active*(target-self.temps).clamp_min(0)).sum(-1)/10),
            (self.temps.mean(-1,keepdim=True).expand(-1,4)-self.temps).unsqueeze(-1)/5,
            b(active.float().mean(-1)),
            torch.full((self.n,4,1),self.config.capacity/8,device=self.device),
            b(((active & (target-self.temps>1)).float().sum(-1))/4),
            (self.area/25).unsqueeze(-1),self.connections,
            ],-1).float()
        if not shared: obs=obs.clone();obs[:,:,27:33]=0
        assert obs.shape[-1]==OBS,obs.shape
        return obs

    def step(self, actions):
        if self.t>=96:raise RuntimeError('Day finished')
        a=torch.as_tensor(actions,device=self.device,dtype=torch.long).reshape(self.n,4)
        if ((a<0)|(a>8)).any():raise ValueError('Actions outside [0,8]')
        self.hvac,self.lights=a//3,a%3
        target,light,active=self.targets()
        c=self.config;slot=self.t
        power=self.kw[self.hvac]+self.lkw[self.lights]
        heat=c.cop*self.kw[self.hvac]
        # Symmetric, conservative heat exchange; complete four-zone coupling.
        coupling=c.coupling*(self.connections*(self.temps[:,None,:]-self.temps[:,:,None])).sum(-1)
        self.temps=self.temps+c.dt/self.capacitance*(self.conductance*(self.outside[:,slot,None]-self.temps)+heat+coupling)
        excess=(torch.abs(self.temps-target)-c.tolerance).clamp_min(0)*active
        comfort=excess.sum(-1)/active.sum(-1).clamp_min(1)
        mismatch=((self.lights!=light)&active).float().sum(-1)/active.sum(-1).clamp_min(1)
        energy=power.sum(-1)*c.dt;cost=energy*self.prices[slot]
        over=(power.sum(-1)-c.capacity).clamp_min(0)
        # Physical safety bound is separate from preferred occupied comfort.
        safety=((12-self.temps).clamp_min(0)+(self.temps-30).clamp_min(0)).mean(-1)
        reward=-c.comfort_weight*comfort-.3*mismatch-(.5+1.5*self.preference[:,8])*cost-c.overload_weight*over.square()-2*safety
        info={'temps':self.temps.clone(),'target':target,'active':active,'hvac':self.hvac.clone(),'lights':self.lights.clone(),
              'power_kw':power.sum(-1),'energy_kwh':energy,'cost':cost,'degree_hours':excess.sum(-1)*c.dt,
              'occupied_hours':active.sum(-1)*c.dt,'violation_hours':((excess>0)&active).sum(-1)*c.dt,
              'overload_kw':over,'light_mismatch':mismatch,'reward':reward}
        self.t+=1
        if self.switches and self.t==48:
            change=self.random(self.n)<.3;self.preference[change]=self.sample_preference()[change]
        return self.observe(),reward,self.t==96,info
