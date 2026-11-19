import csv
import time
import uuid
from pathlib import Path
import numpy as np
import traci
from traffic_marl.env.network import generate
from traffic_marl.env.safety import Signal
from traffic_marl.metrics.collector import Collector
from traffic_marl.utils.provenance import metadata, save_json

class SumoEnv:
    """Thin TraCI adapter with time-limit truncation and safe signal execution."""
    def __init__(self,config,directory,replay=False):
        self.config=config; self.directory=Path(directory); self.replay=replay; self.c=None
    def reset(self,seed=None,demand_config=None):
        self.close()
        if seed is not None: self.config.seed=int(seed)
        if demand_config is not None: raise ValueError('construct Config with demand settings before reset')
        self.started=time.monotonic(); self.directory.mkdir(parents=True,exist_ok=True)
        self.manifest=generate(self.config,self.directory); save_json(self.directory/'config.json',self.config.dict())
        save_json(self.directory/'provenance.json',metadata()); label=uuid.uuid4().hex
        args=['sumo','--net-file',str(self.directory/'network.net.xml'),'--route-files',str(self.directory/'demand.rou.xml'),
              '--seed',str(self.config.seed),'--step-length','1','--time-to-teleport','-1','--no-step-log','true',
              '--duration-log.disable','true','--no-warnings','true','--tripinfo-output',str(self.directory/'tripinfo.xml'),
              '--tripinfo-output.write-unfinished','true','--log',str(self.directory/'sumo.log')]
        try:
            with open(self.directory/'stdout.log','w') as log: traci.start(args,label=label,stdout=log,doSwitch=False)
            self.c=traci.getConnection(label); self.agents=sorted(self.c.trafficlight.getIDList())
            self.lanes={}; self.links={}; self.groups={}; self.signals={}
            for a in self.agents:
                links=self.c.trafficlight.getControlledLinks(a); self.links[a]=[ls[0] for ls in links]
                self.lanes[a]=[x[0] for x in self.links[a]]
                groups=[]
                for inc,out,via in self.links[a]:
                    shape=self.c.lane.getShape(inc); dx,dy=np.subtract(shape[-1],shape[0])
                    groups.append(0 if abs(dx)>abs(dy) else 1)
                self.groups[a]=groups; self.signals[a]=Signal(self.config.min_green,self.config.yellow,self.config.all_red)
                self.c.trafficlight.setRedYellowGreenState(a,self.signals[a].state(groups))
            self.metrics=Collector(self.config,self.manifest['scheduled']); self.executed=[]; self.frames=[]
            self.positions={j:self.c.junction.getPosition(j) for j in self.c.junction.getIDList() if not j.startswith(':')}
            self.edge_shapes={e:self.c.lane.getShape(e+'_0') for e in self.c.edge.getIDList() if not e.startswith(':')}
            self.last_rewards=np.zeros(len(self.agents))
            return self.observe(),self.info()
        except BaseException: self.close(); raise
    def lane_stats(self,a):
        q=[]; occ=[]; wait=[]; downstream=[]
        for inc,out,_ in self.links[a]:
            q.append(self.c.lane.getLastStepHaltingNumber(inc)); occ.append(self.c.lane.getLastStepOccupancy(inc)/100)
            wait.append(self.c.lane.getWaitingTime(inc)); storage=max(self.c.lane.getLength(out)/7.5,1)
            downstream.append(self.c.lane.getLastStepVehicleNumber(out)/storage)
        return np.array(q),np.array(occ),np.array(wait),np.array(downstream)
    def observe(self):
        rows=[]
        for a in self.agents:
            q,o,w,d=self.lane_stats(a)
            order=sorted(range(4),key=lambda k:self.heading(self.links[a][k][0]))
            features=np.stack([q/40,o,w/600,d if self.config.downstream_obs else np.zeros(4)],axis=1)
            if self.config.queue_only: features[:,1:3]=0
            features=features[order].ravel(); s=self.signals[a]
            rows.append(np.r_[features,[s.phase==0,s.phase==1,s.stage=='green',s.stage=='yellow',s.stage=='red'],min(s.elapsed/60,2)])
        obs=np.array(rows,dtype=np.float32)
        if self.config.neighbors:
            base=obs.copy()
            for i,a in enumerate(self.agents):
                x,y=self.positions[a]; ns=[j for j,b in enumerate(self.agents) if j!=i and np.linalg.norm(np.subtract(self.positions[b],(x,y)))<self.config.spacing*1.1]
                if ns: obs[i,:16]+=0.5*base[ns,:16].mean(axis=0)
        return obs
    def heading(self,lane):
        shape=self.c.lane.getShape(lane); dx,dy=np.subtract(shape[-1],shape[0])
        return int(round(np.arctan2(dy,dx)/(np.pi/2)))%4
    def masks(self): return np.array([self.signals[a].mask() for a in self.agents],dtype=bool)
    def info(self): return {'agents':self.agents,'action_mask':self.masks(),'time':self.c.simulation.getTime(),'demand_sha256':self.manifest['demand_sha256']}
    def step(self,actions):
        if len(actions)!=len(self.agents): raise ValueError('one action per agent required')
        for a,action in zip(self.agents,actions): self.signals[a].request(int(action))
        rewards=np.zeros(len(self.agents)); request={a:int(act) for a,act in zip(self.agents,actions)}
        ticks=min(self.config.decision,self.config.duration-int(self.c.simulation.getTime()))
        for tick in range(ticks):
            for a in self.agents: self.c.trafficlight.setRedYellowGreenState(a,self.signals[a].state(self.groups[a]))
            self.c.simulationStep(); qs=[]; spillback=0; states={}
            for i,a in enumerate(self.agents):
                q,_,_,d=self.lane_stats(a); rewards[i]-=q.sum()/40; qs.extend(q.tolist()); spillback+=int(np.sum(d>=0.8))
                s=self.signals[a]; states[a]={'phase':s.phase,'stage':s.stage,'elapsed':s.elapsed,'state':s.state(self.groups[a])}
                self.executed.append({'time':self.c.simulation.getTime(),'agent':a,**states[a],'requested':request[a] if tick==0 else None})
                s.tick()
            self.metrics.update(self.c,qs,spillback)
            if self.replay: self.frames.append({'time':self.c.simulation.getTime(),'signals':states,
                'vehicles':{v:{'xy':self.c.vehicle.getPosition(v),'speed':self.c.vehicle.getSpeed(v)} for v in self.c.vehicle.getIDList()},'metrics':self.metrics.series[-1]})
        if not self.config.local_reward: rewards[:]=rewards.mean()
        self.last_rewards=rewards; truncated=self.c.simulation.getTime()>=self.config.duration
        return self.observe(),rewards,False,truncated,self.info()
    def finish(self):
        summary=self.metrics.summary(); summary.update(wall_seconds=time.monotonic()-self.started,
            steps_per_second=self.config.duration/(time.monotonic()-self.started),signal_switches=sum(s.switches for s in self.signals.values()),illegal_requests=0)
        save_json(self.directory/'summary.json',summary)
        for name,rows in [('timeseries',self.metrics.series),('executed_actions',self.executed),('vehicles',self.metrics.completed+list(self.metrics.vehicles.values()))]:
            if rows:
                with open(self.directory/f'{name}.csv','w') as f:
                    writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows)))); writer.writeheader(); writer.writerows(rows)
        if self.replay: save_json(self.directory/'replay.json',{'positions':self.positions,'edges':self.edge_shapes,'frames':self.frames})
        self.close(); return summary
    def close(self):
        if self.c is not None:
            try: self.c.close()
            finally: self.c=None
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
