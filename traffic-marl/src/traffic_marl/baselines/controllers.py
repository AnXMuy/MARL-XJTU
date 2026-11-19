import numpy as np
class Controller:
    def __init__(self,name):
        if name not in ('fixed','actuated','max_pressure','green_wave'): raise ValueError(name)
        self.name=name
    def actions(self,env):
        actions=[]
        for a in env.agents:
            s=env.signals[a]; switch=False
            if s.mask()[1]:
                q,occ,wait,d=env.lane_stats(a); groups=np.array(env.groups[a]); current=groups==s.phase
                if self.name=='fixed': switch=s.elapsed>=env.config.fixed_green
                elif self.name=='green_wave':
                    cycle=2*(env.config.fixed_green+env.config.yellow+env.config.all_red)
                    offset=env.positions[a][0]/13.89
                    target=int(((env.c.simulation.getTime()-offset)%cycle)>=cycle/2); switch=target!=s.phase
                elif self.name=='actuated': switch=(q[~current].sum()>0 and (occ[current].max()<.05 or s.elapsed>=env.config.max_green))
                else:
                    down=np.array([env.c.lane.getLastStepHaltingNumber(link[1]) for link in env.links[a]])
                    p=q-down; scores=[p[groups==g].sum() for g in (0,1)]
                    switch=scores[1-s.phase]>scores[s.phase]+.1
            actions.append(int(switch))
        return np.array(actions)
