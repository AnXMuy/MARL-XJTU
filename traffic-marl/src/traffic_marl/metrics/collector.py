"""One authoritative 1-second metric implementation; vehicle cohorts exclude warmup departures."""
import numpy as np

def mean(xs): return float(np.mean(xs)) if len(xs) else None
def percentile(xs,p): return float(np.percentile(xs,p)) if len(xs) else None
def gini(xs):
    a=np.sort(np.asarray(xs,dtype=float))
    if not len(a): return None
    if not a.sum(): return 0.0
    return float((2*np.dot(np.arange(1,len(a)+1),a)/a.sum()-len(a)-1)/len(a))

class Collector:
    def __init__(self,config,scheduled):
        self.config=config; self.scheduled=scheduled; self.vehicles={}; self.completed=[]
        self.series=[]; self.departed=0; self.arrived=0; self.teleports=0
    def update(self,c,queues,spillback):
        t=c.simulation.getTime()
        for vid in c.simulation.getDepartedIDList():
            self.departed+=1; edges=c.vehicle.getRoute(vid)
            free=sum(c.lane.getLength(e+'_0')/c.lane.getMaxSpeed(e+'_0') for e in edges)
            self.vehicles[vid]={'id':vid,'depart':t,'wait':0,'stops':0,'stopped':False,'free_flow':free,'speed_sum':0,'samples':0}
        for vid in c.vehicle.getIDList():
            if vid not in self.vehicles: continue
            v=self.vehicles[vid]; stopped=c.vehicle.getSpeed(vid)<.1
            v['wait']+=int(stopped); v['stops']+=int(stopped and not v['stopped']); v['stopped']=stopped
            v['speed_sum']+=c.vehicle.getSpeed(vid); v['samples']+=1
        for vid in c.simulation.getArrivedIDList():
            self.arrived+=1; v=self.vehicles.pop(vid); v.update(arrive=t,travel_time=t-v['depart'])
            v['delay']=v['travel_time']-v['free_flow']; self.completed.append(v)
        self.teleports+=c.simulation.getStartingTeleportNumber()
        self.series.append({'time':t,'queue':sum(queues),'max_lane_queue':max(queues,default=0),
            'arrived':self.arrived,'departed':self.departed,'in_network':c.vehicle.getIDCount(),
            'teleports':self.teleports,'spillback_lanes':spillback,
            'mean_speed':mean([c.vehicle.getSpeed(v) for v in c.vehicle.getIDList()]) or 0})
    def summary(self):
        rows=[r for r in self.series if r['time']>self.config.warmup]
        cohort=[v for v in self.completed if v['depart']>self.config.warmup]
        waits=[v['wait'] for v in cohort]; all_v=self.completed+list(self.vehicles.values())
        speed_sum=sum(v['speed_sum'] for v in all_v); samples=sum(v['samples'] for v in all_v)
        return {'scheduled':self.scheduled,'departed':self.departed,'arrived':self.arrived,
            'completed_cohort':len(cohort),'in_network':len(self.vehicles),'uninserted':self.scheduled-self.departed,
            'completion_rate':self.arrived/self.scheduled if self.scheduled else 1.,
            'travel_time_mean':mean([v['travel_time'] for v in cohort]),'travel_time_p95':percentile([v['travel_time'] for v in cohort],95),
            'delay_mean':mean([v['delay'] for v in cohort]),'waiting_mean':mean(waits),'waiting_p95':percentile(waits,95),
            'waiting_gini':gini(waits),'waiting_jain':float(sum(waits)**2/(len(waits)*sum(x*x for x in waits))) if waits and sum(x*x for x in waits) else (1. if waits else None),
            'total_waiting_vehicle_seconds':sum(v['wait'] for v in all_v),
            'queue_mean':mean([r['queue'] for r in rows]),'queue_p95':percentile([r['queue'] for r in rows],95),
            'max_lane_queue':max((r['max_lane_queue'] for r in rows),default=0),
            'throughput_per_hour':sum(v['arrive']>self.config.warmup for v in self.completed)*3600/(self.config.duration-self.config.warmup),
            'speed_mean':speed_sum/samples if samples else None,'stops_mean':mean([v['stops'] for v in cohort]),
            'spillback_lane_seconds':sum(r['spillback_lanes'] for r in rows),'teleports':self.teleports,
            'status':'severe_congestion' if self.teleports else 'ok'}
