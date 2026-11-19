"""Generated straight-through grid: one lane per approach, two conflict-free phases."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from traffic_marl.utils.provenance import sha256

def generate(config,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); n,d=config.size,config.spacing
    nodes=ET.Element('nodes'); edges=ET.Element('edges')
    def node(name,x,y,t='priority'): ET.SubElement(nodes,'node',id=name,x=str(x),y=str(y),type=t)
    def edge(a,b): ET.SubElement(edges,'edge',id=f'{a}_{b}',attrib={'from':a,'to':b},numLanes='1',speed='13.89')
    for x in range(n):
        for y in range(n): node(f'J{x}_{y}',(x+1)*d,(y+1)*d,'traffic_light')
    routes={}
    for i in range(n):
        lines={'W':[f'W{i}']+[f'J{x}_{i}' for x in range(n)]+[f'E{i}'],
               'E':[f'E{i}']+[f'J{x}_{i}' for x in reversed(range(n))]+[f'W{i}'],
               'S':[f'S{i}']+[f'J{i}_{y}' for y in range(n)]+[f'N{i}'],
               'N':[f'N{i}']+[f'J{i}_{y}' for y in reversed(range(n))]+[f'S{i}']}
        for s,ns in lines.items():
            if s=='W': node(f'{s}{i}',0,(i+1)*d)
            elif s=='E': node(f'{s}{i}',(n+1)*d,(i+1)*d)
            elif s=='S': node(f'{s}{i}',(i+1)*d,0)
            else: node(f'{s}{i}',(i+1)*d,(n+1)*d)
            es=[f'{a}_{b}' for a,b in zip(ns,ns[1:])]; routes[f'{s}{i}']=es
            for a,b in zip(ns,ns[1:]): edge(a,b)
    # Movement connections align each route's incoming lane to its straight continuation.
    connections=ET.Element('connections')
    for es in routes.values():
        for a,b in zip(es,es[1:]): ET.SubElement(connections,'connection',attrib={'from':a,'to':b,'fromLane':'0','toLane':'0'})
    for name,root in [('nodes',nodes),('edges',edges),('connections',connections)]: ET.ElementTree(root).write(directory/f'{name}.xml')
    cmd=['netconvert','--node-files',str(directory/'nodes.xml'),'--edge-files',str(directory/'edges.xml'),
         '--connection-files',str(directory/'connections.xml'),'--output-file',str(directory/'network.net.xml'),
         '--no-turnarounds','true','--tls.guess','true','--tls.default-type','static','--tls.green.time','30','--tls.yellow.time','3']
    subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=60)
    rng=np.random.default_rng(config.seed); arrivals=[]; end=config.demand_end or config.duration
    for name,es in sorted(routes.items()):
        for t in range(end):
            factor=1.0
            if config.scenario in ('morning','evening'):
                factor=1.7 if name[0] in ('WS' if config.scenario=='morning' else 'EN') else .3
            if config.scenario=='pulse' and end//3<=t<end//3+min(300,end//3) and name=='W0': factor=2
            for _ in range(rng.poisson(config.entrance_rates.get(name,config.rate)*factor/3600)):
                arrivals.append((t+float(rng.uniform(0,.99)),name))
    root=ET.Element('routes'); ET.SubElement(root,'vType',id='car',vClass='passenger',carFollowModel='Krauss',sigma='.5',accel='2.6',decel='4.5',length='5',minGap='2.5',maxSpeed='13.89')
    for name,es in sorted(routes.items()): ET.SubElement(root,'route',id=name,edges=' '.join(es))
    for k,(t,name) in enumerate(sorted(arrivals)): ET.SubElement(root,'vehicle',id=f'v{k}',type='car',route=name,depart=f'{t:.3f}',departLane='0',departSpeed='max')
    ET.ElementTree(root).write(directory/'demand.rou.xml')
    manifest={'netconvert_command':cmd,'network_sha256':sha256(directory/'network.net.xml'),'demand_sha256':sha256(directory/'demand.rou.xml'),'scheduled':len(arrivals),'routes':routes,'seed':config.seed}
    (directory/'demand_manifest.json').write_text(json.dumps(manifest,indent=2)); return manifest
