import hashlib
import json
import os
from pathlib import Path
import shutil
import pytest
from traffic_marl.config import Config
from traffic_marl.env.sumo_env import SumoEnv
from traffic_marl.baselines.controllers import Controller

WORKSPACE = Path(__file__).resolve().parents[2]
SUMO = shutil.which('sumo') or (WORKSPACE/'.venv/bin/sumo')
NETCONVERT = shutil.which('netconvert') or (WORKSPACE/'.venv/bin/netconvert')
pytestmark=pytest.mark.skipif(not Path(SUMO).exists() or not Path(NETCONVERT).exists(),reason='SUMO CLI unavailable')

def run_once(cfg,path,policy):
    with SumoEnv(cfg,path) as env:
        obs,info=env.reset(); controller=Controller(policy); done=False
        while not done:
            obs,_,_,done,info=env.step(controller.actions(env))
        result=env.finish()
        demand=env.manifest['demand_sha256']
    return result,demand

def test_single_junction_smoke_and_reproducible_demand(tmp_path):
    cfg=Config(size=1,duration=120,warmup=20,rate=120,seed=93)
    a,ha=run_once(cfg,tmp_path/'a','max_pressure'); b,hb=run_once(cfg,tmp_path/'b','max_pressure')
    assert ha==hb
    assert a['scheduled']==b['scheduled'] and a['arrived']==b['arrived']
    assert a['teleports']==0 and a['illegal_requests']==0
    assert a['in_network']+a['arrived']<=a['scheduled']

def test_paired_demand_across_rules_on_grid(tmp_path):
    cfg=Config(size=3,duration=80,warmup=10,rate=80,seed=7)
    a,ha=run_once(cfg,tmp_path/'fixed','fixed'); b,hb=run_once(cfg,tmp_path/'pressure','max_pressure')
    assert ha==hb
    assert a['scheduled']==b['scheduled']
    assert a['teleports']==b['teleports']==0
    assert a['illegal_requests']==b['illegal_requests']==0
