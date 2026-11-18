import numpy as np
from demo.redesign import Session,snapshot

def test_layout_and_commands_are_paired():
    s=Session(None)
    s.configure([16,24,32,36],['0-1: door','1-2: door'])
    assert s.env.area.tolist()==[[16,24,32,36]]
    assert np.array_equal(s.env.connections.numpy(),s.other.connections.numpy())
    for command in ['我到家了','客厅有点冷','省电模式','我要睡觉了']:
        s.command(command);s.step(4)
        assert np.array_equal(s.env.preference.numpy(),s.other.preference.numpy())
    assert s.env.preference[0,10]==1
    s.manual([22,21,20,19],[1,.5,0,1],.8)
    assert s.env.preference[0,8]>.79
    assert len(snapshot(s))==6
    s.day();assert len(s.rows)==96

def test_named_baselines_reset_same_day_without_changing_rules():
    s=Session(None,baseline='thermostat')
    assert not s.rule.coordinated
    assert np.array_equal(s.env.temps.numpy(),s.other.temps.numpy())
    s.step(2)
    s.baseline_kind='coordinated';s.reset()
    assert s.rule.coordinated and s.env.t==s.other.t==0
    assert np.array_equal(s.env.temps.numpy(),s.other.temps.numpy())
