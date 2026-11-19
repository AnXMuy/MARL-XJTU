import pytest
from traffic_marl.env.safety import Signal

def test_switch_has_yellow_and_all_red_clearance():
    s=Signal(min_green=10,yellow=3,all_red=2)
    for _ in range(10): s.tick()
    assert s.mask()==[True,True]
    s.request(1)
    assert s.state([0,1,0,1])=='yryr'
    for _ in range(3): s.tick()
    assert s.stage=='red' and s.state([0,1,0,1])=='rrrr'
    for _ in range(2): s.tick()
    assert s.phase==1 and s.stage=='green' and s.state([0,1,0,1])=='rGrG'

def test_illegal_early_switch_fails_closed():
    s=Signal(min_green=10)
    with pytest.raises(ValueError): s.request(1)

def test_keep_action_does_not_transition():
    s=Signal(min_green=10)
    s.request(0)
    assert s.stage=='green' and s.phase==0
