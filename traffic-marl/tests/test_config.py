import pytest
from traffic_marl.config import Config

def test_config_rejects_unknown_entrance():
    with pytest.raises(ValueError): Config(entrance_rates={'Q0': 100})

def test_seed_does_not_change_config_digest_content_except_seed():
    a,b=Config(),Config(seed=2)
    assert a.digest()!=b.digest()
