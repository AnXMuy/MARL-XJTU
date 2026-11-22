import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from traffic_signal_marl.agents.ppo import ActorCritic, update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extended import transform_route, evaluation_tasks


def test_phase_actor_cannot_read_traffic_but_critic_can():
    torch.manual_seed(3)
    model = ActorCritic(11, 2, 16, centralized=True, actor_observation="phase-only")
    observations = torch.randn(2, 16, 11)
    changed = observations.clone()
    changed[..., 3:] += 10
    assert torch.equal(model.logits(observations), model.logits(changed))
    assert not torch.equal(model.values(observations), model.values(changed))
    inputs = observations.clone().requires_grad_()
    model.logits(inputs).sum().backward()
    assert torch.count_nonzero(inputs.grad[..., 3:]) == 0


def test_ppo_diagnostics_finite_and_updates_actor():
    torch.manual_seed(2)
    model = ActorCritic(11, 2, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    observations = torch.randn(8, 3, 11)
    with torch.no_grad():
        distribution = torch.distributions.Categorical(logits=model.logits(observations))
        actions = distribution.sample()
        old = distribution.log_prob(actions)
        values = model.values(observations)
    before = model.actor[0].weight.detach().clone()
    config = json.loads((ROOT / "configs/train.json").read_text())
    loss = update(model, optimizer, observations, actions, old, torch.randn(8, 3), values,
                  torch.zeros(3), config)
    assert np.isfinite(loss)
    assert all(value is None or np.isfinite(value) for value in model.last_update_metrics.values())
    assert 0 <= model.last_update_metrics["clip_fraction"] <= 1
    assert not torch.equal(before, model.actor[0].weight)


def test_ood_transform_preserves_count_and_repeatability(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text('<routes>' + ''.join(f'<vehicle id="v{index}" depart="{index*0.5}"><route edges="A1C2 C2D3"/></vehicle>'
                                         for index in range(600)) + '</routes>')
    for mode in ("burst", "hotspot"):
        first, second = tmp_path / f"{mode}1.xml", tmp_path / f"{mode}2.xml"
        transform_route(source, first, mode, 42)
        transform_route(source, second, mode, 42)
        assert first.read_bytes() == second.read_bytes()
        nodes = ET.parse(first).getroot().findall("vehicle")
        assert len(nodes) == 600
        departures = [float(node.get("depart")) for node in nodes]
        assert departures == sorted(departures)
        if mode == "burst":
            assert sum(departure < 120 for departure in departures) == 420


def test_extended_matrix_counts_and_seed_disjointness(tmp_path):
    protocol = json.loads((ROOT / "configs/extended.json").read_text())
    splits = [set(protocol[key]) for key in ("train_demand_seeds", "validation_seeds", "test_demand_seeds", "ood_seeds")]
    for index, split in enumerate(splits):
        assert all(not split.intersection(other) for other in splits[index+1:])
    assert len(evaluation_tasks(tmp_path, protocol, "baselines")) == 330
    assert len(evaluation_tasks(tmp_path, protocol, "evaluate")) == 750
