import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from sumo_rl import parallel_env

from traffic_signal_marl.baselines.controllers import sumo_environment
from traffic_signal_marl.baselines.controllers import FixedTimeController
from traffic_signal_marl.evaluation.episode import run_episode

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("internal_size", [1, 2])
def test_phase_safety_small_network(tmp_path, internal_size):
    count = internal_size + 2
    ids = [f"{chr(65 + column)}{row}" for column in range(1, count - 1) for row in range(1, count - 1)]
    network = tmp_path / "grid.net.xml"
    subprocess.run(["netgenerate", "--grid", "--grid.number", str(count), "--grid.length", "200",
                    "--tls.set", ",".join(ids), "--no-turnarounds", "--output-file", str(network)], check=True)
    route = tmp_path / "empty.rou.xml"
    ET.ElementTree(ET.Element("routes")).write(route)
    env = parallel_env(net_file=str(network), route_file=str(route), num_seconds=30,
                       delta_time=5, yellow_time=3, min_green=5, sumo_seed=1,
                       additional_sumo_cmd="--no-step-log true")
    try:
        observations, _ = env.reset(seed=1)
        base = sumo_environment(env)
        assert set(observations) == set(ids)
        for _ in range(2):
            observations, rewards, *_ = env.step({agent: 1 for agent in ids})
            assert all(reward == 0 for reward in rewards.values())
            for agent, observation in observations.items():
                assert env.observation_space(agent).contains(observation)
                assert env.action_space(agent).contains(1)
        previous = {agent: signal.green_phase for agent, signal in base.traffic_signals.items()}
        states = []
        original_step = base._sumo_step

        def record_step():
            states.append({agent: base.sumo.trafficlight.getRedYellowGreenState(agent) for agent in ids})
            original_step()

        base._sumo_step = record_step
        env.step({agent: 1 for agent in ids})
        assert all(phase == 0 for phase in previous.values())
        assert all(signal.green_phase == 1 for signal in base.traffic_signals.values())
        assert all("y" in states[0][agent] for agent in ids)
        assert all("y" in states[2][agent] for agent in ids)
        assert all("y" not in states[3][agent] for agent in ids)
    finally:
        env.close()


@pytest.mark.parametrize("count", [0, 1, 100])
def test_single_approach_metrics(tmp_path, count):
    route = tmp_path / "approach.rou.xml"
    vehicles = "".join(f'<vehicle id="car{index}" depart="0"><route edges="B0B1 B1B2"/></vehicle>'
                       for index in range(count))
    route.write_text(f"<routes>{vehicles}</routes>")
    result = run_episode(network_file=ROOT / "scenarios/network/grid4x4.net.xml", route_file=route,
                         episode_seconds=90, controller=FixedTimeController(), seed=7, result_dir=tmp_path)
    metrics = result["metrics"]
    assert metrics["scheduled_vehicles"] == count
    assert metrics["completed_vehicles"] + metrics["unfinished_vehicles"] == count
    assert metrics["departed_vehicles"] + metrics["not_inserted_vehicles"] == count
    assert metrics["total_stopped_vehicle_seconds"] == pytest.approx(90 * metrics["mean_network_stopped_vehicles"])
    if count == 0:
        assert metrics["completed_mean_travel_time_seconds"] is None
    elif count == 1:
        assert metrics["completed_vehicles"] == 1
        assert metrics["completed_mean_travel_time_seconds"] > 0
    else:
        assert metrics["unfinished_vehicles"] > 0
        assert metrics["not_inserted_vehicles"] > 0
        assert metrics["pending_vehicle_seconds"] > 0
        assert metrics["intersection_queue_integrals"]["B1"] > 0
