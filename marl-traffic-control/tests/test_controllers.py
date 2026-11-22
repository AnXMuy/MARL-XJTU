from types import SimpleNamespace

from traffic_signal_marl.baselines.controllers import FixedTimeController, MaxPressureController


def test_fixed_time_rotates_phases():
    base = SimpleNamespace(traffic_signals={"J0": SimpleNamespace(num_green_phases=3)}, sumo=None)
    assert FixedTimeController(20).actions({"J0": []}, 40, base) == {"J0": 2}


def test_max_pressure_uses_downstream_counts():
    counts = {"in0": 4, "out0": 1, "in1": 2, "out1": 3}
    lane = SimpleNamespace(getLastStepVehicleNumber=lambda lane_id: counts[lane_id])
    trafficlight = SimpleNamespace(getControlledLinks=lambda _: [[("in0", "out0", "")], [("in1", "out1", "")]])
    signal = SimpleNamespace(green_phases=[SimpleNamespace(state="Gr"), SimpleNamespace(state="rG")], green_phase=1)
    base = SimpleNamespace(traffic_signals={"J0": signal}, sumo=SimpleNamespace(lane=lane, trafficlight=trafficlight))
    assert MaxPressureController().actions({"J0": []}, 0, base) == {"J0": 0}
