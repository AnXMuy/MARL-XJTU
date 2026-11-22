from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def sumo_environment(parallel_env: Any) -> Any:
    current = parallel_env
    for _ in range(8):
        if hasattr(current, "traffic_signals") and hasattr(current, "sumo"):
            return current
        if hasattr(current, "unwrapped"):
            current = current.unwrapped
            if hasattr(current, "env") and not hasattr(current, "traffic_signals"):
                current = current.env
            continue
        if hasattr(current, "aec_env"):
            current = current.aec_env
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    raise TypeError("Could not resolve the underlying SUMO-RL environment")


class SignalController(ABC):
    name: str

    @abstractmethod
    def actions(self, observations: dict[str, Any], elapsed_seconds: int, parallel_env: Any) -> dict[str, int]:
        raise NotImplementedError


class FixedTimeController(SignalController):
    name = "fixed-time"

    def __init__(self, phase_duration_seconds: int = 30):
        if phase_duration_seconds <= 0:
            raise ValueError("phase_duration_seconds must be positive")
        self.phase_duration_seconds = phase_duration_seconds

    def actions(self, observations: dict[str, Any], elapsed_seconds: int, parallel_env: Any) -> dict[str, int]:
        environment = sumo_environment(parallel_env)
        selected_phase = elapsed_seconds // self.phase_duration_seconds
        return {
            agent: selected_phase % environment.traffic_signals[agent].num_green_phases
            for agent in observations
        }


class MaxPressureController(SignalController):
    name = "max-pressure"

    def actions(self, observations: dict[str, Any], elapsed_seconds: int, parallel_env: Any) -> dict[str, int]:
        del elapsed_seconds
        environment = sumo_environment(parallel_env)
        decisions = {}
        for agent in observations:
            signal = environment.traffic_signals[agent]
            controlled_links = environment.sumo.trafficlight.getControlledLinks(agent)
            pressures = []
            for phase in signal.green_phases:
                pressure = 0.0
                for link_index, connections in enumerate(controlled_links):
                    if link_index >= len(phase.state) or phase.state[link_index].lower() != "g":
                        continue
                    for incoming_lane, outgoing_lane, _ in connections:
                        incoming_count = environment.sumo.lane.getLastStepVehicleNumber(incoming_lane)
                        outgoing_count = environment.sumo.lane.getLastStepVehicleNumber(outgoing_lane)
                        pressure += incoming_count - outgoing_count
                pressures.append(pressure)
            maximum = max(pressures)
            current_phase = signal.green_phase
            decisions[agent] = current_phase if pressures[current_phase] == maximum else pressures.index(maximum)
        return decisions
