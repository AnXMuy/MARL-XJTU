from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sumo
from sumo_rl import parallel_env

from traffic_signal_marl.baselines.controllers import FixedTimeController, MaxPressureController, SignalController, sumo_environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sumo_version() -> str:
    return subprocess.run(["sumo", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]


def _tripinfo_metrics(path: Path, scheduled_vehicles: int) -> dict[str, float | int]:
    trips = [node.attrib for node in ET.parse(path).getroot().iter("tripinfo")]
    count = len(trips)
    return {
        "completed_vehicles": count,
        "unfinished_vehicles": max(0, scheduled_vehicles - count),
        "completed_mean_waiting_time_seconds": sum(float(trip.get("waitingTime", 0)) for trip in trips) / count if count else None,
        "completed_mean_travel_time_seconds": sum(float(trip.get("duration", 0)) for trip in trips) / count if count else None,
        "completed_mean_time_loss_seconds": sum(float(trip.get("timeLoss", 0)) for trip in trips) / count if count else None,
    }


def run_episode(
    *,
    network_file: Path,
    route_file: Path,
    controller: SignalController,
    seed: int,
    episode_seconds: int,
    delta_time: int = 5,
    yellow_time: int = 3,
    min_green: int = 5,
    max_green: int = 50,
    result_dir: Path,
    record_actions: bool = False,
) -> dict[str, Any]:
    sumo_home = Path(sumo.__file__).resolve().parent
    os.environ["SUMO_HOME"] = str(sumo_home)
    network_file = network_file.resolve()
    route_file = route_file.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tripinfo_file = (result_dir / f"tripinfo_{controller.name}_seed{seed}_{run_id}.xml").resolve()
    config = {
        "controller": controller.name,
        "network": str(network_file),
        "network_sha256": _sha256(network_file),
        "route_file": str(route_file),
        "route_sha256": _sha256(route_file),
        "seed": seed,
        "episode_seconds": episode_seconds,
        "delta_time_seconds": delta_time,
        "yellow_time_seconds": yellow_time,
        "min_green_seconds": min_green,
        "max_green_seconds": max_green,
        "reward_function": "queue",
        "all_red_seconds": 0,
        "fixed_phase_duration_seconds": getattr(controller, "phase_duration_seconds", None),
        "sumo_version": _sumo_version(),
    }
    environment = parallel_env(
        net_file=str(network_file),
        route_file=str(route_file),
        use_gui=False,
        num_seconds=episode_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        max_green=max_green,
        reward_fn="queue",
        sumo_seed=seed,
        sumo_warnings=False,
        additional_sumo_cmd=f"--tripinfo-output={tripinfo_file} --no-step-log true --duration-log.disable true",
    )
    started_at = time.time()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    observations, _ = environment.reset(seed=seed)
    base_environment = sumo_environment(environment)
    trace = []
    action_trace = []
    departed = set()
    local_queue_integrals = defaultdict(float)
    original_step = base_environment._sumo_step

    def measured_step():
        original_step()
        connection = base_environment.sumo
        departed.update(connection.simulation.getDepartedIDList())
        vehicles = connection.vehicle.getIDList()
        stopped = sum(connection.vehicle.getSpeed(vehicle) < 0.1 for vehicle in vehicles)
        for signal_id, signal in base_environment.traffic_signals.items():
            local_queue_integrals[signal_id] += signal.get_total_queued()
        trace.append({"second": connection.simulation.getTime(), "vehicles": len(vehicles), "stopped": stopped,
                      "pending": len(connection.simulation.getPendingVehicles())})

    base_environment._sumo_step = measured_step
    previous_phases = {key: signal.green_phase for key, signal in base_environment.traffic_signals.items()}
    phase_switches = defaultdict(int)
    elapsed_seconds = 0
    try:
        while environment.agents:
            actions = controller.actions(observations, elapsed_seconds, environment)
            if record_actions:
                action_trace.append({"second": elapsed_seconds, "agent_ids": sorted(observations),
                                     "observations": [observations[agent].tolist() for agent in sorted(observations)],
                                     "actions": [int(actions[agent]) for agent in sorted(observations)]})
            observations, _, terminations, truncations, _ = environment.step(actions)
            for agent, signal in base_environment.traffic_signals.items():
                if signal.green_phase != previous_phases[agent]:
                    phase_switches[agent] += 1
                    previous_phases[agent] = signal.green_phase
            elapsed_seconds += delta_time
            if all(terminations.values()) or all(truncations.values()):
                break
    finally:
        environment.close()

    route_root = ET.parse(route_file).getroot()
    scheduled = sum(1 for child in route_root if child.tag in {"vehicle", "trip"})
    metric_summary = _tripinfo_metrics(tripinfo_file, scheduled)
    total_stopped_seconds = sum(row["stopped"] for row in trace)
    step_rows = base_environment.metrics
    queue_samples = [float(row.get("system_total_stopped", 0)) for row in step_rows]
    signal_ids = list(base_environment.traffic_signals)
    per_signal_waiting = {
        signal_id: max(float(row.get(f"{signal_id}_accumulated_waiting_time", 0)) for row in step_rows)
        if step_rows else 0.0
        for signal_id in signal_ids
    }
    metric_summary.update(
        {
            "scheduled_vehicles": scheduled,
            "departed_vehicles": len(departed),
            "total_stopped_vehicle_seconds": total_stopped_seconds,
            "restricted_wait_seconds_per_scheduled_vehicle": total_stopped_seconds / scheduled if scheduled else 0.0,
            "not_inserted_vehicles": scheduled - len(departed),
            "pending_vehicle_seconds": sum(row["pending"] for row in trace),
            "final_network_vehicles": trace[-1]["vehicles"] if trace else 0,
            "mean_network_vehicles": sum(row["vehicles"] for row in trace) / len(trace) if trace else 0,
            "throughput_vehicles_per_hour": metric_summary["completed_vehicles"] * 3600 / max(1, episode_seconds),
            "mean_network_stopped_vehicles": total_stopped_seconds / len(trace) if trace else 0,
            "max_network_stopped_vehicles": max((row["stopped"] for row in trace), default=0),
            "worst_intersection_queue_integral": max(local_queue_integrals.values(), default=0),
            "intersection_queue_integrals": dict(local_queue_integrals),
            "worst_intersection_waiting_time_seconds": max(per_signal_waiting.values(), default=0.0),
            "phase_switches": sum(phase_switches.values()),
            "phase_switches_by_intersection": dict(phase_switches),
            "intersection_waiting_time_seconds": per_signal_waiting,
        }
    )
    trace_file = result_dir / f"trace_{controller.name}_{seed}_{run_id}.json"
    trace_file.write_text(json.dumps(trace))
    action_file = result_dir / f"actions_{controller.name}_{seed}_{run_id}.json"
    if record_actions:
        action_file.write_text(json.dumps(action_trace))
    return {
        "config": config,
        "metrics": metric_summary,
        "runtime_seconds": time.time() - started_at,
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "tripinfo_file": str(tripinfo_file),
        "trace_file": str(trace_file),
        "action_trace_file": str(action_file) if record_actions else None,
    }


def controller_from_name(name: str) -> SignalController:
    if name == FixedTimeController.name:
        return FixedTimeController()
    if name == MaxPressureController.name:
        return MaxPressureController()
    raise ValueError(f"Unsupported baseline controller: {name}")
