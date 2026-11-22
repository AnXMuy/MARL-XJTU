from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate_scenario import generate_routes
from experiment_snapshot import snapshot

NETWORK = ROOT / "scenarios/network/grid4x4.net.xml"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def transform_route(source, destination, mode, seed):
    root = ET.parse(source).getroot()
    vehicles = root.findall("vehicle")
    generator = np.random.default_rng(seed)
    if mode == "burst":
        for vehicle in vehicles:
            departure = float(vehicle.get("depart"))
            vehicle.set("depart", f"{departure * (120 / 210) if departure < 210 else 120 + (departure - 210) * 2:.2f}")
    elif mode == "hotspot":
        candidates = [vehicle.find("route") for vehicle in vehicles]
        weights = np.array([8 if any(edge.startswith("C2") or edge.endswith("C2") for edge in route.get("edges").split())
                            else 1 for route in candidates], dtype=float)
        weights /= weights.sum()
        selected = generator.choice(len(candidates), size=len(vehicles), p=weights)
        for vehicle, index in zip(vehicles, selected):
            vehicle.remove(vehicle.find("route"))
            vehicle.append(copy.deepcopy(candidates[index]))
    else:
        raise ValueError(mode)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="UTF-8", xml_declaration=True)
    atomic_json(destination.with_suffix(".manifest.json"), dict(source_sha256=sha(source), route_sha256=sha(destination),
                mode=mode, seed=seed, scheduled_vehicles=len(vehicles)))


def generate(output, protocol):
    if (output / "protocol.json").exists():
        raise FileExistsError(f"Frozen study already exists: {output}")
    routes = output / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    for split, seeds in (("train", protocol["train_demand_seeds"]), ("validation", protocol["validation_seeds"]),
                         ("test", protocol["test_demand_seeds"])):
        for profile, period in protocol["profiles"].items():
            for seed in seeds:
                generate_routes(NETWORK, routes / f"{split}_{profile}_{seed}.rou.xml", seed, period, 300)
    for seed in protocol["ood_seeds"]:
        source = routes / f"ood_uniform_{seed}.rou.xml"
        generate_routes(NETWORK, source, seed, 0.5, 300)
        for mode in ("burst", "hotspot"):
            transform_route(source, routes / f"ood_{mode}_{seed}.rou.xml", mode, seed)
        generate_routes(NETWORK, routes / f"ood_sustained_{seed}.rou.xml", seed, 0.5, 600)
    snapshot(output, ROOT)
    atomic_json(output / "inputs.json", dict(network_sha256=sha(NETWORK), routes={path.name: sha(path) for path in routes.glob("*.xml")}))
    atomic_json(output / "protocol.json", protocol)


def training_task(task):
    from train import train
    config, routes, output = task
    return str(train(config, routes, output, resume=(output / "checkpoint.pt").exists()))


def evaluation_task(task):
    import torch
    import traci
    from traffic_signal_marl.agents.ppo import PPOController
    from traffic_signal_marl.baselines.controllers import FixedTimeController, MaxPressureController
    from traffic_signal_marl.evaluation.episode import run_episode

    torch.set_num_threads(1)
    backend = "libsumo" if traci.isLibsumo() else "traci"
    if "LIBSUMO_AS_TRACI" in os.environ and backend != "libsumo":
        raise RuntimeError("Requested libsumo but could not load it")
    output, route, method, initialization, checkpoint, horizon, group, profile, seed = task
    path = output / "evaluation" / group / profile / str(seed) / method / str(initialization) / "result.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous["config"]["route_sha256"] != sha(route) or
                previous.get("checkpoint_sha256") != (sha(checkpoint) if checkpoint else None) or
                previous.get("backend", "traci") != backend):
            raise ValueError(f"Stale cache: {path}")
        return str(path)
    if checkpoint:
        controller = PPOController(checkpoint)
    elif method == "max-pressure":
        controller = MaxPressureController()
    else:
        controller = FixedTimeController(int(method.split("-")[-1]) if method != "hold" else horizon + 1)
    controller.name = method
    try:
        result = run_episode(network_file=NETWORK, route_file=route, controller=controller, seed=seed,
                             episode_seconds=horizon, result_dir=path.parent, record_actions=bool(checkpoint))
    except Exception as error:
        raise RuntimeError(f"group={group} scenario={profile} seed={seed} method={method} initialization={initialization}") from error
    result.update(group=group, profile=profile, method=method, training_seed=initialization, backend=backend,
                  checkpoint=str(checkpoint) if checkpoint else None, checkpoint_sha256=sha(checkpoint) if checkpoint else None)
    atomic_json(path, result)
    return str(path)


def baseline_names(protocol):
    return [f"fixed-{period}" for period in protocol["fixed_periods"]] + ["max-pressure", "hold"]


def evaluation_tasks(output, protocol, stage):
    tasks = []
    if stage == "baselines":
        for split, seeds in (("validation", protocol["validation_seeds"]), ("test", protocol["test_demand_seeds"])):
            methods = baseline_names(protocol) if split == "test" else baseline_names(protocol)[:-2]
            for profile in protocol["profiles"]:
                for seed in seeds:
                    for method in methods:
                        tasks.append((output, output / f"routes/{split}_{profile}_{seed}.rou.xml", method, None, None,
                                      600, split, profile, seed))
    elif stage == "evaluate":
        for group in ("test", "budget"):
            seeds = protocol["test_demand_seeds"] if group == "test" else protocol["test_demand_seeds"][:5]
            methods = protocol["methods"] if group == "test" else ["ippo", "mappo"]
            for profile in protocol["profiles"]:
                for seed in seeds:
                    for method in methods:
                        for initialization in protocol["initialization_seeds"]:
                            checkpoint = output / f"models/{method}/{initialization}/checkpoint_{60 if group == 'test' else 24}.pt"
                            tasks.append((output, output / f"routes/test_{profile}_{seed}.rou.xml", method, initialization,
                                          checkpoint, 600, group, profile, seed))
    elif stage == "ood":
        selections = json.loads((output / "selection.json").read_text())
        for profile in ("burst", "hotspot", "sustained"):
            fixed = selections["C2"]["method"]
            for seed in protocol["ood_seeds"]:
                for method in (fixed, "max-pressure", "ippo", "mappo"):
                    for initialization in (protocol["initialization_seeds"] if method in ("ippo", "mappo") else [None]):
                        checkpoint = output / f"models/{method}/{initialization}/checkpoint_60.pt" if initialization else None
                        tasks.append((output, output / f"routes/ood_{profile}_{seed}.rou.xml", method, initialization,
                                      checkpoint, 900 if profile == "sustained" else 600, "ood", profile, seed))
    return tasks


def select(output, protocol):
    selections = {}
    for profile in protocol["profiles"]:
        scores = {}
        for period in protocol["fixed_periods"]:
            method = f"fixed-{period}"
            results = [json.loads((output / f"evaluation/validation/{profile}/{seed}/{method}/None/result.json").read_text())
                       for seed in protocol["validation_seeds"]]
            scores[method] = float(np.mean([result["metrics"][protocol["primary_metric"]] for result in results]))
        best = min(scores, key=scores.get)
        selections[profile] = dict(method=best, validation_scores=scores)
    atomic_json(output / "selection.json", selections)
    print(json.dumps(selections, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Frozen extended experiment; never selects models on test results")
    parser.add_argument("--stage", required=True, choices=["generate", "train", "baselines", "select", "evaluate", "ood"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-only", action="store_true", help="Evaluate only checkpoints already archived")
    args = parser.parse_args()
    if args.ready_only and args.stage not in ("evaluate", "ood"):
        parser.error("--ready-only requires evaluate or ood")
    protocol = json.loads(args.protocol.read_text())
    output = ROOT / protocol["output"]
    if args.stage == "generate":
        generate(output, protocol)
        return
    if protocol != json.loads((output / "protocol.json").read_text()):
        raise ValueError("Protocol changed")
    if args.stage == "select":
        select(output, protocol)
        return
    if args.stage == "train":
        tasks = []
        routes = [output / f"routes/train_{profile}_{seed}.rou.xml" for seed in protocol["train_demand_seeds"] for profile in protocol["profiles"]]
        for method in protocol["methods"]:
            for initialization in protocol["initialization_seeds"]:
                config = json.loads((ROOT / "configs/train.json").read_text())
                config.update(seed=initialization, episodes=protocol["training_episodes"], checkpoint_episodes=protocol["checkpoint_episodes"],
                              algorithm="ippo" if method == "ippo" else "mappo", actor_observation="phase-only" if method == "mappo-phase" else "local",
                              reward="network-queue" if method == "mappo-network" else "queue")
                tasks.append((config, routes, output / f"models/{method}/{initialization}"))
        worker = training_task
    else:
        tasks = evaluation_tasks(output, protocol, args.stage)
        if args.ready_only:
            tasks = [task for task in tasks if (task[4] is None or task[4].is_file()) and
                     not (task[0] / f"evaluation/{task[6]}/{task[7]}/{task[8]}/{task[2]}/{task[3]}/result.json").exists()]
        worker = evaluation_task
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
