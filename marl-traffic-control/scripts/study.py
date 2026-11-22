from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate_scenario import generate_routes
from traffic_signal_marl.provenance import git_metadata


def train_worker(task):
    from train import train
    config, routes, output = task
    checkpoint = Path(output) / "checkpoint.pt"
    return str(train(config, [Path(route) for route in routes], Path(output), resume=checkpoint.exists()))


def eval_worker(task):
    from traffic_signal_marl.agents.ppo import PPOController
    from traffic_signal_marl.evaluation.episode import controller_from_name, run_episode
    import torch
    torch.set_num_threads(1)
    route, method, checkpoint, seed, output, profile, training_seed, horizon = task
    output = Path(output)
    if (output / "result.json").exists():
        return str(output / "result.json")
    controller = PPOController(checkpoint) if checkpoint else controller_from_name(method)
    controller.name = method
    try:
        result = run_episode(network_file=ROOT / "scenarios/network/grid4x4.net.xml", route_file=Path(route),
                             controller=controller, seed=seed, episode_seconds=horizon, result_dir=output)
    except Exception as error:
        raise RuntimeError(f"Failed method={method} scenario={profile} seed={seed} checkpoint={checkpoint}") from error
    result.update(profile=profile, method=method, training_seed=training_seed, checkpoint=checkpoint,
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output / "result.json")
    return str(output / "result.json")


def snapshot(output):
    output.mkdir(parents=True, exist_ok=True)
    revision, status = git_metadata(ROOT)
    manifest = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
                for name in ["numpy", "torch", "sumo-rl", "pettingzoo", "gymnasium", "traci", "sumolib"]},
                "git_commit": revision, "git_status": status,
                "source_hashes": {}}
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["src", "scripts", "configs", "tests"]:
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    relative = str(path.relative_to(ROOT))
                    manifest["source_hashes"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.write(path, relative)
    (output / "environment.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "train", "evaluate"], required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--eval-set", choices=["all", "baseline", "learned"], default="all")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = ROOT / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        if (output / "protocol.json").exists():
            raise FileExistsError(f"Study already frozen: {output}; choose a new protocol output")
        snapshot(output)
        for split, seeds in [("train", protocol["train_demand_seeds"]), ("test", protocol["test_demand_seeds"])]:
            for profile, period in protocol["profiles"].items():
                for seed in seeds:
                    route = ROOT / f"scenarios/routes/{split}_{profile}_{seed}.rou.xml"
                    generate_routes(ROOT / "scenarios/network/grid4x4.net.xml", route, seed, period, protocol["demand_seconds"])
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        return
    stored = json.loads((output / "protocol.json").read_text())
    if protocol != stored:
        raise ValueError("Protocol changed since generation; choose a new output")
    tasks = []
    if args.stage == "train":
        routes = [str(ROOT / f"scenarios/routes/train_{profile}_{seed}.rou.xml")
                  for seed in protocol["train_demand_seeds"] for profile in protocol["profiles"]]
        for method in ["ippo", "mappo", "mappo-network"]:
            for seed in protocol["initialization_seeds"]:
                config = json.loads((ROOT / "configs/train.json").read_text())
                config.update(seed=seed, algorithm="ippo" if method == "ippo" else "mappo", episodes=protocol["training_episodes"],
                              reward="network-queue" if method == "mappo-network" else "queue")
                tasks.append((config, routes, str(output / "models" / method / str(seed))))
        worker = train_worker
    else:
        for profile in protocol["profiles"]:
            for seed in protocol["test_demand_seeds"]:
                route = str(ROOT / f"scenarios/routes/test_{profile}_{seed}.rou.xml")
                for method in ["fixed-time", "max-pressure", "ippo", "mappo", "mappo-network"]:
                    learned = method in ["ippo", "mappo", "mappo-network"]
                    if (args.eval_set == "baseline" and learned) or (args.eval_set == "learned" and not learned):
                        continue
                    training_seeds = protocol["initialization_seeds"] if learned else [None]
                    for initialization in training_seeds:
                        checkpoint = str(output / "models" / method / str(initialization) / "checkpoint.pt") if initialization else None
                        result_dir = output / "evaluation" / profile / str(seed) / method / str(initialization)
                        tasks.append((route, method, checkpoint, seed, str(result_dir), profile, initialization,
                                      protocol["episode_seconds"]))
        worker = eval_worker
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
