from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_signal_marl.evaluation.episode import controller_from_name, run_episode
from traffic_signal_marl.provenance import git_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible SUMO-RL episode with a baseline or checkpoint")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/smoke.json")
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument("--controller", choices=["fixed-time", "max-pressure"])
    agent.add_argument("--checkpoint", type=Path, help="Trusted local checkpoint produced by scripts/train.py")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/raw")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.checkpoint:
        from traffic_signal_marl.agents.ppo import PPOController
        import torch
        torch.set_num_threads(1)
        controller = PPOController(args.checkpoint)
    else:
        controller = controller_from_name(args.controller or config["controller"])
    controller_name = controller.name
    seed = args.seed if args.seed is not None else config["sumo_seed"]
    result = run_episode(
        network_file=PROJECT_ROOT / config["network_file"],
        route_file=PROJECT_ROOT / config["route_file"],
        controller=controller,
        seed=seed,
        episode_seconds=config["episode_seconds"],
        delta_time=config["delta_time_seconds"],
        yellow_time=config["yellow_time_seconds"],
        min_green=config["min_green_seconds"],
        max_green=config["max_green_seconds"],
        result_dir=args.output.resolve(),
    )
    result["checkpoint"] = str(args.checkpoint.resolve()) if args.checkpoint else None
    revision, status = git_metadata(PROJECT_ROOT)
    result["code_revision"] = revision
    result["working_tree_dirty"] = bool(status) if status is not None else None
    result["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("sumo-rl", "pettingzoo", "gymnasium", "traci", "sumolib", "numpy")
    }
    if args.checkpoint:
        result["packages"]["torch"] = torch.__version__
    output_file = args.output / f"{controller_name}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output_file), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
