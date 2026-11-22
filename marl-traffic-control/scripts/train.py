from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traffic_signal_marl.agents.ppo import ActorCritic, update
from sumo_rl import parallel_env


def reconcile_log(output, episode):
    log = output / "train.jsonl"
    if not log.exists():
        if episode >= 0:
            raise ValueError(f"Missing training log for checkpoint: {output}")
        return
    records = log.read_text().splitlines()
    committed = [json.loads(record) for record in records[:episode + 1]]
    if [record["episode"] for record in committed] != list(range(episode + 1)):
        raise ValueError(f"Training log does not match checkpoint: {output}")
    if len(records) > len(committed):
        temporary = output / "train.jsonl.tmp"
        temporary.write_text("".join(json.dumps(record) + "\n" for record in committed))
        temporary.replace(log)


def train(config, routes, output, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and not resume:
        raise FileExistsError(f"Use --resume or a new output directory: {output}")
    torch.set_num_threads(config["torch_threads"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = torch.device(config["device"])
    scenario_hashes = {str(route): hashlib.sha256(route.read_bytes()).hexdigest() for route in routes}
    network_hash = hashlib.sha256((ROOT / config["network_file"]).read_bytes()).hexdigest()
    payload = None
    if resume:
        payload = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
        if payload["config"] != config or payload["routes"] != [str(route) for route in routes]:
            raise ValueError("Resume configuration/scenarios differ from checkpoint")
        if payload.get("scenario_hashes", scenario_hashes) != scenario_hashes or payload.get("network_sha256", network_hash) != network_hash:
            raise ValueError("Resume scenario bytes differ from checkpoint")
    model = None
    start = payload["episode"] + 1 if payload else 0
    reconcile_log(output, start - 1)
    for episode in range(start, config["episodes"]):
        started = time.monotonic()
        route = routes[episode % len(routes)]
        env = parallel_env(net_file=str(ROOT / config["network_file"]), route_file=str(route),
                           num_seconds=config["episode_seconds"], delta_time=config["delta_time"],
                           yellow_time=config["yellow_time"], min_green=config["min_green"],
                           max_green=config["max_green"], reward_fn="queue", sumo_seed=seed + episode,
                           sumo_warnings=False, additional_sumo_cmd="--no-step-log true --duration-log.disable true")
        try:
            observations, _ = env.reset(seed=seed + episode)
            agent_ids = sorted(observations)
            dimensions = {(env.observation_space(agent).shape[0], env.action_space(agent).n) for agent in agent_ids}
            if len(dimensions) != 1:
                raise ValueError("Shared policy requires homogeneous agent spaces")
            observation_size, action_size = dimensions.pop()
            spec = dict(observation_size=observation_size, action_size=action_size, agents=len(agent_ids),
                        hidden_size=config["hidden_size"], centralized=config["algorithm"] == "mappo",
                        actor_observation=config.get("actor_observation", "local"))
            if model is None:
                model = ActorCritic(**spec).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                if payload:
                    model.load_state_dict(payload["model"])
                    optimizer.load_state_dict(payload["optimizer"])
                    torch.set_rng_state(payload["torch_rng"].cpu())
                    if device.type == "cuda":
                        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
            buffer = {name: [] for name in ["observations", "actions", "log_probs", "values", "rewards"]}
            while env.agents:
                inputs = torch.as_tensor(np.stack([observations[agent] for agent in agent_ids]), device=device)
                with torch.no_grad():
                    distribution = Categorical(logits=model.logits(inputs))
                    actions = distribution.sample()
                    values = model.values(inputs)
                next_observations, rewards, terminated, truncated, _ = env.step(dict(zip(agent_ids, actions.cpu().tolist())))
                reward = torch.tensor([rewards[agent] for agent in agent_ids], dtype=torch.float32, device=device)
                if config["reward"] == "network-queue":
                    reward = reward.mean().expand_as(reward)
                reward = reward * config["reward_scale"]
                for name, value in zip(buffer, [inputs, actions, distribution.log_prob(actions), values, reward]):
                    buffer[name].append(value.detach())
                observations = next_observations
                if all(terminated.values()) or all(truncated.values()):
                    break
            with torch.no_grad():
                final_inputs = torch.as_tensor(np.stack([observations[agent] for agent in agent_ids]), device=device)
                bootstrap = model.values(final_inputs)
            loss = update(model, optimizer, *(torch.stack(buffer[key]) for key in
                          ["observations", "actions", "log_probs", "rewards", "values"]), bootstrap, config)
        finally:
            env.close()
        record = {"episode": episode, "route": str(route), "loss": loss,
                  "mean_reward": float(torch.stack(buffer["rewards"]).mean()),
                  "utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.monotonic() - started,
                  **model.last_update_metrics}
        with (output / "train.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_spec": spec,
                 "config": config, "agent_ids": agent_ids, "episode": episode,
                 "routes": [str(route) for route in routes], "torch_rng": torch.get_rng_state(),
                 "scenario_hashes": scenario_hashes, "network_sha256": network_hash,
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "packages": {name: importlib.metadata.version(name) for name in ["torch", "sumo-rl", "numpy"]}}
        torch.save(state, output / "checkpoint.tmp")
        (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
        if episode + 1 in config.get("checkpoint_episodes", []):
            snapshot = output / f"checkpoint_{episode + 1}.pt"
            torch.save(state, snapshot.with_suffix(".tmp"))
            snapshot.with_suffix(".tmp").replace(snapshot)
        (output / "config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps(record), flush=True)
    return output / "checkpoint.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--algorithm", choices=["ippo", "mappo"])
    parser.add_argument("--routes", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.algorithm:
        config["algorithm"] = args.algorithm
    train(config, [route.resolve() for route in args.routes], args.output, args.resume)


if __name__ == "__main__":
    main()
