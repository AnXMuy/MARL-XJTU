from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from traffic_signal_marl.baselines.controllers import SignalController


def mlp(input_size, hidden_size, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size),
                         nn.Tanh(), nn.Linear(hidden_size, output_size))


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=64, centralized=False,
                 actor_observation="local"):
        super().__init__()
        self.centralized = centralized
        self.agents = agents
        if actor_observation not in {"local", "phase-only"}:
            raise ValueError(f"Unknown actor observation: {actor_observation}")
        self.actor_observation = actor_observation
        self.phase_features = int(action_size) + 1
        self.actor = mlp(observation_size, hidden_size, action_size)
        critic_size = observation_size * agents + agents if centralized else observation_size
        self.critic = mlp(critic_size, hidden_size, 1)

    def logits(self, observations):
        if self.actor_observation == "phase-only":
            observations = observations.clone()
            observations[..., self.phase_features:] = 0
        return self.actor(observations)

    def critic_inputs(self, observations):
        if not self.centralized:
            return observations
        prefix = observations.shape[:-2]
        global_state = observations.flatten(-2).unsqueeze(-2).expand(*prefix, self.agents, -1)
        ids = torch.eye(self.agents, device=observations.device).expand(*prefix, self.agents, self.agents)
        return torch.cat([global_state, ids], dim=-1)

    def values(self, observations):
        return self.critic(self.critic_inputs(observations)).squeeze(-1)


def gae(rewards, values, bootstrap, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        carry = delta + gamma * gae_lambda * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


def update(model, optimizer, observations, actions, old_log_probs, rewards, values, bootstrap, config):
    advantages, returns = gae(rewards, values, bootstrap, config["gamma"], config["gae_lambda"])
    inputs = model.critic_inputs(observations).flatten(0, 1)
    observations = observations.flatten(0, 1)
    actions = actions.flatten()
    old_log_probs = old_log_probs.flatten()
    returns = returns.flatten()
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    losses = []
    diagnostics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl")}
    for _ in range(config["epochs"]):
        permutation = torch.randperm(len(actions), device=actions.device)
        for indices in permutation.split(config["minibatch_size"]):
            distribution = Categorical(logits=model.logits(observations[indices]))
            ratios = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
            unclipped = ratios * advantages[indices]
            clipped = ratios.clamp(1 - config["clip_epsilon"], 1 + config["clip_epsilon"]) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (model.critic(inputs[indices]).squeeze(-1) - returns[indices]).square().mean()
            loss = policy_loss + config["value_coefficient"] * value_loss - config["entropy_coefficient"] * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                for key, value in zip(diagnostics, (policy_loss, value_loss, distribution.entropy().mean(),
                        ((ratios - 1).abs() > config["clip_epsilon"]).float().mean(),
                        ((ratios - 1) - ratios.log()).mean())):
                    diagnostics[key].append(float(value))
    model.last_update_metrics = {key: float(np.mean(values)) for key, values in diagnostics.items()}
    variance = returns.var(unbiased=False)
    model.last_update_metrics["explained_variance_pre_update"] = float(
        1 - (returns - values.flatten()).var(unbiased=False) / variance) if float(variance) > 0 else None
    return float(np.mean(losses))


class PPOController(SignalController):
    def __init__(self, checkpoint, device="cpu"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.spec = payload["model_spec"]
        self.model = ActorCritic(**self.spec).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = device
        self.name = payload["config"]["algorithm"]
        self.agent_ids = payload["agent_ids"]

    def actions(self, observations, elapsed_seconds, parallel_env):
        with torch.no_grad():
            inputs = torch.as_tensor(np.stack([observations[agent] for agent in self.agent_ids]), device=self.device)
            selected = self.model.logits(inputs).argmax(-1).cpu().tolist()
        return dict(zip(self.agent_ids, selected))
