import torch

from traffic_signal_marl.agents.ppo import ActorCritic, gae


def test_gae_bootstraps_time_limit():
    rewards = torch.ones(2, 3)
    advantage, returns = gae(rewards, torch.zeros_like(rewards), torch.ones(3), 1.0, 1.0)
    assert torch.allclose(returns, torch.tensor([[3., 3., 3.], [2., 2., 2.]]))


def test_centralized_critic_and_local_actor():
    model = ActorCritic(5, 2, 4, centralized=True)
    observations = torch.randn(3, 4, 5)
    assert model.values(observations).shape == (3, 4)
    original = model.actor(observations[:, 0]).detach()
    observations[:, 1] += 100
    assert torch.equal(model.actor(observations[:, 0]), original)


def test_model_save_load(tmp_path):
    model = ActorCritic(5, 2, 4)
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = ActorCritic(5, 2, 4)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
