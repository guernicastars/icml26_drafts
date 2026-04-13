"""
Minimal smoke test: run 1 seed, 5 meta-steps, K=4 batch, H=10 horizon.
Validates that the full pipeline runs without errors.

Usage:
    python -m experiments.smoke_test
"""

import torch
import numpy as np
from experiments.envs.iterated_matrix import (
    make_ipd, make_rps, generate_ipd_personas, split_personas,
    TabularPeerPolicy,
)
from experiments.agents.policy import LSTMPolicy
from experiments.agents.value import LSTMValue
from experiments.training.meta_learner import MetaConfig, meta_train, meta_test
from experiments.training.evidence import EvidenceTracker
from experiments.training.gae import compute_gae
from experiments.training.dice import dice_objective as compute_dice_loss


def test_env():
    print("Testing environment...")
    env = make_ipd()
    assert env.n_states == 5
    assert env.n_actions_1 == 2
    assert env.obs_dim == 5

    obs = env.initial_obs()
    assert obs.shape == (5,)
    assert obs[0] == 1.0

    obs2, r1, r2 = env.step(0, 0, obs)
    assert r1 == 0.5 and r2 == 0.5  # CC

    obs3, r1, r2 = env.step(1, 0, obs2)
    assert r1 == 1.5 and r2 == -1.5  # DC
    print("  env OK")


def test_rps_env():
    print("Testing RPS environment...")
    env = make_rps()
    assert env.n_states == 10
    assert env.n_actions_1 == 3
    _, r1, r2 = env.step(0, 2, env.initial_obs())  # Rock beats Scissors
    assert r1 == 1.0 and r2 == -1.0
    print("  rps env OK")


def test_policy():
    print("Testing LSTM policy...")
    device = torch.device("cpu")
    policy = LSTMPolicy(obs_dim=5, n_actions=2)
    h = policy.init_hidden(device)
    obs = torch.zeros(1, 5)
    obs[0, 0] = 1.0
    probs, h_new = policy(obs, h)
    assert probs.shape == (1, 2)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    print("  policy OK")


def test_value():
    print("Testing LSTM value network...")
    device = torch.device("cpu")
    vnet = LSTMValue(obs_dim=5)
    h = vnet.init_hidden(device)
    obs = torch.zeros(1, 5)
    obs[0, 0] = 1.0
    val, h_new = vnet(obs, h)
    assert val.shape == (1,)
    print("  value net OK")


def test_gae():
    print("Testing GAE...")
    rewards = torch.tensor([1.0, 0.5, 0.2, 0.1])
    values = torch.tensor([0.8, 0.4, 0.15, 0.05])
    adv, ret = compute_gae(rewards, values, gamma=0.96, lam=0.95)
    assert adv.shape == (4,)
    assert ret.shape == (4,)
    print("  gae OK")


def test_dice():
    print("Testing DiCE loss...")
    rewards = torch.tensor([1.0, 0.5, 0.2])
    log_probs = torch.tensor([-0.5, -0.3, -0.7], requires_grad=True)
    loss = compute_dice_loss(rewards, log_probs, gamma=0.96)
    loss.backward()
    assert log_probs.grad is not None
    print("  dice OK")


def test_evidence():
    print("Testing evidence tracker...")
    et = EvidenceTracker(n_agents=2)
    et.update(torch.tensor([1.0, 5.0]))
    et.update(torch.tensor([1.0, 5.0]))
    w = et.weights()
    assert w.shape == (2,)
    assert w[0] > w[1]  # agent 0 has lower variance -> higher weight
    ratio = et.variance_ratio()
    assert ratio <= 1.0
    print(f"  evidence OK (HM/AM ratio = {ratio:.3f})")


def test_personas():
    print("Testing persona generation...")
    rng = np.random.default_rng(42)
    personas = generate_ipd_personas(10, rng)
    assert len(personas) == 10
    for p in personas:
        assert p.shape == (5, 2)
        assert np.allclose(p.sum(axis=1), 1.0)
    print("  personas OK")


def test_collect_rollout():
    print("Testing trajectory collection...")
    from experiments.training.meta_learner import collect_rollout
    env = make_ipd()
    device = torch.device("cpu")
    policy = LSTMPolicy(5, 2)
    rng = np.random.default_rng(42)
    personas = generate_ipd_personas(10, rng)
    peer = TabularPeerPolicy(personas[0])

    rewards, lp_agent, lp_peer = collect_rollout(
        policy, peer, env, horizon=10, device=device
    )
    assert rewards.shape == (10,)
    assert lp_agent.shape == (10,)
    assert lp_peer.shape == (10,)
    print("  rollout OK")


def test_meta_train_smoke():
    print("Testing meta-training (smoke, 3 steps)...")
    env = make_ipd()
    rng = np.random.default_rng(42)
    personas = generate_ipd_personas(20, rng)
    train_p, val_p, test_p = split_personas(personas, 10, 5, 5)

    config = MetaConfig(
        horizon=10, chain_length=2, batch_size=4,
        n_meta_steps=3, eval_interval=1, device="cpu",
        lr_inner=0.1, lr_outer_actor=1e-3, lr_outer_critic=1e-3,
    )

    for method in ["reinforce", "meta_mapg", "ew_lola_pg"]:
        torch.manual_seed(42)
        policy = LSTMPolicy(env.obs_dim, env.n_actions_1)
        value_net = LSTMValue(env.obs_dim)
        rewards = meta_train(policy, value_net, env, train_p, config,
                             np.random.default_rng(42), method=method)
        assert len(rewards) == 3
        print(f"  {method}: {[f'{r:.2f}' for r in rewards]}")

    print("  meta-train smoke OK")


def test_meta_test_smoke():
    print("Testing meta-testing (smoke)...")
    env = make_ipd()
    rng = np.random.default_rng(42)
    personas = generate_ipd_personas(20, rng)
    _, _, test_p = split_personas(personas, 10, 5, 5)

    config = MetaConfig(
        horizon=10, chain_length=2, batch_size=4,
        n_meta_steps=3, eval_interval=1, device="cpu",
    )

    policy = LSTMPolicy(env.obs_dim, env.n_actions_1)
    value_net = LSTMValue(env.obs_dim)
    results = meta_test(policy, value_net, env, test_p[:2], config)
    assert results["mean"].shape == (2,)
    assert results["ci_95"].shape == (2,)
    print(f"  test results: mean={results['mean']}")
    print("  meta-test smoke OK")


if __name__ == "__main__":
    test_env()
    test_rps_env()
    test_policy()
    test_value()
    test_gae()
    test_dice()
    test_evidence()
    test_personas()
    test_collect_rollout()
    test_meta_train_smoke()
    test_meta_test_smoke()
    print("\nAll smoke tests passed.")
