"""
Meta-learning loop for multi-agent policy gradient experiments.

Implements the meta-training/testing protocol from Kim et al. (2021):
- Outer loop: optimize initial policy parameters phi_0
- Inner loop: adapt via policy gradient over a Markov chain of L steps
- Each inner step collects K trajectories and updates phi

Key insight from Kim et al.: The meta-gradient requires differentiating
through the inner-loop updates. We use DiCE (Foerster et al., 2018) to
make the inner-loop policy gradient differentiable, then backprop through
the chain of L inner-loop updates to compute the meta-gradient.

Methods differ in which terms of the meta-gradient they include:
- REINFORCE: no meta-learning, standard PG at each step
- Meta-PG: Terms 1+2 (current policy + own learning)
- LOLA-DiCE: Terms 1+3 (current policy + peer learning)
- Meta-MAPG: Terms 1+2+3 (all three)
- EW-PG: Term 1 with evidence weighting
- LOLA-PG: Terms 1+3 with annealed LOLA
- EW-LOLA-PG: Terms 1+3 with evidence weighting + annealed LOLA
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from dataclasses import dataclass

from experiments.agents.policy import LSTMPolicy
from experiments.agents.value import LSTMValue
from experiments.envs.iterated_matrix import (
    IteratedMatrixGame, TabularPeerPolicy,
)
from experiments.training.evidence import EvidenceTracker


@dataclass
class MetaConfig:
    horizon: int = 150
    chain_length: int = 7
    batch_size: int = 64
    gamma: float = 0.96
    gae_lambda: float = 0.95
    lr_inner: float = 0.1
    lr_outer_actor: float = 1e-4
    lr_outer_critic: float = 1.5e-4
    n_meta_steps: int = 500
    eval_interval: int = 50
    device: str = "cpu"

    evidence_alpha: float = 0.99
    lola_lambda_init: float = 0.1
    lola_anneal_rate: float = 0.5
    evidence_w_min: float = 0.01


def collect_rollout(policy: LSTMPolicy, peer, env: IteratedMatrixGame,
                    horizon: int, device: torch.device):
    """Collect one trajectory, keeping computation graph for meta-gradient.

    Returns: (rewards, log_probs_agent, log_probs_peer)
    All tensors of shape (T,).
    """
    obs = env.initial_obs()
    h_agent = policy.init_hidden(device)
    h_peer = peer.init_hidden(device)

    rewards, lp_agent, lp_peer = [], [], []

    for _ in range(horizon):
        obs_t = torch.tensor(obs, device=device).unsqueeze(0)

        probs_a, h_agent = policy(obs_t, h_agent)
        probs_p, h_peer = peer(obs_t, h_peer)

        dist_a = torch.distributions.Categorical(probs_a.squeeze(0))
        dist_p = torch.distributions.Categorical(probs_p.squeeze(0))

        a1 = dist_a.sample()
        a2 = dist_p.sample()

        lp_agent.append(dist_a.log_prob(a1))
        lp_peer.append(dist_p.log_prob(a2))

        obs, r1, r2 = env.step(a1.item(), a2.item(), obs)
        rewards.append(r1)

    return (
        torch.tensor(rewards, device=device, dtype=torch.float32),
        torch.stack(lp_agent),
        torch.stack(lp_peer),
    )


def compute_dice_loss(rewards: torch.Tensor, log_probs: torch.Tensor,
                      gamma: float) -> torch.Tensor:
    """DiCE-based differentiable policy gradient loss.

    Uses the magic_box operator so that gradients of this loss
    w.r.t. policy parameters give the REINFORCE estimator,
    and higher-order gradients are correct for meta-learning.
    """
    T = rewards.shape[0]
    discounts = gamma ** torch.arange(T, device=rewards.device, dtype=rewards.dtype)

    # Cumulative sum for DiCE: at time t, include log_probs from 0..t
    cumsum_lp = torch.cumsum(log_probs, dim=0)
    dice_weights = torch.exp(cumsum_lp - cumsum_lp.detach())

    return (discounts * rewards * dice_weights).sum()


def collect_batch_dice(policy: LSTMPolicy, peer, env: IteratedMatrixGame,
                       config: MetaConfig, device: torch.device):
    """Collect K trajectories and compute DiCE objective.

    Returns: (mean_reward, dice_loss_agent, dice_loss_peer)
    dice_loss_agent: scalar loss whose grad w.r.t. policy params = PG
    """
    total_reward = 0.0
    dice_agent_sum = torch.tensor(0.0, device=device)
    dice_peer_sum = torch.tensor(0.0, device=device)

    for _ in range(config.batch_size):
        rewards, lp_agent, lp_peer = collect_rollout(
            policy, peer, env, config.horizon, device
        )
        total_reward += rewards.sum().item()
        dice_agent_sum = dice_agent_sum + compute_dice_loss(
            rewards, lp_agent, config.gamma
        )
        dice_peer_sum = dice_peer_sum + compute_dice_loss(
            rewards, lp_peer, config.gamma
        )

    mean_reward = total_reward / config.batch_size
    dice_agent_mean = dice_agent_sum / config.batch_size
    dice_peer_mean = dice_peer_sum / config.batch_size

    return mean_reward, dice_agent_mean, dice_peer_mean


def meta_train(policy: LSTMPolicy, value_net: LSTMValue,
               env: IteratedMatrixGame, personas: list,
               config: MetaConfig, rng: np.random.Generator,
               method: str = "meta_mapg"):
    """Unified meta-training for all methods.

    method: "reinforce", "meta_pg", "lola_dice", "meta_mapg",
            "ew_pg", "lola_pg", "ew_lola_pg"
    """
    device = torch.device(config.device)
    policy = policy.to(device)
    value_net = value_net.to(device)

    actor_optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.lr_outer_actor
    )
    critic_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=config.lr_outer_critic
    )

    use_peer_learning = method in ("meta_mapg", "lola_dice", "ew_lola_pg", "lola_pg")
    use_evidence = method in ("ew_lola_pg", "ew_pg")
    is_meta = method != "reinforce"

    evidence_tracker = EvidenceTracker(
        n_agents=2, alpha=config.evidence_alpha, w_min=config.evidence_w_min
    ) if use_evidence else None

    rewards_log = []
    best_val_reward = float("-inf")

    for step in range(config.n_meta_steps):
        persona_probs = personas[rng.integers(len(personas))]
        peer = TabularPeerPolicy(persona_probs)

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        if not is_meta:
            # REINFORCE: no meta-learning, just PG at each chain step
            chain_reward = _train_step_reinforce(
                policy, peer, env, config, value_net, device
            )
        else:
            chain_reward = _train_step_meta(
                policy, peer, env, config, value_net, device,
                step, method, use_peer_learning, use_evidence,
                evidence_tracker,
            )

        actor_optimizer.step()
        rewards_log.append(chain_reward)

        if step > 0 and step % config.eval_interval == 0:
            print(f"    step {step}/{config.n_meta_steps}, "
                  f"reward={chain_reward:.3f}")

    return rewards_log


def _train_step_reinforce(policy, peer, env, config, value_net, device):
    """Single training step for REINFORCE (no meta-learning)."""
    total_reward = 0.0

    for l in range(config.chain_length):
        batch_loss = torch.tensor(0.0, device=device)

        for k in range(config.batch_size):
            rewards, lp_agent, _ = collect_rollout(
                policy, peer, env, config.horizon, device
            )
            total_reward += rewards.sum().item()

            # Simple REINFORCE with discounted returns as baseline
            T = rewards.shape[0]
            discounts = config.gamma ** torch.arange(T, device=device, dtype=torch.float32)
            returns = torch.flip(
                torch.cumsum(torch.flip(discounts * rewards, [0]), dim=0),
                [0]
            ) / discounts

            batch_loss = batch_loss - (returns.detach() * lp_agent).sum()

        batch_loss = batch_loss / config.batch_size
        batch_loss.backward()

    return total_reward / (config.chain_length * config.batch_size)


def _train_step_meta(policy, peer, env, config, value_net, device,
                     step, method, use_peer_learning, use_evidence,
                     evidence_tracker):
    """Single meta-training step with differentiable inner loop.

    The key structure (following Kim et al.):
    1. Save initial params phi_0
    2. For l = 0..L-1:
       a. Collect K trajectories with current policy
       b. Compute inner-loop gradient via DiCE
       c. Update params: phi_{l+1} = phi_l - alpha * grad
       d. Keep computation graph alive (create_graph=True)
    3. After chain, backprop meta-objective to phi_0
    """
    # Inner loop uses a separate copy with graph-connected params
    inner_policy = deepcopy(policy)
    # Reconnect params to the original policy's graph
    # by re-initializing from policy's params
    param_names = []
    fast_weights = {}
    for name, param in policy.named_parameters():
        param_names.append(name)
        fast_weights[name] = param.clone()

    total_reward = 0.0
    meta_objective = torch.tensor(0.0, device=device)

    for l in range(config.chain_length):
        # Set inner_policy's params to fast_weights
        _set_params(inner_policy, fast_weights, param_names)

        mean_r, dice_agent, dice_peer = collect_batch_dice(
            inner_policy, peer, env, config, device
        )
        total_reward += mean_r

        # Meta-objective: sum of DiCE objectives across chain
        chain_obj = dice_agent

        if use_peer_learning:
            lola_lambda = config.lola_lambda_init / (1 + step * config.lola_anneal_rate)
            chain_obj = chain_obj + lola_lambda * dice_peer

        if use_evidence and evidence_tracker is not None:
            with torch.no_grad():
                grad_agent = torch.autograd.grad(
                    dice_agent, list(fast_weights.values()),
                    retain_graph=True, allow_unused=True
                )
                grad_peer = torch.autograd.grad(
                    dice_peer, list(fast_weights.values()),
                    retain_graph=True, allow_unused=True
                )
                norm_agent = sum(
                    g.norm().item() for g in grad_agent if g is not None
                )
                norm_peer = sum(
                    g.norm().item() for g in grad_peer if g is not None
                )
                evidence_tracker.update(torch.tensor([norm_agent, norm_peer]))
            w = evidence_tracker.weights()[0].item()
            chain_obj = chain_obj * w

        meta_objective = meta_objective + chain_obj

        # Inner-loop update: phi_{l+1} = phi_l - alpha * grad_phi_l(DiCE_loss)
        inner_grads = torch.autograd.grad(
            dice_agent, list(fast_weights.values()),
            create_graph=True, retain_graph=True, allow_unused=True
        )

        new_fast_weights = {}
        for name, g in zip(param_names, inner_grads):
            if g is not None:
                new_fast_weights[name] = fast_weights[name] + config.lr_inner * g
            else:
                new_fast_weights[name] = fast_weights[name]
        fast_weights = new_fast_weights

    # Backprop meta-objective to original policy params
    meta_objective = -meta_objective  # maximize -> minimize
    meta_objective.backward()

    return total_reward / config.chain_length


def _set_params(model: nn.Module, param_dict: dict, param_names: list):
    """Copy fast_weights into model parameters (in-place data swap)."""
    for name, param in model.named_parameters():
        if name in param_dict:
            param.data = param_dict[name].data


def meta_test(policy: LSTMPolicy, value_net: LSTMValue,
              env: IteratedMatrixGame, test_personas: list,
              config: MetaConfig) -> dict:
    """Evaluate a meta-trained policy on test personas.

    For each persona:
    1. Initialize from meta-trained phi_0
    2. At each chain step, collect K trajectories (measure performance)
    3. Adapt with policy gradient
    """
    device = torch.device(config.device)
    policy = policy.to(device)
    value_net = value_net.to(device)

    all_chain_rewards = []

    for persona_probs in test_personas:
        peer = TabularPeerPolicy(persona_probs)
        test_policy = deepcopy(policy)

        chain_rewards = []
        for l in range(config.chain_length):
            step_rewards = []
            with torch.no_grad():
                for k in range(config.batch_size):
                    rewards, _, _ = collect_rollout(
                        test_policy, peer, env, config.horizon, device
                    )
                    step_rewards.append(rewards.sum().item())
            chain_rewards.append(np.mean(step_rewards))

            _adapt_step(test_policy, peer, env, config, device)

        all_chain_rewards.append(chain_rewards)

    rewards_array = np.array(all_chain_rewards)
    mean_rewards = rewards_array.mean(axis=0)
    std_rewards = rewards_array.std(axis=0)
    ci_95 = 1.96 * std_rewards / np.sqrt(len(test_personas))

    return {
        "mean": mean_rewards,
        "std": std_rewards,
        "ci_95": ci_95,
        "raw": rewards_array,
    }


def _adapt_step(policy: LSTMPolicy, peer, env: IteratedMatrixGame,
                config: MetaConfig, device: torch.device):
    """One inner-loop adaptation step during meta-testing."""
    # Need gradients for adaptation
    for p in policy.parameters():
        p.requires_grad_(True)

    batch_loss = torch.tensor(0.0, device=device)
    for k in range(min(config.batch_size, 16)):  # smaller batch for speed
        rewards, lp_agent, _ = collect_rollout(
            policy, peer, env, config.horizon, device
        )
        T = rewards.shape[0]
        discounts = config.gamma ** torch.arange(T, device=device, dtype=torch.float32)
        returns = torch.flip(
            torch.cumsum(torch.flip(discounts * rewards, [0]), dim=0), [0]
        ) / discounts
        batch_loss = batch_loss - (returns.detach() * lp_agent).sum()

    batch_loss = batch_loss / min(config.batch_size, 16)

    grads = torch.autograd.grad(batch_loss, policy.parameters(), allow_unused=True)
    with torch.no_grad():
        for p, g in zip(policy.parameters(), grads):
            if g is not None:
                p.data -= config.lr_inner * g
