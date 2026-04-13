"""
Generalized Advantage Estimation (Schulman et al., 2016).
"""

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float, lam: float) -> tuple:
    """Compute GAE advantages and returns.

    rewards: (T,)
    values: (T,) — value estimates from critic
    gamma: discount factor
    lam: GAE lambda

    Returns: (advantages, returns) each (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else torch.tensor(0.0, device=rewards.device)
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
