"""
Differentiable Monte-Carlo Estimator (DiCE) from Foerster et al. (2018).

The magic_box operator creates a surrogate that evaluates to 1 in forward
pass but has the correct gradient for backprop through stochastic nodes.
This enables computing the peer learning gradient via autograd.
"""

import torch


def magic_box(log_probs: torch.Tensor) -> torch.Tensor:
    """DiCE magic_box operator.

    Forward: returns exp(tau - tau.detach()) = 1
    Backward: has gradient equal to the REINFORCE estimator

    log_probs: cumulative sum of log-probabilities up to current step
    """
    tau = log_probs.sum()
    return torch.exp(tau - tau.detach())


def dice_objective(rewards: torch.Tensor, log_probs: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """Compute the DiCE objective for a single trajectory.

    rewards: (T,) rewards
    log_probs: (T,) log-probabilities of actions taken
    gamma: discount factor

    Returns scalar loss whose gradient is the policy gradient estimator.
    """
    T = rewards.shape[0]
    cumulative_lp = torch.cumsum(log_probs, dim=0)
    discounts = torch.tensor(
        [gamma ** t for t in range(T)],
        device=rewards.device, dtype=rewards.dtype,
    )
    deps = magic_box(cumulative_lp)
    return (discounts * rewards * deps).sum()
