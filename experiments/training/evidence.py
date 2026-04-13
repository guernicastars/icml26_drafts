"""
Keynesian evidence weight tracker for EW-PG.

Tracks running variance of gradient estimates per agent.
Evidence weight w_i = V_min / V_i, ensuring low-evidence agents
take smaller steps (the Keynesian fragility principle).

Variance improvement factor: HM(V)/AM(V) <= 1 by AM-HM inequality.
"""

import torch


class EvidenceTracker:
    def __init__(self, n_agents: int, alpha: float = 0.99,
                 w_min: float = 0.01):
        self.n_agents = n_agents
        self.alpha = alpha
        self.w_min = w_min
        self.running_var = torch.ones(n_agents)
        self.running_mean = torch.zeros(n_agents)

    def update(self, grad_norms: torch.Tensor):
        """Update running variance estimate from gradient norms.

        grad_norms: (n_agents,) norm of each agent's gradient estimate
        """
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * grad_norms
        self.running_var = (self.alpha * self.running_var
                           + (1 - self.alpha) * (grad_norms - self.running_mean) ** 2)

    def weights(self) -> torch.Tensor:
        """Compute evidence weights w_i = V_min / V_i.

        Returns: (n_agents,) weights in [w_min, 1.0]
        """
        v = self.running_var.clamp(min=1e-8)
        v_min = v.min()
        w = v_min / v
        return w.clamp(min=self.w_min)

    def variance_ratio(self) -> float:
        """Compute HM(V)/AM(V) — the theoretical improvement factor."""
        v = self.running_var.clamp(min=1e-8)
        am = v.mean()
        hm = self.n_agents / (1.0 / v).sum()
        return (hm / am).item()
