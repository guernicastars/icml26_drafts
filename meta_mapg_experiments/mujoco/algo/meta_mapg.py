import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """Gaussian Policy Network."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

def compute_meta_mapg_loss(pi_own, pi_peer, obs_own, act_own, ret_own, obs_peer, act_peer, ret_peer, lr_inner=0.01):
    """
    Exact Meta-MAPG Gradient surrogate loss.
    own_learning + peer_learning (differentiating through peer's update).
    """
    # 1. Own-learning Term
    dist_own = pi_own(obs_own)
    log_prob_own = dist_own.log_prob(act_own).sum(-1)
    loss_own = -(log_prob_own * ret_own).mean()

    # 2. Peer-learning Term
    dist_peer = pi_peer(obs_peer)
    log_prob_peer = dist_peer.log_prob(act_peer).sum(-1)
    loss_peer = -(log_prob_peer * ret_peer).mean()
    
    # Compute peer gradient
    grad_peer = torch.autograd.grad(loss_peer, pi_peer.parameters(), create_graph=True)
    
    # Meta-return surrogate (DiCE / LOLA-like inner product)
    peer_term = 0.0
    for p_peer, g_peer in zip(pi_peer.parameters(), grad_peer):
        peer_term += (g_peer * p_peer).sum()
        
    # Scale peer term by inner learning rate
    total_loss = loss_own + lr_inner * peer_term
    return total_loss
