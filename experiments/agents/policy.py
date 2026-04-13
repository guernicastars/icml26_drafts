"""
LSTM policy network for iterated matrix games.

Architecture matches Kim et al. (2021) Appendix D:
  FC(obs_dim, 64) -> LSTM(64, 64) -> FC(64, n_actions) -> softmax
"""

import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_actions)

    def forward(self, obs: torch.Tensor, hidden: tuple) -> tuple:
        """
        obs: (batch, obs_dim) or (1, obs_dim)
        hidden: (h, c) each (1, hidden_size)
        Returns: (action_probs, new_hidden)
        """
        x = torch.relu(self.fc_in(obs))
        h, c = self.lstm(x, hidden)
        logits = self.fc_out(h)
        probs = torch.softmax(logits, dim=-1)
        return probs, (h, c)

    def init_hidden(self, device: torch.device) -> tuple:
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        return (h, c)

    def get_log_probs(self, obs_seq: torch.Tensor, actions: torch.Tensor,
                      hidden: tuple) -> torch.Tensor:
        """Compute log-probs for a sequence of observations and actions.

        obs_seq: (T, obs_dim)
        actions: (T,)
        Returns: (T,) log probabilities
        """
        log_probs = []
        h, c = hidden
        for t in range(obs_seq.shape[0]):
            probs, (h, c) = self.forward(obs_seq[t:t+1], (h, c))
            dist = torch.distributions.Categorical(probs.squeeze(0))
            log_probs.append(dist.log_prob(actions[t]))
        return torch.stack(log_probs)
