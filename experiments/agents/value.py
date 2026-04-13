"""
LSTM value network for GAE computation.

Separate network from policy (Kim et al. found this more stable).
"""

import torch
import torch.nn as nn


class LSTMValue(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, hidden: tuple) -> tuple:
        x = torch.relu(self.fc_in(obs))
        h, c = self.lstm(x, hidden)
        value = self.fc_out(h)
        return value.squeeze(-1), (h, c)

    def init_hidden(self, device: torch.device) -> tuple:
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        return (h, c)

    def evaluate_trajectory(self, obs_seq: torch.Tensor,
                            hidden: tuple) -> torch.Tensor:
        """Compute values for a trajectory.

        obs_seq: (T, obs_dim)
        Returns: (T,) values
        """
        values = []
        h, c = hidden
        for t in range(obs_seq.shape[0]):
            v, (h, c) = self.forward(obs_seq[t:t+1], (h, c))
            values.append(v.squeeze(0))
        return torch.stack(values)
