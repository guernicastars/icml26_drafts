"""
Iterated matrix game environments for meta-learning in MARL.

Follows Kim et al. (2021) experimental protocol:
- State = last joint action (one-hot encoded)
- IPD: 2 actions x 2 agents -> 5 states (initial + 4 joint)
- RPS: 3 actions x 2 agents -> 10 states (initial + 9 joint)
- Peer personas sampled from tabular policy populations
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Trajectory:
    observations: torch.Tensor   # (T, obs_dim)
    actions: torch.Tensor        # (T,) int
    rewards: torch.Tensor        # (T,)
    log_probs: torch.Tensor      # (T,)
    dones: torch.Tensor          # (T,)
    values: torch.Tensor         # (T,) from value network, optional


class IteratedMatrixGame:
    def __init__(self, R1: np.ndarray, R2: np.ndarray, gamma: float,
                 name: str = ""):
        self.R1 = R1
        self.R2 = R2
        self.n_actions_1 = R1.shape[0]
        self.n_actions_2 = R1.shape[1]
        self.n_joint = self.n_actions_1 * self.n_actions_2
        self.n_states = 1 + self.n_joint
        self.obs_dim = self.n_states
        self.gamma = gamma
        self.name = name

    def initial_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = 1.0
        return obs

    def step(self, a1: int, a2: int, prev_obs: np.ndarray):
        r1 = self.R1[a1, a2]
        r2 = self.R2[a1, a2]
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        state_idx = 1 + a1 * self.n_actions_2 + a2
        obs[state_idx] = 1.0
        return obs, r1, r2

    def collect_trajectory(self, policy_1, policy_2, horizon: int,
                           device: torch.device) -> tuple:
        """Collect a single trajectory from two policies.

        policy_1, policy_2: callable(obs_tensor, hidden) -> (action_probs, hidden)
            For LSTM policies. For tabular peers, use TabularPeerPolicy wrapper.

        Returns (traj_1, traj_2) as Trajectory objects.
        """
        obs = self.initial_obs()

        obs_list_1, act_list_1, rew_list_1, lp_list_1 = [], [], [], []
        obs_list_2, act_list_2, rew_list_2, lp_list_2 = [], [], [], []

        h1 = policy_1.init_hidden(device)
        h2 = policy_2.init_hidden(device)

        for _ in range(horizon):
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)

            probs_1, h1 = policy_1(obs_t, h1)
            probs_2, h2 = policy_2(obs_t, h2)

            dist_1 = torch.distributions.Categorical(probs_1.squeeze(0))
            dist_2 = torch.distributions.Categorical(probs_2.squeeze(0))

            a1 = dist_1.sample()
            a2 = dist_2.sample()

            lp1 = dist_1.log_prob(a1)
            lp2 = dist_2.log_prob(a2)

            obs_next, r1, r2 = self.step(a1.item(), a2.item(), obs)

            obs_list_1.append(obs_t.squeeze(0))
            act_list_1.append(a1)
            rew_list_1.append(r1)
            lp_list_1.append(lp1)

            obs_list_2.append(obs_t.squeeze(0))
            act_list_2.append(a2)
            rew_list_2.append(r2)
            lp_list_2.append(lp2)

            obs = obs_next

        traj_1 = Trajectory(
            observations=torch.stack(obs_list_1),
            actions=torch.stack(act_list_1),
            rewards=torch.tensor(rew_list_1, device=device, dtype=torch.float32),
            log_probs=torch.stack(lp_list_1),
            dones=torch.zeros(horizon, device=device),
            values=torch.zeros(horizon, device=device),
        )
        traj_2 = Trajectory(
            observations=torch.stack(obs_list_2),
            actions=torch.stack(act_list_2),
            rewards=torch.tensor(rew_list_2, device=device, dtype=torch.float32),
            log_probs=torch.stack(lp_list_2),
            dones=torch.zeros(horizon, device=device),
            values=torch.zeros(horizon, device=device),
        )
        return traj_1, traj_2

    def collect_batch(self, policy_1, policy_2, horizon: int,
                      batch_size: int, device: torch.device) -> tuple:
        """Collect K trajectories. Returns lists of (traj_1, traj_2)."""
        batch_1, batch_2 = [], []
        for _ in range(batch_size):
            t1, t2 = self.collect_trajectory(policy_1, policy_2, horizon, device)
            batch_1.append(t1)
            batch_2.append(t2)
        return batch_1, batch_2


class TabularPeerPolicy:
    """Wraps a tabular policy (numpy array of probabilities) as a callable
    compatible with collect_trajectory.

    probs: np.ndarray of shape (n_states, n_actions)
    """

    def __init__(self, probs: np.ndarray):
        self.probs_np = probs

    def __call__(self, obs: torch.Tensor, hidden):
        state_idx = obs.squeeze(0).argmax().item()
        p = torch.tensor(self.probs_np[state_idx], dtype=torch.float32,
                         device=obs.device).unsqueeze(0)
        return p, hidden

    def init_hidden(self, device: torch.device):
        return None


def generate_ipd_personas(n_total: int, rng: np.random.Generator) -> list:
    """Generate IPD peer personas: half cooperating, half defecting.

    Each persona is a (5, 2) probability array over 5 states and 2 actions.
    Action 0 = Cooperate, Action 1 = Defect.
    """
    n_states = 5
    personas = []
    n_half = n_total // 2

    for _ in range(n_half):
        p_coop = rng.uniform(0.5, 1.0, size=n_states)
        probs = np.stack([p_coop, 1.0 - p_coop], axis=1).astype(np.float32)
        personas.append(probs)

    for _ in range(n_total - n_half):
        p_coop = rng.uniform(0.0, 0.5, size=n_states)
        probs = np.stack([p_coop, 1.0 - p_coop], axis=1).astype(np.float32)
        personas.append(probs)

    rng.shuffle(personas)
    return personas


def generate_rps_personas(n_total: int, rng: np.random.Generator) -> list:
    """Generate RPS peer personas: rock/paper/scissors biased.

    Each persona is a (10, 3) probability array.
    """
    n_states = 10
    personas = []
    n_per_type = n_total // 3

    for bias_action in range(3):
        for _ in range(n_per_type if bias_action < 2 else n_total - 2 * n_per_type):
            probs = np.zeros((n_states, 3), dtype=np.float32)
            for s in range(n_states):
                p_bias = rng.uniform(1.0 / 3.0, 1.0)
                remaining = 1.0 - p_bias
                other = remaining / 2.0
                p = np.full(3, other)
                p[bias_action] = p_bias
                probs[s] = p
            personas.append(probs)

    rng.shuffle(personas)
    return personas


def split_personas(personas: list, n_train: int, n_val: int, n_test: int):
    assert len(personas) >= n_train + n_val + n_test
    return (personas[:n_train],
            personas[n_train:n_train + n_val],
            personas[n_train + n_val:n_train + n_val + n_test])


def make_ipd() -> IteratedMatrixGame:
    R1 = np.array([[0.5, -1.5],
                    [1.5, -0.5]], dtype=np.float32)
    R2 = np.array([[0.5, 1.5],
                    [-1.5, -0.5]], dtype=np.float32)
    return IteratedMatrixGame(R1, R2, gamma=0.96, name="IPD")


def make_rps() -> IteratedMatrixGame:
    R1 = np.array([[0.0, -1.0, 1.0],
                    [1.0, 0.0, -1.0],
                    [-1.0, 1.0, 0.0]], dtype=np.float32)
    R2 = -R1
    return IteratedMatrixGame(R1, R2, gamma=0.90, name="RPS")
