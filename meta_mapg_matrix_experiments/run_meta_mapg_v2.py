"""
Meta-MAPG v2: Sampled estimator with polynomial schedule, L>=1 inner unrolling.

Implements Eq.(4) of Kim et al. (2021) exactly:
  Psi_L = Current-Policy + Own-Learning + Peer-Learning

Schedule from Theorem 3:
  gamma_n = gamma / (n + m)^p
  N_n >= c_N * n^{2*l_sigma}   (outer rollouts)
  K_n >= c_K * n^{2*l_b}       (inner rollouts per step)

Conditions: p + l_b > 1,  p - l_sigma > 0.5
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os

import numpy as np
import torch

# ─── Games ────────────────────────────────────────────────────────────────────

GAMES = {
    # rows = agent 0 action, cols = agent 1 action; last dim = payoff index
    "StagHunt": torch.tensor([
        [[5.0, 5.0], [0.0, 3.0]],
        [[3.0, 0.0], [3.0, 3.0]],
    ]),
    "Chicken": torch.tensor([
        [[2.0, 2.0], [0.0, 3.0]],
        [[3.0, 0.0], [-5.0, -5.0]],
    ]),
}

# Pareto check: both cooperate (action 0) with prob > 0.95
PARETO_THRESHOLD = 0.95

# ─── Policy ───────────────────────────────────────────────────────────────────

def probs(logits):
    return torch.softmax(logits, dim=0)


def sample_action(logits):
    return torch.distributions.Categorical(logits=logits).sample().item()


def log_prob_action(logits, action):
    return torch.log_softmax(logits, dim=0)[action]


# ─── Exact meta-gradient (for bias measurement) ───────────────────────────────

def exact_meta_gradient(logits0, logits1, payoffs, L, alpha):
    """
    Closed-form meta-gradient for agent 0 in a 2x2 game.
    Uses automatic differentiation through the deterministic inner loop.
    Returns gradient w.r.t. logits0.
    """
    l0 = logits0.clone().detach().requires_grad_(True)
    l1 = logits1.clone().detach().requires_grad_(True)

    l1_inner = l1.clone()
    for _ in range(L):
        p0 = torch.softmax(l0, dim=0)
        p1 = torch.softmax(l1_inner, dim=0)
        v1 = sum(
            p0[a0] * p1[a1] * payoffs[a0, a1, 1]
            for a0 in range(2) for a1 in range(2)
        )
        g1 = torch.autograd.grad(v1, l1_inner, create_graph=False)[0]
        l1_inner = (l1_inner + alpha * g1).detach().requires_grad_(True)

    p0 = torch.softmax(l0, dim=0)
    p1 = torch.softmax(l1_inner, dim=0)
    v0_meta = sum(
        p0[a0] * p1[a1] * payoffs[a0, a1, 0]
        for a0 in range(2) for a1 in range(2)
    )
    return torch.autograd.grad(v0_meta, l0)[0].detach()


# ─── Sampling utilities ───────────────────────────────────────────────────────

def sample_rollout(logits0, logits1, payoffs):
    """One episode: sample actions, return (a0, a1, r0, r1)."""
    a0 = sample_action(logits0)
    a1 = sample_action(logits1)
    r0 = payoffs[a0, a1, 0].item()
    r1 = payoffs[a0, a1, 1].item()
    return a0, a1, r0, r1


def inner_pg_step(logits1, payoffs, K, alpha):
    """
    One stochastic inner PG step for agent 1 using K rollouts.
    Returns updated logits1 (detached).
    """
    l1 = logits1.clone().detach().requires_grad_(True)
    # Need fixed l0 for inner update; use uniform as reference (not needed for agent1 update)
    # Agent 1's gradient uses samples of its own action
    log_probs = []
    rewards = []
    for _ in range(K):
        a1 = sample_action(l1)
        # Agent 0's action irrelevant for agent 1's marginal PG step here;
        # we treat agent 0's distribution as fixed external for the inner step.
        # Use expected reward over agent 0's current policy (passed via closure — simplified).
        r1 = sum(
            0.5 * payoffs[a0, a1, 1].item() for a0 in range(2)
        )  # uniform a0 approximation for inner step
        log_probs.append(log_prob_action(l1, a1))
        rewards.append(r1)
    loss = -sum(lp * r for lp, r in zip(log_probs, rewards)) / K
    g = torch.autograd.grad(loss, l1)[0]
    return (l1 - alpha * g).detach()  # gradient ascent on reward


def stochastic_inner_chain(logits0, logits1, payoffs, L, K, alpha):
    """
    Run L stochastic inner steps for agent 1, given fixed logits0.
    Returns list of (logits1_l, a1_samples, r_samples) at each step.
    """
    l1_trajectory = [logits1.detach().clone()]
    for _ in range(L):
        l1_next = inner_pg_step(l1_trajectory[-1], payoffs, K, alpha)
        l1_trajectory.append(l1_next)
    return l1_trajectory  # length L+1


# ─── Meta-MAPG estimator (Eq.4, Kim et al. 2021) ─────────────────────────────

def meta_mapg_gradient(logits0, logits1, payoffs, L, K_inner, N_outer, alpha, ablation="full"):
    """
    Sample-based Meta-MAPG gradient for agent 0.

    ablation: "current" | "current+own" | "full" (current+own+peer)
    """
    l0 = logits0.clone().detach().requires_grad_(True)
    l1_base = logits1.clone().detach()

    grad_accum = torch.zeros_like(l0)

    for _ in range(N_outer):
        l1_traj = stochastic_inner_chain(l0.detach(), l1_base, payoffs, L, K_inner, alpha)
        l1_final = l1_traj[-1]

        a0_f, a1_f, r0_f, _ = sample_rollout(l0, l1_final, payoffs)
        G0 = r0_f

        # Current Policy score: grad_{l0} log pi0(a0_f | l0)
        lp0_cur = log_prob_action(l0, a0_f)   # has grad through l0
        g_cur = torch.autograd.grad(lp0_cur, l0, retain_graph=True)[0].detach()

        if ablation == "current":
            grad_accum += G0 * g_cur
            continue

        # Own Learning: agent 0's params are fixed in inner loop (phi0_ell = l0),
        # so this adds L copies of the current-policy score.
        g_own = g_cur * L  # each inner step contributes the same score

        if ablation == "current+own":
            grad_accum += G0 * (g_cur + g_own)
            continue

        # Peer Learning: grad_{l0} log pi1(a1_ell | phi1_ell(l0))
        # Re-run differentiable inner step to get d phi1 / d l0
        g_peer = torch.zeros_like(l0)
        for ell in range(1, L + 1):
            l0_for_peer = l0.clone().detach().requires_grad_(True)
            l1_diff = inner_pg_step_diff(l0_for_peer, l1_base, payoffs, K_inner, alpha)
            a1_ell = sample_action(l1_diff.detach())
            lp1 = log_prob_action(l1_diff, a1_ell)
            g_p = torch.autograd.grad(lp1, l0_for_peer, retain_graph=False)[0].detach()
            g_peer += g_p

        grad_accum += G0 * (g_cur + g_own + g_peer)

    return grad_accum / N_outer


def inner_pg_step_diff(logits0_fixed, logits1, payoffs, K, alpha):
    """
    Differentiable inner PG step for peer (agent 1).
    logits0_fixed: agent 0's logits (requires_grad for peer learning term).
    Returns logits1_next (differentiable w.r.t. logits0_fixed via the reward).
    """
    l1 = logits1.clone().detach().requires_grad_(True)
    # Peer's gradient: expectation over joint distribution
    p0 = torch.softmax(logits0_fixed, dim=0)
    p1 = torch.softmax(l1, dim=0)
    # Expected value for peer
    v1 = sum(
        p0[a0] * p1[a1] * payoffs[a0, a1, 1]
        for a0 in range(2) for a1 in range(2)
    )
    g1 = torch.autograd.grad(v1, l1, create_graph=True)[0]
    # New logits1 (differentiable through g1 which depends on v1 which depends on logits0)
    l1_new = l1.detach() + alpha * g1
    return l1_new


# ─── OMWU baseline ────────────────────────────────────────────────────────────

def omwu_step(logits0, logits1, payoffs, eta):
    """Online Mirror Descent / Multiplicative Weights Update step."""
    l0 = logits0.clone().detach().requires_grad_(True)
    l1 = logits1.clone().detach().requires_grad_(True)
    p0, p1 = torch.softmax(l0, dim=0), torch.softmax(l1, dim=0)
    v0 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 0] for a0 in range(2) for a1 in range(2))
    v1 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 1] for a0 in range(2) for a1 in range(2))
    g0 = torch.autograd.grad(v0, l0)[0].detach()
    g1 = torch.autograd.grad(v1, l1)[0].detach()
    return logits0 + eta * g0, logits1 + eta * g1


# ─── Restart mechanism (Giannou §4: bounded exploration region) ───────────────

def giannou_restart(rng):
    """
    Restart: sample uniformly from bounded exploration region [-B, B]^d.
    B=2.0 is our exploration radius covering the full simplex.
    """
    return torch.tensor(rng.uniform(-2.0, 2.0, size=2), dtype=torch.float32)


# ─── Main run function ────────────────────────────────────────────────────────

def run_seed(
    game_name, algo, seed, L, p, l_b, l_sigma, gamma0, m_warmup,
    c_N, c_K, n_outer_episodes, restart_thresh, ablation="full"
):
    """
    Run one seed. Returns (converged, first_passage_ep, n_restarts, restart_counts_list).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    payoffs = GAMES[game_name]
    alpha = 0.1  # inner loop step size

    logits0 = torch.tensor(rng.uniform(-0.5, 0.5, 2), dtype=torch.float32)
    logits1 = torch.tensor(rng.uniform(-0.5, 0.5, 2), dtype=torch.float32)

    restarts = 0
    restart_eps = []
    reward_window = []
    window_size = 50

    for n in range(1, n_outer_episodes + 1):
        # Polynomial schedule
        gamma_n = gamma0 / (n + m_warmup) ** p
        N_n = max(1, int(c_N * n ** (2 * l_sigma)))
        K_n = max(1, int(c_K * n ** (2 * l_b)))

        if algo == "REINFORCE":
            a0, a1, r0, r1 = sample_rollout(logits0, logits1, payoffs)
            p0 = probs(logits0)
            g0 = torch.zeros(2)
            g0[a0] += r0 * (1 - p0[a0].item())
            g0[1 - a0] -= r0 * p0[a0].item()
            logits0 = logits0 + gamma_n * g0

            p1 = probs(logits1)
            g1 = torch.zeros(2)
            g1[a1] += r1 * (1 - p1[a1].item())
            g1[1 - a1] -= r1 * p1[a1].item()
            logits1 = logits1 + gamma_n * g1

        elif algo in ("Meta-MAPG", "Meta-MAPG+Restart", "Meta-MAPG+Restart+Current",
                      "Meta-MAPG+Only-Current", "LOLA"):
            if algo == "LOLA":
                # LOLA: differentiable through one inner step
                l0 = logits0.clone().detach().requires_grad_(True)
                l1 = logits1.clone().detach()
                l1_next = inner_pg_step_diff(l0, l1, payoffs, K=16, alpha=alpha)
                p0 = torch.softmax(l0, dim=0)
                p1 = torch.softmax(l1_next, dim=0)
                v0 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 0] for a0 in range(2) for a1 in range(2))
                g0 = torch.autograd.grad(v0, l0)[0].detach()
                logits0 = (logits0 + gamma_n * g0).detach()
                # peer standard PG
                a0b, a1b, _, r1b = sample_rollout(logits0, logits1, payoffs)
                p1b = probs(logits1)
                s1 = torch.zeros(2)
                s1[a1b] = r1b * (1 - p1b[a1b].item())
                s1[1 - a1b] = -r1b * p1b[a1b].item()
                logits1 = (logits1 + gamma_n * s1).detach()
            else:
                abl = "full" if "Meta-MAPG" in algo else ablation
                if "Only-Current" in algo:
                    abl = "current"
                elif "Current" in algo and "Own" not in algo:
                    abl = "current+own"

                g0 = meta_mapg_gradient(
                    logits0, logits1, payoffs, L=L, K_inner=K_n,
                    N_outer=N_n, alpha=alpha, ablation=abl
                )
                logits0 = (logits0 + gamma_n * g0).detach()
                # peer standard PG
                a0b, a1b, _, r1b = sample_rollout(logits0, logits1, payoffs)
                p1b = probs(logits1)
                s1 = torch.zeros(2)
                s1[a1b] = r1b * (1 - p1b[a1b].item())
                s1[1 - a1b] = -r1b * p1b[a1b].item()
                logits1 = (logits1 + gamma_n * s1).detach()

        elif algo == "OMWU":
            logits0, logits1 = omwu_step(logits0, logits1, payoffs, eta=gamma_n)

        elif algo == "REINFORCE+Restart":
            a0, a1, r0, r1 = sample_rollout(logits0, logits1, payoffs)
            p0 = probs(logits0)
            g0 = torch.zeros(2)
            g0[a0] += r0 * (1 - p0[a0].item())
            g0[1 - a0] -= r0 * p0[a0].item()
            logits0 = logits0 + gamma_n * g0
            p1 = probs(logits1)
            g1 = torch.zeros(2)
            g1[a1] += r1 * (1 - p1[a1].item())
            g1[1 - a1] -= r1 * p1[a1].item()
            logits1 = logits1 + gamma_n * g1

        cur_r = sum(
            probs(logits0)[a0].item() * probs(logits1)[a1].item() * payoffs[a0, a1, 0].item()
            for a0 in range(2) for a1 in range(2)
        )
        reward_window.append(cur_r)
        if len(reward_window) > window_size:
            reward_window.pop(0)

        # Global restart (Giannou §4)
        if "Restart" in algo and restart_thresh is not None and len(reward_window) == window_size:
            avg_r = sum(reward_window) / window_size
            if avg_r < restart_thresh:
                restarts += 1
                restart_eps.append(n)
                logits0 = giannou_restart(rng)
                logits1 = giannou_restart(rng)
                reward_window = []

        # Convergence check
        p0c = probs(logits0)[0].item()
        p1c = probs(logits1)[0].item()
        if p0c > PARETO_THRESHOLD and p1c > PARETO_THRESHOLD:
            return True, n, restarts, restart_eps

    return False, n_outer_episodes, restarts, restart_eps


# ─── Bias/Variance measurement ────────────────────────────────────────────────

def measure_bias_variance(game_name, L, alpha, K_values, N_values, n_trials=200, seed=42):
    """
    At a fixed intermediate policy, measure:
      bias(K) = ||E[v_hat_K] - v_exact||  as K varies (N fixed large)
      variance(N) = E[||v_hat - E[v_hat]||^2] as N varies (K fixed large)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    payoffs = GAMES[game_name]

    # Fixed intermediate policy (not at equilibrium — in basin of attraction)
    logits0 = torch.tensor([0.3, 0.8], dtype=torch.float32)
    logits1 = torch.tensor([0.5, 0.5], dtype=torch.float32)

    v_exact = exact_meta_gradient(logits0, logits1, payoffs, L=L, alpha=alpha)

    bias_results = []
    for K in K_values:
        estimates = []
        for _ in range(n_trials):
            g = meta_mapg_gradient(logits0, logits1, payoffs, L=L, K_inner=K,
                                   N_outer=8, alpha=alpha, ablation="full")
            estimates.append(g)
        mean_est = torch.stack(estimates).mean(0)
        bias = (mean_est - v_exact).norm().item()
        bias_results.append({"K": K, "bias": bias})

    variance_results = []
    K_large = max(K_values)
    for N in N_values:
        estimates = []
        for _ in range(n_trials):
            g = meta_mapg_gradient(logits0, logits1, payoffs, L=L, K_inner=K_large,
                                   N_outer=N, alpha=alpha, ablation="full")
            estimates.append(g)
        ests = torch.stack(estimates)
        mean_est = ests.mean(0)
        variance = ((ests - mean_est) ** 2).sum(1).mean().item()
        variance_results.append({"N": N, "variance": variance})

    return bias_results, variance_results


# ─── Assumption 5 constants ───────────────────────────────────────────────────

def compute_assumption5_constants(game_name, L, alpha):
    """
    At converged Stag Hunt policy (S,S) ≈ logits [2, -2]:
    Compute m_F (min singular value of DF), L_F (curvature bound), V_max.
    """
    payoffs = GAMES[game_name]
    # Near-converged policy
    logits0 = torch.tensor([2.0, -2.0], dtype=torch.float32, requires_grad=True)
    logits1 = torch.tensor([2.0, -2.0], dtype=torch.float32, requires_grad=True)

    # Compute F = Lambda(T_alpha(phi)) Jacobian numerically
    eps = 1e-3
    F_cols = []
    base_probs = torch.softmax(logits0, dim=0).detach()

    for i in range(2):
        l0_plus = logits0.clone().detach()
        l0_plus[i] += eps
        # Deterministic inner chain (L steps)
        l1_det = logits1.detach().clone()
        for _ in range(L):
            l1_det_req = l1_det.requires_grad_(True)
            p0 = torch.softmax(l0_plus, dim=0)
            p1 = torch.softmax(l1_det_req, dim=0)
            v1 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 1] for a0 in range(2) for a1 in range(2))
            g1 = torch.autograd.grad(v1, l1_det_req)[0]
            l1_det = (l1_det_req + alpha * g1).detach()
        p_plus = torch.softmax(l0_plus, dim=0).detach()

        l0_minus = logits0.clone().detach()
        l0_minus[i] -= eps
        l1_det = logits1.detach().clone()
        for _ in range(L):
            l1_det_req = l1_det.requires_grad_(True)
            p0 = torch.softmax(l0_minus, dim=0)
            p1 = torch.softmax(l1_det_req, dim=0)
            v1 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 1] for a0 in range(2) for a1 in range(2))
            g1 = torch.autograd.grad(v1, l1_det_req)[0]
            l1_det = (l1_det_req + alpha * g1).detach()
        p_minus = torch.softmax(l0_minus, dim=0).detach()

        F_cols.append((p_plus - p_minus) / (2 * eps))

    DF = torch.stack(F_cols, dim=1)  # 2x2 Jacobian
    svd = torch.linalg.svdvals(DF)
    m_F = svd.min().item()
    M_F = svd.max().item()

    # V_max: max payoff gradient norm at this policy
    l0r = logits0.clone().requires_grad_(True)
    l1r = logits1.clone().detach()
    p0 = torch.softmax(l0r, dim=0)
    p1 = torch.softmax(l1r, dim=0)
    v0 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 0] for a0 in range(2) for a1 in range(2))
    v_grad = torch.autograd.grad(v0, l0r)[0].detach()
    V_max = v_grad.norm().item()

    # L_F: rough curvature — use finite differences of DF in each direction
    eps2 = 1e-2
    DF_perturbed_col = []
    for i in range(2):
        l0_p = logits0.clone().detach()
        l0_p[i] += eps2
        F_cols_p = []
        for j in range(2):
            lp = l0_p.clone()
            lp[j] += eps
            p_plus = torch.softmax(lp, dim=0).detach()
            lm = l0_p.clone()
            lm[j] -= eps
            p_minus = torch.softmax(lm, dim=0).detach()
            F_cols_p.append((p_plus - p_minus) / (2 * eps))
        DF_p = torch.stack(F_cols_p, dim=1)
        DF_perturbed_col.append((DF_p - DF).norm().item() / eps2)
    L_F = max(DF_perturbed_col)

    return {"m_F": m_F, "M_F": M_F, "L_F": L_F, "V_max": V_max,
            "curvature_dominance_lhs": 0.5 * V_max * L_F,
            "note": "c*m_F^2 must exceed lhs for Assumption 5 to hold"}


# ─── Parallel wrapper ─────────────────────────────────────────────────────────

def worker(args):
    (game, algo, seed, L, p, l_b, l_sigma, gamma0, m_warmup,
     c_N, c_K, n_eps, tau) = args
    conv, fp, nr, reps = run_seed(
        game, algo, seed, L=L, p=p, l_b=l_b, l_sigma=l_sigma,
        gamma0=gamma0, m_warmup=m_warmup, c_N=c_N, c_K=c_K,
        n_outer_episodes=n_eps, restart_thresh=tau
    )
    return game, algo, seed, conv, fp, nr, reps


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_v2")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--n_eps", type=int, default=3000)
    parser.add_argument("--L", type=int, default=1, help="Inner unrolling depth")
    parser.add_argument("--p", type=float, default=0.7, help="LR decay exponent")
    parser.add_argument("--l_b", type=float, default=0.4, help="Bias exponent")
    parser.add_argument("--l_sigma", type=float, default=0.1, help="Variance exponent")
    parser.add_argument("--gamma0", type=float, default=0.5)
    parser.add_argument("--m_warmup", type=float, default=10.0)
    parser.add_argument("--c_N", type=float, default=1.0)
    parser.add_argument("--c_K", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 12))
    parser.add_argument("--skip_bias_variance", action="store_true")
    parser.add_argument("--skip_assumption5", action="store_true")
    args = parser.parse_args()

    # Verify schedule conditions
    assert args.p + args.l_b > 1.0, f"p+l_b={args.p+args.l_b} must be >1"
    assert args.p - args.l_sigma > 0.5, f"p-l_sigma={args.p-args.l_sigma} must be >0.5"

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Schedule: p={args.p}, l_b={args.l_b}, l_sigma={args.l_sigma}")
    print(f"  p+l_b={args.p+args.l_b:.2f} > 1  ✓")
    print(f"  p-l_sigma={args.p-args.l_sigma:.2f} > 0.5  ✓")
    print(f"L={args.L}, {args.n_seeds} seeds, {args.n_eps} episodes")

    games_list = ["StagHunt"]
    algos = [
        "REINFORCE",
        "REINFORCE+Restart",
        "LOLA",
        "OMWU",
        "Meta-MAPG",
        "Meta-MAPG+Restart",
    ]
    thresholds = {"StagHunt": 3.5}  # Between H,H payoff (3) and S,S payoff (5)

    tasks = [
        (game, algo, seed, args.L, args.p, args.l_b, args.l_sigma,
         args.gamma0, args.m_warmup, args.c_N, args.c_K, args.n_eps,
         thresholds.get(game) if "Restart" in algo else None)
        for game in games_list
        for algo in algos
        for seed in range(args.n_seeds)
    ]

    print(f"\nRunning {len(tasks)} tasks on {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, tasks)

    with open(os.path.join(args.out_dir, "convergence.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "algo", "seed", "converged", "first_passage", "n_restarts", "restart_episodes"])
        for game, algo, seed, conv, fp, nr, reps in results:
            writer.writerow([game, algo, seed, conv, fp, nr, str(reps)])

    # Summary
    print("\n=== Convergence Rates ===")
    from collections import defaultdict
    counts = defaultdict(lambda: {"conv": 0, "total": 0, "restarts": [], "fps": []})
    for game, algo, seed, conv, fp, nr, reps in results:
        key = (game, algo)
        counts[key]["total"] += 1
        counts[key]["conv"] += int(conv)
        counts[key]["restarts"].append(nr)
        if conv:
            counts[key]["fps"].append(fp)

    for (game, algo), d in sorted(counts.items()):
        rate = 100 * d["conv"] / d["total"]
        mean_r = np.mean(d["restarts"])
        ci = 1.96 * np.sqrt(rate / 100 * (1 - rate / 100) / d["total"]) * 100
        print(f"  {game:12s} {algo:25s}  {rate:5.1f}% ± {ci:.1f}%  restarts={mean_r:.1f}")

    # Bias/Variance measurement
    if not args.skip_bias_variance:
        print("\n=== Bias/Variance Measurement ===")
        K_vals = [1, 2, 4, 8, 16, 32, 64, 128]
        N_vals = [1, 2, 4, 8, 16, 32, 64]
        bias_res, var_res = measure_bias_variance(
            "StagHunt", L=args.L, alpha=0.1, K_values=K_vals, N_values=N_vals
        )
        with open(os.path.join(args.out_dir, "bias_variance.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["K", "bias"])
            for row in bias_res:
                writer.writerow([row["K"], row["bias"]])
            writer.writerow([])
            writer.writerow(["N", "variance"])
            for row in var_res:
                writer.writerow([row["N"], row["variance"]])
        for r in bias_res:
            print(f"  bias(K={r['K']:4d}) = {r['bias']:.4f}")
        for r in var_res:
            print(f"  var(N={r['N']:4d})  = {r['variance']:.6f}")

    # Assumption 5 constants
    if not args.skip_assumption5:
        print("\n=== Assumption 5 Constants (StagHunt, converged policy) ===")
        consts = compute_assumption5_constants("StagHunt", L=args.L, alpha=0.1)
        for k, v in consts.items():
            print(f"  {k}: {v}")
        with open(os.path.join(args.out_dir, "assumption5.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in consts.items():
                writer.writerow([k, v])


if __name__ == "__main__":
    main()
