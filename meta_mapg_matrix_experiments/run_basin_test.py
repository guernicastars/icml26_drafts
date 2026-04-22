import argparse
import multiprocessing as mp
import os
import csv
import torch
import numpy as np

# A simplified runner for the basin experiment
# We test convergence probability (to Pareto Nash) from varying initial positions.
# t=0: near Risk-Dominant (H,H), t=1: near Pareto (S,S)
# phi(t) = (1-t) * [-2, 2] + t * [2, -2]

GAMES = {
    "StagHunt": torch.tensor([
        [[5.0, 5.0], [0.0, 3.0]],
        [[3.0, 0.0], [3.0, 3.0]],
    ])
}

PARETO_THRESHOLD = 0.95

def probs(logits):
    return torch.softmax(logits, dim=0)

def sample_action(logits):
    return torch.distributions.Categorical(logits=logits).sample().item()

def log_prob_action(logits, action):
    return torch.log_softmax(logits, dim=0)[action]

def sample_rollout(logits0, logits1, payoffs):
    a0 = sample_action(logits0)
    a1 = sample_action(logits1)
    r0 = payoffs[a0, a1, 0].item()
    r1 = payoffs[a0, a1, 1].item()
    return a0, a1, r0, r1

def inner_pg_step(logits0, logits1, payoffs, K, alpha):
    l1 = logits1.clone().detach().requires_grad_(True)
    p0 = torch.softmax(logits0.detach(), dim=0)
    log_probs = []
    rewards = []
    for _ in range(K):
        a1 = sample_action(l1)
        r1 = sum(p0[a0].item() * payoffs[a0, a1, 1].item() for a0 in range(2))
        log_probs.append(log_prob_action(l1, a1))
        rewards.append(r1)
    loss = -sum(lp * r for lp, r in zip(log_probs, rewards)) / K
    g = torch.autograd.grad(loss, l1)[0]
    return (l1 - alpha * g).detach()

def stochastic_inner_chain(logits0, logits1, payoffs, L, K, alpha):
    l1_trajectory = [logits1.detach().clone()]
    for _ in range(L):
        l1_next = inner_pg_step(logits0, l1_trajectory[-1], payoffs, K, alpha)
        l1_trajectory.append(l1_next)
    return l1_trajectory

def inner_pg_step_diff(logits0_fixed, logits1, payoffs, K, alpha):
    l1 = logits1.clone().detach().requires_grad_(True)
    p0 = torch.softmax(logits0_fixed, dim=0)
    p1 = torch.softmax(l1, dim=0)
    v1 = sum(p0[a0] * p1[a1] * payoffs[a0, a1, 1] for a0 in range(2) for a1 in range(2))
    g1 = torch.autograd.grad(v1, l1, create_graph=True)[0]
    l1_new = l1.detach() + alpha * g1
    return l1_new

def meta_mapg_gradient(logits0, logits1, payoffs, L, K_inner, N_outer, alpha, ablation="full"):
    l0 = logits0.clone().detach().requires_grad_(True)
    l1_base = logits1.clone().detach()

    grad_accum = torch.zeros_like(l0)

    for _ in range(N_outer):
        l1_traj = stochastic_inner_chain(l0.detach(), l1_base, payoffs, L, K_inner, alpha)
        l1_final = l1_traj[-1]

        a0_f, a1_f, r0_f, _ = sample_rollout(l0, l1_final, payoffs)
        G0 = r0_f

        lp0_cur = log_prob_action(l0, a0_f)
        g_cur = torch.autograd.grad(lp0_cur, l0, retain_graph=True)[0].detach()

        g_own = g_cur * L

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

def run_seed_basin(args):
    algo, seed, t_pos, n_eps = args
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    payoffs = GAMES["StagHunt"]
    alpha = 0.1
    L = 1
    
    # Initialize based on t_pos
    # t_pos = 0.0 -> [-2, 2] (near H,H)
    # t_pos = 1.0 -> [2, -2] (near S,S)
    p_init = (1 - t_pos) * np.array([-2.0, 2.0]) + t_pos * np.array([2.0, -2.0])
    
    # Add a small uniform noise to initialization
    noise0 = rng.uniform(-0.1, 0.1, 2)
    noise1 = rng.uniform(-0.1, 0.1, 2)
    
    logits0 = torch.tensor(p_init + noise0, dtype=torch.float32)
    logits1 = torch.tensor(p_init + noise1, dtype=torch.float32)

    gamma0 = 0.5
    m_warmup = 10.0
    p = 0.7
    l_b = 0.4
    l_sigma = 0.1
    c_N = 1.0
    c_K = 1.0

    for n in range(1, n_eps + 1):
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

        elif algo == "Meta-MAPG":
            g0 = meta_mapg_gradient(
                logits0, logits1, payoffs, L=L, K_inner=K_n,
                N_outer=N_n, alpha=alpha, ablation="full"
            )
            logits0 = (logits0 + gamma_n * g0).detach()
            
            a0b, a1b, _, r1b = sample_rollout(logits0, logits1, payoffs)
            p1b = probs(logits1)
            s1 = torch.zeros(2)
            s1[a1b] = r1b * (1 - p1b[a1b].item())
            s1[1 - a1b] = -r1b * p1b[a1b].item()
            logits1 = (logits1 + gamma_n * s1).detach()

        p0c = probs(logits0)[0].item()
        p1c = probs(logits1)[0].item()
        if p0c > PARETO_THRESHOLD and p1c > PARETO_THRESHOLD:
            return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_basin")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    t_values = np.linspace(0.0, 1.0, 11)
    algos = ["REINFORCE", "Meta-MAPG"]
    
    tasks = []
    for algo in algos:
        for t in t_values:
            for seed in range(args.n_seeds):
                tasks.append((algo, seed, t, 3000))
                
    print(f"Running {len(tasks)} basin evaluation tasks...")
    
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_seed_basin, tasks)
        
    out_file = os.path.join(args.out_dir, "basin_results.csv")
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "t_pos", "seed", "converged"])
        for i, task in enumerate(tasks):
            writer.writerow([task[0], task[2], task[1], results[i]])
            
    # Aggregate
    agg = {}
    for i, task in enumerate(tasks):
        algo, _, t_pos, _ = task
        conv = results[i]
        key = (algo, t_pos)
        if key not in agg:
            agg[key] = {"conv": 0, "total": 0}
        agg[key]["conv"] += int(conv)
        agg[key]["total"] += 1
        
    print("\n=== Basin Convergence rates ===")
    for algo in algos:
        print(f"Algorithm: {algo}")
        for t in t_values:
            key = (algo, t)
            rate = agg[key]["conv"] / agg[key]["total"] * 100
            print(f"  t={t:4.2f}: {rate:5.1f}%")
            
    print(f"Details saved to {out_file}")

if __name__ == "__main__":
    main()
