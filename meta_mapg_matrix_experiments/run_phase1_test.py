import argparse
import multiprocessing as mp
import os
import csv
import torch
import numpy as np

GAMES = {
    "StagHunt": torch.tensor([
        [[5.0, 5.0], [0.0, 3.0]],
        [[3.0, 0.0], [3.0, 3.0]],
    ])
}

BASIN_THRESHOLD = 0.8  # p_coop > 0.8 is entering the "safe" nested basin

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

def meta_mapg_gradient(logits0, logits1, payoffs, L, K_inner, N_outer, alpha, lambda_c):
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

        grad_accum += G0 * (g_cur + g_own + lambda_c * g_peer)

    return grad_accum / N_outer

def run_seed_phase1(args):
    lambda_c, seed, max_eps = args
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    payoffs = GAMES["StagHunt"]
    alpha = 0.1
    L = 1
    
    # Initialize from a fixed point far away from basin entry target
    # e.g., uniform mixing where p_coop is 0.5
    noise0 = rng.uniform(-0.1, 0.1, 2)
    noise1 = rng.uniform(-0.1, 0.1, 2)
    logits0 = torch.tensor([0.0, 0.0] + noise0, dtype=torch.float32)
    logits1 = torch.tensor([0.0, 0.0] + noise1, dtype=torch.float32)

    alpha_0 = 0.05  # Constant learning rate for Phase 1
    N_n = 4 # Small constant sample sizes for fast entry
    K_n = 4

    for n in range(1, max_eps + 1):
        g0 = meta_mapg_gradient(
            logits0, logits1, payoffs, L=L, K_inner=K_n,
            N_outer=N_n, alpha=alpha, lambda_c=lambda_c
        )
        logits0 = (logits0 + alpha_0 * g0).detach()
        
        a0b, a1b, _, r1b = sample_rollout(logits0, logits1, payoffs)
        p1b = probs(logits1)
        s1 = torch.zeros(2)
        s1[a1b] = r1b * (1 - p1b[a1b].item())
        s1[1 - a1b] = -r1b * p1b[a1b].item()
        logits1 = (logits1 + alpha_0 * s1).detach()

        p0c = probs(logits0)[0].item()
        if p0c > BASIN_THRESHOLD:
            return n

    return max_eps  # Failed to enter within max_eps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_phase1")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0]
    # PG is essentially lambda_c = 0.0 since discrete sampling peer learning has structural issues.
    
    tasks = []
    for lc in lambda_values:
        for seed in range(args.n_seeds):
            tasks.append((lc, seed, 800))
                
    print(f"Running {len(tasks)} basin-entry tasks (max eps=800) with 16 workers...")
    
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_seed_phase1, tasks)
        
    out_file = os.path.join(args.out_dir, "phase1_results.csv")
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda_c", "seed", "N_0"])
        for i, task in enumerate(tasks):
            writer.writerow([task[0], task[1], results[i]])
            
    # Aggregate (Median and Mean N_0 for successful entries)
    print("\n=== Basin Entry Times (N_0) ===")
    for lc in lambda_values:
        times = [results[i] for i, t in enumerate(tasks) if t[0] == lc and results[i] < 800]
        if len(times) == 0:
            print(f"  lambda_c={lc:4.1f}: NO SUCCESSFUL ENTRIES")
        else:
            success_rate = len(times) / args.n_seeds * 100
            print(f"  lambda_c={lc:4.1f}: Median N_0 = {np.median(times):.1f}, Mean N_0 = {np.mean(times):.1f} (Success: {success_rate:.0f}%)")

if __name__ == "__main__":
    main()
