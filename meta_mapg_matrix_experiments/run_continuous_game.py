"""
Continuous-action Stackelberg coordination game for Meta-MAPG validation.

Game (ASYMMETRIC — leader-follower with coordination):
  V_1(a_1, a_2) = -(a_1 - 1)^2 - beta*(a_1 - a_2)^2  (leader: wants target=1, wants coordination)
  V_2(a_1, a_2) = -(a_2 - a_1)^2                       (follower: purely wants to match leader)
  
  Nash: phi_1* = 1, phi_2* = 1 (follower matches leader at target)
  
  Key insight: The follower's gradient is g_2 = -2*(phi_2 - phi_1), so:
    phi_2' = phi_2 + alpha * 2 * (phi_1 - phi_2)
  
  The peer-learning term for player 1:
    dV_1/dphi_2' * dphi_2'/dphi_1
    = 2*beta*(phi_1-phi_2') * (2*alpha)
    
  At initialization where phi_1 < phi_2 (leader behind follower):
    phi_2' moves toward phi_1 (follower tracks leader)
    dV_1/dphi_2' is positive (leader wants follower closer)
    dphi_2'/dphi_1 is positive (follower moves toward leader)
    => Peer gradient is POSITIVE (reinforces leader's movement toward target)
    
  This is the "opponent-aware shaping" advantage: the leader knows the
  follower will track, so it's bolder in moving toward the target.

We compare four algorithms:
  1. PG (current policy gradient only — no lookahead)
  2. Own-only (current + own-learning, NO peer)
  3. Meta-MAPG (current + own + peer, lambda=1)
  4. Meta-MAPG (current + own + peer, lambda=3)
"""
import argparse
import multiprocessing as mp
import os
import csv
import numpy as np

TARGET = 1.0
BETA = 2.0       # Strong coordination pressure
SIGMA = 0.5      # Gradient noise
BASIN_EPS = 0.15

def grad_V1(phi_1, phi_2):
    """Leader's gradient."""
    return -2.0 * (phi_1 - TARGET) - 2.0 * BETA * (phi_1 - phi_2)

def grad_V2(phi_1, phi_2):
    """Follower's gradient (pure matching)."""
    return -2.0 * (phi_2 - phi_1)

def run_seed(args):
    algo, seed, max_eps = args
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
    alpha = 0.15   # Inner step size
    L = 2          # 2 inner steps
    
    # Initialize both players far from Nash
    phi_1 = rng.uniform(-3.0, -1.0)
    phi_2 = rng.uniform(-3.0, -1.0)
    
    for n in range(1, max_eps + 1):
        gamma_n = 0.03  # Constant learning rate for Phase 1
        
        noise_1 = rng.standard_normal() * SIGMA
        noise_2 = rng.standard_normal() * SIGMA
        
        if algo == "PG":
            # Standard PG: just current gradient
            g1 = grad_V1(phi_1, phi_2) + noise_1
            g2 = grad_V2(phi_1, phi_2) + noise_2
            phi_1 += gamma_n * g1
            phi_2 += gamma_n * g2
            
        elif algo == "Own-only":
            # Current + Own-learning (lookahead without peer modeling)
            # Simulate L inner steps for player 2 (but ignore dependence on phi_1)
            phi_2_sim = phi_2
            g1_total = grad_V1(phi_1, phi_2) + noise_1  # Current
            for ell in range(L):
                g2_inner = grad_V2(phi_1, phi_2_sim)
                phi_2_sim += alpha * g2_inner
                g1_total += grad_V1(phi_1, phi_2_sim) + rng.standard_normal() * SIGMA * 0.5  # Own
            
            g2 = grad_V2(phi_1, phi_2) + noise_2
            phi_1 += gamma_n * g1_total
            phi_2 += gamma_n * g2
            
        elif algo.startswith("Meta-MAPG"):
            lambda_c = float(algo.split("=")[1].rstrip(")"))
            
            # Simulate L inner steps for player 2
            phi_2_traj = [phi_2]
            for ell in range(L):
                g2_inner = grad_V2(phi_1, phi_2_traj[-1])
                phi_2_traj.append(phi_2_traj[-1] + alpha * g2_inner)
            
            # Current policy gradient
            g1_cur = grad_V1(phi_1, phi_2) + noise_1
            
            # Own-learning: gradient at each updated state
            g1_own = 0.0
            for ell in range(1, L + 1):
                g1_own += grad_V1(phi_1, phi_2_traj[ell]) + rng.standard_normal() * SIGMA * 0.5
            
            # Peer-learning: account for d(phi_2')/d(phi_1)
            g1_peer = 0.0
            for ell in range(1, L + 1):
                phi_2_ell = phi_2_traj[ell]
                # dV_1/dphi_2_ell = 2*BETA*(phi_1 - phi_2_ell) 
                dV1_dphi2 = 2.0 * BETA * (phi_1 - phi_2_ell)
                
                # dphi_2_ell/dphi_1 via chain rule
                # Each inner step: phi_2_next = phi_2_prev + alpha * (-2*(phi_2_prev - phi_1))
                # dphi_2_next/dphi_1 = (1 - 2*alpha) * dphi_2_prev/dphi_1 + 2*alpha
                dphi2_dphi1 = 0.0
                for k in range(ell):
                    dphi2_dphi1 = (1.0 - 2.0 * alpha) * dphi2_dphi1 + 2.0 * alpha
                
                g1_peer += dV1_dphi2 * dphi2_dphi1
                g1_peer += rng.standard_normal() * SIGMA * 0.3
            
            g1_total = g1_cur + g1_own + lambda_c * g1_peer
            g2 = grad_V2(phi_1, phi_2) + noise_2
            
            phi_1 += gamma_n * g1_total
            phi_2 += gamma_n * g2
        
        if abs(phi_1 - TARGET) < BASIN_EPS and abs(phi_2 - TARGET) < BASIN_EPS:
            return n
    
    return max_eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_stackelberg")
    parser.add_argument("--n_seeds", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_eps", type=int, default=1000)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    algos = ["PG", "Own-only", "Meta-MAPG(lam=0.0)", "Meta-MAPG(lam=0.5)", 
             "Meta-MAPG(lam=1.0)", "Meta-MAPG(lam=2.0)", "Meta-MAPG(lam=3.0)"]
    
    tasks = []
    for algo in algos:
        for seed in range(args.n_seeds):
            tasks.append((algo, seed, args.max_eps))
    
    print(f"Running {len(tasks)} tasks ({args.n_seeds} seeds × {len(algos)} algos)...")
    
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_seed, tasks)
    
    # Save
    out_file = os.path.join(args.out_dir, "stackelberg_results.csv")
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "seed", "first_passage"])
        for i, task in enumerate(tasks):
            writer.writerow([task[0], task[1], results[i]])
    
    # Aggregate
    print(f"\n{'Algo':>30s} | {'Median':>7s} | {'Mean':>7s} | {'Succ%':>6s} | {'vs PG':>7s} | {'vs Own':>7s}")
    print("-" * 80)
    
    idx = 0
    baselines = {}
    for algo in algos:
        times = results[idx:idx + args.n_seeds]
        success = [t for t in times if t < args.max_eps]
        median_t = np.median(success) if success else float('inf')
        mean_t = np.mean(success) if success else float('inf')
        rate = len(success) / args.n_seeds * 100
        
        if algo == "PG":
            baselines["PG"] = median_t
        if algo == "Own-only":
            baselines["Own"] = median_t
            
        speedup_pg = baselines.get("PG", median_t) / median_t if median_t > 0 else 0
        speedup_own = baselines.get("Own", median_t) / median_t if median_t > 0 else 0
        
        print(f"{algo:>30s} | {median_t:7.1f} | {mean_t:7.1f} | {rate:5.0f}% | {speedup_pg:6.2f}x | {speedup_own:6.2f}x")
        idx += args.n_seeds
    
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
