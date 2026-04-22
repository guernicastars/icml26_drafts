import numpy as np
import torch
import torch.optim as optim
import os
import argparse
import itertools

class IPDEnv:
    """Iterated Prisoner's Dilemma matrix game."""
    def __init__(self):
        # 0: Stag (Cooperate), 1: Hare (Defect)
        # Payoffs: Stag-Stag=(5,5), Stag-Hare=(0,3), Hare-Stag=(3,0), Hare-Hare=(3,3)
        self.payoffs = np.array([
            [[5.0, 5.0], [0.0, 3.0]],
            [[3.0, 0.0], [3.0, 3.0]]
        ])
    def step(self, a0, a1):
        return self.payoffs[a0, a1, 0], self.payoffs[a0, a1, 1]

class TabularPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Softmax param for probability of Cooperate vs Defect
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    def log_probs(self):
        return torch.nn.functional.log_softmax(self.logits, dim=0)
    def sample(self):
        return torch.distributions.Categorical(logits=self.logits).sample().item()

def run_ipd_seed(seed, threshold, max_eps=2000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = IPDEnv()
    
    pi_meta = TabularPolicy()
    pi_peer = TabularPolicy()
    
    opt_meta = optim.Adam(pi_meta.parameters(), lr=0.01)
    opt_peer = optim.Adam(pi_peer.parameters(), lr=0.01)
    
    restarts = 0
    recent_rews = []
    last_restart = 0
    
    for ep in range(max_eps):
        # Sample act
        a0 = pi_meta.sample()
        a1 = pi_peer.sample()
        r0, r1 = env.step(a0, a1)
        
        # Peer update
        opt_peer.zero_grad()
        loss_peer = - (pi_peer.log_probs()[a1] * r1)
        loss_peer.backward()
        opt_peer.step()
        
        # Meta Update (Standard REINFORCE to show basin capture)
        opt_meta.zero_grad()
        loss_meta = - (pi_meta.log_probs()[a0] * r0)
        loss_meta.backward()
        opt_meta.step()

        
        recent_rews.append(r0)
        if len(recent_rews) > 100:
            recent_rews.pop(0)
            avg_r = sum(recent_rews) / 100.0
            if avg_r < threshold and ep > last_restart + 100:
                restarts += 1
                pi_meta.logits.data = torch.randn(2) * 2.0 
                # Meta-MAPG randomly jumps to break out of DD Nash basin
                opt_meta = optim.Adam(pi_meta.parameters(), lr=0.05) 
                last_restart = ep
            
        coop_prob = torch.nn.functional.softmax(pi_meta.logits, dim=0)[0].item()
        
        if coop_prob > 0.90 and torch.nn.functional.softmax(pi_peer.logits, dim=0)[0].item() > 0.90:
            return True, ep, restarts # Converged to CC Nash globally
            
    return False, max_eps, restarts

import multiprocessing as mp

def run_wrapper(args):
    s, t = args
    conv, eps, rest = run_ipd_seed(s, threshold=t)
    return t, s, conv, eps, rest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Grid search over restart thresholds
    thresholds = [0.0, 1.0, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0, 6.0]
    seeds = range(100)
    
    tasks = list(itertools.product(seeds, thresholds))
    results_str = "threshold,seed,converged,episodes,restarts\n"
    
    cores = mp.cpu_count()
    print(f"Running IPD sweeps across {cores} CPU cores...")
    
    with mp.Pool(cores) as pool:
        results = pool.map(run_wrapper, tasks)
        
    # Aggregate
    success_counts = {t: 0 for t in thresholds}
    for r in results:
        t, s, conv, eps, rest = r
        results_str += f"{t},{s},{conv},{eps},{rest}\n"
        if conv: success_counts[t] += 1
        
    for t in thresholds:
        print(f"Threshold: {t} | Convergence to CC Nash: {success_counts[t]}%")
            
    with open(f"{args.out_dir}/ipd_grid_results.csv", "w") as f:
        f.write(results_str)
        
    print(f"Done. Saved to {args.out_dir}/ipd_grid_results.csv")

if __name__ == "__main__":
    main()

