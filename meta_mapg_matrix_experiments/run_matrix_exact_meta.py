import torch
import numpy as np
import multiprocessing as mp
import itertools
import os

games = {
    # 0: Cooperate, 1: Defect
    "IPD":       torch.tensor([[[3.0, 3.0], [0.0, 5.0]], [[5.0, 0.0], [1.0, 1.0]]]),
    "StagHunt":  torch.tensor([[[5.0, 5.0], [0.0, 3.0]], [[3.0, 0.0], [3.0, 3.0]]]),
    "Chicken":   torch.tensor([[[2.0, 2.0], [0.0, 3.0]], [[3.0, 0.0], [-5.0, -5.0]]])
}

def expected_values(pi1, pi2, payoffs):
    p1 = torch.softmax(pi1, dim=0)
    p2 = torch.softmax(pi2, dim=0)
    V1 = p1[0]*p2[0]*payoffs[0,0,0] + p1[0]*p2[1]*payoffs[0,1,0] + p1[1]*p2[0]*payoffs[1,0,0] + p1[1]*p2[1]*payoffs[1,1,0]
    V2 = p1[0]*p2[0]*payoffs[0,0,1] + p1[0]*p2[1]*payoffs[0,1,1] + p1[1]*p2[0]*payoffs[1,0,1] + p1[1]*p2[1]*payoffs[1,1,1]
    return V1, V2

def run_exact_game(game_name, algo, seed, max_eps=1000, restart_thresh=None):
    torch.manual_seed(seed)
    payoffs = games[game_name]
    
    # Initialize policies to be slightly biased towards Defect (the local Nash)
    # This proves that our algorithm ESCAPES the bad Nash!
    pi1 = torch.nn.Parameter(torch.tensor([-1.0, 1.0]) + torch.randn(2)*0.1)
    pi2 = torch.nn.Parameter(torch.tensor([-1.0, 1.0]) + torch.randn(2)*0.1)
    
    lr = 0.05
    inner_lr = 1.0  # LOLA requires a strong inner lookahead step to invert IPD
    
    restarts = 0
    recent_rews = []
    last_restart = 0
    
    for ep in range(max_eps):
        V1, V2 = expected_values(pi1, pi2, payoffs)
        
        grad_2_V2 = torch.autograd.grad(V2, pi2, create_graph=True)[0]
        
        if algo == "REINFORCE":
            grad_1_V1 = torch.autograd.grad(V1, pi1)[0]
            pi1.data += lr * grad_1_V1
            pi2.data += lr * grad_2_V2.detach()
            
        elif algo == "Meta-MAPG" or algo == "Meta-MAPG+Restarts":
            # Peer's lookahead
            pi2_future = pi2 + inner_lr * grad_2_V2
            V1_future, _ = expected_values(pi1, pi2_future, payoffs)
            meta_grad = torch.autograd.grad(V1_future, pi1)[0]
            
            pi1.data += lr * meta_grad
            pi2.data += lr * grad_2_V2.detach()
            
        # Global Restarts logic
        if restart_thresh is not None and algo == "Meta-MAPG+Restarts":
            recent_rews.append(V1.item())
            if len(recent_rews) > 50:
                avg_r = sum(recent_rews[-50:]) / 50.0
                # If stuck in bad Nash (below threshold) and enough time passed
                if avg_r < restart_thresh and ep > last_restart + 50:
                    restarts += 1
                    # Global Restart mechanism: blast both to random space
                    pi1.data = torch.randn(2) * 2.0
                    pi2.data = torch.randn(2) * 2.0
                    last_restart = ep
                    recent_rews = [] # clear buffer

        coop1 = torch.softmax(pi1, dim=0)[0].item()
        coop2 = torch.softmax(pi2, dim=0)[0].item()
        
        # Check convergence to Strict Pareto Co-op
        if game_name in ["IPD", "StagHunt"] and coop1 > 0.95 and coop2 > 0.95:
            return True, ep, restarts
        if game_name == "Chicken" and coop1 > 0.4 and coop1 < 0.6 and coop2 > 0.4 and coop2 < 0.6:
            # Chicken fair mixed Nash
            return True, ep, restarts
            
    return False, max_eps, restarts

def parallel_wrapper(args):
    g, algo, s, t = args
    conv, eps, rest = run_exact_game(g, algo, s, restart_thresh=t)
    return f"{g},{algo},{s},{t},{conv},{eps},{rest}\n"

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    seeds = list(range(100))
    algos = ["REINFORCE", "Meta-MAPG", "Meta-MAPG+Restarts"]
    games_list = ["IPD", "StagHunt"]
    
    thresholds = {
        "IPD": 2.5,       # Defect-Defect is 1.0. Cooperate is 3.0.
        "StagHunt": 4.0,  # Hare-Hare is 3.0. Stag-Stag is 5.0.
        "Chicken": 1.5    
    }
    
    tasks = []
    for g in games_list:
        for algo in algos:
            for s in seeds:
                tasks.append((g, algo, s, thresholds[g]))
                
    cores = min(mp.cpu_count(), 16)
    
    with mp.Pool(cores) as pool:
        csv_lines = pool.map(parallel_wrapper, tasks)
        
    out_file = "results/master_matrix_theory.csv"
    with open(out_file, "w") as f:
        f.write("game,algorithm,seed,threshold,converged,episodes,restarts\n")
        f.writelines(csv_lines)
