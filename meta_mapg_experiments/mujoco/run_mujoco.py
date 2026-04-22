import os
import torch
import torch.optim as optim
import numpy as np
import argparse
import time
from env.halfcheetah import TwoAgentHalfCheetah
from algo.meta_mapg import PolicyNet, compute_meta_mapg_loss
from algo.restarts import GlobalRestartManager

def compute_returns(rewards, gamma=0.99, device="cpu"):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    return (returns - returns.mean()) / (returns.std() + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1500.0)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    
    run_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = TwoAgentHalfCheetah()
    
    pi_0 = PolicyNet(env.obs_dim, env.action_dim).to(device)
    pi_1 = PolicyNet(env.obs_dim, env.action_dim).to(device)
    
    opt_0 = optim.Adam(pi_0.parameters(), lr=3e-4) 
    opt_1 = optim.Adam(pi_1.parameters(), lr=3e-4) 
    
    restarter = GlobalRestartManager(patience=args.patience, threshold=args.threshold) 
    
    log_rewards, log_restarts = [], []

    print(f"[Seed {args.seed}] Starting MAMuJoCo HalfCheetah...")
    print(f"Target Threshold: {args.threshold} | Device: {device}")
    start_time = time.time()
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        traj = {"o0": [], "a0": [], "r0": [], "o1": [], "a1": [], "r1": []}
        ep_rew = 0
        
        while not done:
            o0 = torch.FloatTensor(obs["agent_0"]).unsqueeze(0).to(device)
            o1 = torch.FloatTensor(obs["agent_1"]).unsqueeze(0).to(device)
            
            dist_0 = pi_0(o0)
            dist_1 = pi_1(o1)
            a0, a1 = dist_0.sample(), dist_1.sample()
            
            next_obs, rewards, dones, _ = env.step(
                {"agent_0": a0.cpu().detach().numpy()[0], "agent_1": a1.cpu().detach().numpy()[0]}
            )
            done = dones["__all__"]
            ep_rew += rewards["agent_0"]
            
            traj["o0"].append(o0); traj["a0"].append(a0); traj["r0"].append(rewards["agent_0"])
            traj["o1"].append(o1); traj["a1"].append(a1); traj["r1"].append(rewards["agent_1"])
            obs = next_obs
            
        log_rewards.append(ep_rew)
        
        o0_t, a0_t = torch.cat(traj["o0"]), torch.cat(traj["a0"])
        o1_t, a1_t = torch.cat(traj["o1"]), torch.cat(traj["a1"])
        ret0_t = compute_returns(traj["r0"], device=device)
        ret1_t = compute_returns(traj["r1"], device=device)
        
        # Peer PG Update
        opt_1.zero_grad()
        loss_1 = -(pi_1(o1_t).log_prob(a1_t.detach()).sum(-1) * ret1_t).mean()
        loss_1.backward()
        opt_1.step()
        
        # Meta-MAPG Update
        opt_0.zero_grad()
        loss_0 = compute_meta_mapg_loss(pi_0, pi_1, o0_t, a0_t.detach(), ret0_t, o1_t, a1_t.detach(), ret1_t)
        loss_0.backward()
        opt_0.step()

        # Check Global Restart
        if restarter.check_and_restart(ep_rew, pi_0):
            print(f"[Seed {args.seed}] Global Restart Triggered at ep {ep}!")
            log_restarts.append(ep)
            pi_1 = PolicyNet(env.obs_dim, env.action_dim).to(device)
            opt_0 = optim.Adam(pi_0.parameters(), lr=3e-4) 
            opt_1 = optim.Adam(pi_1.parameters(), lr=3e-4)

        # Log with Flush to avoid "Small File" buffer issues
        with open(os.path.join(run_dir, "logs", "metrics.csv"), "a") as f:
            f.write(f"{ep},{ep_rew:.2f},{restarter.restart_count},{time.time()-start_time:.2f}\n")
            f.flush()

        if ep % 10 == 0:
            print(f"[Seed {args.seed}] Ep {ep} | Reward: {ep_rew:.2f} | Restarts: {restarter.restart_count}")

        # Sparse Checkpoints
        if ep % args.save_freq == 0:
            torch.save({
                'pi_0': pi_0.state_dict(),
                'pi_1': pi_1.state_dict(),
                'opt_0': opt_0.state_dict(),
                'opt_1': opt_1.state_dict(),
                'restarts': restarter.restart_count
            }, os.path.join(run_dir, "checkpoints", f"ckpt_ep{ep}.pt"))

    # Final dump
    np.savetxt(os.path.join(run_dir, "restarts.txt"), log_restarts)
    print(f"[Seed {args.seed}] Done. Saved to: {run_dir}")

if __name__ == "__main__":
    main()
