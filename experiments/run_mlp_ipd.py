#!/usr/bin/env python3
"""
Non-tabular IPD with MLP policy — proof-of-transport experiment.

Repeats the IPD ablation (PG vs Meta-PG vs LOLA-style vs Meta-MAPG)
with a 2-layer MLP policy (16 hidden units, tanh) instead of tabular Bernoulli.

Uses DiCE-style surrogate losses for correct higher-order gradients.

Eugene's spec: §1–§12 of the coauthor handoff document.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════════════
# IPD Environment
# ══════════════════════════════════════════════════════════════════════════════

# Payoffs: row = my action, col = opponent action.  C=0, D=1.
# [[CC, CD], [DC, DD]] = [[3,0],[5,1]]
R_MATRIX = torch.tensor([[3.0, 0.0],
                          [5.0, 1.0]])

HORIZON = 12
GAMMA = 0.96
N_STATES = 5  # CC=0, CD=1, DC=2, DD=3, START=4
SUCCESS_THRESHOLD = 0.82


def state_onehot(state_idx, device="cpu"):
    """One-hot encode a single state index."""
    v = torch.zeros(N_STATES, device=device)
    v[state_idx] = 1.0
    return v


def joint_action_to_state(a0, a1):
    """Map joint action (a0, a1) to next state index: CC=0, CD=1, DC=2, DD=3."""
    return a0 * 2 + a1


def rollout_batch(policy0, policy1, batch_size, horizon=HORIZON, gamma=GAMMA):
    """
    Sample a batch of IPD trajectories.

    Returns dict with:
      states:  (B, H, 5) one-hot states
      actions: (B, H, 2) joint actions [a0, a1]
      rewards: (B, H, 2) rewards [r0, r1]
      log_probs: (B, H, 2) log π(a_i | s)
      returns:   (B, 2) discounted returns per agent
    """
    device = next(policy0.parameters()).device
    B, H = batch_size, horizon

    all_states = torch.zeros(B, H, N_STATES, device=device)
    all_actions = torch.zeros(B, H, 2, dtype=torch.long, device=device)
    all_rewards = torch.zeros(B, H, 2, device=device)
    all_log_probs = torch.zeros(B, H, 2, device=device)

    # Start state
    state = state_onehot(4, device).unsqueeze(0).expand(B, -1)  # (B, 5)

    for t in range(H):
        all_states[:, t] = state

        # Get cooperation logits
        logit0 = policy0(state).squeeze(-1)  # (B,)
        logit1 = policy1(state).squeeze(-1)  # (B,)

        # Sample actions: 0 = cooperate, 1 = defect
        prob_coop0 = torch.sigmoid(logit0)
        prob_coop1 = torch.sigmoid(logit1)

        a0 = torch.bernoulli(1.0 - prob_coop0).long()  # 0=coop, 1=defect
        a1 = torch.bernoulli(1.0 - prob_coop1).long()

        # Log probs
        lp0 = torch.where(a0 == 0, torch.log(prob_coop0 + 1e-8),
                           torch.log(1.0 - prob_coop0 + 1e-8))
        lp1 = torch.where(a1 == 0, torch.log(prob_coop1 + 1e-8),
                           torch.log(1.0 - prob_coop1 + 1e-8))

        all_actions[:, t, 0] = a0
        all_actions[:, t, 1] = a1
        all_log_probs[:, t, 0] = lp0
        all_log_probs[:, t, 1] = lp1

        # Rewards
        r0 = R_MATRIX[a0, a1]
        r1 = R_MATRIX[a1, a0]  # Symmetric game, transpose for player 1
        all_rewards[:, t, 0] = r0
        all_rewards[:, t, 1] = r1

        # Next state
        next_state_idx = a0 * 2 + a1  # CC=0, CD=1, DC=2, DD=3
        state = torch.zeros(B, N_STATES, device=device)
        for b in range(B):
            state[b, next_state_idx[b]] = 1.0

    # Compute discounted returns
    returns = torch.zeros(B, 2, device=device)
    for t in range(H):
        returns += (gamma ** t) * all_rewards[:, t]

    return {
        "states": all_states,
        "actions": all_actions,
        "rewards": all_rewards,
        "log_probs": all_log_probs,
        "returns": returns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MLP Policy
# ══════════════════════════════════════════════════════════════════════════════

class MLPPolicy(nn.Module):
    """
    2-layer MLP: 5 → 16 → 16 → 1 with tanh activations.
    Output is a single logit for Bernoulli cooperation probability.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATES, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        # Scale final layer by 0.05 so initial coop prob ≈ 0.5 and less aggressive collapse
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.mul_(0.0);

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# DiCE surrogate losses
# ══════════════════════════════════════════════════════════════════════════════

def dice_operator(log_probs):
    """
    DiCE magic box: has value 1, gradient exp(x - x.detach()) * grad(x).
    Input: (B, H) tensor of log probs.
    Output: (B, H) tensor with correct higher-order gradient.
    """
    return (log_probs - log_probs.detach()).exp()


def compute_losses(policy0, policy1, trajs, method,
                   peer_coef=1.5, own_coef=0.35, inner_lr=0.1):
    """
    Compute DiCE surrogate losses for both agents with higher-order terms.
    """
    states = trajs["states"]       # (B, H, 5)
    actions = trajs["actions"]     # (B, H, 2)
    rewards = trajs["rewards"]     # (B, H, 2)
    B, H = states.shape[0], states.shape[1]

    # Re-evaluate log probs with gradient flow
    logit0 = policy0(states).squeeze(-1)  # (B, H)
    logit1 = policy1(states).squeeze(-1)

    prob_coop0 = torch.sigmoid(logit0)
    prob_coop1 = torch.sigmoid(logit1)

    a0, a1 = actions[:, :, 0], actions[:, :, 1]
    lp0 = torch.where(a0 == 0, torch.log(prob_coop0 + 1e-8), torch.log(1.0 - prob_coop0 + 1e-8))
    lp1 = torch.where(a1 == 0, torch.log(prob_coop1 + 1e-8), torch.log(1.0 - prob_coop1 + 1e-8))

    cum_lp0 = torch.cumsum(lp0, dim=1)
    cum_lp1 = torch.cumsum(lp1, dim=1)

    discounts = GAMMA ** torch.arange(H, dtype=torch.float32, device=states.device).unsqueeze(0)
    disc_r0 = rewards[:, :, 0] * discounts
    disc_r1 = rewards[:, :, 1] * discounts

    # Basic DiCE surrogates (E[R * dice_0 * dice_1])
    # Differentiating this w.r.t theta0 gives base PG.
    # Differentiating the gradient w.r.t theta1 gives the LOLA cross-term.
    dice0 = dice_operator(cum_lp0)
    dice1 = dice_operator(cum_lp1)

    v0_surr = (disc_r0.detach() * dice0 * dice1).sum(dim=1).mean()
    v1_surr = (disc_r1.detach() * dice0 * dice1).sum(dim=1).mean()

    # --- Base REINFORCE losses ---
    # simple backward on v0_surr would give grad_0(V0) + grad_1(V0).
    # To get JUST grad_0(V0), we should ideally detach dice1 if we only want 1st order.
    # But for Meta-MAPG we need the graph.
    params0 = list(policy0.parameters())
    params1 = list(policy1.parameters())

    g0_0 = torch.autograd.grad(v0_surr, params0, create_graph=True, retain_graph=True)
    g1_1 = torch.autograd.grad(v1_surr, params1, create_graph=True, retain_graph=True)

    # 1. Base PG gradients (as losses)
    # To apply as loss, we use the property that grad_0( <g0_0_detached, params0> ) = g0_0_detached.
    l0 = -sum((g.detach() * p).sum() for g, p in zip(g0_0, params0))
    l1 = -sum((g.detach() * p).sum() for g, p in zip(g1_1, params1))

    if method == "standard_pg":
        return l0, l1

    # 2. Own-learning terms (Meta-PG)
    # grad_0 [ 0.5 * alpha * ||g0_0||^2 ] = alpha * H_00 * g0_0
    # To maximize V0 + 0.5*alpha*||g0_0||^2, we minimize -V0 - 0.5*alpha*||g0_0||^2
    own_l0 = -0.5 * own_coef * inner_lr * sum((g ** 2).sum() for g in g0_0)
    own_l1 = -0.5 * own_coef * inner_lr * sum((g ** 2).sum() for g in g1_1)

    if method == "meta_pg":
        return l0 + own_l0, l1 + own_l1

    # 3. Peer-learning terms (LOLA)
    # grad_0 [ alpha * <grad_1 V0, grad_1 V1> ] = alpha * H_10 * g1
    # To maximize this, we minimize the negative.
    g0_1 = torch.autograd.grad(v0_surr, params1, create_graph=True, retain_graph=True)
    g1_0 = torch.autograd.grad(v1_surr, params0, create_graph=True, retain_graph=True)

    peer_l0 = -peer_coef * inner_lr * sum((g0 * g1).sum() for g0, g1 in zip(g0_1, g1_1))
    peer_l1 = -peer_coef * inner_lr * sum((g1 * g0).sum() for g1, g0 in zip(g1_0, g0_0))

    if method == "lola_style":
        return l0 + peer_l0, l1 + peer_l1

    if method == "meta_mapg":
        return l0 + own_l0 + peer_l0, l1 + own_l1 + peer_l1

    raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_seed(method, seed, n_steps=260, batch_size=384,
                   peer_coef=1.5, own_coef=0.35, lr=0.9, lr_power=0.24,
                   log_every=10, inner_lr=0.1):
    """
    Train one seed. Returns dict with metrics.
    """
    actual_seed = 1000 + 37 * seed
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)

    policy0 = MLPPolicy()
    policy1 = MLPPolicy()

    # Verify initial coop prob ≈ 0.5
    with torch.no_grad():
        start_state = state_onehot(4).unsqueeze(0)
        init_coop0 = torch.sigmoid(policy0(start_state)).item()
        init_coop1 = torch.sigmoid(policy1(start_state)).item()

    coop_history = []

    for step in range(1, n_steps + 1):
        # Learning rate schedule: lr / step^lr_power
        current_lr = lr / (step ** lr_power)

        # 1. Sample trajectories
        with torch.no_grad():
            trajs = rollout_batch(policy0, policy1, batch_size)

        # 2. Compute DiCE surrogate losses
        loss0, loss1 = compute_losses(
            policy0, policy1, trajs, method,
            peer_coef=peer_coef, own_coef=own_coef, inner_lr=inner_lr
        )

        # 3. Compute gradients
        policy0.zero_grad()
        policy1.zero_grad()

        # We need both losses' gradients; backward them separately
        loss0.backward(retain_graph=True)
        # Save policy0 grads, zero policy1's spurious grads
        grads0 = {name: p.grad.clone() for name, p in policy0.named_parameters()
                  if p.grad is not None}
        policy0.zero_grad()
        policy1.zero_grad()

        loss1.backward()
        grads1 = {name: p.grad.clone() for name, p in policy1.named_parameters()
                  if p.grad is not None}

        # 4. Apply SGD step with scheduled LR
        with torch.no_grad():
            for name, p in policy0.named_parameters():
                if name in grads0:
                    p -= current_lr * grads0[name]
            for name, p in policy1.named_parameters():
                if name in grads1:
                    p -= current_lr * grads1[name]

        # 5. Log cooperation rate at start state
        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                start_state = state_onehot(4).unsqueeze(0)
                coop0 = torch.sigmoid(policy0(start_state)).item()
                coop1 = torch.sigmoid(policy1(start_state)).item()
                coop_history.append((step, coop0, coop1))

    # Final metrics
    final_coop0 = coop_history[-1][1]
    final_coop1 = coop_history[-1][2]
    final_coop = (final_coop0 + final_coop1) / 2.0
    success = final_coop >= SUCCESS_THRESHOLD

    # Compute final return via rollout
    with torch.no_grad():
        eval_trajs = rollout_batch(policy0, policy1, 1000)
        final_return = eval_trajs["returns"].mean(dim=0)  # (2,)

    # Time to convergence: first step where coop >= threshold and stays
    time_to_conv = n_steps
    for i, (s, c0, c1) in enumerate(coop_history):
        avg = (c0 + c1) / 2.0
        if avg >= SUCCESS_THRESHOLD:
            # Check if it stays
            stays = all((coop_history[j][1] + coop_history[j][2]) / 2.0 >= SUCCESS_THRESHOLD * 0.95
                        for j in range(i, len(coop_history)))
            if stays:
                time_to_conv = s
                break

    return {
        "method": method,
        "seed": seed,
        "actual_seed": actual_seed,
        "final_coop": final_coop,
        "final_coop0": final_coop0,
        "final_coop1": final_coop1,
        "final_return0": final_return[0].item(),
        "final_return1": final_return[1].item(),
        "success": success,
        "time_to_conv": time_to_conv,
        "init_coop0": init_coop0,
        "init_coop1": init_coop1,
        "coop_history": coop_history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MLP IPD ablation experiment")
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--n_steps", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.9)
    parser.add_argument("--lr_power", type=float, default=0.24)
    parser.add_argument("--peer_coef", type=float, default=1.5)
    parser.add_argument("--own_coef", type=float, default=0.35)
    parser.add_argument("--out_dir", type=str, default="artifacts/mlp")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["standard_pg", "meta_pg", "lola_style", "meta_mapg"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"MLP IPD Ablation: {args.n_seeds} seeds × {len(args.methods)} methods")
    print(f"  Steps: {args.n_steps}, Batch: {args.batch_size}")
    print(f"  LR: {args.lr}, LR power: {args.lr_power}")
    print(f"  Peer coef: {args.peer_coef}, Own coef: {args.own_coef}")
    print(f"  Seed schedule: 1000 + 37*i")
    print()

    all_results = []
    t0 = time.time()

    for method in args.methods:
        print(f"--- {method} ---")
        method_results = []
        for seed in range(args.n_seeds):
            result = train_one_seed(
                method=method,
                seed=seed,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                peer_coef=args.peer_coef,
                own_coef=args.own_coef,
                lr=args.lr,
                lr_power=args.lr_power,
            )
            method_results.append(result)
            all_results.append(result)
            if (seed + 1) % 10 == 0:
                successes = sum(1 for r in method_results if r["success"])
                print(f"  seed {seed+1}/{args.n_seeds}: "
                      f"{successes}/{len(method_results)} success so far "
                      f"(last coop={result['final_coop']:.3f})")

        successes = sum(1 for r in method_results if r["success"])
        coops = [r["final_coop"] for r in method_results]
        print(f"  DONE: {successes}/{args.n_seeds} ({100*successes/args.n_seeds:.0f}%) | "
              f"mean coop = {np.mean(coops):.3f} ± {np.std(coops):.3f}")
        print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save CSV ──
    csv_path = os.path.join(args.out_dir, "mlp_ipd_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "seed", "final_coop", "final_return",
            "success", "time_to_conv"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "method": r["method"],
                "seed": r["seed"],
                "final_coop": f"{r['final_coop']:.4f}",
                "final_return": f"{(r['final_return0'] + r['final_return1'])/2:.4f}",
                "success": int(r["success"]),
                "time_to_conv": r["time_to_conv"],
            })
    print(f"Saved: {csv_path}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"{'Method':>15s} | {'Success':>8s} | {'95% CI':>14s} | {'Mean Coop':>10s} | {'Mean Ret':>10s}")
    print("-" * 70)
    for method in args.methods:
        mrs = [r for r in all_results if r["method"] == method]
        n = len(mrs)
        succ = sum(1 for r in mrs if r["success"])
        rate = succ / n
        # Wilson score CI
        z = 1.96
        denom = 1 + z**2/n
        center = (rate + z**2/(2*n)) / denom
        margin = z * np.sqrt((rate*(1-rate) + z**2/(4*n))/n) / denom
        ci_lo = max(0, center - margin)
        ci_hi = min(1, center + margin)

        mean_coop = np.mean([r["final_coop"] for r in mrs])
        mean_ret = np.mean([(r["final_return0"]+r["final_return1"])/2 for r in mrs])
        label = {"standard_pg": "PG", "meta_pg": "Meta-PG",
                 "lola_style": "LOLA", "meta_mapg": "Meta-MAPG"}[method]
        print(f"{label:>15s} | {100*rate:6.0f}%  | [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%] | "
              f"{mean_coop:9.3f}  | {mean_ret:9.2f}")

    # ── Generate bar plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = []
        rates = []
        ci_los = []
        ci_his = []
        colors = ["#9B9B9B", "#4878CF", "#F4A261", "#E84855"]

        for i, method in enumerate(args.methods):
            mrs = [r for r in all_results if r["method"] == method]
            n = len(mrs)
            succ = sum(1 for r in mrs if r["success"])
            rate = succ / n
            z = 1.96
            denom = 1 + z**2/n
            center = (rate + z**2/(2*n)) / denom
            margin = z * np.sqrt((rate*(1-rate) + z**2/(4*n))/n) / denom

            label = {"standard_pg": "PG", "meta_pg": "Meta-PG",
                     "lola_style": "LOLA\n(peer-only)",
                     "meta_mapg": "Meta-MAPG\n(full)"}[method]
            labels.append(label)
            rates.append(100 * rate)
            ci_los.append(100 * max(0, rate - (center - max(0, center - margin))))
            ci_his.append(100 * (min(1, center + margin) - rate))

        bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.5,
                      width=0.6, alpha=0.85)
        ax.errorbar(range(len(labels)), rates,
                    yerr=[ci_los, ci_his],
                    fmt="none", ecolor="black", capsize=5, linewidth=1.5)

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel("Cooperative Success Rate (%)", fontsize=12)
        ax.set_title("Non-tabular IPD: MLP Policy (2×16 tanh), 30 seeds", fontsize=13)
        ax.set_ylim(0, 110)
        ax.axhline(y=SUCCESS_THRESHOLD*100, color="gray", linestyle="--", alpha=0.5,
                   label=f"Threshold ({SUCCESS_THRESHOLD*100:.0f}%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "mlp_ipd.pdf")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
