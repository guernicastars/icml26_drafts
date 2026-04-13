"""
Main experiment runner for EW-LOLA-PG replication.

Usage:
    python -m experiments.run_experiment --env ipd --methods all --seeds 10
    python -m experiments.run_experiment --env rps --methods ew_lola_pg,meta_mapg --seeds 5
"""

import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path

from experiments.envs.iterated_matrix import (
    make_ipd, make_rps,
    generate_ipd_personas, generate_rps_personas,
    split_personas,
)
from experiments.agents.policy import LSTMPolicy
from experiments.agents.value import LSTMValue
from experiments.training.meta_learner import (
    MetaConfig, meta_train, meta_test,
)

RESULTS_DIR = Path(__file__).parent / "results"
METHODS = [
    "reinforce",
    "meta_pg",
    "lola_dice",
    "meta_mapg",
    "ew_pg",
    "lola_pg",
    "ew_lola_pg",
]


def load_config(env_name: str) -> dict:
    config_path = Path(__file__).parent / "configs" / f"{env_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_name: str):
    if env_name == "ipd":
        return make_ipd()
    elif env_name == "rps":
        return make_rps()
    raise ValueError(f"Unknown env: {env_name}")


def make_personas(env_name: str, cfg: dict, rng: np.random.Generator):
    n_train = cfg["n_personas_train"]
    n_val = cfg["n_personas_val"]
    n_test = cfg["n_personas_test"]
    n_total = n_train + n_val + n_test

    if env_name == "ipd":
        personas = generate_ipd_personas(n_total, rng)
    elif env_name == "rps":
        personas = generate_rps_personas(n_total, rng)
    else:
        raise ValueError(f"Unknown env: {env_name}")

    return split_personas(personas, n_train, n_val, n_test)


def run_single_seed(env_name: str, method: str, seed: int,
                    cfg: dict) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = make_env(env_name)
    train_personas, val_personas, test_personas = make_personas(env_name, cfg, rng)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MetaConfig(
        horizon=cfg["horizon"],
        chain_length=cfg["chain_length"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        lr_inner=cfg["lr_inner"],
        lr_outer_actor=cfg["lr_outer_actor"],
        lr_outer_critic=cfg["lr_outer_critic"],
        n_meta_steps=cfg["n_meta_steps"],
        eval_interval=cfg["eval_interval"],
        device=device,
        evidence_alpha=cfg.get("evidence_alpha", 0.99),
        lola_lambda_init=cfg.get("lola_lambda_init", 0.1),
        lola_anneal_rate=cfg.get("lola_anneal_rate", 0.5),
        evidence_w_min=cfg.get("evidence_w_min", 0.01),
    )

    policy = LSTMPolicy(env.obs_dim, env.n_actions_1)
    value_net = LSTMValue(env.obs_dim)

    print(f"  [{method}] seed={seed} training on {device}...")

    train_rewards = meta_train(
        policy, value_net, env, train_personas, config, rng,
        method=method,
    )

    print(f"  [{method}] seed={seed} evaluating...")
    test_results = meta_test(policy, value_net, env, test_personas, config)

    return {
        "method": method,
        "seed": seed,
        "env": env_name,
        "train_rewards": train_rewards,
        "test_mean": test_results["mean"].tolist(),
        "test_std": test_results["std"].tolist(),
        "test_ci_95": test_results["ci_95"].tolist(),
    }


def run_experiment(env_name: str, methods: list, n_seeds: int):
    cfg = load_config(env_name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method in methods:
        print(f"\n=== {method.upper()} on {env_name.upper()} ===")
        method_results = []
        for seed in range(n_seeds):
            result = run_single_seed(env_name, method, seed, cfg)
            method_results.append(result)
        all_results[method] = method_results

    out_path = RESULTS_DIR / f"{env_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def run_single(env_name: str, method: str, seed: int):
    cfg = load_config(env_name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_single_seed(env_name, method, seed, cfg)
    out_path = RESULTS_DIR / f"{env_name}_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["ipd", "rps"])
    parser.add_argument("--methods", default="all",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Offset for seed numbering (for parallel runs)")
    parser.add_argument("--single-seed", type=int, default=None,
                        help="Run a single seed (for GPU parallelism)")
    parser.add_argument("--single-method", default=None,
                        help="Run a single method (for GPU parallelism)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id to use")
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.methods == "all":
        methods = METHODS
    else:
        methods = args.methods.split(",")
        for m in methods:
            if m not in METHODS:
                raise ValueError(f"Unknown method: {m}. Choose from {METHODS}")

    if args.single_seed is not None and args.single_method is not None:
        run_single(args.env, args.single_method, args.single_seed)
    elif args.single_seed is not None:
        for method in methods:
            run_single(args.env, method, args.single_seed)
    else:
        seed_start = args.seed_offset
        seed_end = seed_start + args.seeds
        cfg = load_config(args.env)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        all_results = {}
        for method in methods:
            print(f"\n=== {method.upper()} on {args.env.upper()} ===")
            method_results = []
            for seed in range(seed_start, seed_end):
                result = run_single_seed(args.env, method, seed, cfg)
                method_results.append(result)
            all_results[method] = method_results
        suffix = f"_offset{seed_start}" if seed_start > 0 else ""
        out_path = RESULTS_DIR / f"{args.env}_results{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
