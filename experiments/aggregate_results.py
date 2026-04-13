"""
Aggregate results from parallel single-seed runs into combined JSON.

Usage:
    python -m experiments.aggregate_results --env ipd --seeds 30
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
METHODS = [
    "reinforce", "meta_pg", "lola_dice", "meta_mapg",
    "ew_pg", "lola_pg", "ew_lola_pg",
]


def aggregate(env_name: str, n_seeds: int, methods: list):
    all_results = {}
    for method in methods:
        method_results = []
        for seed in range(n_seeds):
            path = RESULTS_DIR / f"{env_name}_{method}_seed{seed}.json"
            if not path.exists():
                print(f"  MISSING: {path}")
                continue
            with open(path) as f:
                result = json.load(f)
            method_results.append(result)
        if method_results:
            all_results[method] = method_results
            print(f"  {method}: {len(method_results)}/{n_seeds} seeds")
        else:
            print(f"  {method}: NO RESULTS")

    out_path = RESULTS_DIR / f"{env_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["ipd", "rps"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--methods", default="all")
    args = parser.parse_args()

    if args.methods == "all":
        methods = METHODS
    else:
        methods = args.methods.split(",")

    aggregate(args.env, args.seeds, methods)


if __name__ == "__main__":
    main()
