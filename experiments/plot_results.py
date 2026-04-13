"""
Publication-quality plotting for ICML 2026 NExT-Game workshop paper.

Generates:
- Adaptation curves (mean + 95% CI) per environment
- AUC comparison table
- Ablation study figure

Usage:
    python -m experiments.plot_results --env ipd
    python -m experiments.plot_results --env rps
    python -m experiments.plot_results --env all
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

COLORS = {
    "reinforce": "#bdc3c7",
    "meta_pg": "#3498db",
    "lola_dice": "#9b59b6",
    "meta_mapg": "#e74c3c",
    "ew_pg": "#2ecc71",
    "lola_pg": "#1abc9c",
    "ew_lola_pg": "#f39c12",
}

LABELS = {
    "reinforce": "REINFORCE",
    "meta_pg": "Meta-PG",
    "lola_dice": "LOLA-DiCE",
    "meta_mapg": "Meta-MAPG",
    "ew_pg": "EW-PG",
    "lola_pg": "LOLA-PG",
    "ew_lola_pg": "EW-LOLA-PG (Ours)",
}

PLOT_ORDER = [
    "reinforce", "meta_pg", "lola_dice", "meta_mapg",
    "ew_pg", "lola_pg", "ew_lola_pg",
]


def load_results(env_name: str) -> dict:
    path = RESULTS_DIR / f"{env_name}_results.json"
    with open(path) as f:
        return json.load(f)


def plot_adaptation_curves(results: dict, env_name: str, title: str,
                           out_path: Path):
    """Plot adaptation performance during meta-testing."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for method in PLOT_ORDER:
        if method not in results:
            continue
        seeds = results[method]
        test_means = np.array([s["test_mean"] for s in seeds])

        mean = test_means.mean(axis=0)
        std = test_means.std(axis=0)
        ci = 1.96 * std / np.sqrt(len(seeds))
        steps = np.arange(1, len(mean) + 1)

        ax.plot(steps, mean, color=COLORS[method], label=LABELS[method],
                linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci,
                        color=COLORS[method], alpha=0.15)

    ax.set_xlabel("Policy Update Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ablation(results: dict, env_name: str, out_path: Path):
    """Ablation: EW-PG vs LOLA-PG vs EW-LOLA-PG."""
    ablation_methods = ["meta_mapg", "ew_pg", "lola_pg", "ew_lola_pg"]
    available = [m for m in ablation_methods if m in results]

    if len(available) < 2:
        print("  Skipping ablation: not enough methods")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for method in available:
        seeds = results[method]
        test_means = np.array([s["test_mean"] for s in seeds])
        mean = test_means.mean(axis=0)
        std = test_means.std(axis=0)
        ci = 1.96 * std / np.sqrt(len(seeds))
        steps = np.arange(1, len(mean) + 1)

        ax.plot(steps, mean, color=COLORS[method], label=LABELS[method],
                linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci,
                        color=COLORS[method], alpha=0.15)

    ax.set_xlabel("Policy Update Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(f"Ablation Study ({env_name.upper()})", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_auc_table(results: dict, env_name: str):
    """Print AUC comparison table."""
    print(f"\n{'='*50}")
    print(f"  AUC Table: {env_name.upper()}")
    print(f"{'='*50}")
    print(f"  {'Method':<20} {'AUC':>10} {'CI 95%':>12}")
    print(f"  {'-'*42}")

    for method in PLOT_ORDER:
        if method not in results:
            continue
        seeds = results[method]
        test_means = np.array([s["test_mean"] for s in seeds])
        aucs = test_means.sum(axis=1)
        mean_auc = aucs.mean()
        ci = 1.96 * aucs.std() / np.sqrt(len(seeds))
        print(f"  {LABELS[method]:<20} {mean_auc:>10.2f} +/- {ci:>8.2f}")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all", choices=["ipd", "rps", "all"])
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    envs = ["ipd", "rps"] if args.env == "all" else [args.env]

    titles = {
        "ipd": "IPD: Adaptation Performance (Mixed Incentive)",
        "rps": "RPS: Adaptation Performance (Competitive)",
    }

    for env_name in envs:
        results_path = RESULTS_DIR / f"{env_name}_results.json"
        if not results_path.exists():
            print(f"No results for {env_name}, skipping")
            continue

        print(f"\nPlotting {env_name.upper()}...")
        results = load_results(env_name)

        plot_adaptation_curves(
            results, env_name, titles[env_name],
            FIGURES_DIR / f"{env_name}_adaptation.pdf",
        )
        plot_ablation(
            results, env_name,
            FIGURES_DIR / f"{env_name}_ablation.pdf",
        )
        print_auc_table(results, env_name)


if __name__ == "__main__":
    main()
