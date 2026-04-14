from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .reward_models import RewardModelPair


@dataclass
class BoNResult:
    n: int
    gold_reward_mean: float
    gold_reward_std: float
    proxy_reward_mean: float
    proxy_reward_std: float
    overopt_gap: float


def generate_candidates(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> list[str]:
    if device is None:
        device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_length = encoded["input_ids"].shape[1]
    candidates = []
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
            )
        text = tokenizer.decode(gen[0][input_length:], skip_special_tokens=True)
        candidates.append(text)
    return candidates


def best_of_n_evaluate(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    reward_pair: RewardModelPair,
    n_values: list[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    base_gold_scores: list[float] | None = None,
    device: torch.device | None = None,
) -> list[BoNResult]:
    if device is None:
        device = next(model.parameters()).device

    max_n = max(n_values)
    all_candidates: list[list[str]] = []
    for prompt in prompts:
        candidates = generate_candidates(
            model, tokenizer, prompt, max_n,
            max_new_tokens=max_new_tokens, temperature=temperature, device=device,
        )
        all_candidates.append(candidates)

    results = []
    for n in sorted(n_values):
        gold_scores_per_prompt = []
        proxy_scores_per_prompt = []

        for prompt_idx, prompt in enumerate(prompts):
            candidates = all_candidates[prompt_idx][:n]
            proxy_scores = reward_pair.proxy.score(
                [prompt] * len(candidates), candidates,
            )
            best_idx = int(np.argmax(proxy_scores))
            gold_score = reward_pair.gold.score([prompt], [candidates[best_idx]])[0]
            gold_scores_per_prompt.append(gold_score)
            proxy_scores_per_prompt.append(proxy_scores[best_idx])

        gold_mean = float(np.mean(gold_scores_per_prompt))
        proxy_mean = float(np.mean(proxy_scores_per_prompt))

        results.append(BoNResult(
            n=n,
            gold_reward_mean=gold_mean,
            gold_reward_std=float(np.std(gold_scores_per_prompt)),
            proxy_reward_mean=proxy_mean,
            proxy_reward_std=float(np.std(proxy_scores_per_prompt)),
            overopt_gap=proxy_mean - gold_mean,
        ))

    return results
