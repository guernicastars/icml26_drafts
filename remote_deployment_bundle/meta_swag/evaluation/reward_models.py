from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RewardModelWrapper:
    model_name: str
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    device: torch.device
    max_length: int = 1024

    @classmethod
    def load(
        cls,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 1024,
    ) -> RewardModelWrapper:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype,
        ).eval().to(device)
        return cls(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str], batch_size: int = 8) -> list[float]:
        scores = []
        for i in range(0, len(prompts), batch_size):
            batch_texts = [p + r for p, r in zip(prompts[i:i + batch_size], responses[i:i + batch_size])]
            encoded = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            batch_scores = logits.squeeze(-1).float().cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
        return scores


@dataclass
class RewardModelPair:
    gold: RewardModelWrapper
    proxy: RewardModelWrapper

    @classmethod
    def load(
        cls,
        gold_name: str,
        proxy_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> RewardModelPair:
        gold = RewardModelWrapper.load(gold_name, device, dtype)
        proxy = RewardModelWrapper.load(proxy_name, device, dtype)
        return cls(gold=gold, proxy=proxy)
