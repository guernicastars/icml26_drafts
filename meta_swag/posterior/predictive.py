from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from ..adapters.state import AdapterStateManifest, restore_adapter_state
from .base import AggregatedAdapterResult


class PosteriorPredictive:
    """Deploy S posterior samples through a model and aggregate outputs.

    For point-estimate schemes (MAP, last_iterate), S=1 and the single
    sample is the mean vector — equivalent to the old restore_aggregated path.

    For stochastic schemes (Meta-SWAG, Laplace-LoRA, etc.), S>1 draws from
    the posterior, merges each into the model, and averages outputs at the
    logit level (true BMA).
    """

    def __init__(
        self,
        result: AggregatedAdapterResult,
        manifest: AdapterStateManifest,
        num_samples: int = 16,
        seed: int = 0,
    ):
        self.result = result
        self.manifest = manifest
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self._is_point_estimate = result.scheme.lower() in ("map", "last_iterate")

    @property
    def effective_num_samples(self) -> int:
        return 1 if self._is_point_estimate else self.num_samples

    def sample_vectors(self) -> np.ndarray:
        if self._is_point_estimate:
            return self.result.mean_vector[None, :]
        return self.result.sample(self.num_samples, self.rng)

    def deploy_iter(self, model: torch.nn.Module) -> Iterator[tuple[int, np.ndarray]]:
        vectors = self.sample_vectors()
        for i, vec in enumerate(vectors):
            restore_adapter_state(model, vec, self.manifest)
            yield i, vec

    @torch.no_grad()
    def average_logits(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        accumulated = None
        n = self.effective_num_samples
        for idx, _vec in self.deploy_iter(model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            if accumulated is None:
                accumulated = logits
            else:
                accumulated = accumulated + logits
        return accumulated / n

    @torch.no_grad()
    def average_softmax(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        accumulated = None
        n = self.effective_num_samples
        for idx, _vec in self.deploy_iter(model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs.logits.float().softmax(dim=-1)
            if accumulated is None:
                accumulated = probs
            else:
                accumulated = accumulated + probs
        return accumulated / n

    @torch.no_grad()
    def sample_generations(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> list[list[torch.Tensor]]:
        all_generations = []
        for idx, _vec in self.deploy_iter(model):
            generations = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            all_generations.append(generations)
        return all_generations

    def compute_predictive_variance(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        logits_list = []
        for idx, _vec in self.deploy_iter(model):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_list.append(outputs.logits.float().cpu())

        stacked = torch.stack(logits_list, dim=0)
        mean_logits = stacked.mean(dim=0)
        within_var = stacked.var(dim=0).mean().item()
        predictive_entropy = -(mean_logits.softmax(-1) * mean_logits.log_softmax(-1)).sum(-1).mean().item()

        return {
            "within_sample_variance": within_var,
            "predictive_entropy": predictive_entropy,
            "num_samples": len(logits_list),
        }
