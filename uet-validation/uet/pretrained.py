from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
]

RTX3060_MODELS = PYTHIA_MODELS[:4]
A100_MODELS = PYTHIA_MODELS

PYTHIA_CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000,
    100000, 120000, 143000,
]


@dataclass
class ModelSpectrum:
    model_name: str
    hidden_dim: int
    n_params: int
    n_tokens_eval: int
    val_loss: float
    eigenvalues: np.ndarray
    d_eff: float
    stable_rank: float


def _load_model_and_tokenizer(
    model_name: str,
    device: str,
    revision: str | None = None,
    dtype: torch.dtype = torch.float16,
):
    kwargs = {}
    if revision is not None:
        kwargs["revision"] = f"step{revision}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        **kwargs,
    )
    if device == "cpu":
        model = model.float()
    model.eval()
    return model, tokenizer


def _tokenize_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    split: str = "test",
    max_tokens: int = 500_000,
    seq_len: int = 1024,
) -> Tensor:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())

    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens]

    n_seqs = len(tokens) // seq_len
    tokens = tokens[: n_seqs * seq_len]
    return torch.tensor(tokens, dtype=torch.long).reshape(n_seqs, seq_len)


@torch.no_grad()
def harvest_activations(
    model_name: str,
    device: str = "cuda",
    max_tokens: int = 500_000,
    seq_len: int = 1024,
    batch_size: int = 8,
    revision: str | None = None,
) -> tuple[np.ndarray, float]:
    model, tokenizer = _load_model_and_tokenizer(model_name, device, revision=revision)
    input_ids = _tokenize_dataset(tokenizer, max_tokens=max_tokens, seq_len=seq_len)

    all_hidden = []
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i : i + batch_size].to(model.device)
        outputs = model(batch, output_hidden_states=True, labels=batch)

        hidden = outputs.hidden_states[-1]
        all_hidden.append(hidden[:, :, :].float().cpu().numpy().reshape(-1, hidden.shape[-1]))

        total_loss += outputs.loss.item() * batch.numel()
        total_tokens += batch.numel()

        if len(all_hidden) % 10 == 0:
            n_collected = sum(h.shape[0] for h in all_hidden)
            logger.info(
                "%s: %d/%d seqs, %d embeddings collected",
                model_name.split("/")[-1], min(i + batch_size, len(input_ids)),
                len(input_ids), n_collected,
            )

    del model
    torch.cuda.empty_cache()

    embeddings = np.concatenate(all_hidden, axis=0)
    avg_loss = total_loss / max(total_tokens, 1)

    logger.info(
        "%s: %d embeddings (d=%d), val_loss=%.4f",
        model_name.split("/")[-1], embeddings.shape[0], embeddings.shape[1], avg_loss,
    )
    return embeddings, avg_loss


def _param_count_from_config(model_name: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    V = getattr(cfg, "vocab_size", 50304)
    d = getattr(cfg, "hidden_size", 768)
    L = getattr(cfg, "num_hidden_layers", 12)
    d_ff = getattr(cfg, "intermediate_size", 4 * d)
    return V * d + L * (4 * d * d + 2 * d * d_ff) + V * d


def compute_model_spectrum(
    model_name: str,
    device: str = "cuda",
    max_tokens: int = 500_000,
    batch_size: int = 8,
    revision: str | None = None,
) -> ModelSpectrum:
    from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank

    embeddings, val_loss = harvest_activations(
        model_name, device=device, max_tokens=max_tokens,
        batch_size=batch_size, revision=revision,
    )

    cov = covariance(embeddings)
    evals = eigenspectrum(cov)

    return ModelSpectrum(
        model_name=model_name,
        hidden_dim=embeddings.shape[1],
        n_params=_param_count_from_config(model_name),
        n_tokens_eval=embeddings.shape[0],
        val_loss=val_loss,
        eigenvalues=evals,
        d_eff=effective_dimension(evals),
        stable_rank=stable_rank(evals),
    )
