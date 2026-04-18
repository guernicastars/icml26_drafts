from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))

logger = logging.getLogger(__name__)


@torch.no_grad()
def harvest_all_layer_activations(
    model_name: str,
    device: str = "cuda",
    max_tokens: int = 200_000,
    seq_len: int = 512,
    batch_size: int = 4,
    revision: str | None = None,
) -> tuple[dict[int, np.ndarray], float]:
    """
    Returns ({layer_idx: (n_tokens, hidden_dim)}, avg_val_loss) for all transformer layers.

    Layer 0 is the embedding layer output; layer L is the final transformer block output.
    """
    from uet.pretrained import _load_model_and_tokenizer, _tokenize_dataset

    model, tokenizer = _load_model_and_tokenizer(model_name, device, revision=revision)
    input_ids = _tokenize_dataset(tokenizer, max_tokens=max_tokens, seq_len=seq_len)

    n_layers = len(model.gpt_neox.layers) + 1  # +1 for embedding output

    layer_hiddens: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    total_loss = 0.0
    total_tokens = 0

    from tqdm import tqdm
    n_batches = (len(input_ids) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(input_ids), batch_size), total=n_batches,
                  desc=f"layers {model_name.split('/')[-1]}", unit="batch"):
        batch = input_ids[i : i + batch_size].to(model.device)
        outputs = model(batch, output_hidden_states=True, labels=batch)

        for layer_idx, hidden in enumerate(outputs.hidden_states):
            layer_hiddens[layer_idx].append(
                hidden.float().cpu().numpy().reshape(-1, hidden.shape[-1])
            )

        total_loss += outputs.loss.item() * batch.numel()
        total_tokens += batch.numel()

    del model
    torch.cuda.empty_cache()

    avg_loss = total_loss / max(total_tokens, 1)
    layer_embeddings = {
        idx: np.concatenate(arrs, axis=0)
        for idx, arrs in layer_hiddens.items()
    }
    logger.info(
        "%s: %d layers harvested, val_loss=%.4f",
        model_name.split("/")[-1], n_layers, avg_loss,
    )
    return layer_embeddings, avg_loss
