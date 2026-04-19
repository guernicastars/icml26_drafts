"""Thin wrapper around v1 harvest_activations with trust_remote_code=True."""
from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def harvest(
    model_name: str,
    device: str = "cuda",
    max_tokens: int = 100_000,
    seq_len: int = 1024,
    batch_size: int = 4,
) -> tuple[np.ndarray, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    n_seqs = len(tokens) // seq_len
    input_ids = torch.tensor(tokens[:n_seqs * seq_len]).reshape(n_seqs, seq_len)

    all_hidden, total_loss, total_tokens = [], 0.0, 0
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), batch_size),
                      desc=f"harvest {model_name.split('/')[-1]}"):
            batch = input_ids[i:i + batch_size].to(device)
            out = model(batch, output_hidden_states=True, labels=batch)
            h = out.hidden_states[-1].float().cpu().numpy().reshape(-1, out.hidden_states[-1].shape[-1])
            all_hidden.append(h)
            total_loss += out.loss.item() * batch.numel()
            total_tokens += batch.numel()

    del model
    torch.cuda.empty_cache()

    H = np.concatenate(all_hidden, axis=0)
    avg_loss = total_loss / max(total_tokens, 1)
    logger.info("%s  n=%d  d=%d  val_loss=%.4f", model_name.split("/")[-1],
                H.shape[0], H.shape[1], avg_loss)
    return H, avg_loss
