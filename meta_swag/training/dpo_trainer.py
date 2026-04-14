from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..adapters.state import AdapterStateManifest, build_manifest, flatten_adapter_state
from .checkpoint import RetainedCheckpoint
from .preference import get_batch_logps, preference_loss
from .retention import build_retention_schedule


class DPODataset(Dataset):
    def __init__(self, data: list[dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["attention_mask"][0].sum().item()

        chosen_labels = chosen_enc["input_ids"][0].clone()
        chosen_labels[:prompt_len] = -100
        chosen_labels[chosen_enc["attention_mask"][0] == 0] = -100

        rejected_labels = rejected_enc["input_ids"][0].clone()
        rejected_labels[:prompt_len] = -100
        rejected_labels[rejected_enc["attention_mask"][0] == 0] = -100

        return {
            "chosen_input_ids": chosen_enc["input_ids"][0],
            "chosen_attention_mask": chosen_enc["attention_mask"][0],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_enc["input_ids"][0],
            "rejected_attention_mask": rejected_enc["attention_mask"][0],
            "rejected_labels": rejected_labels,
        }


def _compute_reference_logps(
    ref_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ref_chosen_logps_all = []
    ref_rejected_logps_all = []
    ref_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            for prefix, storage in [("chosen", ref_chosen_logps_all), ("rejected", ref_rejected_logps_all)]:
                input_ids = batch[f"{prefix}_input_ids"].to(device)
                attention_mask = batch[f"{prefix}_attention_mask"].to(device)
                labels = batch[f"{prefix}_labels"].to(device)
                outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                logps = get_batch_logps(outputs.logits, labels)
                storage.append(logps.cpu())
    return ref_chosen_logps_all, ref_rejected_logps_all


def train_dpo_with_retention(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    train_dataset: DPODataset,
    device: torch.device,
    lr: float = 5e-6,
    beta: float = 0.1,
    n_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 150,
    keep_last: int = 50,
    tail_fraction: float = 0.5,
    loss_type: str = "dpo",
    label_smoothing: float = 0.0,
    checkpoint_id_prefix: str = "dpo",
    cache_ref_logps: bool = True,
) -> tuple[list[RetainedCheckpoint], AdapterStateManifest]:
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = max(1, n_epochs * steps_per_epoch)
    retention_steps = set(build_retention_schedule(num_training_steps, keep_last, tail_fraction))

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    manifest = build_manifest(model)
    retained: list[RetainedCheckpoint] = []

    ref_chosen_cache: list[torch.Tensor] | None = None
    ref_rejected_cache: list[torch.Tensor] | None = None

    if cache_ref_logps:
        ref_chosen_cache, ref_rejected_cache = _compute_reference_logps(ref_model, dataloader, device)

    progress = tqdm(total=num_training_steps, desc="DPO Training")
    current_step = 0

    for epoch in range(n_epochs):
        model.train()
        for batch_index, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            policy_chosen_logps = get_batch_logps(chosen_out.logits, chosen_labels)
            policy_rejected_logps = get_batch_logps(rejected_out.logits, rejected_labels)

            if ref_chosen_cache is not None and ref_rejected_cache is not None:
                ref_chosen_logps = ref_chosen_cache[batch_index % len(ref_chosen_cache)].to(device)
                ref_rejected_logps = ref_rejected_cache[batch_index % len(ref_rejected_cache)].to(device)
            else:
                with torch.no_grad():
                    ref_chosen_out = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                    ref_rejected_out = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                    ref_chosen_logps = get_batch_logps(ref_chosen_out.logits, chosen_labels)
                    ref_rejected_logps = get_batch_logps(ref_rejected_out.logits, rejected_labels)

            losses, _chosen_rewards, _rejected_rewards = preference_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
                beta=beta,
                loss_type=loss_type,
                label_smoothing=label_smoothing,
            )

            loss = losses.mean() / gradient_accumulation_steps
            loss.backward()

            should_step = (batch_index + 1) % gradient_accumulation_steps == 0 or (batch_index + 1) == len(dataloader)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_step += 1
            progress.update(1)
            progress.set_description(
                f"epoch {epoch} | loss {float(losses.mean()):.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

            if current_step in retention_steps:
                flat_vector, _ = flatten_adapter_state(model, manifest)
                retained.append(
                    RetainedCheckpoint(
                        checkpoint_id=f"{checkpoint_id_prefix}_step_{current_step:05d}",
                        step=current_step,
                        epoch=epoch,
                        train_loss=float(losses.mean().detach().cpu()),
                        adapter_vector=flat_vector,
                        adapter_dimension=int(flat_vector.size),
                    )
                )

    progress.close()
    return retained, manifest
