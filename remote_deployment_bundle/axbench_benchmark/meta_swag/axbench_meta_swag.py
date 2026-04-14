from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from meta_swag.posterior.base import AggregatedAdapterResult
    from meta_swag.posterior.meta_swag import aggregate_adapter_checkpoints
    from meta_swag.training.retention import build_retention_schedule
    from meta_swag.adapters.state import AdapterStateManifest, build_manifest, flatten_adapter_state
except ImportError:
    from .adapter_posterior import AggregatedAdapterResult, aggregate_adapter_checkpoints, build_retention_schedule
    from .adapter_state import AdapterStateManifest, build_manifest, flatten_adapter_state


@dataclass
class RetainedCheckpoint:
    checkpoint_id: str
    step: int
    epoch: int
    train_loss: float
    adapter_vector: np.ndarray
    adapter_dimension: int
    selected_factor: float | None = None
    weighting_metric: float | None = None
    validation_factor_sweep: list[dict[str, float]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "selected_factor": self.selected_factor,
            "weighting_metric": self.weighting_metric,
            "adapter_dimension": self.adapter_dimension,
            "validation_factor_sweep": self.validation_factor_sweep,
        }


@dataclass
class FinalMethodResult:
    scheme: str
    selected_factor: float
    validation_composite: float
    test_composite: float
    concept_relevance: float
    instruction_relevance: float
    fluency: float
    perplexity: float | None
    delta_over_unsteered: float
    diagnostics: dict[str, float]


def distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def harmonic_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def split_validation_test(df, validation_ratio: float = 0.5):
    validation_ratio = float(np.clip(validation_ratio, 0.0, 1.0))
    if "input_id" not in df.columns:
        raise KeyError("Expected an input_id column when splitting steering data.")
    boundary = int(round((df["input_id"].max() + 1) * validation_ratio))
    validation_df = df[df["input_id"] < boundary].copy()
    test_df = df[df["input_id"] >= boundary].copy()
    if validation_df.empty:
        validation_df = df.copy()
    if test_df.empty:
        test_df = df.copy()
    return validation_df, test_df


def choose_factor_from_factor_sweep(factor_rows: list[dict[str, float]]) -> tuple[float, float]:
    if not factor_rows:
        raise ValueError("factor_rows cannot be empty.")

    def sort_key(row: dict[str, float]) -> tuple[float, float, float, float]:
        perplexity = row.get("perplexity")
        perplexity_value = float(perplexity) if perplexity is not None and not np.isnan(perplexity) else float("inf")
        return (
            float(row.get("composite", 0.0)),
            float(row.get("instruction_relevance", 0.0)),
            float(row.get("fluency", 0.0)),
            -perplexity_value,
        )

    best_row = max(
        factor_rows,
        key=sort_key,
    )
    return float(best_row["factor"]), float(best_row["composite"])


def weighting_metric_from_row(row: dict[str, float]) -> float:
    return harmonic_mean(
        [
            float(row.get("instruction_relevance", 0.0)),
            float(row.get("fluency", 0.0)),
        ]
    )


def attach_validation_metrics(record: RetainedCheckpoint, factor_rows: list[dict[str, float]]) -> RetainedCheckpoint:
    selected_factor, _ = choose_factor_from_factor_sweep(factor_rows)
    chosen_row = next(row for row in factor_rows if float(row["factor"]) == float(selected_factor))
    record.selected_factor = selected_factor
    record.weighting_metric = weighting_metric_from_row(chosen_row)
    record.validation_factor_sweep = factor_rows
    return record


def aggregate_checkpoint_records(
    records: list[RetainedCheckpoint],
    scheme: str,
    beta: float = 1.0,
    threshold_quantile: float = 0.75,
    low_rank_rank: int | None = None,
) -> AggregatedAdapterResult:
    if not records:
        raise ValueError("At least one retained checkpoint is required.")
    scores = np.asarray(
        [
            record.weighting_metric if record.weighting_metric is not None else 0.0
            for record in records
        ],
        dtype=np.float32,
    )
    checkpoints = np.stack([record.adapter_vector for record in records], axis=0).astype(np.float32)
    target_ess = max(8, int(math.ceil(len(records) / 2)))
    return aggregate_adapter_checkpoints(
        checkpoints=checkpoints,
        scores=scores,
        scheme=scheme,
        beta=beta,
        target_ess=target_ess,
        threshold_quantile=threshold_quantile,
        low_rank_rank=low_rank_rank,
    )


def _capture_checkpoint(
    records: list[RetainedCheckpoint],
    checkpoint_id_prefix: str,
    step: int,
    epoch: int,
    train_loss: float,
    module: torch.nn.Module,
    manifest: AdapterStateManifest,
) -> None:
    flat_vector, _ = flatten_adapter_state(module, manifest)
    records.append(
        RetainedCheckpoint(
            checkpoint_id=f"{checkpoint_id_prefix}_step_{step:05d}",
            step=step,
            epoch=epoch,
            train_loss=float(train_loss),
            adapter_vector=flat_vector,
            adapter_dimension=int(flat_vector.size),
        )
    )


def train_lora_with_retention(
    model,
    examples,
    keep_last: int,
    tail_fraction: float,
    checkpoint_id_prefix: str,
    **kwargs,
) -> tuple[list[RetainedCheckpoint], AdapterStateManifest]:
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    train_dataloader = model.make_dataloader(examples, **kwargs)
    optimizer = torch.optim.AdamW(
        model.ax_model.parameters(),
        lr=model.training_args.lr,
        weight_decay=model.training_args.weight_decay,
    )
    grad_accum = max(1, int(model.training_args.gradient_accumulation_steps))
    num_training_steps = max(1, model.training_args.n_epochs * math.ceil(len(train_dataloader) / grad_accum))
    retention_steps = set(build_retention_schedule(num_training_steps, keep_last, tail_fraction))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    manifest = build_manifest(model.ax_model)
    retained: list[RetainedCheckpoint] = []
    rank = distributed_rank()
    progress = tqdm(range(num_training_steps), position=rank, leave=True)
    current_step = 0

    for epoch in range(model.training_args.n_epochs):
        for batch_index, batch in enumerate(train_dataloader):
            inputs = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model.ax_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = outputs.loss.mean()
            (loss / grad_accum).backward()

            should_step = (batch_index + 1) % grad_accum == 0 or (batch_index + 1) == len(train_dataloader)
            if not should_step:
                continue

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_step += 1
            progress.update(1)
            progress.set_description(f"lr {optimizer.param_groups[0]['lr']:.6f} || loss {float(loss):.6f}")

            if current_step in retention_steps:
                _capture_checkpoint(
                    retained,
                    checkpoint_id_prefix=checkpoint_id_prefix,
                    step=current_step,
                    epoch=epoch,
                    train_loss=float(loss.detach().cpu()),
                    module=model.ax_model,
                    manifest=manifest,
                )
    progress.close()
    return retained, manifest


def _preference_batch_metrics(
    model,
    chosen_logps,
    rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    chosen_rewards,
    rejected_rewards,
    losses,
):
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    return {
        "rewards_train/steer_accuracies": reward_accuracies.mean().cpu().numpy().tolist(),
        "loss/steer": losses.mean().detach().cpu().float().numpy().tolist(),
        "logps_train/steer_chosen": chosen_logps.detach().mean().cpu().float().numpy().tolist(),
        "logps_train/steer_rejected": rejected_logps.detach().mean().cpu().float().numpy().tolist(),
    }


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return (per_token_logps * loss_mask).sum(-1)


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    gemma: float,
    simpo_scaler: float,
    winning_lens: torch.LongTensor,
    losing_lens: torch.LongTensor,
    label_smoothing: float = 0.0,
    loss_type: str = "dpo",
    reference_free: bool = False,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps
    if reference_free:
        ref_logratios = 0
    logits = pi_logratios - ref_logratios

    if loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpo":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "simpo":
        losses = -F.logsigmoid((beta / winning_lens) * policy_chosen_logps - (beta / losing_lens) * policy_rejected_logps - gemma)
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(ref_logratios_reverse * simpo_scaler, torch.ones_like(ref_logratios_reverse)) / winning_lens
        ) * policy_chosen_logps
        scaled_policy_rejected_logps = (1.0 / losing_lens) * policy_rejected_logps
        losses = -F.logsigmoid(scaled_policy_chosen_logps - scaled_policy_rejected_logps)
    elif loss_type == "apo_zero":
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * chosen_logratios) + F.logsigmoid(beta * rejected_logratios)
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def train_preference_lora_with_retention(
    model,
    examples,
    keep_last: int,
    tail_fraction: float,
    checkpoint_id_prefix: str,
    **kwargs,
) -> tuple[list[RetainedCheckpoint], AdapterStateManifest]:
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    train_dataloader = model.make_preference_dataloader(examples, **kwargs)
    optimizer = torch.optim.AdamW(
        model.ax_model.parameters(),
        lr=model.training_args.lr,
        weight_decay=model.training_args.weight_decay,
    )
    grad_accum = max(1, int(model.training_args.gradient_accumulation_steps))
    num_training_steps = max(1, model.training_args.n_epochs * math.ceil(len(train_dataloader) / grad_accum))
    retention_steps = set(build_retention_schedule(num_training_steps, keep_last, tail_fraction))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    manifest = build_manifest(model.ax_model)
    retained: list[RetainedCheckpoint] = []
    rank = distributed_rank()
    progress = tqdm(range(num_training_steps), position=rank, leave=True)
    current_step = 0

    for epoch in range(model.training_args.n_epochs):
        for batch_index, batch in enumerate(train_dataloader):
            expanded_batch_size = model.training_args.batch_size * len(model.preference_pairs)
            minibatch_size = model.training_args.batch_size
            num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size

            winning_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "intervention_locations": [],
                "steering_factors": [],
            }
            losing_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "intervention_locations": [],
                "steering_factors": [],
            }

            for i in range(model.training_args.batch_size):
                for pair in model.preference_pairs:
                    winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                    winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                    winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                    winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])

                    losing_inputs["input_ids"].append(batch[f"{pair}_losing_input_ids"][i])
                    losing_inputs["attention_mask"].append(batch[f"{pair}_losing_attention_mask"][i])
                    losing_inputs["labels"].append(batch[f"{pair}_losing_labels"][i])
                    losing_inputs["intervention_locations"].append(batch[f"{pair}_losing_intervention_locations"][i])

                    if "_add" in pair:
                        winning_inputs["steering_factors"].append(torch.tensor(np.random.choice(model.training_args.steering_factors)))
                        losing_inputs["steering_factors"].append(torch.tensor(np.random.choice(model.training_args.steering_factors)))
                    else:
                        if model.training_args.substraction_type == "null_it_out":
                            winning_inputs["steering_factors"].append(torch.tensor(0.0))
                            losing_inputs["steering_factors"].append(torch.tensor(0.0))
                        else:
                            sampled = float(np.random.choice(model.training_args.steering_factors))
                            winning_inputs["steering_factors"].append(torch.tensor(-sampled))
                            losing_inputs["steering_factors"].append(torch.tensor(-sampled))

            loss_sum = 0.0
            batch_metrics = {}

            for minibatch_index in range(num_minibatches):
                start_idx = minibatch_index * minibatch_size
                end_idx = min((minibatch_index + 1) * minibatch_size, expanded_batch_size)
                if start_idx >= expanded_batch_size:
                    break

                minibatch_inputs = {
                    key: torch.stack(
                        winning_inputs[key][start_idx:end_idx] + losing_inputs[key][start_idx:end_idx],
                        dim=0,
                    ).to(model.device)
                    for key in winning_inputs
                }

                if isinstance(model.ax, list):
                    unit_locations = {
                        "sources->base": (
                            None,
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist() * len(model.ax),
                        )
                    }
                else:
                    unit_locations = {
                        "sources->base": (
                            None,
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                        )
                    }

                subspaces = [
                    {
                        "k": model.training_args.topk,
                        "steering_factor": minibatch_inputs["steering_factors"],
                    }
                ]
                if isinstance(model.ax, list):
                    subspaces = subspaces * len(model.ax)

                ref_outputs, policy_outputs_orig = model.ax_model(
                    base={
                        "input_ids": minibatch_inputs["input_ids"],
                        "attention_mask": minibatch_inputs["attention_mask"],
                    },
                    unit_locations=unit_locations,
                    output_original_output=True,
                    subspaces=subspaces,
                    use_cache=False,
                )

                policy_outputs_orig_logps = _get_batch_logps(policy_outputs_orig.logits, minibatch_inputs["labels"])
                ref_logps = _get_batch_logps(ref_outputs.logits, minibatch_inputs["labels"])

                minibatch_size_actual = minibatch_inputs["input_ids"].shape[0]
                half = minibatch_size_actual // 2
                steer_chosen_logps = policy_outputs_orig_logps[:half]
                steer_rejected_logps = policy_outputs_orig_logps[half:]
                steer_ref_chosen_logps = ref_logps[:half]
                steer_ref_rejected_logps = ref_logps[half:]

                winning_lens = minibatch_inputs["attention_mask"][:half].sum(dim=-1)
                losing_lens = minibatch_inputs["attention_mask"][half:].sum(dim=-1)
                steer_losses, steer_chosen_rewards, steer_rejected_rewards = preference_loss(
                    steer_chosen_logps,
                    steer_rejected_logps,
                    steer_ref_chosen_logps,
                    steer_ref_rejected_logps,
                    beta=model.training_args.beta,
                    gemma=model.training_args.gemma,
                    simpo_scaler=model.training_args.simpo_scaler,
                    reference_free=model.training_args.reference_free,
                    label_smoothing=model.training_args.label_smoothing,
                    loss_type=model.training_args.loss_type,
                    winning_lens=winning_lens,
                    losing_lens=losing_lens,
                )

                steer_loss = steer_losses.mean()
                normalized_loss = steer_loss / (num_minibatches * grad_accum)
                normalized_loss.backward()
                loss_sum += float(steer_loss.detach()) * (end_idx - start_idx)

                minibatch_metrics = _preference_batch_metrics(
                    model,
                    steer_chosen_logps,
                    steer_rejected_logps,
                    steer_ref_chosen_logps,
                    steer_ref_rejected_logps,
                    steer_chosen_rewards,
                    steer_rejected_rewards,
                    steer_losses,
                )
                for key, value in minibatch_metrics.items():
                    batch_metrics.setdefault(key, []).append(float(value) * (end_idx - start_idx))

            metrics = {
                key: sum(values) / expanded_batch_size
                for key, values in batch_metrics.items()
            }
            loss = loss_sum / expanded_batch_size
            metrics["loss/train"] = loss

            should_step = (batch_index + 1) % grad_accum == 0 or (batch_index + 1) == len(train_dataloader)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.ax_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_step += 1
            progress.update(1)
            progress.set_description(
                "lr %.6f || loss %.6f || steer acc %.6f"
                % (
                    optimizer.param_groups[0]["lr"],
                    float(loss),
                    metrics.get("rewards_train/steer_accuracies", 0.0),
                )
            )

            if current_step in retention_steps:
                _capture_checkpoint(
                    retained,
                    checkpoint_id_prefix=checkpoint_id_prefix,
                    step=current_step,
                    epoch=epoch,
                    train_loss=float(loss),
                    module=model.ax_model,
                    manifest=manifest,
                )

    progress.close()
    return retained, manifest
