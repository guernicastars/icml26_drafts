from __future__ import annotations

import torch
import torch.nn.functional as F


def get_batch_logps(
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
    gemma: float = 0.0,
    simpo_scaler: float = 1.0,
    winning_lens: torch.LongTensor | None = None,
    losing_lens: torch.LongTensor | None = None,
    label_smoothing: float = 0.0,
    loss_type: str = "dpo",
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps
    if reference_free:
        ref_logratios = 0
    logits = pi_logratios - ref_logratios

    if loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpo":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "simpo":
        losses = -F.logsigmoid(
            (beta / winning_lens) * policy_chosen_logps
            - (beta / losing_lens) * policy_rejected_logps
            - gemma
        )
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(ref_logratios_reverse * simpo_scaler, torch.ones_like(ref_logratios_reverse))
            / winning_lens
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
