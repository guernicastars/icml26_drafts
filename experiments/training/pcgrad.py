"""
Projecting Conflicting Gradients (PCGrad) from Yu et al. (2020).

Used during outer-loop optimization to handle conflicting gradients
from diverse persona batches.
"""

import torch


def pcgrad_update(grads: list[torch.Tensor]) -> torch.Tensor:
    """Project conflicting gradients and return the combined gradient.

    grads: list of gradient tensors (one per task/persona), each same shape
    Returns: combined gradient, same shape
    """
    n = len(grads)
    if n == 1:
        return grads[0]

    projected = [g.clone() for g in grads]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = torch.dot(projected[i].flatten(), grads[j].flatten())
            if dot < 0:
                proj = dot / (torch.dot(grads[j].flatten(), grads[j].flatten()) + 1e-12)
                projected[i] = projected[i] - proj * grads[j]

    return torch.stack(projected).mean(dim=0)
