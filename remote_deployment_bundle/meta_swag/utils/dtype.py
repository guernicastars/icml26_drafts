"""dtype selection that is safe across Volta (V100), Ampere (A100), and Hopper (H100).

Volta (sm_70) does not have native bfloat16; PyTorch emulates it in fp32, which
is both slow and occasionally produces NaNs under long DPO trajectories. Ampere
and newer (sm_80+) run bf16 natively.
"""
from __future__ import annotations

import torch


def supports_bf16(device: torch.device | str | int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    idx = 0 if device is None else torch.device(device).index or 0
    major, _ = torch.cuda.get_device_capability(idx)
    return major >= 8


def autodetect_dtype(device: torch.device | str | int | None = None) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if supports_bf16(device) else torch.float16


def parse_dtype(name: str, device: torch.device | str | int | None = None) -> torch.dtype:
    key = name.lower()
    if key in ("auto", ""):
        return autodetect_dtype(device)
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unknown dtype name: {name!r}. Use auto|fp16|bf16|fp32.")
