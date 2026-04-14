from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class AdapterParameterSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


@dataclass(frozen=True)
class AdapterStateManifest:
    parameters: tuple[AdapterParameterSpec, ...]

    @property
    def total_params(self) -> int:
        return sum(spec.numel for spec in self.parameters)


def iter_trainable_parameters(module: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def build_manifest(module: torch.nn.Module) -> AdapterStateManifest:
    specs = []
    for name, parameter in iter_trainable_parameters(module):
        specs.append(
            AdapterParameterSpec(
                name=name,
                shape=tuple(int(dim) for dim in parameter.shape),
                dtype=str(parameter.detach().cpu().dtype).replace("torch.", ""),
                numel=int(parameter.numel()),
            )
        )
    return AdapterStateManifest(tuple(specs))


def flatten_adapter_state(
    module: torch.nn.Module,
    manifest: AdapterStateManifest | None = None,
) -> tuple[np.ndarray, AdapterStateManifest]:
    resolved_manifest = manifest or build_manifest(module)
    chunks = []
    live_parameters = dict(iter_trainable_parameters(module))
    for spec in resolved_manifest.parameters:
        if spec.name not in live_parameters:
            raise KeyError(f"Trainable parameter {spec.name!r} was not found on the module.")
        parameter = live_parameters[spec.name].detach().cpu().reshape(-1).to(torch.float32)
        chunks.append(parameter)
    if chunks:
        vector = torch.cat(chunks).numpy()
    else:
        vector = np.zeros(0, dtype=np.float32)
    return vector.astype(np.float32, copy=False), resolved_manifest


def restore_adapter_state(
    module: torch.nn.Module,
    flat_vector: np.ndarray,
    manifest: AdapterStateManifest,
) -> None:
    vector = np.asarray(flat_vector, dtype=np.float32)
    if vector.size != manifest.total_params:
        raise ValueError(
            f"Flat vector has {vector.size} entries but manifest expects {manifest.total_params}."
        )

    offset = 0
    live_parameters = dict(iter_trainable_parameters(module))
    with torch.no_grad():
        for spec in manifest.parameters:
            if spec.name not in live_parameters:
                raise KeyError(f"Trainable parameter {spec.name!r} was not found on the module.")
            next_offset = offset + spec.numel
            view = vector[offset:next_offset].reshape(spec.shape)
            parameter = live_parameters[spec.name]
            restored = torch.from_numpy(view).to(dtype=parameter.dtype, device=parameter.device)
            parameter.copy_(restored)
            offset = next_offset


def save_manifest(manifest: AdapterStateManifest, path: str | Path) -> None:
    payload = {"parameters": [asdict(spec) for spec in manifest.parameters]}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_manifest(path: str | Path) -> AdapterStateManifest:
    payload = json.loads(Path(path).read_text())
    parameters = []
    for spec in payload["parameters"]:
        parameters.append(
            AdapterParameterSpec(
                name=spec["name"],
                shape=tuple(spec["shape"]),
                dtype=spec["dtype"],
                numel=int(spec["numel"]),
            )
        )
    return AdapterStateManifest(tuple(parameters))
