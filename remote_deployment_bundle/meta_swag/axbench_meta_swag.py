from __future__ import annotations

"""
AxBench helpers for Meta-SWAG.

The deployment bundle historically carried these helpers under
`axbench_benchmark/meta_swag/axbench_meta_swag.py`. The benchmark runner imports
them as `meta_swag.axbench_meta_swag`, so we provide this thin loader wrapper
that re-exports the implementation without duplicating code.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_IMPL_PATH = (
    Path(__file__).resolve().parents[1]
    / "axbench_benchmark"
    / "meta_swag"
    / "axbench_meta_swag.py"
)

if not _IMPL_PATH.exists():
    raise ModuleNotFoundError(f"AxBench Meta-SWAG implementation not found at {_IMPL_PATH}")

_spec = spec_from_file_location("_meta_swag_axbench_meta_swag_impl", _IMPL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load module spec from {_IMPL_PATH}")

_mod = module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

for _k, _v in _mod.__dict__.items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v

