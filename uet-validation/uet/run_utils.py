from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path


def setup_run_dir(output_dir: Path, experiment: str, run_name: str | None = None) -> Path:
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / experiment / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, log_name: str = "run.log") -> logging.Logger:
    log_path = run_dir / "logs" / log_name

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_h = logging.FileHandler(log_path, mode="w")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(fmt)

    root.setLevel(logging.INFO)
    root.addHandler(file_h)
    root.addHandler(stream_h)
    return root


def dump_config(run_dir: Path, args: Namespace, extra: dict | None = None) -> None:
    config = {k: _jsonable(v) for k, v in vars(args).items()}
    if extra:
        config.update({k: _jsonable(v) for k, v in extra.items()})
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    return v


def dump_metadata(run_dir: Path, metadata: dict) -> None:
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
