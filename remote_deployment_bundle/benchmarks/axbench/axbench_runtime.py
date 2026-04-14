from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def benchmark_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return benchmark_root().parents[1]


def axbench_repo_path() -> Path:
    axbench_in_benchmark = benchmark_root() / "external" / "axbench"
    if axbench_in_benchmark.exists():
        return axbench_in_benchmark
    old_path = project_root() / "axbench_benchmark" / "external" / "axbench"
    if old_path.exists():
        return old_path
    raise FileNotFoundError("AxBench repo not found. Run setup.sh first.")


def _git_output(path: Path, *args: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return output.strip() or None


def describe_external_repo(name: str):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ExternalRepoSpec:
        name: str
        path: Path
        git_sha: str | None
        remote_url: str | None

        def as_json(self) -> dict[str, str | None]:
            return {
                "name": self.name,
                "path": str(self.path),
                "git_sha": self.git_sha,
                "remote_url": self.remote_url,
            }

    path = benchmark_root() / "external" / name
    if not path.exists():
        path = project_root() / "axbench_benchmark" / "external" / name
    return ExternalRepoSpec(
        name=name,
        path=path,
        git_sha=_git_output(path, "rev-parse", "HEAD"),
        remote_url=_git_output(path, "remote", "get-url", "origin"),
    )


def ensure_import_path(repo_path: Path, src_subdir: str | None = None) -> None:
    target = repo_path / src_subdir if src_subdir else repo_path
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)


def import_axbench():
    repo_path = axbench_repo_path()
    ensure_import_path(repo_path)
    import axbench  # type: ignore
    return axbench


def import_alpaca_eval():
    for base in [benchmark_root(), project_root() / "axbench_benchmark"]:
        repo_path = base / "external" / "alpaca_eval"
        if repo_path.exists():
            ensure_import_path(repo_path, "src")
            import alpaca_eval  # type: ignore
            return alpaca_eval
    raise ImportError("alpaca_eval not found. Clone it into external/alpaca_eval.")
