from __future__ import annotations

import gzip
import io
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

_CACHE = Path.home() / ".cache" / "uet-data"

_MNIST_URLS = {
    "train_x": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_y": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_x":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_y":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

_CIFAR10_URL  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _parse_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 0x0803:  # images
            h, w = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h * w)
        else:               # labels
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(root: Path = _CACHE / "mnist", flat: bool = True) -> tuple[TensorDataset, TensorDataset]:
    root = Path(root)
    paths = {k: _fetch(v, root / Path(v).name) for k, v in _MNIST_URLS.items()}
    X_tr = _parse_idx(paths["train_x"]).astype(np.float32) / 255.0
    y_tr = _parse_idx(paths["train_y"])
    X_te = _parse_idx(paths["test_x"]).astype(np.float32) / 255.0
    y_te = _parse_idx(paths["test_y"])
    if not flat:
        X_tr = X_tr.reshape(-1, 1, 28, 28)
        X_te = X_te.reshape(-1, 1, 28, 28)
    return (TensorDataset(torch.from_numpy(X_tr.copy()), torch.from_numpy(y_tr.copy()).long()),
            TensorDataset(torch.from_numpy(X_te.copy()), torch.from_numpy(y_te.copy()).long()))


def _load_cifar_batches(tar_path: Path, batch_names: list[str], label_key: bytes) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in batch_names:
            member = next(m for m in tf.getmembers() if m.name.endswith(name))
            batch = pickle.load(io.BytesIO(tf.extractfile(member).read()), encoding="bytes")
            Xs.append(batch[b"data"].astype(np.float32) / 255.0)
            ys.extend(batch[label_key])
    return np.concatenate(Xs), np.array(ys, dtype=np.int64)


def load_cifar10(root: Path = _CACHE / "cifar10", flat: bool = True) -> tuple[TensorDataset, TensorDataset]:
    root = Path(root)
    tar = _fetch(_CIFAR10_URL, root / "cifar-10-python.tar.gz")
    train_batches = [f"data_batch_{i}" for i in range(1, 6)]
    X_tr, y_tr = _load_cifar_batches(tar, train_batches, b"labels")
    X_te, y_te = _load_cifar_batches(tar, ["test_batch"], b"labels")
    if not flat:
        X_tr = X_tr.reshape(-1, 3, 32, 32)
        X_te = X_te.reshape(-1, 3, 32, 32)
    return (TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)))


def load_cifar100(root: Path = _CACHE / "cifar100", flat: bool = True) -> tuple[TensorDataset, TensorDataset]:
    root = Path(root)
    tar = _fetch(_CIFAR100_URL, root / "cifar-100-python.tar.gz")
    X_tr, y_tr = _load_cifar_batches(tar, ["train"], b"fine_labels")
    X_te, y_te = _load_cifar_batches(tar, ["test"], b"fine_labels")
    if not flat:
        X_tr = X_tr.reshape(-1, 3, 32, 32)
        X_te = X_te.reshape(-1, 3, 32, 32)
    return (TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)))
