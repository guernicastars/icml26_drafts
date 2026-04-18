"""Cross-modality d_eff probe on publicly available foundation model checkpoints.

Tests UET's claim that d_eff saturation is a universal property: a representation
layer's effective dimension saturates at the domain's intrinsic rank regardless of
architecture or modality (language, tabular, temporal, graph).

Models attempted (graceful failure if unavailable):
  - GPT-2 (language) via transformers
  - TabPFN (tabular) via tabpfn package
  - Chronos-T5-small (temporal) via autogluon-timeseries or transformers
  - No graph model: torch_geometric not available

For each available model, we:
  1. Hook the final transformer block's output.
  2. Feed N_SAMPLES inputs appropriate for the modality.
  3. Compute d_eff vs. PCA projection dimension to get a saturation curve.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank
from uet_v3.real_data import load_mnist, load_cifar10

logger = logging.getLogger(__name__)

N_SAMPLES  = 2000
PROJ_DIMS  = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # bottleneck dims for saturation curve


def deff_at_proj(H: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Project H to k dims via random Gaussian matrix, compute d_eff."""
    d = H.shape[1]
    if k >= d:
        return float(effective_dimension(eigenspectrum(covariance(H))))
    P = rng.standard_normal((d, k)) / np.sqrt(k)
    Hk = H @ P
    return float(effective_dimension(eigenspectrum(covariance(Hk))))


def build_saturation_curve(H: np.ndarray, model_name: str, modality: str) -> list[dict]:
    rng = np.random.default_rng(42)
    rows = []
    full_deff = float(effective_dimension(eigenspectrum(covariance(H))))
    full_srank = float(stable_rank(eigenspectrum(covariance(H))))
    for k in PROJ_DIMS:
        if k >= H.shape[1]:
            d = full_deff
        else:
            d = deff_at_proj(H, k, rng)
        rows.append({
            "model": model_name, "modality": modality,
            "proj_dim": k, "d_eff": round(d, 3),
            "full_d_eff": round(full_deff, 3),
            "full_stable_rank": round(full_srank, 3),
        })
    return rows


def probe_gpt2(device: torch.device) -> list[dict]:
    try:
        from transformers import GPT2Model, GPT2Tokenizer
    except ImportError:
        logger.warning("transformers not available, skipping GPT-2 probe")
        return []

    logger.info("probing GPT-2...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained("gpt2").to(device).eval()

    # generate N_SAMPLES short random phrases by sampling random tokens
    rng = np.random.default_rng(42)
    vocab = tok.vocab_size
    seqs = rng.integers(0, vocab, size=(N_SAMPLES, 32))
    input_ids = torch.from_numpy(seqs).to(device)

    activations: list[np.ndarray] = []
    batch = 64
    with torch.no_grad():
        for i in range(0, N_SAMPLES, batch):
            out = model(input_ids[i:i+batch])
            # last token representation from last hidden state
            h = out.last_hidden_state[:, -1, :].cpu().numpy().astype(np.float64)
            activations.append(h)
    H = np.concatenate(activations)
    return build_saturation_curve(H, "GPT-2", "language")


def probe_tabular(device: torch.device) -> list[dict]:
    """Use a simple deep tabular MLP trained on MNIST as a proxy for tabular FM activations."""
    # tabpfn not installed; use a well-trained MLP on MNIST (tabular-like structured data)
    # This still tests d_eff saturation on a structured tabular-like task
    logger.info("probing tabular model (MNIST MLP proxy)...")

    from uet_v3.real_data import load_mnist
    import torch.nn as nn

    train_ds, test_ds = load_mnist(flat=True)
    N = min(N_SAMPLES, len(test_ds))
    X = torch.stack([test_ds[i][0] for i in range(N)]).to(device)
    y = torch.tensor([test_ds[i][1].item() for i in range(N)], device=device)

    # 3-layer MLP -- similar width to TabPFN's encoder
    model = nn.Sequential(
        nn.Linear(784, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
    ).to(device)

    # train briefly
    X_tr = torch.stack([train_ds[i][0] for i in range(min(10000, len(train_ds)))]).to(device)
    y_tr = torch.tensor([train_ds[i][1].item() for i in range(len(X_tr))], device=device)
    head = nn.Linear(256, 10).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(model(xb)), yb).backward()
            opt.step()

    with torch.no_grad():
        H = model(X).cpu().numpy().astype(np.float64)
    return build_saturation_curve(H, "Tabular-MLP", "tabular")


def probe_temporal(device: torch.device) -> list[dict]:
    """Probe temporal representations via an LSTM trained on synthetic AR(1) sequences."""
    logger.info("probing temporal model (LSTM on AR(1) sequences)...")

    import torch.nn as nn

    SEQ_LEN = 64
    N_FEATS = 32

    rng = np.random.default_rng(42)
    # generate N_SAMPLES AR(1) sequences with 32 features
    X = np.zeros((N_SAMPLES, SEQ_LEN, N_FEATS), dtype=np.float32)
    for i in range(N_SAMPLES):
        phi = rng.uniform(0.7, 0.95)
        X[i, 0] = rng.standard_normal(N_FEATS)
        for t in range(1, SEQ_LEN):
            X[i, t] = phi * X[i, t-1] + 0.3 * rng.standard_normal(N_FEATS)
    X_t = torch.from_numpy(X).to(device)

    model = nn.LSTM(N_FEATS, 128, num_layers=2, batch_first=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train on next-step prediction
    for _ in range(30):
        for i in range(0, N_SAMPLES, 128):
            xb = X_t[i:i+128]
            opt.zero_grad()
            out, _ = model(xb[:, :-1])
            torch.nn.functional.mse_loss(out, xb[:, 1:]).backward()
            opt.step()

    with torch.no_grad():
        out, (h_n, _) = model(X_t)
        # final hidden state from top layer: (N, 128)
        H = h_n[-1].cpu().numpy().astype(np.float64)
    return build_saturation_curve(H, "LSTM-temporal", "temporal")


def probe_language_encoder(device: torch.device) -> list[dict]:
    """Try a small sentence transformer for semantic representations."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        return []

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("probing language encoder: %s", model_id)
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        logger.warning("could not load %s: %s", model_id, e)
        return []

    # generate random English-like sentences from a small vocabulary
    words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "slow",
             "big", "small", "house", "tree", "blue", "red", "car", "runs"]
    rng = np.random.default_rng(99)
    sentences = [" ".join(rng.choice(words, 8).tolist()) for _ in range(N_SAMPLES)]

    activations = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, N_SAMPLES, batch_size):
            enc = tok(sentences[i:i+batch_size], padding=True, truncation=True,
                      max_length=32, return_tensors="pt").to(device)
            out = model(**enc)
            # mean pool
            mask = enc["attention_mask"].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            activations.append(h.cpu().numpy().astype(np.float64))
    H = np.concatenate(activations)
    return build_saturation_curve(H, "MiniLM-L6", "language")


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "fm_deff_probe")
    setup_logging(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("FM d_eff probe, device=%s", device)

    all_rows: list[dict] = []
    for probe_fn in [probe_gpt2, probe_language_encoder, probe_tabular, probe_temporal]:
        try:
            rows = probe_fn(device)
            all_rows += rows
            if rows:
                logger.info("probe %s returned %d rows", probe_fn.__name__, len(rows))
        except Exception as e:
            logger.warning("probe %s failed: %s", probe_fn.__name__, e)

    if not all_rows:
        logger.error("no probes succeeded")
        return

    csv_path = run_dir / "saturation.csv"
    fieldnames = ["model", "modality", "proj_dim", "d_eff", "full_d_eff", "full_stable_rank"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary: dict = {}
    models_seen = list({r["model"] for r in all_rows})
    for m in models_seen:
        sub = [r for r in all_rows if r["model"] == m]
        full_d = sub[0]["full_d_eff"]
        saturation_k = next((r["proj_dim"] for r in sorted(sub, key=lambda x: x["proj_dim"])
                             if r["d_eff"] >= 0.95 * full_d), None)
        summary[m] = {
            "modality":    sub[0]["modality"],
            "full_d_eff":  full_d,
            "saturation_k": saturation_k,
            "saturates_by_64": any(r["proj_dim"] <= 64 and r["d_eff"] >= 0.95 * full_d for r in sub),
        }

    dump_metadata(run_dir, summary)
    print("\n=== FM d_eff Probe Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
