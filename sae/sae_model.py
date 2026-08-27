"""TopK sparse autoencoder used by the rest of sae/. Follows Mariotte et al. 2025
(arXiv 2509.24793) and Gao et al. 2025:

    z     = TopK(ReLU(W_e x + b_e))
    x_hat = W_d z
    loss  = MSE(x_hat, x)

Tied init and unit-norm decoder rows are standard TopK stabilizers the paper
does not spell out. Inputs are z-scored with train-split statistics, stored in
the checkpoint so codes decode consistently at load time.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baselines.embedding_probes import DATASET_CONFIGS, _speaker_split
from data.dataset import extract_speaker_id


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_dict, k):
        super().__init__()
        self.d_in, self.n_dict, self.k = d_in, n_dict, k
        self.encoder = nn.Linear(d_in, n_dict)
        decoder = self.encoder.weight.detach().clone()  # (n_dict, d_in)
        decoder /= decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.decoder = nn.Parameter(decoder)

    def encode(self, x):
        a = torch.relu(self.encoder(x))
        top_vals, top_idx = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, top_idx, top_vals)

    def decode(self, z):
        return z @ self.decoder

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm_decoder_(self):
        self.decoder.data /= self.decoder.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def sparsity_to_k(sparsity_pct, n_dict):
    """e.g. 95% sparsity with n_dict=4096 -> k=204 active units."""
    return max(1, int((1 - sparsity_pct / 100.0) * n_dict))


def train_sae(X_train, X_val, n_dict, k, epochs=150, lr=1e-3, batch_size=32,
              device="cpu", seed=0, log_every=0):
    """Adam, lr 1e-3, batch 32 as in the paper; keeps the best-val-MSE epoch."""
    torch.manual_seed(seed)
    model = TopKSAE(X_train.shape[1], n_dict, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train.to(device)),
                        batch_size=batch_size, shuffle=True)
    X_val_dev = X_val.to(device)

    best_mse, best_state, best_epoch = math.inf, None, -1
    for epoch in range(epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, _ = model(xb)
            F.mse_loss(x_hat, xb).backward()
            opt.step()
            model.renorm_decoder_()

        model.eval()
        with torch.no_grad():
            x_hat, z_val = model(X_val_dev)
            val_mse = F.mse_loss(x_hat, X_val_dev).item()
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = {key: val.clone() for key, val in model.state_dict().items()}
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch + 1}/{epochs}  val_mse {val_mse:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, z_val = model(X_val_dev)
        dead_frac = ((z_val > 0).sum(dim=0) == 0).float().mean().item()
    return model, {"val_mse": best_mse, "best_epoch": best_epoch, "dead_frac": dead_frac}


@torch.no_grad()
def encode_dataset(model, X, mu, sd, device, batch_size=2048):
    """Standardize with stored train stats, encode in batches, return CPU codes."""
    out = []
    for i in range(0, len(X), batch_size):
        xb = ((X[i:i + batch_size] - mu) / sd).to(device)
        out.append(model.encode(xb).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruction_mse(model, X, mu, sd, device, batch_size=2048):
    total, count = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = ((X[i:i + batch_size] - mu) / sd).to(device)
        x_hat, _ = model(xb)
        total += F.mse_loss(x_hat, xb, reduction="sum").item()
        count += xb.numel()
    return total / count


def save_checkpoint(path, model, mu, sd, meta):
    payload = dict(meta)
    payload.update(
        state=model.state_dict(), d_in=model.d_in, n_dict=model.n_dict,
        k=model.k, mu=mu.cpu(), sd=sd.cpu(),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TopKSAE(payload["d_in"], payload["n_dict"], payload["k"])
    model.load_state_dict(payload["state"])
    model = model.to(device).eval()
    return model, payload


def checkpoint_path(out_dir, dataset, encoder, layer, sparsity):
    return os.path.join(out_dir, "checkpoints",
                        f"{dataset}_{encoder}_L{layer}_s{sparsity}.pt")


def load_layerwise(dataset, encoder, embeddings_dir="embeddings", path=None):
    cfg = DATASET_CONFIGS[dataset]
    path = path or os.path.join(
        embeddings_dir,
        f"{cfg['embeddings_prefix']}{encoder}_layerwise_pooled_embeddings.pt",
    )
    data = torch.load(path, weights_only=False)
    if not data.get("layerwise"):
        raise ValueError(f"{path} is not a layerwise embeddings file.")
    return data


def split_indices(data, dataset):
    cfg = DATASET_CONFIGS[dataset]
    speaker_ids = [extract_speaker_id(p) for p in data["file_paths"]]
    return _speaker_split(speaker_ids, cfg["val_speakers"], cfg["test_speakers"])


def resolve_layer(spec, dataset, encoder, n_layers, sweep_dir="results"):
    """'best' = highest linear-probe UAR layer from the sweep CSV;
    'final' = last row; otherwise an explicit integer index."""
    if spec == "final":
        return n_layers - 1
    if spec == "best":
        path = os.path.join(sweep_dir, f"layer_sweep_{dataset}_{encoder}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run the layer sweep first or pass an "
                "explicit layer index instead of 'best'.")
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["probe"] == "linear"]
        best = max(rows, key=lambda r: float(r["uar"]))
        return int(best["layer"])
    return int(spec)


def sweep_reference_uar(dataset, encoder, layer, sweep_dir="results"):
    """Linear-probe UAR on the original representation at this layer, for the
    'does the sparse code keep the task information' comparison."""
    path = os.path.join(sweep_dir, f"layer_sweep_{dataset}_{encoder}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["probe"] == "linear" and int(row["layer"]) == layer:
                return float(row["uar"])
    return None
