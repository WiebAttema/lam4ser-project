"""
Disentanglement of SAE codes against eGeMAPS voice factors: one Lasso fit per
factor, reporting informativeness (held-out R²) and completeness (DCI
compactness of the Lasso weights), plus the same fit on the original
representation as a reference. Mirrors the paper's Fig. 2-4.

How to run (after train_sae.py and extract_egemaps.py):
python sae/disentanglement.py
python sae/disentanglement.py --encoders wavlm-large --sparsities 90 --cv
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import digamma
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neighbors import NearestNeighbors

from baselines.layer_sweep import ENCODER_COLORS, FALLBACK_COLOR
from sae.sae_model import (
    encode_dataset, load_checkpoint, load_layerwise, pick_device, split_indices,
)

FAMILY_RULES = [  # first keyword hit wins; mirrors the paper's seven families
    ("pitch", ["F0semitone", "pitch"]),
    ("formants", ["F1", "F2", "F3", "formant"]),
    ("mfcc", ["mfcc"]),
    ("quality", ["jitter", "shimmer", "HNR"]),
    ("spectral", ["alphaRatio", "slope", "Flux", "Hammarberg"]),
    ("loudness", ["loudness", "equivalentSoundLevel"]),
    ("rhythm", ["Segments", "Pause", "rate", "Length"]),
]


def factor_family(name):
    for family, keys in FAMILY_RULES:
        if any(k.lower() in name.lower() for k in keys):
            return family
    return "other"


def knn_entropy(x, k=3):
    """Kozachenko-Leonenko estimator for a 1-D variable."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = len(x)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dist, _ = nn.kneighbors(x)
    r = np.clip(dist[:, k], 1e-12, None)
    return digamma(n) - digamma(k) + np.log(2.0) + np.mean(np.log(r))


def fit_factor(F_col, X, holdout, alpha, use_cv, seed):
    rng = np.random.RandomState(seed)
    n = len(F_col)
    order = rng.permutation(n)
    n_fit = int(n * (1 - holdout))
    fit_idx, eval_idx = order[:n_fit], order[n_fit:]

    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    f_mu, f_sd = F_col[fit_idx].mean(), max(F_col[fit_idx].std(), 1e-8)
    f = (F_col - f_mu) / f_sd

    if use_cv:
        reg = LassoCV(alphas=np.logspace(-3, 0, 6), cv=3, max_iter=5000, n_jobs=-1)
    else:
        reg = Lasso(alpha=alpha, max_iter=5000)
    reg.fit(Xs[fit_idx], f[fit_idx])
    r2 = reg.score(Xs[eval_idx], f[eval_idx])

    # DCI completeness: 1.0 = one dimension carries the factor, 0 = spread evenly.
    w = np.abs(reg.coef_)
    total = w.sum()
    if total < 1e-12:
        return r2, 0.0, reg.coef_
    rho = w / total
    nz = rho[rho > 0]
    completeness = 1.0 + np.sum(nz * np.log(nz)) / np.log(len(w))
    return r2, float(completeness), reg.coef_


def load_egemaps(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[2:]
        files, rows = [], []
        for row in reader:
            files.append(row[0])
            rows.append([float(v) for v in row[2:]])
    F = np.asarray(rows, dtype=np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return files, names, F


def analyze(X, F_test, factor_names, entropies, tag, args, seed):
    """X: (n_test_sub, dims) numpy. Returns one row dict per factor."""
    rows = []
    coef_matrix = np.zeros((len(factor_names), X.shape[1]), dtype=np.float32)
    for i, name in enumerate(factor_names):
        r2, comp, coefs = fit_factor(F_test[:, i], X, args.holdout, args.alpha,
                                     args.cv, seed)
        coef_matrix[i] = coefs
        rows.append({**tag, "factor": name, "family": factor_family(name),
                     "r2": r2, "completeness": comp, "entropy": entropies[i]})
    return rows, coef_matrix


def make_figures(rows, args):
    keys = []
    for r in rows:
        key = (r["encoder"], r["layer"])
        if key not in keys:
            keys.append(key)

    def top10(sub, metric):
        best = sorted(sub, key=lambda r: r[metric], reverse=True)[:10]
        vals = [r[metric] for r in best]
        return np.mean(vals), np.std(vals)

    # Fig 2 analog: top-10 informativeness and completeness vs sparsity.
    for metric, fname in [("r2", "disentangle_informativeness"),
                          ("completeness", "disentangle_completeness")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for encoder, layer in keys:
            color = ENCODER_COLORS.get(encoder, FALLBACK_COLOR)
            sae_rows = [r for r in rows if r["encoder"] == encoder
                        and r["layer"] == layer and r["sparsity"] != "ref"]
            sparsities = sorted({r["sparsity"] for r in sae_rows})
            means, stds = [], []
            for s in sparsities:
                m, sd = top10([r for r in sae_rows if r["sparsity"] == s], metric)
                means.append(m)
                stds.append(sd)
            ax.errorbar(sparsities, means, yerr=stds, color=color, linewidth=2,
                        capsize=2, label=f"{encoder} L{layer}")
            ref_rows = [r for r in rows if r["encoder"] == encoder
                        and r["layer"] == layer and r["sparsity"] == "ref"]
            if ref_rows:
                m, _ = top10(ref_rows, metric)
                ax.axhline(m, color=color, linewidth=1, linestyle="--", alpha=0.45)
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel(f"Top 10 {'R²' if metric == 'r2' else 'completeness'}")
        ax.set_title(f"{'Informativeness' if metric == 'r2' else 'Completeness'} "
                     "(dashed = original representation)", fontsize=11.5)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{fname}_{args.dataset}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved → {out}")

    # Fig 3 analog: completeness vs factor entropy at the scatter sparsity.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for encoder, layer in keys:
        sub = [r for r in rows if r["encoder"] == encoder and r["layer"] == layer
               and r["sparsity"] == args.scatter_sparsity]
        if not sub:
            continue
        ax.scatter([r["entropy"] for r in sub], [r["completeness"] for r in sub],
                   s=14, alpha=0.65, color=ENCODER_COLORS.get(encoder, FALLBACK_COLOR),
                   label=f"{encoder} L{layer}")
    ax.set_xlabel("Factor entropy estimate")
    ax.set_ylabel("Completeness")
    ax.set_title(f"Completeness vs factor entropy ({args.scatter_sparsity}% sparsity)",
                 fontsize=11.5)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"disentangle_entropy_{args.dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")

    # Fig 4 analog: family counts of the 10 best-predicted factors, averaged
    # across sparsity levels, one panel per (encoder, layer).
    families = [f for f, _ in FAMILY_RULES] + ["other"]
    n = len(keys)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax, (encoder, layer) in zip(axes.flat, keys):
        sae_rows = [r for r in rows if r["encoder"] == encoder
                    and r["layer"] == layer and r["sparsity"] != "ref"]
        sparsities = sorted({r["sparsity"] for r in sae_rows})
        counts = np.zeros((len(sparsities), len(families)))
        for si, s in enumerate(sparsities):
            best = sorted([r for r in sae_rows if r["sparsity"] == s],
                          key=lambda r: r["r2"], reverse=True)[:10]
            for r in best:
                counts[si, families.index(r["family"])] += 1
        ax.bar(range(len(families)), counts.mean(axis=0),
               color=ENCODER_COLORS.get(encoder, FALLBACK_COLOR))
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{encoder} L{layer}", fontsize=10)
        ax.set_ylabel("Avg count in top 10")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"disentangle_families_{args.dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eGeMAPS disentanglement of SAE codes.")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "emodb"])
    parser.add_argument("--encoders", nargs="+", default=None,
                        help="Default: every encoder with a checkpoint.")
    parser.add_argument("--sparsities", nargs="+", type=int, default=None,
                        help="Default: every sparsity found in checkpoints.")
    parser.add_argument("--egemaps", default="sae/outputs/aibo_egemaps.csv")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--embeddings", default=None,
                        help="Explicit layerwise .pt path (single encoder only).")
    parser.add_argument("--out-dir", default="sae/outputs")
    parser.add_argument("--max-samples", type=int, default=4000,
                        help="Subsample of the test split used for the Lasso fits.")
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cv", action="store_true",
                        help="LassoCV instead of a fixed alpha (much slower).")
    parser.add_argument("--scatter-sparsity", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    paths = sorted(glob.glob(os.path.join(ckpt_dir, f"{args.dataset}_*.pt")))
    if not paths:
        sys.exit(f"No checkpoints in {ckpt_dir}; run train_sae.py first.")

    ege_files, factor_names, F_all = load_egemaps(args.egemaps)
    print(f"eGeMAPS: {F_all.shape[0]} clips x {len(factor_names)} factors")

    # Group checkpoints per (encoder, layer) so embeddings load once.
    groups = {}
    for path in paths:
        _, meta = load_checkpoint(path)  # cheap enough; reload later on device
        if args.encoders and meta["encoder"] not in args.encoders:
            continue
        if args.sparsities and meta["sparsity"] not in args.sparsities:
            continue
        groups.setdefault((meta["encoder"], meta["layer"]), []).append(path)

    all_rows = []
    coefs_dir = os.path.join(args.out_dir, "lasso_coefs")
    os.makedirs(coefs_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    for (encoder, layer), ckpts in groups.items():
        data = load_layerwise(args.dataset, encoder, args.embeddings_dir,
                              path=args.embeddings)
        _, _, test_idx = split_indices(data, args.dataset)
        emb_files = [os.path.basename(data["file_paths"][i]) for i in test_idx]
        ege_base = [os.path.basename(p) for p in ege_files]
        assert emb_files[:20] == [ege_base[i] for i in test_idx[:20]], \
            "eGeMAPS rows do not line up with the embeddings index."

        sub = rng.permutation(len(test_idx))[:args.max_samples]
        test_sub = [test_idx[i] for i in sub]
        F_test = F_all[test_sub]
        entropies = [knn_entropy(F_test[:, i]) for i in range(F_test.shape[1])]

        X = data["embeddings"][:, layer, :].float()
        X_test = X[torch.tensor(test_sub)]

        # Reference: the original (standardized) representation.
        mu, sd = None, None
        for path in sorted(ckpts):
            model, meta = load_checkpoint(path, device)
            if mu is None:
                mu, sd = meta["mu"], meta["sd"]
                Xs = ((X_test - mu) / sd).numpy()
                print(f"\n=== {encoder} L{layer}: reference (original representation)")
                rows, _ = analyze(
                    Xs, F_test, factor_names, entropies,
                    {"encoder": encoder, "layer": layer, "sparsity": "ref"},
                    args, args.seed,
                )
                all_rows.extend(rows)
            Z = encode_dataset(model, X_test, meta["mu"], meta["sd"], device).numpy()
            print(f"=== {encoder} L{layer}: SAE at {meta['sparsity']}% sparsity")
            rows, coef_matrix = analyze(
                Z, F_test, factor_names, entropies,
                {"encoder": encoder, "layer": layer, "sparsity": meta["sparsity"]},
                args, args.seed,
            )
            all_rows.extend(rows)
            np.savez_compressed(
                os.path.join(coefs_dir,
                             f"{args.dataset}_{encoder}_L{layer}_s{meta['sparsity']}.npz"),
                coefs=coef_matrix, factor_names=np.array(factor_names),
            )

    csv_path = os.path.join(args.out_dir, f"disentanglement_{args.dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved → {csv_path}")

    make_figures(all_rows, args)
