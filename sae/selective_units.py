"""
Class-selective SAE units and a causal ablation test. For each encoder: rank
dictionary units by how much more they fire on one class than the rest
(selectivity), name the top units via the eGeMAPS Lasso coefficients from
disentanglement.py, then zero the top target-class units at test time and
check whether recall for that class drops while the other classes hold.

The headline output is the cross-encoder summary: how many target-selective
units each encoder's dictionary contains, and how much recall they carry.

How to run (after train_sae.py; unit naming needs disentanglement.py):
python sae/selective_units.py --target emphatic --sparsity 90 --ablate 1 5 10 20 40
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
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight

from baselines.embedding_probes import LinearProbe, _train
from baselines.layer_sweep import ENCODER_COLORS, FALLBACK_COLOR
from sae.sae_model import (
    encode_dataset, load_checkpoint, load_layerwise, pick_device, split_indices,
)


def load_unit_factors(coefs_dir, dataset, encoder, layer, sparsity):
    path = os.path.join(coefs_dir, f"{dataset}_{encoder}_L{layer}_s{sparsity}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data["coefs"], list(data["factor_names"])


def top_factors_for_unit(unit, coefs, factor_names, n=3):
    weights = np.abs(coefs[:, unit])
    if weights.max() < 1e-12:
        return ""
    order = np.argsort(weights)[::-1][:n]
    return "; ".join(factor_names[i] for i in order if weights[i] > 1e-12)


def main(args):
    device = pick_device()
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    coefs_dir = os.path.join(args.out_dir, "lasso_coefs")

    unit_rows, ablation_rows, summary = [], [], []
    panels = []

    for encoder in args.encoders:
        pattern = os.path.join(ckpt_dir, f"{args.dataset}_{encoder}_L*_s{args.sparsity}.pt")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"[skip] no checkpoint matches {pattern}")
            continue
        if args.layer is not None:
            paths = [p for p in paths if f"_L{args.layer}_" in p]
        # Lowest-numbered layer checkpoint ("best" layer from train_sae.py;
        # the final-layer checkpoint sorts to the larger index).
        path = paths[0]
        model, meta = load_checkpoint(path, device)
        layer = meta["layer"]
        print(f"\n=== {encoder} L{layer} ({meta['layer_label']}), "
              f"{args.sparsity}% sparsity, dict {model.n_dict}")

        data = load_layerwise(args.dataset, encoder, args.embeddings_dir,
                              path=args.embeddings)
        y = torch.tensor(data["labels"], dtype=torch.long)
        idx2label = data["idx2label"]
        num_classes = len(idx2label)
        class_names = [idx2label[i] for i in range(num_classes)]
        if args.target not in class_names:
            sys.exit(f"target class {args.target!r} not in {class_names}")
        target = class_names.index(args.target)

        train_idx, val_idx, test_idx = split_indices(data, args.dataset)
        X = data["embeddings"][:, layer, :].float()
        Z_train = encode_dataset(model, X[train_idx], meta["mu"], meta["sd"], device)
        Z_val = encode_dataset(model, X[val_idx], meta["mu"], meta["sd"], device)
        Z_test = encode_dataset(model, X[test_idx], meta["mu"], meta["sd"], device)
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        naming = load_unit_factors(coefs_dir, args.dataset, encoder, layer,
                                   args.sparsity)

        # --- 1. selectivity on the train split
        # selectivity(j, c) in [-1, 1]: how much more unit j fires on class c
        # than on the rest. Kept on train so the test-split ablation stays honest.
        sel = np.zeros((num_classes, model.n_dict), dtype=np.float64)
        active = np.zeros_like(sel)
        for c in range(num_classes):
            mask = (y_train == c).numpy()
            mu_c = Z_train[mask].mean(dim=0).numpy()
            mu_o = Z_train[~mask].mean(dim=0).numpy()
            sel[c] = (mu_c - mu_o) / (mu_c + mu_o + 1e-9)
            active[c] = (Z_train[mask] > 0).float().mean(dim=0).numpy()

        counts = {}
        for c, name in enumerate(class_names):
            usable = (sel[c] > args.threshold) & (active[c] > args.min_active)
            counts[name] = int(usable.sum())
            order = np.argsort(sel[c])[::-1]
            kept = [j for j in order if active[c][j] > args.min_active][:args.top_units]
            for rank, j in enumerate(kept):
                unit_rows.append({
                    "encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                    "class": name, "rank": rank, "unit": int(j),
                    "selectivity": float(sel[c][j]),
                    "active_rate_class": float(active[c][j]),
                    "top_factors": (top_factors_for_unit(j, *naming)
                                    if naming else ""),
                })
        print("  selective units per class "
              f"(selectivity > {args.threshold}, active > {args.min_active}): "
              + "  ".join(f"{n} {counts[n]}" for n in class_names))

        # --- 2. ablation of the top target-class units
        cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                                  y=y_train.numpy())
        probe = _train(
            LinearProbe(model.n_dict, num_classes),
            Z_train, y_train, Z_val, y_val,
            torch.tensor(cw, dtype=torch.float), device,
            epochs=args.probe_epochs, batch_size=256,
            select="uar" if args.dataset == "aibo" else "acc",
        )
        probe.eval()

        order = np.argsort(sel[target])[::-1]
        ranked = [j for j in order if active[target][j] > args.min_active]

        def recalls(codes):
            with torch.no_grad():
                preds = probe(codes.to(device)).argmax(dim=-1).cpu().numpy()
            return recall_score(y_test.numpy(), preds, average=None,
                                labels=np.arange(num_classes), zero_division=0)

        curve = {}
        for m in [0] + args.ablate:
            codes = Z_test.clone()
            if m:
                codes[:, ranked[:m]] = 0.0
            per_class = recalls(codes)
            curve[m] = per_class
            row = {"encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                   "target": args.target, "ablated_units": m,
                   "uar": float(np.mean(per_class))}
            row.update({f"recall_{n}": float(r)
                        for n, r in zip(class_names, per_class)})
            ablation_rows.append(row)
        base = curve[0][target]
        worst = min(curve[m][target] for m in args.ablate)
        print(f"  {args.target} recall: {base:.3f} baseline -> "
              f"{worst:.3f} after ablating up to {max(args.ablate)} units")

        summary.append({"encoder": encoder, "layer": layer,
                        f"{args.target}_selective_units": counts[args.target],
                        f"{args.target}_recall_baseline": float(base),
                        f"{args.target}_recall_ablated": float(worst)})
        panels.append((encoder, layer, class_names, curve))

    # --- outputs
    for name, rows in [("selective_units", unit_rows),
                       ("selective_units_ablation", ablation_rows),
                       ("selective_units_summary", summary)]:
        if not rows:
            continue
        path = os.path.join(args.out_dir, f"{name}_{args.dataset}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved → {path}")

    if panels:
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False,
                                 sharey=True)
        for ax, (encoder, layer, class_names, curve) in zip(axes[0], panels):
            ms = sorted(curve)
            for c, cname in enumerate(class_names):
                is_target = cname == args.target
                ax.plot(ms, [100 * curve[m][c] for m in ms],
                        "-o" if is_target else "-",
                        linewidth=2.4 if is_target else 1.2,
                        markersize=4,
                        color=(ENCODER_COLORS.get(encoder, FALLBACK_COLOR)
                               if is_target else "#b9bec6"),
                        label=cname if is_target else None)
            ax.set_title(f"{encoder} L{layer}", fontsize=10)
            ax.set_xlabel(f"Top {args.target} units ablated")
            ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        axes[0][0].set_ylabel("Recall (%)")
        fig.suptitle(f"Ablating {args.target}-selective SAE units "
                     "(grey lines: the other classes)", fontsize=11.5)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"selective_units_ablation_{args.dataset}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-selective SAE units and ablation.")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "emodb"])
    parser.add_argument("--encoders", nargs="+", default=[
        "wavlm-large", "wav2vec2-large-emotion", "qwen2-audio", "audio-flamingo-3"])
    parser.add_argument("--sparsity", type=int, default=90)
    parser.add_argument("--layer", type=int, default=None,
                        help="Explicit layer; default = first checkpoint found.")
    parser.add_argument("--target", default="emphatic")
    parser.add_argument("--ablate", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--top-units", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-active", type=float, default=0.02)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--embeddings", default=None,
                        help="Explicit layerwise .pt path (single encoder only).")
    parser.add_argument("--out-dir", default="sae/outputs")
    args = parser.parse_args()

    if args.embeddings and len(args.encoders) != 1:
        parser.error("--embeddings requires exactly one --encoders entry.")
    main(args)
