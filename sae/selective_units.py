"""
Class-selective SAE units and a causal ablation test. For each encoder: rank
dictionary units by how much more they fire on one class than the rest
(selectivity), name the top units via the eGeMAPS Lasso coefficients from
disentanglement.py, then zero the top units of a target class at test time and
check whether recall for that class drops while the other classes hold.

The headline output is the cross-encoder summary: how many selective units each
encoder's dictionary contains per class, and how much recall they carry.

How to run (after train_sae.py; unit naming needs disentanglement.py):
python sae/selective_units.py                       # every class
python sae/selective_units.py --targets emphatic positive
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


def class_selectivity(Z_train, y_train, num_classes):
    """selectivity(c, j) in [-1, 1]: how much more unit j fires on class c than
    on the rest, plus the fraction of class-c clips on which it fires at all.
    Computed on train only so the test-split ablation stays honest."""
    sel = np.zeros((num_classes, Z_train.shape[1]), dtype=np.float64)
    active = np.zeros_like(sel)
    for c in range(num_classes):
        mask = (y_train == c).numpy()
        mu_c = Z_train[mask].mean(dim=0).numpy()
        mu_o = Z_train[~mask].mean(dim=0).numpy()
        sel[c] = (mu_c - mu_o) / (mu_c + mu_o + 1e-9)
        active[c] = (Z_train[mask] > 0).float().mean(dim=0).numpy()
    return sel, active


def ablation_curve(probe, Z_test, y_test, ranked, steps, num_classes, device):
    """Per-class recall after zeroing the top-m ranked units, for each m."""
    curve = {}
    for m in [0] + steps:
        codes = Z_test.clone()
        if m:
            codes[:, ranked[:m]] = 0.0
        with torch.no_grad():
            preds = probe(codes.to(device)).argmax(dim=-1).cpu().numpy()
        curve[m] = recall_score(y_test.numpy(), preds, average=None,
                                labels=np.arange(num_classes), zero_division=0)
    return curve


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def plot_ablation(panels, target, args):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False, sharey=True)
    for ax, (encoder, layer, class_names, curve) in zip(axes[0], panels):
        ms = sorted(curve)
        for c, cname in enumerate(class_names):
            is_target = cname == target
            ax.plot(ms, [100 * curve[m][c] for m in ms],
                    "-o" if is_target else "-",
                    linewidth=2.4 if is_target else 1.2, markersize=4,
                    color=(ENCODER_COLORS.get(encoder, FALLBACK_COLOR)
                           if is_target else "#b9bec6"),
                    label=cname if is_target else None)
        ax.set_title(f"{encoder} L{layer}", fontsize=10)
        ax.set_xlabel(f"Top {target} units ablated")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("Recall (%)")
    fig.suptitle(f"Ablating {target}-selective SAE units "
                 "(grey lines: the other classes)", fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(args.out_dir,
                       f"selective_units_ablation_{args.dataset}_{target}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_counts(counts, class_names, args):
    """How many selective units each encoder holds per class: the 'why only
    emphatic' figure."""
    encoders = list(counts)
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 3.4))
    width = 0.8 / len(encoders)
    for i, encoder in enumerate(encoders):
        ax.bar(np.arange(len(class_names)) + (i - (len(encoders) - 1) / 2) * width,
               [counts[encoder][c] for c in class_names], width,
               color=ENCODER_COLORS.get(encoder, FALLBACK_COLOR), label=encoder)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Selective units")
    ax.set_title(f"Class-selective dictionary units "
                 f"(selectivity > {args.threshold}, active > {args.min_active})",
                 fontsize=10.5)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"selective_units_counts_{args.dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def main(args):
    device = pick_device()
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    coefs_dir = os.path.join(args.out_dir, "lasso_coefs")

    unit_rows, summary = [], []
    ablation_rows = {}   # target -> rows
    panels = {}          # target -> panels
    counts_by_encoder = {}
    class_names = None

    for encoder in args.encoders:
        pattern = os.path.join(ckpt_dir, f"{args.dataset}_{encoder}_L*_s{args.sparsity}.pt")
        paths = sorted(glob.glob(pattern))
        if args.layer is not None:
            paths = [p for p in paths if f"_L{args.layer}_" in p]
        if not paths:
            print(f"[skip] no checkpoint matches {pattern}")
            continue
        # Lowest-numbered layer checkpoint ("best" layer from train_sae.py;
        # the final-layer checkpoint sorts to the larger index).
        model, meta = load_checkpoint(paths[0], device)
        layer = meta["layer"]
        print(f"\n=== {encoder} L{layer} ({meta['layer_label']}), "
              f"{args.sparsity}% sparsity, dict {model.n_dict}")

        data = load_layerwise(args.dataset, encoder, args.embeddings_dir,
                              path=args.embeddings)
        y = torch.tensor(data["labels"], dtype=torch.long)
        idx2label = data["idx2label"]
        num_classes = len(idx2label)
        class_names = [idx2label[i] for i in range(num_classes)]

        targets = class_names if args.targets == ["all"] else args.targets
        unknown = [t for t in targets if t not in class_names]
        if unknown:
            sys.exit(f"target class(es) {unknown} not in {class_names}")

        train_idx, val_idx, test_idx = split_indices(data, args.dataset)
        X = data["embeddings"][:, layer, :].float()
        Z_train = encode_dataset(model, X[train_idx], meta["mu"], meta["sd"], device)
        Z_val = encode_dataset(model, X[val_idx], meta["mu"], meta["sd"], device)
        Z_test = encode_dataset(model, X[test_idx], meta["mu"], meta["sd"], device)
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        naming = load_unit_factors(coefs_dir, args.dataset, encoder, layer, args.sparsity)

        # --- 1. selectivity per class
        sel, active = class_selectivity(Z_train, y_train, num_classes)
        counts = {}
        for c, name in enumerate(class_names):
            counts[name] = int(((sel[c] > args.threshold)
                                & (active[c] > args.min_active)).sum())
            order = np.argsort(sel[c])[::-1]
            kept = [j for j in order if active[c][j] > args.min_active][:args.top_units]
            for rank, j in enumerate(kept):
                unit_rows.append({
                    "encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                    "class": name, "rank": rank, "unit": int(j),
                    "selectivity": float(sel[c][j]),
                    "active_rate_class": float(active[c][j]),
                    "top_factors": top_factors_for_unit(j, *naming) if naming else "",
                })
        counts_by_encoder[encoder] = counts
        print(f"  selective units per class "
              f"(selectivity > {args.threshold}, active > {args.min_active}): "
              + "  ".join(f"{n} {counts[n]}" for n in class_names))

        # --- 2. one probe per encoder, reused for every target
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

        # --- 3. ablate the top units of each target class
        for target in targets:
            t = class_names.index(target)
            order = np.argsort(sel[t])[::-1]
            ranked = [j for j in order if active[t][j] > args.min_active]
            curve = ablation_curve(probe, Z_test, y_test, ranked, args.ablate,
                                   num_classes, device)

            for m in sorted(curve):
                row = {"encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                       "target": target, "ablated_units": m,
                       "uar": float(np.mean(curve[m]))}
                row.update({f"recall_{n}": float(r)
                            for n, r in zip(class_names, curve[m])})
                ablation_rows.setdefault(target, []).append(row)

            base, worst = curve[0][t], min(curve[m][t] for m in args.ablate)
            # Largest drop in any non-target class: tells you where the ablation
            # stops being specific and starts hurting the model generally.
            collateral = max(
                (curve[0][c] - min(curve[m][c] for m in args.ablate))
                for c in range(num_classes) if c != t
            )
            summary.append({
                "encoder": encoder, "layer": layer, "target": target,
                "selective_units": counts[target],
                "recall_baseline": float(base), "recall_ablated": float(worst),
                "max_other_class_drop": float(collateral),
                # A count means nothing without the bar it was counted against,
                # so the thresholds travel with the numbers.
                "threshold": args.threshold, "min_active": args.min_active,
                "sparsity": args.sparsity, "max_ablated": max(args.ablate),
            })
            panels.setdefault(target, []).append((encoder, layer, class_names, curve))
            print(f"  {target:>9} recall: {base:.3f} → {worst:.3f} "
                  f"after ablating up to {max(args.ablate)} units "
                  f"(worst other-class drop {collateral:.3f})")

    # --- outputs
    write_csv(os.path.join(args.out_dir, f"selective_units_{args.dataset}.csv"), unit_rows)
    write_csv(os.path.join(args.out_dir, f"selective_units_summary_{args.dataset}.csv"),
              summary)
    for target, rows in ablation_rows.items():
        write_csv(os.path.join(args.out_dir,
                               f"selective_units_ablation_{args.dataset}_{target}.csv"),
                  rows)
        plot_ablation(panels[target], target, args)
    if counts_by_encoder and class_names:
        plot_counts(counts_by_encoder, class_names, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-selective SAE units and ablation.")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "emodb"])
    parser.add_argument("--encoders", nargs="+", default=[
        "wavlm-large", "wav2vec2-large-emotion", "qwen2-audio", "audio-flamingo-3"])
    parser.add_argument("--sparsity", type=int, default=90)
    parser.add_argument("--layer", type=int, default=None,
                        help="Explicit layer; default = first checkpoint found.")
    parser.add_argument("--targets", nargs="+", default=["all"],
                        help="Class names, or 'all' for every class.")
    parser.add_argument("--ablate", nargs="+", type=int,
                        default=[1, 5, 10, 20, 40, 80, 160])
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
