"""Factor-driven SAE ablation. Where selective_units.py ranks dictionary units by
the emotion class they fire on, this ranks them by which eGeMAPS property loads
on them, reusing the Lasso weights from disentanglement.py. It zeroes those
units at test time and reports the effect on every emotion class.

Each ablation is compared against zeroing the same number of random active
units, so a drop only counts if it beats that control.

    python sae/factor_units.py
    python sae/factor_units.py --families loudness quality --ablate 5 10 20 40
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
from sklearn.utils.class_weight import compute_class_weight

from baselines.embedding_probes import LinearProbe, _train
from baselines.layer_sweep import ENCODER_COLORS, FALLBACK_COLOR
from sae.disentanglement import FAMILY_RULES, factor_family
from sae.sae_model import (
    encode_dataset, load_checkpoint, load_layerwise, pick_device, split_indices,
)
from sae.selective_units import (
    ablation_curve, class_selectivity, load_unit_factors, write_csv,
)


def resolve_groups(factor_names, families, factors):
    """[(group name, factor indices)] for whole families and single factors."""
    groups = []
    for family in families:
        idx = [i for i, n in enumerate(factor_names) if factor_family(n) == family]
        if idx:
            groups.append((family, idx))
        else:
            print(f"[skip] no eGeMAPS factor in family {family!r}")
    for name in factors:
        if name in factor_names:
            groups.append((name, [factor_names.index(name)]))
        else:
            print(f"[skip] unknown factor {name!r}")
    return groups


def factor_scores(coefs, idx):
    """Per-unit score for a factor group. Each factor's absolute Lasso weights
    are normalised to sum 1 first, so a factor with large raw coefficients does
    not dominate the group."""
    w = np.abs(coefs[idx])
    total = w.sum(axis=1, keepdims=True)
    total[total < 1e-12] = 1.0
    return (w / total).mean(axis=0)


def random_control(probe, Z_test, y_test, pool, steps, num_classes, device, n_draws, seed):
    """Mean per-class recall after zeroing m units drawn at random from `pool`."""
    rng = np.random.RandomState(seed)
    curves = []
    for _ in range(n_draws):
        ranked = list(rng.permutation(pool))
        curves.append(ablation_curve(probe, Z_test, y_test, ranked, steps,
                                     num_classes, device))
    return {m: np.mean([c[m] for c in curves], axis=0) for m in curves[0]}


def plot_group(panels, group, class_names, args):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False, sharey=True)
    for ax, (encoder, layer, curve, control) in zip(axes[0], panels):
        ms = sorted(curve)
        for c, cname in enumerate(class_names):
            ax.plot(ms, [100 * curve[m][c] for m in ms], "-o", markersize=3,
                    linewidth=1.8, label=cname)
        ax.plot(ms, [100 * np.mean(control[m]) for m in ms], "--", linewidth=1.4,
                color="#898781", label="UAR, random units")
        ax.set_title(f"{encoder} L{layer}", fontsize=10)
        ax.set_xlabel(f"Top {group} units ablated")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0][0].set_ylabel("Recall (%)")
    axes[0][-1].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Ablating units that carry {group} (dashed: random-unit control)",
                 fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"factor_units_ablation_{args.dataset}_{group}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def main(args):
    device = pick_device()
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    coefs_dir = os.path.join(args.out_dir, "lasso_coefs")

    top_rows, ablation_rows, overlap_rows = [], [], []
    panels = {}
    class_names = None

    for encoder in args.encoders:
        pattern = os.path.join(ckpt_dir, f"{args.dataset}_{encoder}_L*_s{args.sparsity}.pt")
        paths = sorted(glob.glob(pattern))
        if args.layer is not None:
            paths = [p for p in paths if f"_L{args.layer}_" in p]
        if not paths:
            print(f"[skip] no checkpoint matches {pattern}")
            continue
        model, meta = load_checkpoint(paths[0], device)
        layer = meta["layer"]

        naming = load_unit_factors(coefs_dir, args.dataset, encoder, layer, args.sparsity)
        if naming is None:
            print(f"[skip] {encoder} L{layer}: no Lasso coefficients; "
                  "run disentanglement.py for this cell first.")
            continue
        coefs, factor_names = naming
        groups = resolve_groups(factor_names, args.families, args.factors)
        if not groups:
            sys.exit("No usable factor groups.")
        print(f"\n=== {encoder} L{layer} ({meta['layer_label']}), "
              f"{args.sparsity}% sparsity, dict {model.n_dict}")

        data = load_layerwise(args.dataset, encoder, args.embeddings_dir,
                              path=args.embeddings)
        y = torch.tensor(data["labels"], dtype=torch.long)
        idx2label = data["idx2label"]
        num_classes = len(idx2label)
        class_names = [idx2label[i] for i in range(num_classes)]

        train_idx, val_idx, test_idx = split_indices(data, args.dataset)
        X = data["embeddings"][:, layer, :].float()
        Z_train = encode_dataset(model, X[train_idx], meta["mu"], meta["sd"], device)
        Z_val = encode_dataset(model, X[val_idx], meta["mu"], meta["sd"], device)
        Z_test = encode_dataset(model, X[test_idx], meta["mu"], meta["sd"], device)
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        sel, active = class_selectivity(Z_train, y_train, num_classes)
        alive = (Z_train > 0).float().mean(dim=0).numpy() > args.min_active

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

        pool = np.flatnonzero(alive)
        control = random_control(probe, Z_test, y_test, pool, args.ablate,
                                 num_classes, device, args.n_random, args.seed)

        for group, idx in groups:
            score = factor_scores(coefs, idx)
            ranked = [j for j in np.argsort(score)[::-1] if alive[j] and score[j] > 0]
            if not ranked:
                print(f"  [skip] {group}: no unit carries this factor group")
                continue
            curve = ablation_curve(probe, Z_test, y_test, ranked, args.ablate,
                                   num_classes, device)

            for m in sorted(curve):
                row = {"encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                       "group": group, "ablated_units": m, "mode": "factor",
                       "uar": float(np.mean(curve[m]))}
                row.update({f"recall_{n}": float(r)
                            for n, r in zip(class_names, curve[m])})
                ablation_rows.append(row)
                ctl = {"encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                       "group": group, "ablated_units": m, "mode": "random",
                       "uar": float(np.mean(control[m]))}
                ctl.update({f"recall_{n}": float(r)
                            for n, r in zip(class_names, control[m])})
                ablation_rows.append(ctl)

            # Which class suffers most, and does it beat the random control?
            drops = curve[0] - curve[max(args.ablate)]
            ctl_drops = control[0] - control[max(args.ablate)]
            hit = int(np.argmax(drops - ctl_drops))

            for rank, j in enumerate(ranked[:args.top_units]):
                owner = [class_names[c] for c in range(num_classes)
                         if sel[c][j] > args.threshold and active[c][j] > args.min_active]
                top_rows.append({
                    "encoder": encoder, "layer": layer, "sparsity": args.sparsity,
                    "group": group, "rank": rank, "unit": int(j),
                    "score": float(score[j]),
                    "selective_for": ";".join(owner),
                })

            top_set = set(ranked[:args.top_units])
            overlap = {"encoder": encoder, "layer": layer, "group": group,
                       "n_units_ranked": len(ranked),
                       "most_affected_class": class_names[hit],
                       "drop": float(drops[hit]),
                       "drop_random_control": float(ctl_drops[hit]),
                       "threshold": args.threshold, "min_active": args.min_active,
                       "sparsity": args.sparsity, "top_units": args.top_units,
                       "max_ablated": max(args.ablate)}
            for c, name in enumerate(class_names):
                selective = {int(j) for j in np.flatnonzero(
                    (sel[c] > args.threshold) & (active[c] > args.min_active))}
                overlap[f"overlap_{name}"] = len(top_set & selective)
            overlap_rows.append(overlap)

            panels.setdefault(group, []).append((encoder, layer, curve, control))
            print(f"  {group:>10}: {len(ranked)} units carry it | "
                  f"largest excess drop {class_names[hit]} "
                  f"{drops[hit]:.3f} vs {ctl_drops[hit]:.3f} random")

    write_csv(os.path.join(args.out_dir, f"factor_units_{args.dataset}.csv"), top_rows)
    write_csv(os.path.join(args.out_dir, f"factor_units_ablation_{args.dataset}.csv"),
              ablation_rows)
    write_csv(os.path.join(args.out_dir, f"factor_units_overlap_{args.dataset}.csv"),
              overlap_rows)
    for group, group_panels in panels.items():
        plot_group(group_panels, group, class_names, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor-driven SAE unit ablation.")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "emodb"])
    parser.add_argument("--encoders", nargs="+", default=[
        "wavlm-large", "wav2vec2-large-emotion", "qwen2-audio", "audio-flamingo-3"])
    parser.add_argument("--sparsity", type=int, default=90)
    parser.add_argument("--layer", type=int, default=None,
                        help="Explicit layer; default = first checkpoint found.")
    parser.add_argument("--families", nargs="+",
                        default=[f for f, _ in FAMILY_RULES],
                        help="eGeMAPS families to ablate as groups.")
    parser.add_argument("--factors", nargs="+", default=[],
                        help="Individual eGeMAPS factor names, each its own group.")
    parser.add_argument("--ablate", nargs="+", type=int,
                        default=[1, 5, 10, 20, 40, 80, 160])
    parser.add_argument("--top-units", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=5,
                        help="Random-control draws to average over.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Class-selectivity bar, for the overlap columns.")
    parser.add_argument("--min-active", type=float, default=0.02)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--embeddings", default=None,
                        help="Explicit layerwise .pt path (single encoder only).")
    parser.add_argument("--out-dir", default="sae/outputs")
    args = parser.parse_args()

    if args.embeddings and len(args.encoders) != 1:
        parser.error("--embeddings requires exactly one --encoders entry.")
    main(args)
