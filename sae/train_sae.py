"""
Train TopK SAEs per (encoder, layer, sparsity) and check the sparse codes
still carry the task: reconstruction MSE (paper Fig. 1 right) and a linear
probe on the codes (Fig. 1 middle). Rows append to the CSV as they finish;
pass --resume after a job time limit. Saves checkpoints used by
disentanglement.py and selective_units.py.

How to run (repo root, needs the layerwise embeddings):
python sae/train_sae.py --resume
python sae/train_sae.py --encoders wavlm-large --layers best --sparsities 90 99 --epochs 50
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from baselines.embedding_probes import LinearProbe, _train
from baselines.layer_sweep import ENCODER_COLORS, FALLBACK_COLOR
from sae.sae_model import (
    checkpoint_path, encode_dataset, load_layerwise, pick_device,
    reconstruction_mse, resolve_layer, save_checkpoint, sparsity_to_k,
    split_indices, sweep_reference_uar, train_sae,
)

BASE_FIELDS = ["encoder", "layer", "layer_label", "is_final", "sparsity", "k",
               "n_dict", "best_epoch", "val_mse", "test_mse", "dead_frac",
               "accuracy", "wf1", "uar"]
STR_FIELDS = {"encoder", "layer_label"}


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key in ("layer", "sparsity", "k", "n_dict", "best_epoch"):
                    row[key] = int(val)
                elif key == "is_final":
                    row[key] = val == "True"
                elif key not in STR_FIELDS:
                    row[key] = float(val)
            rows.append(row)
    return rows


def run_encoder(encoder, args, device, writer, csv_file, done, rows_out):
    data = load_layerwise(args.dataset, encoder, args.embeddings_dir, path=args.embeddings)
    X_all = data["embeddings"].float()
    y = torch.tensor(data["labels"], dtype=torch.long)
    n_layers = X_all.shape[1]
    layer_labels = data.get("layer_labels") or [str(i) for i in range(n_layers)]
    idx2label = data["idx2label"]
    num_classes = len(idx2label)
    class_names = [idx2label[i] for i in range(num_classes)]

    train_idx, val_idx, test_idx = split_indices(data, args.dataset)
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=y[train_idx].numpy())
    class_weights = torch.tensor(cw, dtype=torch.float)
    select = "uar" if args.dataset == "aibo" else "acc"

    layers = []
    for spec in args.layers:
        idx = resolve_layer(spec, args.dataset, encoder, n_layers, args.sweep_dir)
        if idx not in layers:
            layers.append(idx)

    for layer in layers:
        X = X_all[:, layer, :]
        mu = X[train_idx].mean(dim=0)
        sd = X[train_idx].std(dim=0).clamp_min(1e-6)
        Xs_train = (X[train_idx] - mu) / sd
        Xs_val = (X[val_idx] - mu) / sd
        print(f"\n=== {encoder} layer {layer} ({layer_labels[layer]}), "
              f"dim {X.shape[1]}, dict {args.dict_size}")

        for sparsity in args.sparsities:
            if (encoder, layer, sparsity) in done:
                continue
            k = sparsity_to_k(sparsity, args.dict_size)
            model, info = train_sae(
                Xs_train, Xs_val, args.dict_size, k,
                epochs=args.epochs, device=device, seed=args.seed,
            )
            test_mse = reconstruction_mse(model, X[test_idx], mu, sd, device)

            Z_train = encode_dataset(model, X[train_idx], mu, sd, device)
            Z_val = encode_dataset(model, X[val_idx], mu, sd, device)
            Z_test = encode_dataset(model, X[test_idx], mu, sd, device)
            probe = _train(
                LinearProbe(args.dict_size, num_classes),
                Z_train, y[train_idx], Z_val, y[val_idx],
                class_weights, device, epochs=args.probe_epochs,
                batch_size=256, select=select,
            )
            probe.eval()
            with torch.no_grad():
                preds = probe(Z_test.to(device)).argmax(dim=-1).cpu().numpy()
            true = y[test_idx].numpy()
            per_class = recall_score(true, preds, average=None,
                                     labels=np.arange(num_classes), zero_division=0)

            row = {
                "encoder": encoder, "layer": layer,
                "layer_label": layer_labels[layer],
                "is_final": layer == n_layers - 1,
                "sparsity": sparsity, "k": k, "n_dict": args.dict_size,
                "best_epoch": info["best_epoch"], "val_mse": info["val_mse"],
                "test_mse": test_mse, "dead_frac": info["dead_frac"],
                "accuracy": accuracy_score(true, preds),
                "wf1": f1_score(true, preds, average="weighted"),
                "uar": recall_score(true, preds, average="macro"),
            }
            row.update({f"recall_{n}": r for n, r in zip(class_names, per_class)})
            writer.writerow(row)
            csv_file.flush()
            rows_out.append(row)

            save_checkpoint(
                checkpoint_path(args.out_dir, args.dataset, encoder, layer, sparsity),
                model, mu, sd,
                meta={"dataset": args.dataset, "encoder": encoder, "layer": layer,
                      "layer_label": layer_labels[layer], "sparsity": sparsity,
                      "val_mse": info["val_mse"], "seed": args.seed},
            )
            print(f"  s={sparsity}% k={k:>5} | mse {test_mse:.5f}  "
                  f"dead {info['dead_frac']:.2f} | probe acc {row['accuracy']:.4f}  "
                  f"wf1 {row['wf1']:.4f}  uar {row['uar']:.4f}")


def plot(rows, args):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    seen = []
    for row in rows:
        key = (row["encoder"], row["layer"])
        if key not in seen:
            seen.append(key)
    for encoder, layer in seen:
        sub = sorted([r for r in rows if r["encoder"] == encoder and r["layer"] == layer],
                     key=lambda r: r["sparsity"])
        if not sub:
            continue
        color = ENCODER_COLORS.get(encoder, FALLBACK_COLOR)
        style = ":" if sub[0]["is_final"] else "-"
        label = f"{encoder} L{layer}" + (" (final)" if sub[0]["is_final"] else "")
        xs = [r["sparsity"] for r in sub]
        axes[0].plot(xs, [r["test_mse"] for r in sub], style, color=color,
                     linewidth=2, label=label)
        axes[1].plot(xs, [100 * r["uar"] for r in sub], style, color=color,
                     linewidth=2, label=label)
        ref = sweep_reference_uar(args.dataset, encoder, layer, args.sweep_dir)
        if ref is not None:
            axes[1].axhline(100 * ref, color=color, linewidth=1,
                            linestyle="--", alpha=0.45)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Reconstruction MSE (test)")
    axes[1].set_ylabel("Probe UAR on sparse codes (%)")
    for ax, title in zip(axes, ["SAE reconstruction", "SAE probing"]):
        ax.set_xlabel("Sparsity (%)")
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"sae_training_{args.dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate TopK SAEs.")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "emodb"])
    parser.add_argument("--encoders", nargs="+", default=[
        "wavlm-large", "wav2vec2-large-emotion", "qwen2-audio", "audio-flamingo-3"])
    parser.add_argument("--layers", nargs="+", default=["best", "final"],
                        help="'best', 'final', or explicit layer indices.")
    parser.add_argument("--sparsities", nargs="+", type=int,
                        default=[75, 80, 85, 90, 95, 99])
    parser.add_argument("--dict-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--embeddings", default=None,
                        help="Explicit layerwise .pt path (single encoder only).")
    parser.add_argument("--sweep-dir", default="results")
    parser.add_argument("--out-dir", default="sae/outputs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.embeddings and len(args.encoders) != 1:
        parser.error("--embeddings requires exactly one --encoders entry.")

    device = pick_device()
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"sae_training_{args.dataset}.csv")

    existing = load_csv(csv_path) if args.resume and os.path.exists(csv_path) else []
    done = {(r["encoder"], r["layer"], r["sparsity"]) for r in existing}
    if existing:
        print(f"Resuming: {len(existing)} rows already in {csv_path}")
    rows = list(existing)

    # Field list needs the class names; peek at the first encoder's file.
    peek = load_layerwise(args.dataset, args.encoders[0], args.embeddings_dir,
                          path=args.embeddings)
    class_names = [peek["idx2label"][i] for i in range(len(peek["idx2label"]))]
    fields = BASE_FIELDS + [f"recall_{n}" for n in class_names]
    del peek

    csv_file = open(csv_path, "a" if existing else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not existing:
        writer.writeheader()
        csv_file.flush()

    for encoder in args.encoders:
        run_encoder(encoder, args, device, writer, csv_file, done, rows)
    csv_file.close()
    print(f"Saved → {csv_path}")

    plot(rows, args)
