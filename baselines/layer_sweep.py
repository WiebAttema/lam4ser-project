"""
Per-layer probe sweep on layerwise mean-pooled embeddings.

Trains a linear and MLP probe per layer per encoder. Layers are z-scored with 
train-split statistics before probing, and checkpoints on AIBO are selected on 
validation UAR. Rows are appended to the CSV as they finish; rerun with --resume 
to skip rows already written.

Needs embeddings extracted with:
python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --layerwise

How to run:
python baselines/layer_sweep.py --dataset aibo --encoders qwen2-audio audio-flamingo-3 wavlm-large
python baselines/layer_sweep.py --dataset aibo --encoders qwen2-audio --resume
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from baselines.embedding_probes import (
    DATASET_CONFIGS,
    LinearProbe,
    MLPProbe,
    _speaker_split,
    _train,
)
from data.dataset import extract_speaker_id

RESULTS_DIR = "results"
FIELDS = ["encoder", "layer", "layer_label", "probe", "accuracy", "wf1", "uar"]
STR_FIELDS = {"encoder", "layer_label", "probe"}

# Fixed per encoder so colors stay stable across different subsets of encoders.
ENCODER_COLORS = {
    "wavlm-large": "#2a78d6",
    "qwen2-audio": "#1baf7a",
    "audio-flamingo-3": "#eda100",
    "wav2vec2-large-emotion": "#008300",
    "hubert-large": "#4a3aa7",
    "wav2vec2-base": "#e34948",
}
FALLBACK_COLOR = "#898781"
PROBE_STYLES = {"linear": "-", "mlp": "--"}

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TICK_INK = "#898781"
LABEL_INK = "#52514e"


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key == "layer":
                    row[key] = int(val)
                elif key not in STR_FIELDS:
                    row[key] = float(val)
            rows.append(row)
    return rows


def sweep(encoder, dataset, probes, epochs, batch_size, device, csv_path,
          embeddings_path=None, resume=False, standardize=True):
    cfg = DATASET_CONFIGS[dataset]
    path = embeddings_path or (
        f"embeddings/{cfg['embeddings_prefix']}{encoder}_layerwise_pooled_embeddings.pt"
    )
    print(f"\n=== {encoder}: loading {path}")
    data = torch.load(path, weights_only=False)
    if not data.get("layerwise"):
        raise ValueError(f"{path} is not a layerwise embeddings file; "
                         "re-run preprocessing with --layerwise.")

    X = data["embeddings"].float()  # (N, n_layers, dim)
    y = torch.tensor(data["labels"], dtype=torch.long)
    n_layers, input_dim = X.shape[1], X.shape[2]
    layer_labels = data.get("layer_labels") or [str(i) for i in range(n_layers)]
    idx2label = data["idx2label"]
    num_classes = len(idx2label)

    speaker_ids = [extract_speaker_id(p) for p in data["file_paths"]]
    train_idx, val_idx, test_idx = _speaker_split(
        speaker_ids, cfg["val_speakers"], cfg["test_speakers"]
    )
    select = "uar" if dataset == "aibo" else "acc"
    print(f"  {X.shape[0]} samples, {n_layers} layers, dim={input_dim} | "
          f"split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"  standardize={standardize}  select=val_{select}  "
          f"batch={batch_size}  epochs={epochs}")

    cw = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y[train_idx].numpy()
    )
    class_weights = torch.tensor(cw, dtype=torch.float)

    class_names = [idx2label[i] for i in range(num_classes)]
    fields = FIELDS + [f"recall_{name}" for name in class_names]

    existing = []
    if resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != fields:
            raise ValueError(
                f"{csv_path} has a different column set (probably from an older "
                "sweep version); delete or rename it, or run without --resume."
            )
        existing = load_csv(csv_path)
    done = {(r["layer"], r["probe"]) for r in existing}
    if existing:
        print(f"  resuming: {len(existing)} rows already in {csv_path}")
    rows = list(existing)

    csv_file = open(csv_path, "a" if existing else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not existing:
        writer.writeheader()
        csv_file.flush()

    for layer in range(n_layers):
        if all((layer, probe) in done for probe in probes):
            continue
        Xl = X[:, layer, :]
        if standardize:
            mu = Xl[train_idx].mean(dim=0)
            sd = Xl[train_idx].std(dim=0).clamp_min(1e-6)
            Xl = (Xl - mu) / sd
        for probe in probes:
            if (layer, probe) in done:
                continue
            probe_cls = LinearProbe if probe == "linear" else MLPProbe
            model = _train(
                probe_cls(input_dim, num_classes),
                Xl[train_idx], y[train_idx], Xl[val_idx], y[val_idx],
                class_weights, device, epochs=epochs,
                batch_size=batch_size, select=select,
            )
            model.eval()
            with torch.no_grad():
                preds = model(Xl[test_idx].to(device)).argmax(dim=-1).cpu().numpy()
            true = y[test_idx].numpy()
            per_class = recall_score(
                true, preds, average=None, labels=np.arange(num_classes),
                zero_division=0,
            )
            row = {
                "encoder": encoder,
                "layer": layer,
                "layer_label": layer_labels[layer],
                "probe": probe,
                "accuracy": accuracy_score(true, preds),
                "wf1": f1_score(true, preds, average="weighted"),
                "uar": recall_score(true, preds, average="macro"),
            }
            row.update({f"recall_{n}": r for n, r in zip(class_names, per_class)})
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            recalls = "  ".join(f"{n[:3]} {r:.2f}" for n, r in zip(class_names, per_class))
            print(f"  {layer_labels[layer]:>9} | {probe:>6} | "
                  f"acc {row['accuracy']:.4f}  wf1 {row['wf1']:.4f}  uar {row['uar']:.4f}"
                  f" | {recalls}")

    csv_file.close()
    print(f"Saved → {csv_path}")
    return rows


def plot(all_rows, dataset, metric, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for encoder in dict.fromkeys(r["encoder"] for r in all_rows):
        color = ENCODER_COLORS.get(encoder, FALLBACK_COLOR)
        for probe, style in PROBE_STYLES.items():
            rows = [r for r in all_rows if r["encoder"] == encoder and r["probe"] == probe]
            if not rows:
                continue
            xs = [r["layer"] for r in rows]
            ys = [100 * r[metric] for r in rows]
            ax.plot(xs, ys, style, color=color, linewidth=2,
                    label=f"{encoder} ({probe})")

    metric_name = metric.upper() if metric == "uar" else metric.capitalize()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Layer index", color=LABEL_INK)
    ax.set_ylabel(f"{metric_name} (%)", color=LABEL_INK)
    ax.set_title(f"Per-layer probing on {dataset.upper()} test set", color=LABEL_INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=TICK_INK, labelcolor=LABEL_INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-layer probe sweep on layerwise embeddings.")
    parser.add_argument("--dataset", default="aibo", choices=list(DATASET_CONFIGS))
    parser.add_argument(
        "--encoders", nargs="+",
        default=["qwen2-audio", "audio-flamingo-3", "wavlm-large"],
        help="Encoders with an existing layerwise embeddings file.",
    )
    parser.add_argument(
        "--probes", nargs="+", default=["linear", "mlp"], choices=["linear", "mlp"],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--no-standardize", action="store_true",
        help="Disable per-layer z-scoring (train-split statistics).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (layer, probe) rows already present in the CSV.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Rebuild the figure from existing CSVs without training.",
    )
    parser.add_argument(
        "--embeddings", default=None,
        help="Override embeddings path (single encoder only).",
    )
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    if args.embeddings and len(args.encoders) != 1:
        parser.error("--embeddings requires exactly one --encoders entry.")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for encoder in args.encoders:
        csv_path = os.path.join(args.output_dir, f"layer_sweep_{args.dataset}_{encoder}.csv")
        if args.plot_only:
            rows = load_csv(csv_path)
        else:
            rows = sweep(
                encoder, args.dataset, args.probes, args.epochs, args.batch_size,
                device, csv_path, embeddings_path=args.embeddings,
                resume=args.resume, standardize=not args.no_standardize,
            )
        all_rows.extend(rows)

    metric = "uar" if args.dataset == "aibo" else "accuracy"
    plot(all_rows, args.dataset, metric,
         os.path.join(args.output_dir, f"layer_sweep_{args.dataset}.png"))
