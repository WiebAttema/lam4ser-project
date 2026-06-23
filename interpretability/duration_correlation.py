"""
Checks whether the attended audio position from logit_lens.py tracks the
true (unpadded) clip duration, or is just a fixed positional shortcut.

Usage:
    python interpretability/duration_correlation.py \\
        --embeddings embeddings/aibo_wavlm-large_embeddings.pt --results <path to results.json>
"""

import argparse
import json
import os

import audiofile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FUSION_LAYERS = [2, 5, 8, 11]
MAX_DURATION_SEC = 5.0


def load_durations(embeddings_path, sample_indices):
    cache_path = embeddings_path.replace(".pt", "_durations.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    file_paths = torch.load(embeddings_path, weights_only=False)["file_paths"]
    durations = {idx: float(audiofile.duration(file_paths[idx])) for idx in sample_indices}

    with open(cache_path, "w") as f:
        json.dump(durations, f)
    return durations


def duration_correlation(results, embeddings_path, out_dir):
    sample_indices = [r["sample_idx"] for r in results]
    durations = load_durations(embeddings_path, sample_indices)
    duration = np.minimum([durations[i] for i in sample_indices], MAX_DURATION_SEC)

    peak = np.full((len(results), 4), np.nan)
    for i, r in enumerate(results):
        for a in range(4):
            w = r["attn_weights"][a]
            if w is not None:
                peak[i, a] = np.array(w[-1]).argmax()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    bin_edges = np.linspace(0, MAX_DURATION_SEC, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    print("Pearson correlation (duration vs peak attention position):")
    for a, (ax, layer) in enumerate(zip(axes, FUSION_LAYERS)):
        x, y = duration, peak[:, a]
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"  Layer {layer}: r = {r:.4f}")

        ax.scatter(x, y, s=2, alpha=0.08, color="steelblue")
        bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, 9)
        means = [y[bin_idx == b].mean() for b in range(10)]
        ax.plot(bin_centers, means, color="tomato", marker="o", label="binned mean")
        ax.set_title(f"Layer {layer} (r={r:.3f})", fontsize=11)
        ax.set_xlabel("True audio duration [s]", fontsize=10)
        ax.tick_params(labelsize=9)
        if a == 0:
            ax.set_ylabel("Peak attended audio token", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_duration_correlation.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    out_dir = os.path.dirname(args.results) or "."
    duration_correlation(results, args.embeddings, out_dir)
