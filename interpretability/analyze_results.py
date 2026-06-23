"""
Produces the class-conditional P(correct) curve, the KL-per-step breakdown
(adapter calls vs frozen GPT-2 blocks), and the per-class attention peak
stats used to check for a positional shortcut.

Usage:
    python interpretability/analyze_results.py --results <path to results.json>
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_LABELS = ["pre-2", "post-2", "pre-5", "post-5", "pre-8", "post-8", "pre-11", "post-11"]
FUSION_LAYERS = [2, 5, 8, 11]
STEP_LABELS = [
    "pre-2\n->post-2", "post-2\n->pre-5", "pre-5\n->post-5", "post-5\n->pre-8",
    "pre-8\n->post-8", "post-8\n->pre-11", "pre-11\n->post-11",
]
STEP_IS_ADAPTER = [True, False, True, False, True, False, True]


def kl(p, q, labels, eps=1e-10):
    pv = np.array([p[l] for l in labels]) + eps
    qv = np.array([q[l] for l in labels]) + eps
    pv, qv = pv / pv.sum(), qv / qv.sum()
    return float((pv * (np.log(pv) - np.log(qv))).sum())


def step_kls(checkpoints, labels):
    label_probs = [ck["label_probs"] for ck in checkpoints]
    return np.array([kl(label_probs[i], label_probs[i + 1], labels) for i in range(7)])


def class_conditional_curves(results, labels, out_dir):
    by_class = defaultdict(list)
    for r in results:
        by_class[r["true_label"]].append(r)

    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        rows = by_class[label]
        real = np.array([[ck["label_probs"][label] for ck in r["real"]["checkpoints"]] for r in rows])
        zero = np.array([[ck["label_probs"][label] for ck in r["zero"]["checkpoints"]] for r in rows])

        x = np.arange(8)
        ax.plot(x, real.mean(0), marker="o", color="steelblue", label="real audio")
        ax.plot(x, zero.mean(0), marker="o", color="tomato", linestyle="--", label="zero audio")
        ax.set_title(f"{label} (n={len(rows)})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(CHECKPOINT_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel("Mean P(correct label)", fontsize=10)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_class_conditional_prob.png"), dpi=300)
    plt.close()

    print("\nFinal (post-11) mean P(correct label) by class:")
    for label in labels:
        rows = by_class[label]
        real_final = np.mean([r["real"]["checkpoints"][-1]["label_probs"][label] for r in rows])
        zero_final = np.mean([r["zero"]["checkpoints"][-1]["label_probs"][label] for r in rows])
        print(f"  {label:<10} n={len(rows):<5} real={real_final:.4f}  zero={zero_final:.4f}  gap={real_final - zero_final:+.4f}")


def temporal_kl_breakdown(results, labels, out_dir):
    """KL at each of the 7 steps between the 8 checkpoints. Odd steps are
    frozen GPT-2 blocks running with no adapter."""
    kl_real = np.array([step_kls(r["real"]["checkpoints"], labels) for r in results])
    kl_zero = np.array([step_kls(r["zero"]["checkpoints"], labels) for r in results])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(7)
    width = 0.35
    ax.bar(x - width / 2, kl_real.mean(0), width, label="real audio", color="steelblue")
    ax.bar(x + width / 2, kl_zero.mean(0), width, label="zero audio", color="tomato")
    for step, is_adapter in enumerate(STEP_IS_ADAPTER):
        if not is_adapter:
            ax.axvspan(step - 0.5, step + 0.5, color="gray", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS, fontsize=9)
    ax.set_ylabel("Mean KL(step_i || step_i+1)  [nats]", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_temporal_kl_breakdown.png"), dpi=300)
    plt.close()

    print("\nMean KL per step (real audio):")
    for step, label in enumerate(STEP_LABELS):
        tag = "adapter" if STEP_IS_ADAPTER[step] else "frozen "
        print(f"  [{tag}] {label.replace(chr(10), ' '):<16} {kl_real[:, step].mean():.5f}")


def positional_shortcut_check(results, labels):
    """Per-class mean/std of the peak attended audio token (last text
    token), per adapter layer. Class-invariant peaks suggest a positional
    shortcut rather than content-driven attention."""
    peaks = {label: [[] for _ in range(4)] for label in labels}
    for r in results:
        for a in range(4):
            w = r["attn_weights"][a]
            if w is not None:
                peaks[r["true_label"]][a].append(int(np.array(w[-1]).argmax()))

    print("\nPeak audio-token index (last text token), mean +/- std by class:")
    for a, layer in enumerate(FUSION_LAYERS):
        print(f"  Layer {layer}:")
        for label in labels:
            p = peaks[label][a]
            if p:
                print(f"    {label:<10} mean={np.mean(p):6.2f}  std={np.std(p):6.2f}  n={len(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    labels = sorted({r["true_label"] for r in results})
    out_dir = os.path.dirname(args.results) or "."

    class_conditional_curves(results, labels, out_dir)
    temporal_kl_breakdown(results, labels, out_dir)
    positional_shortcut_check(results, labels)
