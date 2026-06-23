"""
Run all baselines and plot a grouped comparison against AudioGPT2.

Run from project root:
    python baselines/compare.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import matplotlib.pyplot as plt

from baselines.svm_mfcc import run as run_svm
from baselines.embedding_probes import run as run_probes

# Test-set results from training runs (accuracy, weighted F1)
_AUDIOGPT2 = {
    "wav2vec2-base":          {"accuracy": 0.426, "f1": 0.406},
    "wav2vec2-large-emotion": {"accuracy": 0.932, "f1": 0.932},
    "wavlm-large":            {"accuracy": 0.883, "f1": 0.881},
    "hubert-large":           {"accuracy": 0.784, "f1": 0.786},
}


def _plot(svm_acc, probe_results, out="baselines/comparison.png"):
    encoders = [e for e in probe_results if e in _AUDIOGPT2]
    n = len(encoders)
    if n == 0:
        return

    w = 0.18
    xs = list(range(n))
    groups = [
        ([svm_acc] * n,                                              "SVM + MFCC"),
        ([probe_results[e]["linear"]["accuracy"] for e in encoders], "linear probe"),
        ([probe_results[e]["mlp"]["accuracy"]    for e in encoders], "MLP probe"),
        ([_AUDIOGPT2[e]["accuracy"]              for e in encoders], "AudioGPT2"),
    ]

    fig, ax = plt.subplots(figsize=(max(7, n * 2.5), 5))
    for i, (vals, label) in enumerate(groups):
        ax.bar([x + (i - 1.5) * w for x in xs], vals, w, label=label)

    ax.axhline(svm_acc, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(encoders, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot → {out}")
    plt.close()


DATASET_SPLITS = {
    "emodb": "train speakers 01-02,04-07,11-16 | val 09-10 | test 03-08",
    "aibo": "train Ohm (minus Ohm_31,32) | val Ohm_31,Ohm_32 | test Mont_01-25",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="aibo",
        choices=["emodb", "aibo"],
        help="Which dataset to run baselines on.",
    )
    args = parser.parse_args()

    prefix = "aibo_" if args.dataset == "aibo" else ""
    emb_files = sorted(glob.glob(f"embeddings/{prefix}*_embeddings.pt"))

    print("=" * 64)
    print(f"RUNNING BASELINES  (dataset={args.dataset})")
    print("=" * 64)

    svm_result = run_svm(args.dataset)

    probe_results = {}
    for path in emb_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        encoder = stem.removesuffix("_embeddings")
        if args.dataset == "aibo":
            encoder = encoder.removeprefix("aibo_")
        probe_results[encoder] = run_probes(path, args.dataset)

    encoders = list(probe_results.keys())
    col = 22

    print("\n\n" + "=" * (24 + col * len(encoders)))
    print("COMPARISON  (test set, speaker-independent split)")
    print("=" * (24 + col * len(encoders)))
    print(f"  {'':22}" + "".join(f"{e:>{col}}" for e in encoders))
    print("  " + "-" * (22 + col * len(encoders)))

    def row(label, vals):
        return f"  {label:<22}" + "".join(f"{v * 100:>{col - 1}.1f}%" for v in vals)

    print(row("SVM + MFCC",   [svm_result["accuracy"]]                           * len(encoders)))
    print(row("linear probe", [probe_results[e]["linear"]["accuracy"]             for e in encoders]))
    print(row("MLP probe",    [probe_results[e]["mlp"]["accuracy"]                for e in encoders]))
    print(row("AudioGPT2",    [_AUDIOGPT2.get(e, {}).get("accuracy", float("nan")) for e in encoders]))
    if args.dataset == "aibo":
        print("  " + "-" * (22 + col * len(encoders)))
        print(row("SVM + MFCC (UAR)",   [svm_result["uar"]]                       * len(encoders)))
        print(row("linear probe (UAR)", [probe_results[e]["linear"]["uar"]        for e in encoders]))
        print(row("MLP probe (UAR)",    [probe_results[e]["mlp"]["uar"]           for e in encoders]))
    print("  " + "-" * (22 + col * len(encoders)))
    print(f"  Split: {DATASET_SPLITS[args.dataset]}")
    print("=" * (24 + col * len(encoders)))

    _plot(svm_result["accuracy"], probe_results)


if __name__ == "__main__":
    main()
