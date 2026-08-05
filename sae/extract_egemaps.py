"""
Extract eGeMAPS functionals (88 voice descriptors: pitch, loudness, jitter,
shimmer, formants, MFCC, spectral shape, rhythm) for every AIBO clip, for
disentanglement.py to regress the SAE codes against.

Runs where the AIBO wavs live (the cluster), same AIBO_DATA_DIR convention as
models/audio_encoder/preprocessing_aibo.py. Rows follow the label-file order,
same as the embeddings, so downstream alignment is positional plus a basename
check.

Needs:  pip install opensmile

How to run:
AIBO_DATA_DIR=/data/... python sae/extract_egemaps.py
python sae/extract_egemaps.py --limit 5     # smoke test
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.audio_encoder.preprocessing_aibo import _load_aibo_index


def main(output_path, limit=None):
    import opensmile  # deferred: only needed on the machine with the wavs

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    index = _load_aibo_index()
    if limit is not None:
        index = index[:limit]
    print(f"Samples : {len(index)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_names = None
    with open(output_path, "w", newline="") as f:
        writer = None
        for i, (wav_path, label) in enumerate(index):
            df = smile.process_file(wav_path)
            if feature_names is None:
                feature_names = list(df.columns)
                print(f"Features: {len(feature_names)} (eGeMAPSv02 functionals)")
                writer = csv.writer(f)
                writer.writerow(["file", "label"] + feature_names)
            writer.writerow([wav_path, label] + [f"{v:.6g}" for v in df.iloc[0].tolist()])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(index)}")
                f.flush()

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract eGeMAPS functionals for AIBO.")
    parser.add_argument("--output", default="sae/outputs/aibo_egemaps.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.output, args.limit)
