"""Extract encoder embeddings from the FAU AIBO corpus (IS2009 Emotion Challenge, 5-class) and
save as a .pt file.

Usage:
    python models/audio_encoder/preprocessing_aibo.py --encoder wavlm-large
    python models/audio_encoder/preprocessing_aibo.py --encoder wav2vec2-large-emotion
    python models/audio_encoder/preprocessing_aibo.py  # defaults to wav2vec2-base

To add a new encoder: add an entry to ENCODERS below.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Type

import audiofile
import numpy as np
import torch
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)

SAMPLING_RATE = 16000
# AIBO chunks are short (median ~1.6s, p99 ~4.3s, max ~24.5s) -- 5s covers ~99% of
# samples without padding most clips to mostly-silence like EMoDB's 8s setting.
MAX_DURATION_SEC = 5.0
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLING_RATE)
EMBEDDINGS_DIR = "embeddings"

# Local: "dataset" (repo-relative). On the cluster, set e.g.
# AIBO_DATA_DIR=/data/chi-gpu1/asl_alm_ss26/data
DATASET_DIR = os.environ.get("AIBO_DATA_DIR", "dataset")
WAV_DIR = os.path.join(DATASET_DIR, "wav")
LABELS_FILE = os.path.join(
    DATASET_DIR, "labels", "IS2009EmotionChallenge", "chunk_labels_5cl_corpus.txt"
)

# IS2009 5-class codes -> full emotion names.
AIBO_LABEL_MAP = {
    "A": "anger",
    "E": "emphatic",
    "N": "neutral",
    "P": "positive",
    "R": "rest",
}


@dataclass
class EncoderSpec:
    model_id: str
    hidden_dim: int
    processor_cls: Type
    model_cls: Type


ENCODERS: dict = {
    "wav2vec2-base": EncoderSpec(
        model_id="facebook/wav2vec2-base-960h",
        hidden_dim=768,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wav2vec2-large-emotion": EncoderSpec(
        model_id="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        hidden_dim=1024,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wavlm-large": EncoderSpec(
        model_id="microsoft/wavlm-large",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=WavLMModel,
    ),
    "hubert-large": EncoderSpec(
        model_id="facebook/hubert-large-ls960-ft",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=HubertModel,
    ),
}


def _load_aibo_index() -> list[tuple[str, str]]:
    """Read the IS2009 5-class label file.

    Each line is "<chunk_name> <label_code> <confidence>". Confidence is ignored
    for now -- all samples are used with their majority-vote label.

    Returns a list of (wav_path, label_name) tuples.
    """
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"AIBO label file not found: {LABELS_FILE}")

    index = []
    with open(LABELS_FILE) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            chunk_name, code = parts[0], parts[1]
            wav_path = os.path.join(WAV_DIR, f"{chunk_name}.wav")
            index.append((wav_path, AIBO_LABEL_MAP[code]))

    return index


def extract(encoder_name: str, output_path: str | None = None, limit: int | None = None) -> str:
    spec = ENCODERS[encoder_name]

    if output_path is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        output_path = os.path.join(EMBEDDINGS_DIR, f"aibo_{encoder_name}_embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder : {encoder_name} ({spec.model_id})")
    print(f"Device  : {device}")
    print(f"Output  : {output_path}")

    print("\nLoading AIBO (IS2009 Emotion Challenge, 5-class)...")
    index = _load_aibo_index()
    if limit is not None:
        index = index[:limit]
    print(f"Samples : {len(index)}")

    emotion_classes = sorted(set(AIBO_LABEL_MAP.values()))
    label2idx = {label: idx for idx, label in enumerate(emotion_classes)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"\nLoading {spec.model_id}...")
    processor = spec.processor_cls.from_pretrained(spec.model_id)
    model = spec.model_cls.from_pretrained(spec.model_id).to(device).eval()

    embeddings, labels, file_paths = [], [], []

    for i, (file_path, emotion) in enumerate(index):
        label_int = label2idx[emotion]

        signal, _ = audiofile.read(file_path, always_2d=False)
        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")

        inputs = processor(
            signal,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            hidden = model(input_values).last_hidden_state.squeeze(0).cpu()

        embeddings.append(hidden)
        labels.append(label_int)
        file_paths.append(file_path)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(index)}")

    T_audio = embeddings[0].shape[0]
    print(f"\nDone. T_audio={T_audio}, hidden_dim={spec.hidden_dim}, samples={len(embeddings)}")

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "file_paths": file_paths,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "T_audio": T_audio,
            "hidden_dim": spec.hidden_dim,
            "encoder": encoder_name,
            "model_id": spec.model_id,
        },
        output_path,
    )
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio encoder embeddings from AIBO.")
    parser.add_argument(
        "--encoder",
        choices=list(ENCODERS),
        default="wav2vec2-base",
        help=f"Encoder to use (default: wav2vec2-base). Options: {', '.join(ENCODERS)}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path (default: embeddings/aibo_<encoder>_embeddings.pt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples (for smoke-testing).",
    )
    args = parser.parse_args()
    extract(args.encoder, args.output, args.limit)
