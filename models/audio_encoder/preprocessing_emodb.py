"""Extract encoder embeddings from EMoDB and save as a .pt file.

Usage:
    python models/audio_encoder/preprocessing_emodb.py --encoder wavlm-large
    python models/audio_encoder/preprocessing_emodb.py --encoder wav2vec2-large-emotion
    python models/audio_encoder/preprocessing_emodb.py  # defaults to wav2vec2-base
    python models/audio_encoder/preprocessing_emodb.py --encoder qwen2-audio --limit 5  # smoke test
"""
import argparse
import os
from dataclasses import dataclass
from typing import Type

import audb
import audiofile
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    HubertModel,
    Qwen2AudioForConditionalGeneration,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)

SAMPLING_RATE = 16000
MAX_DURATION_SEC = 8.0
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLING_RATE)
EMBEDDINGS_DIR = "embeddings"


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

# Audio encoders pulled out of LALMs
LALM_MODELS: dict[str, str] = {
    "qwen2-audio": "Qwen/Qwen2-Audio-7B",
}
# d_model of the Whisper-style encoder used by all current LALM_MODELS
LALM_HIDDEN_DIM = 1280


def extract(encoder_name: str, output_path: str | None = None, limit: int | None = None) -> str:
    is_lalm = encoder_name in LALM_MODELS
    if is_lalm:
        model_id = LALM_MODELS[encoder_name]
        hidden_dim = LALM_HIDDEN_DIM
    else:
        spec = ENCODERS[encoder_name]
        model_id = spec.model_id
        hidden_dim = spec.hidden_dim

    if output_path is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        output_path = os.path.join(EMBEDDINGS_DIR, f"{encoder_name}_embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder : {encoder_name} ({model_id})")
    print(f"Device  : {device}")
    print(f"Output  : {output_path}")

    print("\nLoading EMoDB...")
    db = audb.load(
        "emodb",
        version="2.0.0",
        sampling_rate=SAMPLING_RATE,
        mixdown=True,
        format="wav",
    )
    df = db["emotion"].get()
    if limit is not None:
        df = df.iloc[:limit]
    print(f"Samples : {len(df)}")

    emotion_classes = sorted(df["emotion"].unique().tolist())
    label2idx = {label: idx for idx, label in enumerate(emotion_classes)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"\nLoading {model_id}...")
    if is_lalm:
        # Load the full model once, keep only the audio encoder submodule,
        # and drop the second half.
        processor = AutoProcessor.from_pretrained(model_id)
        feature_extractor = processor.feature_extractor
        full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        # transformers <5.0 keeps audio_tower/language_model directly on the
        # ForConditionalGeneration class; >=5.0 nests them under `.model`. Handle both.
        backbone = full_model.model if hasattr(full_model, "model") else full_model
        model = backbone.audio_tower
        del backbone.language_model
        model = model.to(device).eval()

        def encode(signal: np.ndarray) -> torch.Tensor:
            inputs = feature_extractor(signal, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=model.dtype)
            with torch.no_grad():
                return model(input_features).last_hidden_state.squeeze(0).float().cpu()
    else:
        processor = spec.processor_cls.from_pretrained(spec.model_id)
        model = spec.model_cls.from_pretrained(spec.model_id).to(device).eval()

        def encode(signal: np.ndarray) -> torch.Tensor:
            inputs = processor(
                signal,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                return model(input_values).last_hidden_state.squeeze(0).cpu()

    embeddings, labels, file_paths = [], [], []

    for i, (idx, row) in enumerate(df.iterrows()):
        file_path = idx if isinstance(idx, str) else idx[0]
        label_int = label2idx[row["emotion"]]

        signal, _ = audiofile.read(file_path, always_2d=False)
        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")

        hidden = encode(signal)

        embeddings.append(hidden)
        labels.append(label_int)
        file_paths.append(file_path)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}")

    T_audio = embeddings[0].shape[0]
    print(f"\nDone. T_audio={T_audio}, hidden_dim={hidden_dim}, samples={len(embeddings)}")

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "file_paths": file_paths,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "T_audio": T_audio,
            "hidden_dim": hidden_dim,
            "encoder": encoder_name,
            "model_id": model_id,
        },
        output_path,
    )
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio encoder embeddings from EMoDB.")
    all_encoders = list(ENCODERS) + list(LALM_MODELS)
    parser.add_argument(
        "--encoder",
        choices=all_encoders,
        default="wav2vec2-base",
        help=f"Encoder to use (default: wav2vec2-base). Options: {', '.join(all_encoders)}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path (default: embeddings/<encoder>_embeddings.pt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples (for smoke-testing).",
    )
    args = parser.parse_args()
    extract(args.encoder, args.output, args.limit)
