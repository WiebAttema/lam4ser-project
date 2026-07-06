"""Extract encoder embeddings from the FAU AIBO corpus (IS2009 Emotion Challenge, 5-class) and
save as a .pt file.

Usage:
    python models/audio_encoder/preprocessing_aibo.py --encoder wavlm-large
    python models/audio_encoder/preprocessing_aibo.py --encoder wav2vec2-large-emotion
    python models/audio_encoder/preprocessing_aibo.py  # defaults to wav2vec2-base
    python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --limit 5  # smoke test
    python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --pooled
"""
import argparse
import os
from dataclasses import dataclass
from typing import Type

import audiofile
import numpy as np
import torch
from transformers import (
    AutoProcessor,
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

# Audio encoders pulled out of LALMs
LALM_MODELS: dict[str, str] = {
    "qwen2-audio": "Qwen/Qwen2-Audio-7B",
    "audio-flamingo-3": "nvidia/audio-flamingo-3-hf",
}
# d_model of the Whisper-style encoder used by all our LALM_MODELS
LALM_HIDDEN_DIM = 1280

# Model class per LALM encoder
def _load_lalm_full_model(encoder_name: str, model_id: str):
    if encoder_name == "qwen2-audio":
        from transformers import Qwen2AudioForConditionalGeneration as ModelCls
    elif encoder_name == "audio-flamingo-3":
        from transformers import AudioFlamingo3ForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"No model class registered for LALM encoder {encoder_name!r}")
    return ModelCls.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )


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


def extract(encoder_name: str, output_path: str | None = None, limit: int | None = None,
    pooled: bool = False) -> str:
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
        suffix = "_pooled" if pooled else ""
        output_path = os.path.join(EMBEDDINGS_DIR, f"aibo_{encoder_name}{suffix}_embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder : {encoder_name} ({model_id})")
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

    print(f"\nLoading {model_id}...")
    if is_lalm:
        # Load the full model once, keep only the audio encoder submodule, 
        # and drop the second half.
        processor = AutoProcessor.from_pretrained(model_id)
        feature_extractor = processor.feature_extractor
        full_model = _load_lalm_full_model(encoder_name, model_id)
        backbone = full_model.model if hasattr(full_model, "model") else full_model
        model = backbone.audio_tower
        del backbone.language_model
        if encoder_name == "audio-flamingo-3":
            # AudioFlamingo3Encoder debug
            model = model.float()
        model = model.to(device).eval()

        def encode(signal: np.ndarray) -> torch.Tensor:
            inputs = feature_extractor(
                signal, sampling_rate=SAMPLING_RATE, return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device=device, dtype=model.dtype)
            mask = inputs.attention_mask.to(device)
            with torch.no_grad():
                # Qwen2AudioEncoder ignores masking and takes `attention_mask`; 
                # AudioFlamingo3Encoder requires a mask and takes it as `input_features_mask`
                if encoder_name == "audio-flamingo-3":
                    out = model(input_features, input_features_mask=mask)
                else:
                    out = model(input_features, attention_mask=mask)
                hidden = out.last_hidden_state.squeeze(0).float().cpu()
            # Mean pool if requested
            return hidden.mean(dim=0, keepdim=True) if pooled else hidden
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
                hidden = model(input_values).last_hidden_state.squeeze(0).cpu()
            return hidden.mean(dim=0, keepdim=True) if pooled else hidden

    embeddings, labels, file_paths = [], [], []

    for i, (file_path, emotion) in enumerate(index):
        label_int = label2idx[emotion]

        signal, _ = audiofile.read(file_path, always_2d=False)
        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")

        hidden = encode(signal)

        embeddings.append(hidden)
        labels.append(label_int)
        file_paths.append(file_path)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(index)}")

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
            "pooled": pooled,
        },
        output_path,
    )
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio encoder embeddings from AIBO.")
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
        help="Output .pt path (default: embeddings/aibo_<encoder>_embeddings.pt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples (for smoke-testing).",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help=(
            "Mean-pool each sample to a single (1, hidden_dim) vector at extraction time."
        ),
    )
    args = parser.parse_args()
    extract(args.encoder, args.output, args.limit, args.pooled)
