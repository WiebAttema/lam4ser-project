"""Extract encoder embeddings from the FAU AIBO corpus (IS2009 Emotion Challenge, 5-class) and
save as a .pt file.

Usage:
    python models/audio_encoder/preprocessing_aibo.py --encoder wavlm-large
    python models/audio_encoder/preprocessing_aibo.py --encoder wav2vec2-large-emotion
    python models/audio_encoder/preprocessing_aibo.py  # defaults to wav2vec2-base
    python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --limit 5  # smoke test
    python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --pooled
    python models/audio_encoder/preprocessing_aibo.py --encoder qwen2-audio --layerwise
"""
import argparse
import math
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
# Whisper-style feature extractors: 10ms mel hop at 16 kHz, so one mel frame
# corresponds to 160 input samples. Used to locate the real clip inside the
# padded window when trimming in layerwise mode.
MEL_HOP_LENGTH = 160
# Floor for un-padded SSL inputs -- the wav2vec2/WavLM conv stack needs >=400
# samples to produce a single output frame.
MIN_SAMPLES = 640
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


def _pool_true_frames(layers, frac: float) -> torch.Tensor:
    """Mean-pool each (1, T, D) hidden state over only the frames that fall
    inside the real (un-padded) clip.

    Encoder output frames map linearly to input time (the conv front-end is
    strictly local; attention mixes content, not position), so the first
    ceil(T * frac) frames of a layer cover the true clip. Layers may run at
    different frame counts (e.g. Qwen2-Audio's post-pooler output), hence the
    per-layer computation.
    """
    pooled = []
    for h in layers:
        n_real = max(1, math.ceil(h.shape[1] * frac))
        pooled.append(h[0, :n_real].float().mean(dim=0).cpu())
    return torch.stack(pooled)


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
    pooled: bool = False, layerwise: bool = False) -> str:
    if pooled and layerwise:
        raise ValueError("--pooled and --layerwise are mutually exclusive.")
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
        suffix = "_layerwise_pooled" if layerwise else ("_pooled" if pooled else "")
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
    shapes_printed = False
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
            nonlocal shapes_printed
            inputs = feature_extractor(
                signal, sampling_rate=SAMPLING_RATE, return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device=device, dtype=model.dtype)
            with torch.no_grad():
                if encoder_name == "audio-flamingo-3":
                    # AudioFlamingo3Encoder requires the feature-extractor mask,
                    # passed as `input_features_mask`.
                    mask = inputs.attention_mask.to(device)
                    out = model(input_features, input_features_mask=mask,
                                output_hidden_states=layerwise)
                else:
                    # Qwen2AudioEncoder does not support input masking (padding
                    # silence in the log-mel is ignored by design). Newer
                    # transformers versions forward `attention_mask` into the
                    # attention layers, which expect a 4D mask and crash on the
                    # 2D feature-extractor mask -- so don't pass it at all.
                    out = model(input_features, output_hidden_states=layerwise)
                if not layerwise:
                    hidden = out.last_hidden_state.squeeze(0).float().cpu()
                    # Mean pool if requested
                    return hidden.mean(dim=0, keepdim=True) if pooled else hidden
            # The feature extractor pads the mel spectrogram to the encoder's
            # fixed window; only frames inside the real clip go into the mean.
            window_samples = input_features.shape[-1] * MEL_HOP_LENGTH
            frac = min(1.0, len(signal) / window_samples)
            # hidden_states = embedding output + one entry per transformer layer
            # (pre-pooler frame rate for Qwen2-Audio); last_hidden_state is the
            # encoder's final output (post-pooler/layer-norm), kept as an extra row.
            layers = list(out.hidden_states) + [out.last_hidden_state]
            if not shapes_printed:
                frames = sorted({h.shape[1] for h in layers})
                print(f"  [layerwise] {len(layers)} rows (incl. final output), "
                      f"frames per layer {frames}, real-frame fraction {frac:.3f}")
                shapes_printed = True
            return _pool_true_frames(layers, frac)
    else:
        processor = spec.processor_cls.from_pretrained(spec.model_id)
        model = spec.model_cls.from_pretrained(spec.model_id).to(device).eval()

        def encode(signal: np.ndarray) -> torch.Tensor:
            nonlocal shapes_printed
            inputs = processor(
                signal,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                out = model(input_values, output_hidden_states=layerwise)
            if not layerwise:
                hidden = out.last_hidden_state.squeeze(0).cpu()
                return hidden.mean(dim=0, keepdim=True) if pooled else hidden
            # Input is un-padded in layerwise mode, so every frame is real speech.
            if not shapes_printed:
                print(f"  [layerwise] {len(out.hidden_states)} rows, "
                      f"{out.hidden_states[0].shape[1]} frames (un-padded input)")
                shapes_printed = True
            return torch.stack(
                [h[0].float().mean(dim=0).cpu() for h in out.hidden_states]
            )

    embeddings, labels, file_paths, durations = [], [], [], []

    for i, (file_path, emotion) in enumerate(index):
        label_int = label2idx[emotion]

        signal, _ = audiofile.read(file_path, always_2d=False)
        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        elif not layerwise:
            # Layerwise mode keeps the true length: SSL encoders take the
            # un-padded signal directly, Whisper-style feature extractors pad
            # to their fixed window internally (zeros either way).
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")
        durations.append(len(signal) / SAMPLING_RATE)
        if layerwise and len(signal) < MIN_SAMPLES:
            signal = np.pad(signal, (0, MIN_SAMPLES - len(signal)), mode="constant")

        hidden = encode(signal)

        embeddings.append(hidden)
        labels.append(label_int)
        file_paths.append(file_path)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(index)}")

    payload = {
        "labels": labels,
        "file_paths": file_paths,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "hidden_dim": hidden_dim,
        "encoder": encoder_name,
        "model_id": model_id,
        "pooled": pooled,
    }
    if layerwise:
        stacked = torch.stack(embeddings)  # (N, n_layers, hidden_dim)
        n_layers = stacked.shape[1]
        if is_lalm:
            layer_labels = (
                ["embed"] + [f"layer_{j}" for j in range(1, n_layers - 1)] + ["final"]
            )
        else:
            layer_labels = ["conv"] + [f"layer_{j}" for j in range(1, n_layers)]
        payload.update(
            embeddings=stacked,
            layerwise=True,
            n_layers=n_layers,
            layer_labels=layer_labels,
            durations=durations,
        )
        print(f"\nDone. n_layers={n_layers}, hidden_dim={hidden_dim}, samples={len(embeddings)}")
    else:
        T_audio = embeddings[0].shape[0]
        payload.update(embeddings=embeddings, T_audio=T_audio)
        print(f"\nDone. T_audio={T_audio}, hidden_dim={hidden_dim}, samples={len(embeddings)}")

    torch.save(payload, output_path)
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
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help=(
            "Save one mean-pooled vector per encoder layer per sample, shape "
            "(n_layers, hidden_dim), pooled over true-speech frames only "
            "(padding trimmed)."
        ),
    )
    args = parser.parse_args()
    extract(args.encoder, args.output, args.limit, args.pooled, args.layerwise)
