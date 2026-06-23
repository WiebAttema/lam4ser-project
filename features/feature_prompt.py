"""
Convert numeric acoustic features into short textual descriptions.

Main purpose:
    numeric acoustic features
    -> textual feature tokens
    -> GPT-2 tokenizer
    -> input_ids
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

import numpy as np


def acoustic_features_to_text(features: Dict[str, float]) -> str:
    """
    Convert numeric acoustic features into a short text description.

    Example:
        {
            "pitch_mean": 260,
            "energy_mean": 0.09,
            "duration": 1.8,
            "tempo": 120,
        }

        -> "high pitch, high energy, short duration, medium tempo"

    Args:
        features:
            Acoustic feature dictionary from extract_acoustic_features.

    Returns:
        Short textual acoustic description.
    """
    pitch = _describe_pitch(features.get("pitch_mean", 0.0))
    pitch_var = _describe_pitch_variation(features.get("pitch_std", 0.0))
    energy = _describe_energy(features.get("energy_mean", 0.0))
    energy_var = _describe_energy_variation(features.get("energy_std", 0.0))
    duration = _describe_duration(features.get("duration", 0.0))
    tempo = _describe_tempo(features.get("tempo", 0.0))

    parts = [
        pitch,
        pitch_var,
        energy,
        energy_var,
        duration,
        tempo,
    ]

    return ", ".join(parts)




def get_speaker_id_from_path(path: str) -> str:
    """
    EmoDB filenames start with the speaker ID, e.g. 03a01Fa.wav.
    """
    return os.path.basename(str(path))[:2]


def compute_speaker_feature_stats(acoustic_feature_cache, file_paths):
    """
    Compute speaker-wise mean/std for selected acoustic features.

    acoustic_feature_cache:
        list of feature dictionaries, indexed by sample index.

    file_paths:
        list of wav paths, same order as acoustic_feature_cache.
    """
    speaker_values = defaultdict(lambda: defaultdict(list))

    keys = [
        "pitch_mean",
        "pitch_std",
        "energy_mean",
        "energy_std",
    ]

    for idx, path in enumerate(file_paths):
        speaker = get_speaker_id_from_path(path)
        features = acoustic_feature_cache[idx]

        for key in keys:
            value = features.get(key, 0.0)

            if value is not None and np.isfinite(value):
                speaker_values[speaker][key].append(float(value))

    speaker_stats = {}

    for speaker, values_by_key in speaker_values.items():
        speaker_stats[speaker] = {}

        for key in keys:
            values = np.array(values_by_key.get(key, []), dtype=float)

            if len(values) == 0:
                speaker_stats[speaker][key] = {"mean": 0.0, "std": 1.0}
                continue

            mean = float(values.mean())
            std = float(values.std())

            if std < 1e-6:
                std = 1.0

            speaker_stats[speaker][key] = {
                "mean": mean,
                "std": std,
            }

    return speaker_stats


def _zscore(value, mean, std):
    if value is None or not np.isfinite(value):
        return 0.0
    if std < 1e-6:
        return 0.0
    return (float(value) - float(mean)) / float(std)


def _relative_descriptor(z, name):
    if z <= -0.75:
        return f"{name} lower than speaker baseline"
    elif z >= 0.75:
        return f"{name} higher than speaker baseline"
    else:
        return f"{name} around speaker baseline"


def acoustic_features_to_speaker_relative_text(features, speaker_id, speaker_stats):
    """
    Verbalize pitch/energy relative to each speaker's own baseline.

    Duration and tempo are still verbalized with the original absolute rules,
    because they are less directly speaker-baseline dependent.
    """
    stats = speaker_stats.get(speaker_id, {})

    pitch_mean_z = _zscore(
        features.get("pitch_mean", 0.0),
        stats.get("pitch_mean", {}).get("mean", 0.0),
        stats.get("pitch_mean", {}).get("std", 1.0),
    )

    pitch_std_z = _zscore(
        features.get("pitch_std", 0.0),
        stats.get("pitch_std", {}).get("mean", 0.0),
        stats.get("pitch_std", {}).get("std", 1.0),
    )

    energy_mean_z = _zscore(
        features.get("energy_mean", 0.0),
        stats.get("energy_mean", {}).get("mean", 0.0),
        stats.get("energy_mean", {}).get("std", 1.0),
    )

    energy_std_z = _zscore(
        features.get("energy_std", 0.0),
        stats.get("energy_std", {}).get("mean", 0.0),
        stats.get("energy_std", {}).get("std", 1.0),
    )

    parts = [
        _relative_descriptor(pitch_mean_z, "pitch"),
        _relative_descriptor(pitch_std_z, "pitch variation"),
        _relative_descriptor(energy_mean_z, "energy"),
        _relative_descriptor(energy_std_z, "energy variation"),
    ]

    # Keep the old absolute verbalization for duration / tempo by reusing existing function.
    # This assumes your existing acoustic_features_to_text(...) returns a comma-separated string.
    absolute_text = acoustic_features_to_text(features)
    for phrase in absolute_text.split(","):
        phrase = phrase.strip()
        if "duration" in phrase or "tempo" in phrase:
            parts.append(phrase)

    return ", ".join(parts)


def _describe_pitch(pitch_mean: float) -> str:
    """
    Describe mean pitch.

    Thresholds are intentionally simple for the first implementation.
    Later you can replace them with dataset-level quantiles.
    """
    if pitch_mean <= 0:
        return "unknown pitch"
    if pitch_mean < 160:
        return "low pitch"
    if pitch_mean < 240:
        return "medium pitch"
    return "high pitch"


def _describe_pitch_variation(pitch_std: float) -> str:
    """
    Describe pitch variation.
    """
    if pitch_std <= 0:
        return "unknown pitch variation"
    if pitch_std < 25:
        return "stable pitch"
    if pitch_std < 60:
        return "moderate pitch variation"
    return "large pitch variation"


def _describe_energy(energy_mean: float) -> str:
    """
    Describe mean energy.
    """
    if energy_mean <= 0:
        return "unknown energy"
    if energy_mean < 0.03:
        return "low energy"
    if energy_mean < 0.08:
        return "medium energy"
    return "high energy"


def _describe_energy_variation(energy_std: float) -> str:
    """
    Describe energy variation.
    """
    if energy_std <= 0:
        return "unknown energy variation"
    if energy_std < 0.01:
        return "stable energy"
    if energy_std < 0.03:
        return "moderate energy variation"
    return "large energy variation"


def _describe_duration(duration: float) -> str:
    """
    Describe utterance duration.
    """
    if duration <= 0:
        return "unknown duration"
    if duration < 2.0:
        return "short duration"
    if duration < 5.0:
        return "medium duration"
    return "long duration"


def _describe_tempo(tempo: float) -> str:
    """
    Describe tempo.

    For short speech utterances, tempo estimation may be noisy.
    This is acceptable for the first feature-token baseline.
    """
    if tempo <= 0:
        return "unknown tempo"
    if tempo < 90:
        return "slow tempo"
    if tempo < 140:
        return "medium tempo"
    return "fast tempo"
