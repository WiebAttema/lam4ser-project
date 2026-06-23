"""
Prompt templates for LAM4SER.

This file centralizes all prompt variants used by both:
1. classifier-based AudioGPT2
2. autoregressive label generation
"""

# EMoDB (7-class):
EMODB_LABELS = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
]

# AIBO (5-class, IS2009 Emotion Challenge):
AIBO_LABELS = [
    "anger",
    "emphatic",
    "neutral",
    "positive",
    "rest",
]

# Default kept for backwards compatibility with evaluation scripts that import LABELS directly.
LABELS = AIBO_LABELS


PROMPTS = {
    "base": (
        "Classify the emotion of this speech:"
    ),

    "label_list": (
        "Classify the emotion of this speech. "
        "Possible labels are {label_text}."
    ),

    "feature": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}."
    ),

    "feature_speaker": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}."
    ),

    "generation": (
        "Classify the emotion of this speech. "
        "Possible labels are {label_text}. "
        "Answer with one label only:"
    ),

    "feature_generation": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}. "
        "Answer with one label only:"
    ),
}


def get_prompt(
    prompt_type: str,
    features: str | None = None,
    labels: list[str] | None = None,
) -> str:
    """
    Build a prompt string from the selected prompt type.

    Args:
        prompt_type:
            One of: base, label_list, feature, generation, feature_generation

        features:
            Textual acoustic feature description, for example:
            "high pitch, high energy, short duration"

        labels:
            Emotion class names to list in the prompt.
            Defaults to LABELS (AIBO 5-class) when not provided.

    Returns:
        Prompt string.
    """
    if prompt_type not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Available prompt types: {list(PROMPTS.keys())}"
        )

    text = PROMPTS[prompt_type]

    if "{label_text}" in text:
        text = text.replace("{label_text}", ", ".join(labels if labels is not None else LABELS))

    if "{features}" in text:
        if features is None:
            raise ValueError(
                f"Prompt type '{prompt_type}' requires acoustic feature text, "
                "but features=None was provided."
            )
        text = text.replace("{features}", features)

    return text
