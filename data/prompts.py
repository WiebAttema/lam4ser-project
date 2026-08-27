"""Prompt templates for AudioGPT2."""

PROMPTS = {
    "base": "Classify the emotion of this speech:",
}


def get_prompt(prompt_type: str) -> str:
    """Return the prompt text for the given type."""
    if prompt_type not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Available prompt types: {list(PROMPTS.keys())}"
        )
    return PROMPTS[prompt_type]
