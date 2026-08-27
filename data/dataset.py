import os

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from data.prompts import PROMPTS, get_prompt


def extract_speaker_id(file_path: str) -> str:
    """Speaker ID from a wav filename. EMoDB "03a01Wa" -> "03", AIBO "Mont_01_000_00" -> "Mont_01"."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if basename and basename[0].isdigit():
        return basename[:2]
    parts = basename.split("_")
    if len(parts) < 2:
        return "unknown"
    return f"{parts[0]}_{parts[1]}"


class EmoDBFusionDataset(Dataset):
    """Pre-extracted audio embeddings paired with a fixed tokenized text prompt.

    Returns input_ids, audio and label per sample. Used for both EMoDB and AIBO.
    """

    def __init__(
        self,
        embeddings_path: str,
        prompt_type: str = "base",
        max_length: int = 32,
    ):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"'{embeddings_path}' not found. Run the matching script in "
                "models/audio_encoder/ first to generate the embeddings file."
            )

        if prompt_type not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_type: {prompt_type}. "
                f"Available prompt types: {list(PROMPTS.keys())}"
            )

        self.embeddings_path = embeddings_path
        self.prompt_type = prompt_type
        self.max_length = max_length

        data = torch.load(embeddings_path, weights_only=False)

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]

        # Original wav paths, when stored, drive the speaker-independent split.
        self.file_paths = None
        self.speaker_ids = None

        for key in ("file_paths", "paths", "files"):
            if key in data:
                self.file_paths = data[key]
                self.speaker_ids = [extract_speaker_id(p) for p in self.file_paths]
                break

        if self.speaker_ids is None:
            print(
                "WARNING: No file paths found in embeddings file.\n"
                "Speaker-independent splitting is not available.\n"
                "Falling back to random 70/15/15 split."
            )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.label_names = [self.idx2label[i] for i in range(len(self.idx2label))]

        encoded = self.tokenizer(
            get_prompt(self.prompt_type),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].squeeze(0)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids,
            "audio": self.embeddings[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def speaker_independent_split(dataset, val_speakers=None, test_speakers=None):
    """Split indices by speaker. Falls back to a seeded random 70/15/15 split."""
    if dataset.speaker_ids is None or (val_speakers is None and test_speakers is None):
        torch.manual_seed(42)
        n = len(dataset)
        indices = torch.randperm(n).tolist()
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        print("Random 70/15/15 split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")

        if not train_indices or not val_indices or not test_indices:
            raise ValueError("One or more splits are empty after random 70/15/15 split.")

        return train_indices, val_indices, test_indices

    test_speakers = set(test_speakers)
    val_speakers = set(val_speakers)

    train_indices, val_indices, test_indices = [], [], []

    for i, spk in enumerate(dataset.speaker_ids):
        if spk in test_speakers:
            test_indices.append(i)
        elif spk in val_speakers:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_speakers = sorted(set(dataset.speaker_ids[i] for i in train_indices))

    print("Speaker split summary:")
    print(f"  Train speakers: {train_speakers} -> {len(train_indices)} samples")
    print(f"  Val   speakers: {sorted(val_speakers)} -> {len(val_indices)} samples")
    print(f"  Test  speakers: {sorted(test_speakers)} -> {len(test_indices)} samples")

    if not train_indices:
        raise ValueError("Train split is empty. Check the speaker IDs in the dataset.")
    if not val_indices:
        raise ValueError("Val split is empty. Check the val_speakers argument.")
    if not test_indices:
        raise ValueError("Test split is empty. Check the test_speakers argument.")

    return train_indices, val_indices, test_indices
