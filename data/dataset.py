import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from data.prompts import PROMPTS, get_prompt
from features.acoustic_features import extract_acoustic_features
from features.feature_prompt import (
    acoustic_features_to_text,
    acoustic_features_to_speaker_relative_text,
    compute_speaker_feature_stats,
    get_speaker_id_from_path,
)


def extract_speaker_id(file_path: str) -> str:
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if basename and basename[0].isdigit():
        # EMoDB: "03a01Wa" -> "03"
        return basename[:2]
    # AIBO: "Mont_01_000_00" -> "Mont_01"
    parts = basename.split("_")
    if len(parts) < 2:
        return "unknown"
    return f"{parts[0]}_{parts[1]}"


class EmoDBFusionDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str,
        prompt_type: str = "base",
        use_feature_prompt: bool = False,
        max_length: int = 64,
    ):
        if not os.path.exists(embeddings_path):
            print(
                f"ERROR: '{embeddings_path}' not found. "
                "Run models/fusion/preprocessing.py first to generate the embeddings file."
            )
            raise FileNotFoundError(f"'{embeddings_path}' not found")

        if prompt_type not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_type: {prompt_type}. "
                f"Available prompt types: {list(PROMPTS.keys())}"
            )

        self.embeddings_path = embeddings_path
        self.prompt_type = prompt_type
        self.use_feature_prompt = use_feature_prompt or ("feature" in prompt_type)
        self.max_length = max_length

        data = torch.load(embeddings_path, weights_only=False)

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]

        # ------------------------------------------------------------
        # Save original wav file paths if available.
        # These are needed for:
        # 1. speaker-independent split
        # 2. acoustic feature extraction
        # ------------------------------------------------------------
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

        if self.use_feature_prompt and self.file_paths is None:
            raise ValueError(
                "Feature prompt requires wav file paths, but no key among "
                "('file_paths', 'paths', 'files') was found in the embeddings file."
            )

        # ------------------------------------------------------------
        # GPT-2 tokenizer
        # ------------------------------------------------------------
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------------------------------------------------
        # Acoustic feature extraction cache
        #
        # Example:
        # embeddings/wavlm-large_embeddings.pt
        # ->
        # embeddings/wavlm-large_acoustic_features.pt
        # ------------------------------------------------------------
        self.acoustic_feature_cache = None
        self.speaker_feature_stats = None

        if self.use_feature_prompt:
            cache_path = embeddings_path.replace(
                "_embeddings.pt",
                "_acoustic_features.pt",
            )

            if os.path.exists(cache_path):
                print(f"Loading cached acoustic features from: {cache_path}")
                self.acoustic_feature_cache = torch.load(
                    cache_path,
                    weights_only=False,
                )
            else:
                print("Extracting acoustic features from wav files...")
                self.acoustic_feature_cache = []

                for i, wav_path in enumerate(self.file_paths):
                    if i % 50 == 0:
                        print(
                            f"  Extracting acoustic features: "
                            f"{i}/{len(self.file_paths)}"
                        )

                    feature_dict = extract_acoustic_features(wav_path)
                    self.acoustic_feature_cache.append(feature_dict)

                torch.save(self.acoustic_feature_cache, cache_path)
                print(f"Saved acoustic feature cache to: {cache_path}")

        # ------------------------------------------------------------
        # Speaker-wise feature statistics
        #
        # Only used for feature_speaker prompts.
        # This is speaker-wise normalization / relative verbalization,
        # not a mixed-effects model.
        # ------------------------------------------------------------
        if self.use_feature_prompt and "speaker" in self.prompt_type:
            self.speaker_feature_stats = compute_speaker_feature_stats(
                self.acoustic_feature_cache,
                self.file_paths,
            )

        # ------------------------------------------------------------
        # Build prompt tokens per sample.
        #
        # Old version:
        #   one fixed self.input_ids for all samples
        #
        # New version:
        #   self.input_ids_list[idx] can be different for each sample,
        #   especially for acoustic feature prompts.
        # ------------------------------------------------------------
        self.label_names = [self.idx2label[i] for i in range(len(self.idx2label))]
        self.input_ids_list = []

        for idx in range(len(self.embeddings)):
            prompt = self._build_prompt_for_sample(idx)

            encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.input_ids_list.append(encoded["input_ids"].squeeze(0))

    def _build_prompt_for_sample(self, idx: int) -> str:
        """
        Build prompt for one sample.

        base / label_list:
            same prompt for all samples.

        feature:
            acoustic features are loaded or extracted,
            converted into text,
            and inserted into the prompt.

        feature_speaker:
            pitch / energy are described relative to each speaker baseline.
        """
        if self.use_feature_prompt:
            features = self.acoustic_feature_cache[idx]

            if self.prompt_type == "feature_speaker":
                speaker_id = get_speaker_id_from_path(self.file_paths[idx])
                feature_text = acoustic_features_to_speaker_relative_text(
                    features,
                    speaker_id,
                    self.speaker_feature_stats,
                )
            else:
                feature_text = acoustic_features_to_text(features)

            return get_prompt(self.prompt_type, features=feature_text, labels=self.label_names)

        return get_prompt(self.prompt_type, labels=self.label_names)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "audio": self.embeddings[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def speaker_independent_split(dataset, val_speakers=None, test_speakers=None):
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
    print(f"  Train speakers: {train_speakers} → {len(train_indices)} samples")
    print(f"  Val   speakers: {sorted(val_speakers)} → {len(val_indices)} samples")
    print(f"  Test  speakers: {sorted(test_speakers)} → {len(test_indices)} samples")

    if not train_indices:
        raise ValueError("Train split is empty — check speaker IDs in the dataset.")
    if not val_indices:
        raise ValueError("Val split is empty — check val_speakers argument.")
    if not test_indices:
        raise ValueError("Test split is empty — check test_speakers argument.")

    return train_indices, val_indices, test_indices
