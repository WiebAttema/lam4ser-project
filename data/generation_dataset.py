import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from data.prompts import get_prompt
from features.acoustic_features import extract_acoustic_features
from features.feature_prompt import acoustic_features_to_text


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


class EmoDBGenerationDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str,
        prompt_type: str = "generation",
        max_length: int = 96,
    ):
        if not os.path.exists(embeddings_path):
            print(
                f"ERROR: '{embeddings_path}' not found. "
                "Run models/audio_encoder/preprocessing.py first to generate the embeddings file."
            )
            raise FileNotFoundError(f"'{embeddings_path}' not found")

        if prompt_type not in ("generation", "feature_generation"):
            raise ValueError(
                "EmoDBGenerationDataset only supports prompt_type='generation' "
                "or prompt_type='feature_generation'. "
                f"Got: {prompt_type}"
            )

        self.embeddings_path = embeddings_path
        self.prompt_type = prompt_type
        self.max_length = max_length
        self.use_feature_prompt = "feature" in prompt_type

        data = torch.load(embeddings_path, weights_only=False)

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]

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
                "feature_generation requires wav file paths, but no key among "
                "('file_paths', 'paths', 'files') was found in the embeddings file."
            )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.acoustic_feature_cache = None

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

        self.label_names = [self.idx2label[i] for i in range(len(self.idx2label))]
        self.input_ids_list = []
        self.lm_labels_list = []
        self.class_labels_list = []

        for idx in range(len(self.embeddings)):
            input_ids, lm_labels = self._build_generation_sample(idx)
            self.input_ids_list.append(input_ids)
            self.lm_labels_list.append(lm_labels)
            self.class_labels_list.append(
                torch.tensor(self.labels[idx], dtype=torch.long)
            )

    def _label_to_text(self, label_idx: int) -> str:
        if isinstance(self.idx2label, dict):
            return str(self.idx2label[int(label_idx)])

        return str(self.idx2label[int(label_idx)])

    def _build_prompt_for_sample(self, idx: int) -> str:
        if self.use_feature_prompt:
            features = self.acoustic_feature_cache[idx]
            feature_text = acoustic_features_to_text(features)
            return get_prompt(self.prompt_type, features=feature_text, labels=self.label_names)

        return get_prompt(self.prompt_type, labels=self.label_names)

    def _build_generation_sample(self, idx: int):
        prompt = self._build_prompt_for_sample(idx)
        label_text = self._label_to_text(self.labels[idx])

        answer = " " + label_text
        full_text = prompt + answer

        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        full_encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_encoded["input_ids"].squeeze(0)
        lm_labels = input_ids.clone()

        prompt_len = prompt_encoded["input_ids"].shape[1]

        lm_labels[:prompt_len] = -100
        lm_labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, lm_labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "labels": self.lm_labels_list[idx],
            "audio": self.embeddings[idx],
            "class_label": self.class_labels_list[idx],
        }
