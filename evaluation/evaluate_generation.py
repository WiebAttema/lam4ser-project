import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data.generation_dataset import EmoDBGenerationDataset
from data.dataset import speaker_independent_split
from data.prompts import get_prompt
from features.acoustic_features import extract_acoustic_features
from features.feature_prompt import acoustic_features_to_text
from models.compression.compressor import AudioCompressor
from models.audio_gpt2_generation import AudioGPT2Generation


DATASET_CONFIGS = {
    "emodb": {
        "embeddings_prefix": "",
        "checkpoint_dir": "checkpoints",
        "val_speakers": ["09", "10"],
        "test_speakers": ["03", "08"],
    },
    "aibo": {
        "embeddings_prefix": "aibo_",
        "checkpoint_dir": "checkpoints_AIBO",
        "val_speakers": ["Ohm_31", "Ohm_32"],
        "test_speakers": [f"Mont_{i:02d}" for i in range(1, 26)],
    },
}


def _build_config(
    encoder: str,
    dataset: str = "aibo",
    prompt_type: str = "generation",
    checkpoint_path: str | None = None,
) -> dict:
    ds = DATASET_CONFIGS[dataset]
    tag = f"{encoder}_{prompt_type}_generation"

    return {
        "dataset": dataset,
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": 128 if "feature" in prompt_type else 96,
        "embeddings_path": f"embeddings/{ds['embeddings_prefix']}{encoder}_embeddings.pt",
        "batch_size": 1,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": ds["val_speakers"],
        "test_speakers": ds["test_speakers"],
        "checkpoint_path": checkpoint_path or f"{ds['checkpoint_dir']}/{tag}_best.pt",
        "max_new_tokens": 5,
    }


def normalize_generated_label(text: str, labels: list[str]) -> str | None:
    """
    Map generated text back to one of the emotion labels.

    Example:
        " anger" -> "anger"
        "The emotion is anger." -> "anger"
        "happy" -> None, because the label is "happiness"
    """
    text = text.lower().strip()

    for label in labels:
        if label in text:
            return label

    return None


def build_prompt_for_eval(dataset, idx: int) -> str:
    """
    Rebuild only the prompt part for generation (without the answer).
    """
    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    if dataset.use_feature_prompt:
        features = dataset.acoustic_feature_cache[idx]
        feature_text = acoustic_features_to_text(features)
        return get_prompt(dataset.prompt_type, features=feature_text, labels=label_names)

    return get_prompt(dataset.prompt_type, labels=label_names)


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    input_ids,
    audio_hidden,
    max_new_tokens: int = 5,
):
    """
    Greedy decoding for a short emotion label.

    Args:
        input_ids:
            Prompt token ids, shape [1, T].

        audio_hidden:
            Compressed audio embeddings, shape [1, T_audio, audio_dim].

    Returns:
        Full generated token ids, including the prompt.
    """
    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated, audio_hidden)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


def evaluate(config):
    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    if not os.path.exists(config["embeddings_path"]):
        raise FileNotFoundError(f"Embeddings file not found: {config['embeddings_path']}")

    device = config["device"]

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    checkpoint_config = checkpoint.get("config", {})

    adapter_dim = checkpoint_config.get("adapter_dim", config["adapter_dim"])
    dropout = checkpoint_config.get("dropout", config["dropout"])
    lora_rank = checkpoint.get("lora_rank", checkpoint_config.get("lora_rank", 0))

    dataset = EmoDBGenerationDataset(
        embeddings_path=config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
    )

    _, _, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )

    test_dataset = Subset(dataset, test_idx)

    loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    audio_dim = dataset.embeddings[0].shape[-1]

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
        lora_rank=lora_rank,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    compressor = AudioCompressor(target_len=config["target_audio_len"]).to(device)

    y_true = []
    y_pred = []
    generated_answers = []
    generated_full_texts = []

    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    print("\nGeneration evaluation configuration:")
    print(f"  Dataset:     {config['dataset']}")
    print(f"  Encoder:     {config['encoder']}")
    print(f"  Prompt type: {config['prompt_type']}")
    print(f"  Device:      {device}")
    print(f"  Checkpoint:  {config['checkpoint_path']}")
    print()

    for local_batch_idx, batch in enumerate(loader):
        original_idx = test_idx[local_batch_idx]

        prompt = build_prompt_for_eval(dataset, original_idx)

        encoded_prompt = tokenizer(
            prompt,
            max_length=config["max_prompt_length"],
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_prompt["input_ids"].to(device)
        audio = batch["audio"].to(device)
        class_label = batch["class_label"].item()

        audio_compressed = compressor(audio)

        generated_ids = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            audio_hidden=audio_compressed,
            max_new_tokens=config["max_new_tokens"],
        )

        full_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        answer_text = full_text[len(prompt):].strip()

        pred_label = normalize_generated_label(answer_text, label_names)

        true_label = dataset.idx2label[int(class_label)]

        y_true.append(true_label)

        if pred_label is None:
            y_pred.append("invalid")
        else:
            y_pred.append(pred_label)

        generated_answers.append(answer_text)
        generated_full_texts.append(full_text)

    all_eval_labels = label_names + ["invalid"]

    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_eval_labels,
        average="weighted",
        zero_division=0,
    )

    invalid_count = sum(1 for pred in y_pred if pred == "invalid")
    validity = 1.0 - invalid_count / max(len(y_pred), 1)

    print("Generation test results:")
    print(f"  Accuracy:                 {acc:.4f}")
    print(f"  Weighted F1:              {weighted_f1:.4f}")
    print(f"  Generated label validity: {validity:.4f}")
    print()

    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=all_eval_labels,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(f"Labels: {all_eval_labels}")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=all_eval_labels,
        )
    )

    print()
    print("Prediction distribution:")
    print(Counter(y_pred))

    print()
    print("Example generations:")
    for i in range(min(10, len(generated_answers))):
        print("-" * 80)
        print(f"True:      {y_true[i]}")
        print(f"Pred:      {y_pred[i]}")
        print(f"Generated: {generated_answers[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="aibo",
        choices=list(DATASET_CONFIGS),
        help="Which dataset the checkpoint was trained on.",
    )

    parser.add_argument(
        "--encoder",
        default="wavlm-large",
        choices=[
            "wav2vec2-base",
            "wav2vec2-large-emotion",
            "wavlm-large",
            "hubert-large",
        ],
        help="Which encoder's embeddings to evaluate on.",
    )

    parser.add_argument(
        "--prompt_type",
        default="generation",
        choices=[
            "generation",
            "feature_generation",
        ],
        help="Generation prompt template to evaluate.",
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional path to generation checkpoint. If not set, uses the default checkpoint name.",
    )

    args = parser.parse_args()

    config = _build_config(
        encoder=args.encoder,
        dataset=args.dataset,
        prompt_type=args.prompt_type,
        checkpoint_path=args.checkpoint_path,
    )

    evaluate(config)
