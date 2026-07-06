import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from data.dataset import EmoDBFusionDataset, speaker_independent_split
from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2


DATASET_CONFIGS = {
    "emodb": {
        "embeddings_prefix": "",
        "checkpoint_dir": "checkpoints",
        "num_classes_hint": 7,
        "val_speakers": ["09", "10"],
        "test_speakers": ["03", "08"],
        "epochs": 60,
        "selection_metric": "f1",
        "min_selection_epoch": 1,
    },
    "aibo": {
        "embeddings_prefix": "aibo_",
        "checkpoint_dir": "checkpoints_AIBO",
        "num_classes_hint": 5,
        "val_speakers": ["Ohm_31", "Ohm_32"],
        "test_speakers": [f"Mont_{i:02d}" for i in range(1, 26)],
        "epochs": 30,
        "selection_metric": "uar",
        "min_selection_epoch": 5,
    },
}


def _build_config(
    encoder: str,
    dataset: str = "aibo",
    lora_rank: int = 0,
    lora_lr: float = 1e-4,
    prompt_type: str = "base",
) -> dict:
    ds = DATASET_CONFIGS[dataset]
    tag = f"{encoder}_{prompt_type}"

    if lora_rank > 0:
        tag += f"_lora{lora_rank}"

    checkpoint_dir = ds["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    return {
        "dataset": dataset,
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": 64 if "feature" in prompt_type else 32,
        "lora_rank": lora_rank,
        "lora_lr": lora_lr,
        "embeddings_path": f"embeddings/{ds['embeddings_prefix']}{encoder}_embeddings.pt",
        "batch_size": 8,
        "lr": 1e-5,
        "epochs": ds["epochs"],
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": ds["val_speakers"],
        "test_speakers": ds["test_speakers"],
        "num_classes_hint": ds["num_classes_hint"],
        "selection_metric": ds["selection_metric"],
        "min_selection_epoch": ds["min_selection_epoch"],
        "checkpoint_path": f"{checkpoint_dir}/{tag}_best.pt",
        "loss_curve_path": f"{checkpoint_dir}/{tag}_loss_curve.png",
    }


def smoke_test(config):
    audio_dim = 768
    prompt_len = config["max_prompt_length"]
    num_classes = config["num_classes_hint"]

    input_ids = torch.randint(0, 50256, (2, prompt_len))
    audio = torch.randn(2, 50, audio_dim)

    model = AudioGPT2(
        num_classes=num_classes,
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    )

    logits = model(input_ids, audio)

    assert logits.shape == (2, num_classes), f"Expected logits shape (2, {num_classes}), got {logits.shape}"

    print("✓ Smoke test passed")


def train(config):
    if not os.path.exists(config["embeddings_path"]):
        preprocessing_script = f"models/audio_encoder/preprocessing_{config['dataset']}.py"
        print(
            f"ERROR: '{config['embeddings_path']}' not found. "
            f"Run {preprocessing_script} first to generate the embeddings file."
        )
        sys.exit(1)

    dataset = EmoDBFusionDataset(
        config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
    )

    train_idx, val_idx, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    device = config["device"]

    compressor = AudioCompressor(target_len=config["target_audio_len"]).to(device)

    num_classes = len(dataset.label2idx)
    audio_dim = dataset.embeddings[0].shape[-1]

    model = AudioGPT2(
        num_classes=num_classes,
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    ).to(device)

    train_labels = [dataset[i]["label"].item() for i in train_idx]

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(num_classes),
        y=train_labels,
    ) ** 0.6

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
        label_smoothing=0.1,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config["lora_rank"] > 0:
        lora_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and n.endswith((".A", ".B"))
        ]

        other_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not n.endswith((".A", ".B"))
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": config["lr"]},
                {"params": lora_params, "lr": config["lora_lr"]},
            ],
            weight_decay=1e-2,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=1e-2,
        )

    epochs = config["epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses, val_losses = [], []
    selection_metric = config["selection_metric"]
    best_metric = -1.0

    print("\nTraining configuration:")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Selection:    val_{selection_metric}")
    print(f"  Encoder:      {config['encoder']}")
    print(f"  Prompt type:  {config['prompt_type']}")
    print(f"  Prompt length:{config['max_prompt_length']}")
    print(f"  LoRA rank:    {config['lora_rank']}")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {config['checkpoint_path']}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                audio_compressed = compressor(audio)
                logits = model(input_ids, audio_compressed)
                loss = criterion(logits, labels)

                epoch_val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        epoch_val_loss /= len(val_loader)

        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        val_uar = recall_score(all_labels, all_preds, average="macro")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_f1: {val_f1:.4f} | "
            f"val_uar: {val_uar:.4f}"
        )

        current_metric = val_uar if selection_metric == "uar" else val_f1

        if epoch >= config["min_selection_epoch"] and current_metric > best_metric:
            best_metric = current_metric

            torch.save(
                {
                    "epoch": epoch,
                    "dataset": config["dataset"],
                    "encoder": config["encoder"],
                    "prompt_type": config["prompt_type"],
                    "max_prompt_length": config["max_prompt_length"],
                    "lora_rank": config["lora_rank"],
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_uar": val_uar,
                    "idx2label": dataset.idx2label,
                    "label2idx": dataset.label2idx,
                    "config": config,
                },
                config["checkpoint_path"],
            )

            print(f"  ✓ Saved best checkpoint (val_{selection_metric}: {current_metric:.4f})")

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)

            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    test_uar = recall_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    print(f"\nTest results (best checkpoint, epoch {checkpoint['epoch']}):")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Encoder:      {config['encoder']}")
    print(f"  Prompt type:  {config['prompt_type']}")
    print(f"  Accuracy:     {test_acc:.4f}")
    print(f"  Weighted F1:  {test_f1:.4f}")
    print(f"  UAR (macro recall): {test_uar:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {label_names}")
    print(cm)

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training and validation loss "
        f"({config['dataset']}, {config['encoder']}, {config['prompt_type']})"
    )
    plt.legend()
    plt.savefig(config["loss_curve_path"])
    plt.close()

    print(f"\nLoss curve saved to {config['loss_curve_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="aibo",
        choices=list(DATASET_CONFIGS),
        help="Which dataset to train on.",
    )

    parser.add_argument(
        "--encoder",
        default="wav2vec2-base",
        choices=[
            "wav2vec2-base",
            "wav2vec2-large-emotion",
            "wavlm-large",
            "hubert-large",
            "qwen2-audio",
            "audio-flamingo-3",
        ],
        help="Which encoder's embeddings to train on.",
    )

    parser.add_argument(
        "--prompt_type",
        default="base",
        choices=[
            "base",
            "label_list",
            "feature",
            "feature_speaker",
        ],
        help="Prompt template to use.",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="LoRA rank for GPT-2 attention layers (0 = disabled).",
    )

    parser.add_argument(
        "--lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters.",
    )

    args = parser.parse_args()

    config = _build_config(
        args.encoder,
        dataset=args.dataset,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
        prompt_type=args.prompt_type,
    )

    smoke_test(config)
    train(config)
