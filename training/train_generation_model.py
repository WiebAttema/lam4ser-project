import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from data.generation_dataset import EmoDBGenerationDataset
from data.dataset import speaker_independent_split
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
    lora_rank: int = 0,
    lora_lr: float = 1e-4,
) -> dict:
    ds = DATASET_CONFIGS[dataset]
    tag = f"{encoder}_{prompt_type}_generation"

    if lora_rank > 0:
        tag += f"_lora{lora_rank}"

    checkpoint_dir = ds["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    return {
        "dataset": dataset,
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": 128 if "feature" in prompt_type else 96,
        "lora_rank": lora_rank,
        "lora_lr": lora_lr,
        "embeddings_path": f"embeddings/{ds['embeddings_prefix']}{encoder}_embeddings.pt",
        "batch_size": 4,
        "lr": 1e-5,
        "epochs": 100,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": ds["val_speakers"],
        "test_speakers": ds["test_speakers"],
        "checkpoint_path": f"{checkpoint_dir}/{tag}_best.pt",
    }


def smoke_test(config):
    audio_dim = 768
    prompt_len = config["max_prompt_length"]

    input_ids = torch.randint(0, 50256, (2, prompt_len))
    audio = torch.randn(2, 50, audio_dim)

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    )

    logits = model(input_ids, audio)

    assert logits.shape[0] == 2, f"Expected batch size 2, got {logits.shape[0]}"
    assert logits.shape[1] == prompt_len, f"Expected seq len {prompt_len}, got {logits.shape[1]}"
    assert logits.shape[2] == 50257, f"Expected GPT-2 vocab size 50257, got {logits.shape[2]}"

    print("✓ Generation smoke test passed")


def train(config):
    if not os.path.exists(config["embeddings_path"]):
        preprocessing_script = f"models/audio_encoder/preprocessing_{config['dataset']}.py"
        print(
            f"ERROR: '{config['embeddings_path']}' not found. "
            f"Run {preprocessing_script} first to generate the embeddings file."
        )
        sys.exit(1)

    dataset = EmoDBGenerationDataset(
        embeddings_path=config["embeddings_path"],
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

    audio_dim = dataset.embeddings[0].shape[-1]

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

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

    best_val_loss = float("inf")

    print("\nGeneration training configuration:")
    print(f"  Dataset:      {config['dataset']}")
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
            labels = batch["labels"].to(device)
            audio = batch["audio"].to(device)

            audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                audio = batch["audio"].to(device)

                audio_compressed = compressor(audio)

                logits = model(input_ids, audio_compressed)

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "encoder": config["encoder"],
                    "prompt_type": config["prompt_type"],
                    "max_prompt_length": config["max_prompt_length"],
                    "lora_rank": config["lora_rank"],
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "idx2label": dataset.idx2label,
                    "label2idx": dataset.label2idx,
                    "config": config,
                },
                config["checkpoint_path"],
            )

            print(f"  ✓ Saved best generation checkpoint (val_loss: {epoch_val_loss:.4f})")

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    print(f"\nBest checkpoint saved at epoch {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")

    test_loss = evaluate_loss(
        model=model,
        compressor=compressor,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test LM loss: {test_loss:.4f}")


def evaluate_loss(model, compressor, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio = batch["audio"].to(device)

            audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            total_loss += loss.item()

    return total_loss / len(loader)


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
        default="wavlm-large",
        choices=[
            "wav2vec2-base",
            "wav2vec2-large-emotion",
            "wavlm-large",
            "hubert-large",
        ],
        help="Which encoder's embeddings to train on.",
    )

    parser.add_argument(
        "--prompt_type",
        default="generation",
        choices=[
            "generation",
            "feature_generation",
        ],
        help="Generation prompt template to use.",
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
        encoder=args.encoder,
        dataset=args.dataset,
        prompt_type=args.prompt_type,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
    )

    smoke_test(config)
    train(config)
