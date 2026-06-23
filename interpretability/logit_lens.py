"""
Logit-lens analysis for AudioGPT2.

Applies the classifier head to the hidden state before and after each of
the 4 audio fusion adapters (GPT-2 layers 2, 5, 8, 11), once with real
audio and once with the audio zeroed out. Saves per-sample results to
results.json for analyze_results.py and duration_correlation.py to use.

Usage:
    python interpretability/logit_lens.py --dataset aibo --encoder wavlm-large
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data.dataset import EmoDBFusionDataset, speaker_independent_split
from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2


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


def build_config(dataset, encoder, prompt_type):
    ds = DATASET_CONFIGS[dataset]
    return {
        "embeddings_path": f"embeddings/{ds['embeddings_prefix']}{encoder}_embeddings.pt",
        "checkpoint_path": f"{ds['checkpoint_dir']}/{encoder}_{prompt_type}_best.pt",
        "val_speakers": ds["val_speakers"],
        "test_speakers": ds["test_speakers"],
        "prompt_type": prompt_type,
        "max_prompt_length": 64 if "feature" in prompt_type else 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": f"interpretability/outputs/{dataset}_{encoder}_{prompt_type}",
    }


class FusionHook:
    """Captures a fusion block's text hidden state before/after the audio is injected."""

    def __init__(self):
        self.pre = None
        self.post = None
        self.attn = None
        self.handles = []

    def attach(self, block):
        self.handles.append(block.register_forward_pre_hook(self._pre))
        self.handles.append(block.register_forward_hook(self._post))

    def detach(self):
        for h in self.handles:
            h.remove()

    def _pre(self, _module, args):
        self.pre = args[0].detach().clone()

    def _post(self, _module, _args, output):
        fused, attn = output
        self.post = fused.detach().clone()
        self.attn = attn.detach().clone()


@torch.no_grad()
def forward_collect(model, input_ids, audio, hooks):
    """One forward pass. Returns the 8 hidden states (pre/post each
    adapter) and the 4 cross-attention maps."""
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    hidden = model.gpt2.wte(input_ids) + model.gpt2.wpe(position_ids)

    fusion_iter = iter(model.fusion_blocks)
    for i, block in enumerate(model.gpt2.h):
        out = block(hidden)
        hidden = out if isinstance(out, torch.Tensor) else out[0]
        if i in model.fusion_indices:
            hidden, _ = next(fusion_iter)(hidden, audio)

    states = []
    for hook in hooks:
        states += [hook.pre, hook.post]
    return states, [hook.attn for hook in hooks]


@torch.no_grad()
def label_probs(states, model, label_names):
    """Classifier head applied to the last token of every checkpoint."""
    out = []
    for hidden in states:
        logits = model.classifier(hidden[:, -1, :])
        probs = F.softmax(logits[0], dim=-1)
        out.append({l: probs[i].item() for i, l in enumerate(label_names)})
    return out


def run(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    device = config["device"]

    dataset = EmoDBFusionDataset(
        config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
    )
    _, _, test_idx = speaker_independent_split(
        dataset, val_speakers=config["val_speakers"], test_speakers=config["test_speakers"],
    )
    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    ckpt = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    model = AudioGPT2(
        num_classes=len(label_names),
        audio_dim=dataset.embeddings[0].shape[-1],
        adapter_dim=ckpt_cfg.get("adapter_dim", 64),
        dropout=ckpt_cfg.get("dropout", 0.3),
        lora_rank=ckpt.get("lora_rank", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {config['checkpoint_path']} (epoch {ckpt.get('epoch', '?')})")

    hooks = [FusionHook() for _ in model.fusion_blocks]
    for hook, block in zip(hooks, model.fusion_blocks):
        hook.attach(block)
    compressor = AudioCompressor(target_len=50).to(device)

    results = []
    print(f"Running logit lens on {len(test_idx)} test samples...")
    for n, sample_idx in enumerate(test_idx):
        if n % 1000 == 0:
            print(f"  {n}/{len(test_idx)}")

        sample = dataset[sample_idx]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        audio = compressor(sample["audio"].unsqueeze(0).to(device))
        true_label = label_names[sample["label"].item()]

        states_real, attn = forward_collect(model, input_ids, audio, hooks)
        states_zero, _ = forward_collect(model, input_ids, torch.zeros_like(audio), hooks)

        real_probs = label_probs(states_real, model, label_names)
        zero_probs = label_probs(states_zero, model, label_names)
        pred_label = max(real_probs[-1], key=real_probs[-1].get)

        results.append({
            "sample_idx": int(sample_idx),
            "true_label": true_label,
            "correct": pred_label == true_label,
            "real": real_probs,
            "zero": zero_probs,
            "attn_weights": [a[0].tolist() if a is not None else None for a in attn],
        })

    for hook in hooks:
        hook.detach()

    out_path = os.path.join(config["output_dir"], "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f)

    acc = sum(r["correct"] for r in results) / len(results)
    print(f"Saved {out_path}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aibo", choices=list(DATASET_CONFIGS))
    parser.add_argument("--encoder", default="wavlm-large")
    parser.add_argument("--prompt_type", default="base")
    args = parser.parse_args()

    run(build_config(args.dataset, args.encoder, args.prompt_type))
