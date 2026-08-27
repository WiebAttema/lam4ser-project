# LAM4SER

Large Audio Models for Speech Emotion Recognition.

This repository holds the code for AudioGPT2, a model that fuses frozen audio
encoder embeddings into a frozen GPT-2 through cross-attention adapters, and for
the interpretability study built on top of it. The study asks where emotion
information actually sits inside an audio encoder: which layer carries it, what
that layer represents in terms of acoustic descriptors, and whether individual
sparse-autoencoder units are causally responsible for single emotion classes.

The short version of the result is that the audio representation decides the
outcome and the fusion layers add little on top of it, so most of the work here
is about the encoder rather than the architecture around it.

## Setup

```bash
pip install -r requirements.txt
```

The two corpora are not distributed here. EMoDB wavs go in `data/`, and AIBO is
read from wherever `AIBO_DATA_DIR` points. Generated files (embeddings,
checkpoints, CSVs, figures) are gitignored, so a fresh clone starts from raw
audio.

## Pipeline

Everything runs off pre-extracted embeddings, so step 1 comes first and the rest
can be run in any order.

### 1. Extract embeddings

Runs offline, once per encoder per dataset. Writes to
`embeddings/{aibo_}{encoder}_embeddings.pt`.

```bash
python models/audio_encoder/preprocessing_emodb.py --encoder wavlm-large
python models/audio_encoder/preprocessing_aibo.py  --encoder wavlm-large
python models/audio_encoder/preprocessing_aibo.py  --encoder qwen2-audio --layerwise
```

`--layerwise` stores one mean-pooled vector per layer instead of only the final
layer. The layer sweep and everything in `sae/` need it. It is available on the
AIBO script only, which is fine because all of that analysis is AIBO only.

Supported encoders:

| Key | Model | Dim | Layers |
|---|---|---|---|
| `wav2vec2-base` | facebook/wav2vec2-base-960h | 768 | 12 |
| `wav2vec2-large-emotion` | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | 1024 | 12 |
| `wavlm-large` | microsoft/wavlm-large | 1024 | 24 |
| `hubert-large` | facebook/hubert-large-ls960-ft | 1024 | 24 |
| `qwen2-audio` | Qwen/Qwen2-Audio-7B (audio tower) | 1280 | 32 |
| `audio-flamingo-3` | nvidia/audio-flamingo-3-hf (audio tower) | 1280 | 32 |

The last two are Whisper-style towers taken pre-projection, so they stay
comparable to the SSL encoders. They pad every clip to a fixed 1500-position
window and break if that padding is disabled, so output frames are trimmed back
to the clip's true duration before pooling. The EMoDB script covers the first
five; `audio-flamingo-3` is wired up for AIBO only.

### 2. Train AudioGPT2

```bash
python training/train_base_model.py --dataset aibo --encoder wavlm-large
python training/train_base_model.py --dataset aibo --encoder wavlm-large --lora_rank 8
```

GPT-2 stays frozen. Four cross-attention adapter blocks are inserted after
blocks 2, 5, 8 and 11, and audio is mean-pooled to 50 tokens before injection.
That gives about 10.1M trainable parameters with a 768-dim encoder and 13.2M
with a 1024-dim one. Rank-8 LoRA on `c_attn` and `c_proj` adds another 442K.

Training uses AdamW at lr 1e-5, weight decay 1e-2, batch size 8, gradient
clipping at 1.0, label smoothing 0.1 and linear warmup over the first 10% of
steps. Class weights are `balanced ** 0.6`. AIBO runs 30 epochs and selects
checkpoints on validation UAR.

### 3. Baselines

```bash
python baselines/svm_mfcc.py        --dataset aibo
python baselines/embedding_probes.py --dataset aibo --encoder wavlm-large
python baselines/compare.py         --dataset aibo
```

`svm_mfcc.py` is an SVM on 84 hand-crafted features and uses no encoder.
`embedding_probes.py` trains a linear and an MLP probe on mean-pooled frozen
embeddings. `compare.py` aggregates everything into one table.

### 4. Evaluate

```bash
python evaluation/evaluate.py         --dataset aibo --encoder wavlm-large
python evaluation/compare_encoders.py --dataset aibo
```

### 5. Logit lens

Where inside the model the decision is formed. Applies the trained classifier
head to intermediate hidden states at 8 checkpoints, once with real audio and
once with the audio tokens zeroed, and measures how far apart the two
distributions are.

```bash
python interpretability/logit_lens.py --dataset aibo --encoder wavlm-large
python interpretability/analyze_results.py --results interpretability/outputs/...
python interpretability/duration_correlation.py --results ... --embeddings ...
```

### 6. Layer sweep

Probes every layer of an encoder, not just the last one.

```bash
python baselines/layer_sweep.py --dataset aibo --encoders wavlm-large qwen2-audio --resume
```

Each probe keeps its best epoch on validation UAR, but the CSV stores test
metrics only, so ranking layers by that CSV ranks them on test. Treat a single
best layer as an optimistic number and prefer an average over a depth band.

### 7. Sparse autoencoders

TopK SAEs on the layer the sweep picked, then two things built on them: what the
dictionary units correspond to acoustically, and whether removing a class's
units breaks that class.

```bash
python sae/train_sae.py       --dataset aibo --encoders wavlm-large --sparsities 75 90 99
AIBO_DATA_DIR=/data/... python sae/extract_egemaps.py
python sae/disentanglement.py --dataset aibo --cv
python sae/selective_units.py --targets emphatic positive
python sae/factor_units.py    --families loudness quality
```

`disentanglement.py --cv` calibrates the Lasso penalty per factor. Without it a
single penalty constrains the 4096-dim sparse code and the 1024-dim original
representation about equally hard, and the compactness difference between them
mostly disappears.

`train_sae.py` has no mechanism to revive a dictionary unit once it stops being
selected, so longer training loses more units rather than fewer. Periodic
resampling of dead units would fix it and is not implemented.

## Layout

```
data/          dataset loading, speaker-independent splits, prompt text
models/
  audio_encoder/   offline embedding extraction per corpus
  compression/     mean-pools audio frames to a fixed 50 tokens
  fusion/          cross-attention adapter and the block wrapping it
  audio_gpt2.py    the full model: frozen GPT-2 plus adapters plus classifier
training/      training loop
evaluation/    checkpoint evaluation and cross-encoder comparison
baselines/     SVM, probes, per-layer sweep, aggregation
interpretability/  logit lens and its plots
sae/           sparse autoencoders, eGeMAPS disentanglement, unit ablations
tests/         shape and smoke tests
```

## Datasets

**EMoDB.** 816 acted German utterances, 10 speakers, 7 emotions. Speaker
independent: train on 11-16 (493), validate on 09-10 (161), test on 03 and 08
(162). Speaker IDs come from the first two characters of the filename.

**FAU AIBO.** 18,216 German segments from children talking to a robot, 5
emotions, following the INTERSPEECH 2009 split: 24 Ohm speakers for training
(9,068), two held-out Ohm speakers for validation (891), all 25 Mont speakers
for test (8,257). The test set is 65% neutral, which is why UAR is the metric
that matters and accuracy on its own is misleading.

## Tests

```bash
python -m pytest tests/
```
