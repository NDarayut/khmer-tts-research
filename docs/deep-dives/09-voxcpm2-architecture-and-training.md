# 9 — VoxCPM2: How It Works, and How to Train It

Companion to [Model 6](../research/06-model-voxcpm2.md), which covers what VoxCPM2 *is* and what it claims. This document covers the two questions that matter once you have decided to use it: **how the model actually works**, and **what it takes to fine-tune it for Khmer** — the data, the hardware, the config, and the failure modes.

VoxCPM2 is one of the two models left standing after the listening evaluation (see `evaluation/results_report.html`, §5). Of the two, it is the one you can actually ship: **Apache-2.0**, commercial use included, no attribution requirement. Its counterpart, [Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md), is research/non-commercial only.

**Sources.** Architecture facts are read from the checkpoint's own `config.json` and from the `voxcpm` 2.0.3 package source installed in this repo's virtualenv, so they describe the weights actually used in the evaluation. Training facts come from the [official fine-tuning guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) and [FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html). Where the two disagree with the marketing copy, the code wins.

---

## 9.1 The one-paragraph version

VoxCPM2 is a **tokenizer-free, diffusion-autoregressive** TTS model. "Tokenizer-free" means it does not quantize speech into a fixed codebook the way Fish Audio S2 or Higgs TTS 3 do. Instead a 2B-parameter language model predicts *continuous* latent vectors, one per 4-frame patch of audio, and a small diffusion transformer turns each predicted latent into the actual acoustic feature. A separately-trained variational autoencoder then decodes those features to a waveform — taking 16 kHz features in and emitting **48 kHz** audio, so the super-resolution is built into the decoder rather than bolted on.

The practical consequence: there is no discrete audio vocabulary to be badly fitted for a low-resource language. The failure mode that wrecked Fish Audio S2 on Khmer — a codebook that never learned to represent Khmer phonation — has no direct analogue here.

---

## 9.2 Architecture

Everything below is from `config.json` in `openbmb/VoxCPM2`.

### The stack

```
Khmer text
  ↓  byte-level tokenizer (vocab 73,448 — no G2P, no phonemizer, no language tag)
┌─────────────────────────────────────────────────┐
│ MiniCPM4 backbone LM        2048 dim × 28 layers │  ← predicts one latent
│ GQA 16 query / 2 KV heads, LongRoPE to 32k       │     per audio patch
└─────────────────────────────────────────────────┘
  ↓  hidden state
┌─────────────────────────────────────────────────┐
│ Residual LM                 8 layers, no RoPE    │  ← refines the latent
└─────────────────────────────────────────────────┘
  ↓  conditioning vector
┌─────────────────────────────────────────────────┐
│ Local DiT       1024 dim × 12 layers, CFM head   │  ← diffusion: latent → feats
│ euler solver, log-norm schedule, CFG 2.0         │     10 steps at inference
└─────────────────────────────────────────────────┘
  ↓  64-dim features, 4 frames per patch
┌─────────────────────────────────────────────────┐
│ AudioVAE V2     enc 16 kHz  →  dec 48 kHz        │  ← waveform
│ encoder rates [2,5,8,8]  decoder [8,6,5,2,2,2]   │
└─────────────────────────────────────────────────┘
```

There is also a **Local Encoder** (1024 dim × 12 layers) on the input side, which encodes reference audio for voice cloning into the same latent space the LM operates in.

### The numbers that matter for training

| Property | Value | Why it matters |
|---|---|---|
| `patch_size` | 4 | One LM step covers 4 VAE frames. Sequence length is roughly `chars + duration×25/4`. |
| `feat_dim` | 64 | Latent width the DiT predicts. |
| AudioVAE frame rate | 25 fps | 1 second of audio ≈ 25 frames ≈ 6.25 LM positions. |
| `max_length` | 8192 | Training sequence cap. Samples longer than this are dropped by the packer. |
| Encoder sample rate | 16 kHz | **Your training audio must be 16 kHz.** The validator rejects mismatches outright. |
| Output sample rate | 48 kHz | Free super-resolution; you do not supply 48 kHz data. |
| `scalar_quantization_latent_dim` | 512 | Scalar quantization on the latent, not a VQ codebook — this is the "tokenizer-free" part. |
| `inference_cfg_rate` | 2.0 | Classifier-free guidance default; the CLI exposes it as `--cfg-value`. |

### What "tokenizer-free" buys Khmer specifically

Khmer's script is an abugida with no inter-word spacing, stacked subscript consonants, and vowel signs that render on all four sides of a base glyph. Every stage of a conventional TTS frontend struggles with it: there is no eSpeak-NG Khmer voice, so no off-the-shelf G2P; word segmentation needs a dedicated tool like [khmertagger](https://github.com/seanghay/khmertagger).

VoxCPM2 sidesteps all of it. Text goes in as raw bytes through a 73k-entry tokenizer, and the model was trained to map those bytes to sound directly. **You do not need a Khmer phonemizer, a lexicon, or a word segmenter to fine-tune this model.** That single fact removes what is normally the largest chunk of work in a low-resource TTS project.

It also means the model has **no language tag** — you cannot tell it "this is Khmer". It infers the language from the script. In practice this works, and it is why the code-switched half of our evaluation set synthesizes at all: the model switches between Khmer and Latin script mid-sentence without being told.

---

## 9.3 Where Khmer already stands

Khmer (`km`) is **one of VoxCPM2's 30 documented languages** — confirmed in the checkpoint's own model card metadata. OpenBMB report **2.05% CER** for Khmer on their internal benchmark, the strongest published Khmer figure of any model surveyed in this repo.

This changes the shape of the fine-tuning problem entirely, and it is the single most important planning fact in this document:

> **You are not adding Khmer to VoxCPM2. You are improving Khmer that is already there.**

The official guide's headline data requirement — *"500+ hours of target-language data"* — applies to **languages the model does not know**. It does not apply here. For a language already in the training mix, you are doing speaker adaptation, domain adaptation, or prosody correction, and the guide's own recommendations for those are two to three orders of magnitude smaller.

---

## 9.4 What data you need

### The manifest

Training data is a **JSONL file**, one JSON object per line:

```jsonl
{"audio": "clips/km_0001.wav", "text": "សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។"}
{"audio": "clips/km_0002.wav", "text": "ថ្ងៃនេះអាកាសធាតុល្អណាស់។", "ref_audio": "clips/km_0001.wav"}
```

| Field | Required | Notes |
|---|---|---|
| `audio` | **yes** | Path to a WAV. Relative paths resolve against the manifest's own directory. |
| `text` | **yes** | Exact transcript. Raw Khmer — no phonemes, no segmentation, no romanization. |
| `ref_audio` | no | A *different* clip from the same speaker. Trains the voice-cloning path. |
| `duration` | no | Seconds. Supplying it lets the length filter skip opening every file — worth it above a few thousand rows. |
| `dataset_id` | no | Integer tag for mixing corpora; lets the model condition on source. |

Validate before you spend GPU hours — the package ships a checker:

```bash
voxcpm validate --manifest data/train.jsonl --sample-rate 16000 --verbose
```

It reports total hours, duration range, text-length distribution, and hard-fails on missing files, sample-rate mismatches, empty transcripts and malformed JSON.

### Audio requirements

| Requirement | Value | Enforcement |
|---|---|---|
| Sample rate | **16 kHz** | Hard failure in `validate` if it does not match. |
| Format | WAV | Recommended; anything `soundfile` reads will load. |
| Clip duration | **3–30 s** | 30 s is a soft warning (OOM risk); under 0.3 s warns as too short. |
| Trailing silence | **< 0.5 s** | See below — this one is not cosmetic. |
| Loudness | Normalized | Consistent level across the corpus. |

**Trailing silence is the single most-cited failure cause in the official FAQ.** Clips that end with a long silent tail teach the model that utterances do not end, and it learns to keep generating — producing exactly the runaway output that made `fish-s2` unusable in our evaluation. Trim aggressively; this is the highest-value preprocessing step you can do.

### How much

The official guidance, mapped onto what you would actually be doing for Khmer:

| Goal | Method | Data | Realistic for Khmer? |
|---|---|---|---|
| One specific Khmer voice | LoRA `r=32` | 5–50 clips | **Yes** — an afternoon of recording. |
| Better Khmer prosody / a domain (news, IVR, education) | LoRA `r=32–64` | 50–500 clips | **Yes** — the highest-value target. |
| Large-scale Khmer customization | Full fine-tune | 1000+ clips | Plausible with SLR42. |
| Adding a language from scratch | Full fine-tune, lr `1e-5` | **500+ hours** | Not applicable — Khmer is already in. |

### Khmer corpora that exist

| Source | Size | Licence | Notes |
|---|---|---|---|
| [OpenSLR SLR42](https://www.openslr.org/42/) | 866 MB (male set) | CC BY-SA 4.0 | Google-collected, manually QC'd. The closest thing to a standard Khmer TTS corpus. **OpenSLR publishes no hour count, speaker count or sample rate** — download it and measure before planning around it. Mirrored on HF as `deepdml/openslr42-khmer-tts`. |
| [`Panhapich/khmer-tts-processed`](https://huggingface.co/datasets/Panhapich/khmer-tts-processed) | 1k–10k rows | — | Pre-segmented; check provenance before relying on it. |
| [`Panhapich/khmer-english-codeswitch-tts`](https://huggingface.co/datasets/Panhapich/khmer-english-codeswitch-tts) | 1k–10k rows | CC BY 4.0 | Directly matches this project's code-switched eval set. **But it is tagged `synthetic` / `tts-generated`** — it is TTS output, so training on it distils another model's errors, accent and artefacts into yours. Useful for text; treat the audio with suspicion. |
| [KLEA](https://github.com/seanghay/KLEA) | ~3,000 words | — | Word-level, not sentences. Good for pronunciation reference, not for TTS fine-tuning. |

The honest summary: **there is no large, clean, permissively-licensed Khmer TTS corpus.** SLR42 is the only well-attested one and it is a single-gender set of unpublished size. If you want more than a few hours you will be recording it or licensing it. Given that VoxCPM2 already speaks Khmer, recording 200–500 clean sentences from one or two good speakers and running LoRA is a far better use of effort than assembling a large corpus.

---

## 9.5 Fine-tuning: the actual procedure

### Environment

Python 3.10–3.11, PyTorch ≥ 2.5.0, CUDA ≥ 12.0, plus `tensorboardX`, `argbind`, `transformers`, `librosa`.

### VRAM

| | VoxCPM2 |
|---|---|
| **LoRA** | ~20 GB |
| **Full fine-tune** | ~40 GB |

At `batch_size=16`, `max_batch_tokens=8192`. Multi-GPU adds roughly **10 GB per card** for gradient buckets and NCCL buffers — it is not free.

> **Hardware reality check for this repo.** The evaluation ran on a 12 GB RTX 3060. That is **below the LoRA minimum**. Reaching ~20 GB means dropping `batch_size` and `max_batch_tokens` hard and raising `grad_accum_steps` to compensate, and even then 12 GB is marginal. Renting a single 24 GB card (A10G, 3090, 4090) for LoRA — or a 48 GB A6000 for a full fine-tune — is the pragmatic path. Note that `mms` is the only model in the comparison that fine-tunes comfortably on 12 GB.

### LoRA config

`conf/voxcpm_v2/voxcpm_finetune_lora.yaml`:

```yaml
pretrained_path: /path/to/VoxCPM2/
train_manifest:  /path/to/train.jsonl
val_manifest:    /path/to/val.jsonl

sample_rate:     16000      # must match your audio
out_sample_rate: 48000
batch_size:      16
grad_accum_steps: 1
num_workers:     2
num_iters:       1000
log_interval:    10
valid_interval:  500
save_interval:   500

learning_rate: 0.0001       # 1e-4 for LoRA
weight_decay:  0.01
warmup_steps:  100
max_steps:     1000
max_batch_tokens: 8192

save_path:   /path/to/checkpoints/lora
tensorboard: /path/to/logs/lora

lambdas:
  loss/diff: 1.0            # diffusion loss on the latent
  loss/stop: 1.0            # end-of-utterance prediction

lora:
  enable_lm:   true         # adapt the MiniCPM4 backbone
  enable_dit:  true         # adapt the diffusion transformer
  enable_proj: false
  r:     32
  alpha: 32
  dropout: 0.0
```

**Choosing the rank.** `r=32` for cloning a speaker; **`r=64` for style or language work** — which is what Khmer prosody correction is. `alpha` at `r` or `2r`. Keep `enable_lm` and `enable_dit` both on: the backbone carries linguistic behaviour, the DiT carries acoustic detail, and Khmer adaptation wants both. `enable_proj` stays false.

For a **full** fine-tune, delete the `lora:` block entirely and drop the learning rate by 10× to `1e-5`.

### Running it

```bash
# single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# there is also a WebUI wrapper
python lora_ft_webui.py
```

Monitor with `tensorboard --logdir /path/to/logs/lora`. Watch `loss/diff` (should fall steadily), `loss/stop` (should stabilize early), `grad_norm`, and `lr`.

**Stop when the audio sounds right — not when the loss bottoms out.** The FAQ is explicit that on small datasets the model begins ignoring the text input "within a few hundred steps". Checkpoint every 500 steps and listen to each one; the best checkpoint is frequently not the last.

### Using the result

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="/path/to/checkpoints/lora/latest",
)
wav = model.generate(text="សូមស្វាគមន៍")
```

LoRA adapters hot-swap at runtime — `model.load_lora(path)`, `model.set_lora_enabled(False)`, `model.unload_lora()` — so one base model can serve several Khmer voices from a single set of 2B weights. The CLI accepts `--lora-path`, `--lora-r`, `--lora-alpha` on every generation subcommand.

A full fine-tune produces a whole checkpoint instead, loaded as `VoxCPM.from_pretrained("/path/to/checkpoints/full/latest")`.

---

## 9.6 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Output ignores the text; reproduces training clips | Overfitting — the classic small-dataset failure, and it arrives **fast** | Fewer steps, lower LR, more data, lower LoRA rank. Checkpoint often. |
| Generation runs away / long trailing noise | **Trailing silence > 0.5 s in the training clips** | Re-trim the corpus. This is the most common cause, per the official FAQ. |
| Loss will not converge | Bad transcripts, wrong sample rate, misaligned audio | Re-run `voxcpm validate`; spot-check alignment by hand. |
| OOM | Batch too large | Lower `batch_size` / `max_batch_tokens`, raise `grad_accum_steps`. |
| Khmer got better, English/Chinese got worse | Catastrophic forgetting | Mix Chinese/English data into the training set. Or just use LoRA and disable it when synthesizing other languages. |

That last row is a genuine advantage of LoRA for this project: because the adapter is separable, a Khmer LoRA **cannot** damage the base model's other 29 languages. You toggle it off and the original model is back, bit for bit. A full fine-tune has no such escape hatch.

---

## 9.7 Recommended path for Khmer

1. **Record or license 200–500 clean Khmer sentences**, one or two speakers, 16 kHz, 3–15 s each, silence trimmed to under 0.5 s. Prefer this over scraping — corpus quality dominates everything downstream. Avoid synthetic (TTS-generated) audio.
2. **Hold out ~10%** as `val.jsonl` and run `voxcpm validate` on both splits.
3. **LoRA, `r=64`, `alpha=64`, lr `1e-4`, 1000 steps**, on a rented 24 GB card. Save every 500.
4. **Listen to every checkpoint.** Do not select on loss.
5. **Re-run the harness** — `python src/khmer_tts/synthesis/synthesize.py --model voxcpm2` against the fixed 100-sentence set — and A/B the LoRA output against the base model's clips already in `evaluation/results/voxcpm2/audio/`.
6. **Judge by ear.** As the evaluation report establishes at length, no automatic metric in this project can rank Khmer TTS. A blind listening test against the base model is the only thing that will tell you whether the fine-tune helped.

---

## Sources

- [VoxCPM Fine-Tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) — commands, YAML, VRAM, manifest schema
- [VoxCPM Fine-Tuning FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html) — data volumes, forgetting, failure modes
- [OpenBMB/VoxCPM on GitHub](https://github.com/OpenBMB/VoxCPM)
- [`openbmb/VoxCPM2` model card](https://huggingface.co/openbmb/VoxCPM2) — language list, 2M-hour claim, 48 kHz output
- `voxcpm` 2.0.3 package source: `training/validate.py`, `training/data.py`, `modules/layers/lora.py`, `cli.py`
- `config.json` from the `openbmb/VoxCPM2` checkpoint — every architecture figure in §9.2
- [OpenSLR SLR42](https://www.openslr.org/42/) — Khmer TTS corpus
- Companion documents: [06 — Model: VoxCPM2](../research/06-model-voxcpm2.md), [02 — What Data TTS Needs](../research/02-training-data-requirements.md), [10 — Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md)
