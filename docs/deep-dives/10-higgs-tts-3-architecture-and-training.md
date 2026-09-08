# 10 — Higgs TTS 3: How It Works, and How to Train It

The second of the two models left standing after the listening evaluation (`evaluation/results_report.html`, §5), and the newest model in this survey. It arrived after the original research pass, which is why [Document 7](../research/07-other-models-checked.md) still lists *Higgs Audio v2* under "no Khmer support" — that verdict was correct for v2 and is superseded here for v3.

This document mirrors [Document 9](09-voxcpm2-architecture-and-training.md) for VoxCPM2: how the model works, and what training it would actually involve. The two answers are very different, and the second one is mostly bad news.

**Sources.** Architecture is read from the checkpoint's `config.json`, the `bosonai/higgs-audio-v2-tokenizer` config, and the `modeling_higgs_multimodal_qwen3.py` remote code shipped with the weights used in the evaluation. Capabilities and licensing come from the [`bosonai/higgs-tts-3-4b` model card](https://huggingface.co/bosonai/higgs-tts-3-4b). The absence of training code was verified by listing the [official repository](https://github.com/boson-ai/higgs-audio) contents directly.

---

## 10.1 Two things to know before reading further

**1. Khmer is undocumented.** Boson report single-digit WER/CER on **102 languages**, split into 85 "polished" and 17 "usable but less polished". Khmer is in **neither list**. It is not a low-tier language for this model; it is an unlisted one. Yet it demonstrably produces intelligible Khmer — verified on the HF Space and confirmed across 100 sentences in our own run.

That makes Khmer an **emergent capability**: real, but unmeasured, unsupported, and carrying no vendor commitment. If a future release silently drops it, nothing was promised. Everything in this document rests on that footing.

**2. The licence is not open in the sense VoxCPM2 is.** Higgs TTS 3 ships under the **Boson Higgs TTS 3 Research and Non-Commercial License**. Production use, hosted APIs, embedding it in a product or service, or reselling it all require a separate commercial licence from Boson. There is a **Creator Use Grant** that permits free use — including monetized podcasts, videos and social posts — provided you credit "Boson AI's Higgs Audio" in the audio or prominently in the accompanying text.

> **The practical fork.** If the deliverable is a product, VoxCPM2 (Apache-2.0) is the only one of the two you can ship without a commercial negotiation. Higgs TTS 3 is for research, evaluation, and creator content. This is a licensing decision, not a quality one, and it is worth settling before any training effort is spent.

---

## 10.2 Architecture

Higgs TTS 3 is a **neural-codec language model** — the same family as Fish Audio S2, and the opposite design choice to VoxCPM2's tokenizer-free continuous latents. Speech is quantized into discrete tokens, and a large autoregressive transformer predicts them exactly as it would predict text.

### The stack

```
Khmer text
  ↓  Qwen3 tokenizer (vocab 151,936) + Higgs control tokens
┌──────────────────────────────────────────────────────┐
│ Qwen3-4B backbone                                     │
│ 36 layers · hidden 2560 · GQA 32 q / 8 kv · head 128 │
│ RoPE θ=1e6 · 32,768 max pos · 8,192 training seq len │
│ tied text embeddings                                  │
└──────────────────────────────────────────────────────┘
   ↑ fused multi-codebook embedding   ↓ fused multi-codebook head
     one [8 × 1026, 2560] tensor,       one [8 × 1026, 2560] linear,
     summed across the codebook axis    reshaped to [L, 8, 1026]
  ↓  8 codebooks × 1026 vocab, 25 fps, delay-patterned
┌──────────────────────────────────────────────────────┐
│ Higgs Audio v2 Tokenizer  (separate checkpoint)       │
│  semantic branch: HuBERT, 12 layers, 768 dim, 16 kHz │
│  acoustic branch: DAC, 1024-entry codebooks,          │
│                   hop 960, downsample ×320            │
└──────────────────────────────────────────────────────┘
  ↓
24 kHz waveform
```

### The tokenizer is the interesting part

Most codec LMs stack a purely acoustic codec (DAC, EnCodec) under the language model, and the LM has to learn what the sounds *mean* on its own. Boson instead trained a **unified tokenizer** that fuses a semantic branch (a HuBERT encoder operating at 16 kHz, which captures phonetic content) with an acoustic branch (a DAC-style residual quantizer, which captures timbre and detail), producing one token stream at 24 kHz.

This is plausibly why Khmer works at all despite being unlisted. A semantically-grounded tokenizer generalizes to phonation it saw little of during LM training, because the phonetic structure is already factored into the token space by the tokenizer rather than having to be learned from scratch by the LM.

### The delay pattern

Eight codebooks must be emitted per 40 ms frame, but an autoregressive model emits one position at a time. The standard solution — used here — is to **stagger** the codebooks so each step predicts one new codebook's token while the others lag behind:

```
codebook c is shifted by c steps
raw [T, 8]  →  delayed [T + 7, 8]
padded with BOC = 1024 before its span and EOC = 1025 after
(hence vocab 1026 = 1024 codes + BOC + EOC)
```

At generation time a small state machine ramps the delay up, watches for EOC, and winds down; the rows are then de-delayed and handed to the codec. **Any training code must apply exactly this transform to its targets** — see §10.4.

### Prompt format

The remote code builds its input sequence as:

```
<|tts|>  [ <|ref_text|> …reference transcript… ]
         [ <|ref_audio|> …one placeholder per ref audio token… ]
         <|text|> …the text to speak…
         <|audio|>   ← generation starts here
```

Audio positions are marked with placeholder id `-100` in the text stream and overwritten with the fused audio embedding before the forward pass. The bracketed parts are the zero-shot voice-cloning path; omit them and the model uses its own default voice, which is how the evaluation ran it.

### Inline control tokens

All tags use `<|category:value|>` syntax and can be inserted mid-utterance:

| Category | Values |
|---|---|
| **Emotion** (21) | `elation`, `amusement`, `enthusiasm`, `determination`, `pride`, `contentment`, `affection`, `relief`, `contemplation`, `confusion`, `surprise`, `awe`, `longing`, `arousal`, `anger`, `fear`, `disgust`, `bitterness`, `sadness`, `shame`, `helplessness` |
| **Style** (3) | `singing`, `shouting`, `whispering` |
| **Sound effects** (9) | `cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`, `sigh`, `sniff`, `sneeze` — pair each with the matching onomatopoeia |
| **Prosody** | speed `very_slow`/`slow`/`fast`/`very_fast` (≈0.65×–1.4×), `pause` (400–700 ms), `long_pause` (700–1500 ms), `pitch_low` (−3 st), `pitch_high` (+2.5 st), `expressive_high`/`expressive_low` |

**Whether these transfer to Khmer is untested.** They were trained on the documented languages; nothing establishes that `<|emotion:sadness|>` produces Khmer-appropriate sad prosody rather than an English-shaped one. This is cheap to check by hand and worth checking before relying on it.

### Versus VoxCPM2, at a glance

| | VoxCPM2 | Higgs TTS 3 |
|---|---|---|
| Params | 2B | 4B |
| Audio representation | Continuous latents (tokenizer-free) | 8 discrete codebooks @ 25 fps |
| Output | 48 kHz | 24 kHz |
| Khmer | **Documented**, 1 of 30, 2.05% CER claimed | **Undocumented**, works anyway |
| Licence | **Apache-2.0** | Research / non-commercial |
| Official fine-tuning | **Yes** — guide, configs, LoRA, validator | **None** |
| Measured RTF (RTX 3060) | 1.382 | **0.745** |
| Measured UTMOS | 2.49 (last) | 2.98 (2nd) |

Neither metric column ranks these models for Khmer — see the evaluation report, §4. They are here to show what was measured, not to pick a winner.

---

## 10.3 The training situation

**Boson ship no training or fine-tuning code for Higgs TTS 3.** This was verified rather than inferred:

- The [official repository](https://github.com/boson-ai/higgs-audio) contains `boson_multimodal/`, `examples/`, and `tech_blogs/`. `examples/` holds `generation.py`, `serve_engine/`, `vllm/`, `voice_prompts/` and `scene_prompts/` — **inference and serving only**. There is no trainer, no dataset loader, no config directory.
- The remote-code class `HiggsMultimodalQwen3ForConditionalGeneration`, shipped with the weights, implements `generate_speech()`, `_build_prompt_ids()`, `_prefill_embeds()` and a sampler state machine. **It has no `forward()` that accepts `labels` and returns a loss.** It is not wired for gradient descent.
- Nothing in the model card, the [Boson blog post](https://www.boson.ai/blog/higgs-audio-v3-tts), or the [LMSYS SGLang-Omni write-up](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/) discusses fine-tuning.
- Training data is undisclosed. v2 was described as pretrained on a 10M-hour corpus ("AudioVerse"); no comparable figure is published for v3, and no per-language breakdown exists for either. There is no way to know how much Khmer, if any, the model saw.

The nearest prior art is [`JimmyMa99/train-higgs-audio`](https://github.com/JimmyMa99/train-higgs-audio), a community LoRA trainer — **for v2, not v3**. v3 changed the architecture (v2's DualFFN is gone; v3 uses a plain Qwen3 backbone with fused multi-codebook embedding and head), so that code is a useful reference for the data pipeline and not a drop-in.

### What writing a trainer would involve

This is tractable — it is a standard codec-LM training loop, and the architecture is fully legible from the shipped remote code — but it is real engineering, not configuration. The pieces:

1. **Encode audio to codes.** Run `bosonai/higgs-audio-v2-tokenizer` over each training clip to get `[T, 8]` integer codes at 25 fps.
2. **Apply the delay pattern.** `apply_delay_pattern()` already exists in the shipped remote code: `[T, 8] → [T+7, 8]`, BOC-padded before each codebook's span and EOC-padded after.
3. **Build the sequence.** `<|tts|> <|text|> …tokens… <|audio|>` followed by the delayed audio positions, with `-100` placeholders where audio embeddings go — reuse `_build_prompt_ids()` and `_prefill_embeds()` verbatim so training and inference agree exactly. A mismatch here is silent and fatal.
4. **Write the missing `forward()`.** Embed via `HiggsFusedMultiTextEmbedding`, run the Qwen3 backbone, project with `HiggsFusedMultiTextHead` to `[L, 8, 1026]`, and take cross-entropy against the next delayed frame — summed or averaged across all 8 codebooks, masking BOC/EOC padding and the entire text span.
5. **Attach LoRA** (via `peft`) to the backbone's attention and MLP projections. Full fine-tuning of 4B params in bf16 needs roughly 60–80 GB with an 8-bit optimizer; LoRA brings it into single-24 GB-card range. Note the fused embedding and head are tied to the text embedding — decide deliberately whether to adapt them.
6. **Validate by generation, not by loss.** Codec-LM loss is a poor guide to output quality; synthesize a held-out set every N steps and listen.

Data requirements would resemble VoxCPM2's — paired 16 kHz-or-better audio and accurate Khmer transcripts, silence-trimmed, 3–30 s clips — and the same corpus scarcity applies. See [§9.4](09-voxcpm2-architecture-and-training.md) for the Khmer corpus survey, which is model-independent.

---

## 10.4 Recommendation

**Do not fine-tune Higgs TTS 3 for Khmer as a first move.** The reasoning is straightforward and does not depend on any judgement about audio quality:

| | VoxCPM2 | Higgs TTS 3 |
|---|---|---|
| Can you ship the result? | Yes, Apache-2.0 | Not without a commercial licence |
| Is there a fine-tuning path? | Official, documented, with a data validator | You would write the trainer |
| Is Khmer supported? | Documented, with a published CER | Undocumented; works, but no commitment |
| Effort to first result | A YAML file and a 24 GB card | Weeks of implementation, then a 24 GB card |

Two models are equally usable by ear; one of them takes a config file to improve and can be shipped, and the other takes a from-scratch trainer and cannot. That is not a close call.

**Where Higgs TTS 3 does earn its place:**

- **As the baseline to beat.** It is the strongest zero-shot Khmer output in this survey obtained with no adaptation at all. Any VoxCPM2 fine-tune should be A/B'd against it; if the fine-tune does not clearly win, the fine-tune is not done.
- **On speed.** 0.745 median RTF against VoxCPM2's 1.382 — nearly twice as fast on the same card, and the only model besides MMS that ran faster than real time. For a latency-bound application that is a real argument.
- **For expressive and creator work.** The control-token vocabulary has no equivalent in VoxCPM2, and the Creator Use Grant covers monetized content with attribution.
- **As the fallback if VoxCPM2 adaptation stalls.** Should Khmer fine-tuning of VoxCPM2 fail to improve on the base model, writing a Higgs trainer becomes the reasonable next investment rather than the first one.

If you do proceed, the first step is not code — it is **confirming with Boson that a commercial licence is obtainable on acceptable terms**, since without one the work cannot be deployed regardless of how well it turns out.

---

## Sources

- [`bosonai/higgs-tts-3-4b` model card](https://huggingface.co/bosonai/higgs-tts-3-4b) — architecture table, 102-language tiers, control tokens, licence and Creator Use Grant
- [Boson AI — Higgs TTS 3 blog post](https://www.boson.ai/blog/higgs-audio-v3-tts)
- [LMSYS — Higgs Audio v3 TTS on SGLang-Omni](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)
- [boson-ai/higgs-audio on GitHub](https://github.com/boson-ai/higgs-audio) — repository contents, verified to contain no training code
- [`JimmyMa99/train-higgs-audio`](https://github.com/JimmyMa99/train-higgs-audio) — community LoRA trainer for **v2**
- `config.json` from `multimodalart/higgs-audio-v3-tts-4b-transformers` — every backbone figure in §10.2
- `config.json` from `bosonai/higgs-audio-v2-tokenizer` — semantic/acoustic branch specs
- `modeling_higgs_multimodal_qwen3.py` — delay pattern, BOC/EOC ids, prompt format, fused embedding and head, and the absence of a loss-returning `forward()`
- Companion documents: [09 — VoxCPM2](09-voxcpm2-architecture-and-training.md), [07 — Other Models Checked](../research/07-other-models-checked.md) (supersedes its Higgs Audio v2 entry), [02 — What Data TTS Needs](../research/02-training-data-requirements.md)
