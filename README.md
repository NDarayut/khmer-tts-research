# Multilingual Open-Source TTS for Khmer (ខ្មែរ)

A research report surveying **open-source, multilingual text-to-speech (TTS) models that are capable of synthesizing Khmer speech**, even though Khmer was not their primary design target. It covers how TTS systems work, what data they need, how they're benchmarked, and a detailed, sourced breakdown of every candidate model found — including which ones turned out **not** to support Khmer at all (useful negative results for anyone doing this search again).

**TL;DR:** Of the ~15 open-source multilingual TTS systems surveyed, only **three** have any real claim to Khmer support, and only **one** is genuinely usable for it:

| Model | Org | Khmer in official language list? | Real-world Khmer quality | License (commercial use) |
|---|---|---|---|---|
| **[VoxCPM2](docs/06-model-voxcpm2.md)** | OpenBMB | ✅ Yes (1 of 30) | **Good** — 2.05% CER on internal benchmark | Apache-2.0 — free |
| **[Fish Audio S2 / S2-Pro](docs/05-model-fish-audio-s2.md)** | Fish Audio | ✅ Yes (1 of 80+) | **Poor** — 75.15% CER (effectively broken) on the same benchmark | Research License — NC only, paid for commercial |
| **[Meta MMS-TTS (khm)](docs/04-model-meta-mms-tts.md)** | Meta AI | ⚠️ Not "multilingual" in the usual sense — one VITS checkpoint per language, 1,107 languages incl. Khmer | Unbenchmarked publicly; architecture is basic VITS, single speaker, robotic prosody | CC-BY-NC-4.0 — **non-commercial only** |

Everything else checked — Fish Audio S1, XTTS-v2, Chatterbox Multilingual, Qwen3-TTS / Qwen-Audio-3.0-TTS, Higgs Audio v2, CosyVoice2, Zonos, Kokoro, Bark, eSpeak-NG — **does not support Khmer** as of this writing (Sept 2026). See [07 — Models That Don't Support Khmer](docs/07-other-models-checked.md) for the full negative-result list, which is worth keeping so this ground doesn't get re-covered.

## Contents

1. [How TTS Works](docs/01-tts-overview.md) — the pipeline (frontend → acoustic model → vocoder) and the major architecture families (VITS, FastSpeech-style, diffusion, flow-matching, neural-codec LMs like the ones behind Fish Audio and VoxCPM2)
2. [What Data TTS Needs](docs/02-training-data-requirements.md) — data requirements from single-speaker fine-tunes to 2M-hour foundation models, and what's actually available for Khmer
3. [How TTS Is Benchmarked](docs/03-evaluation-benchmarking.md) — MOS/CMOS/SMOS, WER/CER, speaker similarity (SIM), UTMOS/DNSMOS, RTF, and the standard benchmark suites (Seed-TTS-eval, CV3-eval, TTS-Arena)
4. [Model: Meta MMS-TTS (Khmer checkpoint)](docs/04-model-meta-mms-tts.md)
5. [Model: Fish Audio S2 / S2-Pro](docs/05-model-fish-audio-s2.md)
6. [Model: VoxCPM2](docs/06-model-voxcpm2.md)
7. [Other Models Checked (No Khmer Support)](docs/07-other-models-checked.md)
8. [Khmer-Specific Considerations & Recommendation](docs/08-khmer-considerations-and-recommendation.md)

## Scope & method

This report only covers models that are (a) open-source with downloadable weights, (b) usable outside a paid API (even if commercial use requires a separate license), and (c) multilingual by design rather than built specifically for Khmer. Khmer-only projects (e.g. [KLEA](https://github.com/seanghay/KLEA), the [OpenSLR SLR42](https://www.openslr.org/42/) dataset, community MMS fine-tunes) are mentioned only as supporting context, not as primary subjects.

All facts are sourced inline in each document, pulled from official model cards, GitHub repos, blog posts, and the underlying papers as of **September 2026**. TTS is moving fast — treat version-specific numbers (parameter counts, language counts, benchmark scores) as a snapshot, and check the linked sources for updates.

---
*Compiled September 2026.*
