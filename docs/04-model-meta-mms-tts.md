# 4. Model: Meta MMS-TTS — Khmer checkpoint (`facebook/mms-tts-khm`)

| | |
|---|---|
| **Org** | Meta AI (FAIR) |
| **Project** | Massively Multilingual Speech (MMS) |
| **Architecture** | VITS (end-to-end flow-based TTS + HiFi-GAN vocoder), ~36.3M parameters |
| **Languages** | 1,107 languages, **one independent checkpoint per language** — Khmer is `khm` |
| **License** | **CC-BY-NC-4.0 — non-commercial only** |
| **Weights** | [huggingface.co/facebook/mms-tts-khm](https://huggingface.co/facebook/mms-tts-khm) |
| **Paper** | [Pratap et al., "Scaling Speech Technology to 1,000+ Languages," JMLR 2024](https://arxiv.org/abs/2305.13516) |

## What it actually is

MMS-TTS is not a single multilingual model in the way VoxCPM2 or Fish Audio are — it's **1,107 separately trained, single-language VITS models** that happen to share a training recipe and a codebase. Khmer users get `facebook/mms-tts-khm`, a small (36M-parameter), self-contained VITS model that takes Khmer text and outputs a waveform directly, with no shared parameters or cross-lingual transfer from any other language in the project.

**VITS architecture:** a posterior encoder + a conditional prior (a Transformer text encoder combined with normalizing-flow coupling layers) + a HiFi-GAN-based decoder, trained end-to-end with adversarial and variational objectives so a single model goes straight from text to waveform (no separate vocoder step at inference). It uses a **stochastic duration predictor**, which means output is non-deterministic — the same input text produces slightly different speech timing on every call unless you fix the random seed.

## Training data

- Sourced from **MMS-lab**: recordings of people reading the **New Testament**, aggregated from Faith Comes By Hearing, goto.bible, and bible.com — chosen because it is one of the only text+audio parallel corpora that exists in translation across well over a thousand languages.
- MMS-lab totals **~44.7K hours across 1,130 languages**, an average of **~40 hours per language** — and for Khmer specifically this is presumably a **single speaker**, since the paper notes MMS-lab recordings are "mostly single speaker."
- Because training 1,107 models had to be computationally tractable, each language's model was trained for **fewer steps** than a dedicated VITS model normally would be, which the authors explicitly note **reduces quality** relative to a fully-tuned single-language system.
- Roughly **85% of the 1,107 languages** met the project's internal CER quality bar; Khmer's individual pass/fail status isn't broken out in public sources.

## How it performs

No Khmer-specific benchmark numbers are published in the model card. What we know about the *methodology* used to evaluate MMS-TTS generally:
- **MCD** (Mel-Cepstral Distortion) against held-out recordings,
- **ASR-based CER** (transcribe the synthesized speech, compare to the input text),
- **MOS** listening tests.

Given the architecture (small VITS, single speaker, ~40h of religious-text audio, reduced training steps), realistic expectations should be: intelligible but **monotone, reading-style prosody**, a single fixed voice with no cloning or expressive control, and audio quality well below what the larger foundation models below produce. It is the most "guaranteed to actually work" option in the sense that it's Khmer-specific rather than an afterthought in a giant multilingual mix, but it is not competitive on naturalness or flexibility.

## Practical notes

- **Licensing is the biggest limitation for real use**: CC-BY-NC-4.0 forbids commercial use outright, unlike VoxCPM2's Apache-2.0.
- Available directly through 🤗 Transformers (`VitsModel`) since v4.33 — the easiest of the three models to get running with minimal setup, and it runs on CPU.
- A community fine-tune exists — [`KrorngAI/mms-tts-khm-finetuned`](https://huggingface.co/KrorngAI/mms-tts-khm-finetuned) — built specifically to improve on the base Khmer checkpoint's quality; worth checking if you land on MMS as your base, though it falls outside this report's "not built specifically for Khmer" scope.
- No voice cloning, no emotional/style control, no streaming — it is a single fixed-voice synthesizer.

## References
- [facebook/mms-tts-khm — Hugging Face model card](https://huggingface.co/facebook/mms-tts-khm)
- [facebook/mms-tts — parent model index](https://huggingface.co/facebook/mms-tts)
- [Pratap et al., "Scaling Speech Technology to 1,000+ Languages," JMLR 2024 (arXiv)](https://arxiv.org/abs/2305.13516)
- [🤗 Transformers MMS docs](https://huggingface.co/docs/transformers/en/model_doc/mms)
- [wannaphong/ttsmms — convenience wrapper for MMS-TTS](https://github.com/wannaphong/ttsmms)
