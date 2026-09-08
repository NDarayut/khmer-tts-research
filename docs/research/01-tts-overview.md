# 1. How TTS Works

## 1.1 The basic pipeline

Almost every modern TTS system, regardless of architecture, breaks the text→speech problem into the same three stages:

```
Raw text ──▶ [Frontend / Text Processing] ──▶ [Acoustic Model] ──▶ [Vocoder] ──▶ Waveform
```

**1. Frontend / text processing.** Normalizes raw text and turns it into a sequence the model can consume:
- **Text normalization** — expanding numbers, dates, abbreviations, currency ("$5" → "five dollars").
- **Grapheme-to-phoneme (G2P) conversion** — mapping written characters to phonemes (sound units), or, in "end-to-end" systems, skipping this and feeding raw characters/graphemes directly to the model.
- **Word/syllable segmentation** — necessary for languages without whitespace-delimited words (Khmer, Thai, Chinese, Japanese, Lao).

**2. Acoustic model.** Converts the linguistic sequence into an intermediate acoustic representation — historically a mel-spectrogram, more recently a sequence of discrete audio "tokens" from a neural codec. This is where most of the modeling complexity, model size, and architectural differentiation lives.

**3. Vocoder.** Converts the intermediate acoustic representation into an actual waveform. Modern GAN-based vocoders (HiFi-GAN, BigVGAN) or codec decoders do this in a single fast forward pass. Many recent systems fuse the acoustic model and vocoder into one end-to-end network (e.g., VITS), or use a neural codec's own decoder.

## 1.2 Architecture families

| Family | How it works | Examples |
|---|---|---|
| **Autoregressive attention (seq2seq)** | An RNN/Transformer generates mel frames one step at a time, attending over the text; needs a separate vocoder | Tacotron 2, older systems |
| **Non-autoregressive duration-based** | A duration predictor decides how many frames each phoneme gets, then all frames are generated in parallel (fast, stable, but historically less expressive) | FastSpeech, FastSpeech2 |
| **Flow-based end-to-end** | Normalizing flows model the mel/waveform distribution directly; VITS fuses this with a GAN vocoder and a stochastic duration predictor, trained end-to-end from text to waveform in one model | **VITS** (used by Meta's MMS-TTS), Glow-TTS |
| **Diffusion-based** | Iteratively denoises a noise signal into a mel-spectrogram or latent, conditioned on text/speaker; typically higher quality but slower unless distilled | Grad-TTS, NaturalSpeech 2/3, StyleTTS2 (hybrid diffusion+GAN) |
| **Flow-matching** | A more efficient, ODE-based alternative to diffusion — fewer sampling steps for comparable quality; dominant in newer zero-shot systems | Voicebox, E2-TTS, F5-TTS, Matcha-TTS |
| **Neural-codec language models** | A neural audio codec (e.g., a residual-VQ codec similar to EnCodec/DAC) first compresses speech into discrete tokens; an autoregressive LM (often a repurposed LLM backbone) then predicts those tokens from text, the way an LLM predicts text tokens. A codec decoder turns tokens back into audio. This is the dominant paradigm for large-scale **zero-shot voice cloning** because in-context "prompting" with a few seconds of reference audio works the same way few-shot prompting works for text LLMs | VALL-E, Bark, **Fish Audio S1/S2** (Dual-AR: a "slow" AR model for semantics + a "fast" AR model for acoustic detail) |
| **Tokenizer-free hybrid (AR + diffusion)** | Avoids hard discrete tokens (which lose acoustic detail) *and* raw continuous mel regression (which accumulates error over long sequences) by using a semi-discrete/continuous latent predicted autoregressively for prosody/semantics, then refined by a local diffusion decoder for fine acoustic detail | **VoxCPM2** (LocEnc → TSLM → RALM → LocDiT design, built on the MiniCPM-4 LLM backbone) |

## 1.3 Why this matters for Khmer

Two architectural facts directly affect Khmer viability:

- **VITS-family models (like MMS-TTS) need per-language training** — there is no shared cross-lingual representation, so quality for any one language is a direct function of how much data *that language* got, with no benefit from data in other languages.
- **Neural-codec-LM and flow-matching/diffusion foundation models (Fish Audio, VoxCPM2) train jointly across dozens of languages on a shared backbone.** In principle this lets a well-resourced language "lift" a low-resource one via cross-lingual transfer — but in practice, as shown in [Model 5](05-model-fish-audio-s2.md), a language can still be included in the tokenizer/language list while remaining severely under-trained if its share of the multi-million-hour corpus is negligible. Being "on the list" is not the same as being well-supported — see [Section 3](03-evaluation-benchmarking.md) and the model reports for how to actually check.

## References
- [Kim et al., "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (2021)](https://arxiv.org/abs/2106.06103)
- [Modal — "The Top Open-Source Text to Speech (TTS) Models"](https://modal.com/blog/open-source-tts)
- [Zilliz — "What are the standard evaluation metrics for TTS quality?"](https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality)
