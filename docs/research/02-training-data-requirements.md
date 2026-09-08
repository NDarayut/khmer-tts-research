# 2. What Data TTS Needs

## 2.1 The core requirement

At minimum, a TTS model needs **paired (text, audio) data**: audio recordings with accurate, time-aligned transcripts. Beyond that, requirements scale with what kind of system you're building:

| Goal | What you need | Typical scale |
|---|---|---|
| Single-speaker, single-language, high quality (classic Tacotron/VITS fine-tune) | Clean studio recordings from one speaker, consistent mic/room, accurate transcripts | **~10–25 hours** is enough for good quality (e.g., the LJSpeech dataset, ~24h, is the standard benchmark for this) |
| Multi-speaker, single language, voice cloning within that language | Many speakers, ideally speaker-labeled, moderate quality acceptable | Tens to hundreds of hours |
| Massively multilingual foundation model (zero-shot, cross-lingual, in-context voice cloning) | Millions of hours across dozens–hundreds of languages, plus rich metadata (emotion/tone tags, speaker diarization, event tags) for the largest systems | **1–10 million+ hours** (VoxCPM2: 2M+ hrs across 30 languages; Fish Audio S2: 10M+ hrs across 80+ languages; Higgs Audio v2: 10M-hr "AudioVerse") |
| Bootstrapping a genuinely **low-resource** language from near-zero | Any available parallel text+audio, even non-studio — read scripture, audiobooks, sermons, radio, forced-aligned subtitles | Tens of hours can work if paired with transfer learning from related/well-resourced languages (see MMS below) |

## 2.2 What "data" actually consists of

1. **Audio** — ideally 22kHz+ sample rate, single-speaker-per-file, minimal background noise/music, consistent loudness.
2. **Transcripts** — orthographically correct text matching the audio. For high-precision systems, transcripts are **forced-aligned** to the audio at the phoneme or word level (tools like the Montreal Forced Aligner, MFA) so the model or duration predictor can learn exact timing.
3. **A G2P system or phoneme lexicon**, unless the model is trained end-to-end on raw graphemes. Building one requires either linguistic expertise or a rule-based/statistical fallback (see [eSpeak-NG](https://github.com/espeak-ng/espeak-ng), which explicitly does **not** yet cover Khmer — see [Section 7](07-other-models-checked.md)).
4. For multi-speaker/cloning systems: **speaker labels or diarization**, and ideally metadata like emotion, speaking style, and recording condition — this is what lets modern systems like Fish Audio S1 (100k+ hours annotated with emotion/tone/speaker tags) do expressive, controllable synthesis.
5. **Word/syllable segmentation** for scripts without whitespace-delimited words — directly relevant to Khmer, whose script (an abugida derived from Brahmi, like Thai/Lao) does not separate words with spaces. Tools like [khmertagger](https://github.com/seanghay/khmertagger) exist specifically to handle Khmer text normalization/segmentation for speech pipelines.

## 2.3 How low-resource languages actually get covered

There are two dominant strategies among the models surveyed here:

**Strategy A — one model per language, small dedicated dataset per language (Meta MMS-TTS).** Meta's MMS project trained **1,107 separate VITS checkpoints**, one per language, using the **MMS-lab** dataset: recordings of people reading the New Testament, sourced from Faith Comes By Hearing, goto.bible, and bible.com, covering ~44.7K hours in total — an average of only **~40 hours per language**, almost always from a **single speaker**. Quality is capped by that single speaker's recording and by reduced training steps used to make training 1,107 models tractable; the MMS authors note this explicitly degrades quality versus a dedicated, fully-tuned single-language VITS. Roughly 85% of the 1,107 languages met the project's own CER quality bar — meaning ~15% (a long tail that plausibly includes very low-resource languages) did not. See [Model 4](04-model-meta-mms-tts.md).

**Strategy B — one shared multilingual foundation model, uneven data at scale (VoxCPM2, Fish Audio S2).** Instead of training per-language models, a single large model is trained jointly on a multi-million-hour, multi-language corpus, hoping cross-lingual transfer covers thinner languages. This works well *when* the thin language still gets a meaningful share of that corpus and the tokenizer/phonemizer handles its script properly — VoxCPM2's own reported 2.05% CER for Khmer suggests this happened. It works poorly when a language is added to the supported list (for coverage/marketing) without a commensurate data share — Fish Audio S2-Pro's 75.15% CER on the same Khmer benchmark is the cautionary example (see [Model 5](05-model-fish-audio-s2.md)). **The takeaway: for foundation models, "listed as supported" and "actually well-trained" are different claims, and only benchmark numbers (not the language list) tell you which one you're getting.**

## 2.4 What exists for Khmer specifically

- **[OpenSLR SLR42](https://www.openslr.org/42/)** — the main open Khmer TTS dataset, collected by Google (2016–2018), multi-speaker, CC BY-SA 4.0, manually quality-checked. This is the closest thing to a "standard" Khmer TTS corpus and is what most Khmer-specific TTS efforts (outside the models covered here) build on.
- **[KLEA](https://github.com/seanghay/KLEA)** — open-source Khmer *word*-level (not sentence-level) speech model, trained on the kheng.info online audio dictionary (~3,000 recordings). Useful for lexicon/pronunciation work, not full TTS.
- Khmer's inclusion in MMS-lab (Bible-reading corpus) is presumably what backs `facebook/mms-tts-khm` — likely on the order of the ~40-hour project average, single speaker, not independently confirmed in the model card.
- Neither Fish Audio nor VoxCPM2 publish a language-by-language breakdown of how many hours of their multi-million-hour corpora are Khmer — this is the single biggest transparency gap in evaluating these systems for Khmer specifically, and is why the empirical benchmark numbers in the model reports matter more than the marketing language lists.

## References
- [Ito & Johnson, LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Pratap et al., "Scaling Speech Technology to 1,000+ Languages" (MMS paper), JMLR 2024](https://arxiv.org/abs/2305.13516)
- [OpenSLR SLR42 — High quality TTS data for Khmer](https://www.openslr.org/42/)
- [seanghay/KLEA — Khmer Word to Speech Model](https://github.com/seanghay/KLEA)
- [seanghay/khmertagger — Inverse Text Normalization for Khmer ASR](https://github.com/seanghay/khmertagger)
