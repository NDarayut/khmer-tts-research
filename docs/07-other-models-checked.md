# 7. Other Models Checked — No Khmer Support

Recorded here as a negative-result reference, so this search doesn't need to be repeated. All checked as of September 2026 against official model cards, GitHub repos, or vendor documentation.

| Model | Org | Languages supported | Khmer? | Notes |
|---|---|---|---|---|
| **Fish Audio S1 / S1-mini** | Fish Audio | 13 (EN, ZH, JA, KO, FR, DE, AR, ES, RU, NL, IT, PL, PT) | ❌ No | See [Model 5](05-model-fish-audio-s2.md) — superseded by S2 for this purpose |
| **XTTS-v2** | Coqui | 17 (EN, ES, FR, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH-CN, JA, HU, KO, HI) | ❌ No | Popular self-hosted voice-cloning model; no SE Asian languages at all |
| **Chatterbox Multilingual** | Resemble AI | 20+ (AR, DA, DE, EL, EN, ES, FI, FR, HE, HI, IT, JA, KO, MS, NL, NO, PL, PT, RU, SV, SW, TR, ZH) | ❌ No | MIT license; strong ElevenLabs-competitive benchmarks, but no Khmer |
| **Qwen3-TTS** | Alibaba | 10 (ZH, EN, JA, KO, DE, FR, RU, PT, ES, IT) | ❌ No | |
| **Qwen-Audio-3.0-TTS** | Alibaba | 16 (AR, ZH, EN, FR, DE, ID, IT, JA, KO, MS, PT, RU, ES, TL, TH, VI) | ❌ No | Newer/broader than Qwen3-TTS, covers other SE Asian languages (Thai, Vietnamese, Indonesian, Tagalog, Malay) but not Khmer |
| **Higgs Audio v2** | Boson AI | Reported anywhere from 5 core (EN, ZH, KO, DE, ES) to "50+" depending on source | ❌ Not confirmed | Apache-2.0; 10M-hour "AudioVerse" corpus, but no official source lists Khmer |
| **CosyVoice2** | Alibaba/FunAudioLLM | ZH, EN, JA, KO (+ Chinese dialects) | ❌ No | Strong for code-switching among its 4 core languages, not broadly multilingual |
| **Zonos** | Zyphra | EN, JA, ZH, FR, DE | ❌ No | Apache-2.0 |
| **Kokoro** | community/StyleTTS2-based | Handful of languages (mainly English, plus a few community-added) | ❌ No | Very small (82M params), efficient, but limited language breadth |
| **Bark** | Suno | ~13 (EN, ZH, FR, DE, HI, IT, JA, KO, PL, PT, RU, ES, TR) | ❌ No | |
| **eSpeak-NG** | community | 127+ languages/accents | ❌ No | Classical **rule-based formant synthesizer**, not neural — included here because its huge language count made it worth checking directly; Khmer is not in its supported-language table. If it ever gains Khmer support it would still only provide robotic, non-neural speech, useful at most as a G2P/phonemizer fallback, not a quality TTS voice |
| **SeamlessM4T v2** | Meta | 96 languages for text; only **36 for speech output** | ⚠️ Not confirmed | Primarily a *speech-to-speech translation* model, not a general TTS system; its 36-language speech-output list was not fully enumerable from available sources — worth re-checking directly against [the model card](https://huggingface.co/facebook/seamless-m4t-v2-large) if translation-shaped Khmer output is ever the actual need, but it wasn't confirmed to include Khmer here and is architecturally a translation model rather than a text-in/speech-out TTS system |

## Why this list matters

The pattern across nearly every general-purpose "multilingual" open-source TTS model released in 2025–2026 is convergence on the **same 10–20 languages** — the ones with the most training data available online (major European languages, Chinese, Japanese, Korean, Arabic) — regardless of how large the underlying training corpus is. Model card language counts (13, 17, 20, 30, 80+) mostly track *how aggressively a lab chose to include long-tail languages*, not how much data backs any individual one of them. Khmer only appears at all once a model's list stretches past ~30 languages (VoxCPM2, Fish S2), and even then, list membership doesn't guarantee usable quality — see [Fish Audio S2](05-model-fish-audio-s2.md)'s 75% CER for the clearest illustration of that gap.

## References
All individual language claims above are sourced from each project's own GitHub README, Hugging Face model card, or official product page, retrieved September 2026:
- [coqui-ai/TTS — XTTS docs](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md)
- [resemble-ai/chatterbox — GitHub](https://github.com/resemble-ai/chatterbox)
- [QwenLM/Qwen3-TTS — GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Alibaba Cloud — Qwen-Audio-3.0-TTS blog](https://www.alibabacloud.com/blog/qwen-audio-3-0-tts-more-multilingual-easier-to-direct_603379)
- [boson-ai/higgs-audio — GitHub](https://github.com/boson-ai/higgs-audio)
- [espeak-ng/espeak-ng — languages.md](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
- [facebook/seamless-m4t-v2-large — Hugging Face](https://huggingface.co/facebook/seamless-m4t-v2-large)
