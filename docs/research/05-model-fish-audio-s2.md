# 5. Model: Fish Audio S2 / S2-Pro (and why S1 doesn't count)

| | |
|---|---|
| **Org** | Fish Audio |
| **Architecture** | Dual-Autoregressive (Dual-AR) neural-codec language model: a 4B-parameter "Slow AR" for semantic prediction + a 400M-parameter "Fast AR" for acoustic detail (~5B params total, BF16), on an in-house neural audio codec |
| **Languages** | **80+**, incl. Khmer — see full list below |
| **License** | **Fish Audio Research License** — free for research/non-commercial use; **commercial use requires a separate paid license** |
| **Weights** | [huggingface.co/fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) |
| **Code** | [github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (Apache-2.0 codebase; weights are separately licensed as above) |
| **Released** | S2-Pro open-sourced March 9, 2026; a follow-up S2.1-Pro extends to 83 languages |

## Important: S1 vs. S2 — the user's original "Fish Audio S2" pointer was correct, S1 is a dead end for Khmer

Fish Audio's earlier flagship, **S1** (4B params, Qwen3-architecture backbone, GRPO-tuned), explicitly lists only **13 languages** — English, Chinese, Japanese, Korean, French, German, Arabic, Spanish, Russian, Dutch, Italian, Polish, Portuguese — and **Khmer is not among them**. S1 tops the HuggingFace TTS-Arena-V2 Elo leaderboard for naturalness/similarity, and was trained on 2M+ hours (100K+ hours with rich emotion/tone/speaker annotation), but it is simply not a Khmer candidate.

**S2 / S2-Pro** is the relevant model: trained on **10M+ hours across 80+ languages**, with Khmer explicitly named in the published language list (alongside e.g. Burmese, Tagalog, Sinhala, Tibetan — other lower-resource Southeast/South Asian languages). This is presumably what the user was recalling.

## Architecture in a bit more detail

Dual-AR is a two-stage neural-codec-LM design: the model first tokenizes speech with a from-scratch audio codec (conceptually similar to Descript's DAC), then a large "Slow AR" transformer predicts coarse semantic/content tokens conditioned on the input text, while a smaller "Fast AR" transformer fills in fine-grained acoustic tokens conditioned on the Slow AR's output. This split lets the bulk of the parameters (4B) focus on getting content and prosody right, while a lighter model (400M) handles acoustic texture — a common pattern for keeping neural-codec-LM inference fast (S2-Pro reports RTF ≈ 0.195, i.e., roughly 5x faster than real time, with ~100ms time-to-first-audio).

## Training data

- **10M+ hours** of audio across **80+ languages**, tiered by data volume/quality: **Tier 1** (Japanese, English, Chinese) gets the best quality; **Tier 2** (Korean, Spanish, Portuguese, Arabic, Russian, French, German) next; everything else — including Khmer — falls into an unranked "additional languages" tier with **no disclosed per-language hour count**.
- Fish Audio does not publish a language-by-language data breakdown, so there's no way to know from official sources how many of the 10M hours are actually Khmer — the benchmark result below is the only real signal available.
- Reinforcement learning (GRPO, carried over from S1's methodology) is used to align output to human preference, but this is generally applied at the aggregate/high-resource-language level, not verified per low-resource language.

## How it performs on Khmer, specifically

Fish Audio does not publish Khmer-specific numbers themselves. The only concrete data point comes from a **third-party comparison run by OpenBMB (VoxCPM2's maker)** as part of their own 30-language internal benchmark:

> **Fish S2-Pro: 75.15% CER on Khmer** (vs. VoxCPM2's 2.05% CER on the same test)

A 75% character error rate is not "lower quality" — it is **functionally broken**: the majority of characters are wrong, indicating the model is likely producing unintelligible or heavily mispronounced Khmer, consistent with Khmer being a nominal list entry rather than a meaningfully-trained language in the 10M-hour mix. Treat this as a strong caution flag rather than a confirmed usable option, pending independent verification.

## Practical notes

- **Licensing**: free for research/personal use; **commercial deployment requires a separate license from Fish Audio** — a real constraint if the end goal is a shipped product.
- 5B-parameter model — meaningfully heavier to self-host than VoxCPM2 (2B) or MMS-TTS-khm (36M).
- Strong general-purpose system (emotion control, voice cloning, streaming, #1 TTS-Arena Elo) — just not, on current evidence, for Khmer.

## References
- [Fish Audio S2 — official product page](https://fish.audio/s2/)
- [fishaudio/s2-pro — Hugging Face model card](https://huggingface.co/fishaudio/s2-pro)
- [fishaudio/openaudio-s1-mini — Hugging Face (S1 language list)](https://huggingface.co/fishaudio/openaudio-s1-mini)
- [Fish Audio Blog — "Launching Fish Audio S1"](https://fish.audio/blog/introducing-s1/)
- [fishaudio/fish-speech — GitHub (codebase, Apache-2.0)](https://github.com/fishaudio/fish-speech)
- [VoxCPM2 README — third-party Khmer CER comparison](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)
