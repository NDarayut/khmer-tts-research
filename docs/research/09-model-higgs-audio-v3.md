# 9. Model: Higgs Audio v3 TTS (Boson AI) — generates Khmer, but doesn't claim it

> **Addendum, added after the original 01–08 report.** Higgs Audio **v2** was checked in [07](07-other-models-checked.md) and recorded as "Khmer not confirmed." Boson AI released **v3** on **2026-06-04**, jumping from ~5 documented languages to a published list of **102**, which made it worth re-checking. This document is that re-check plus a full write-up of the model.

**Bottom line: Higgs Audio v3 does produce Khmer speech — confirmed by direct testing — even though Khmer appears nowhere in its official 102-language list.** It is an *undocumented* capability, not an unsupported one. That distinction matters: there is no vendor quality claim to hold it to, so its Khmer output has to be measured rather than trusted. Details and disposition in §9.5.

| | |
|---|---|
| **Org** | Boson AI |
| **Released** | 2026-06-04 |
| **Architecture** | ~4B autoregressive decoder on a **Qwen3** backbone (36 layers, hidden 2560, GQA 32/8) + discrete audio codec head |
| **Audio representation** | 8 codebooks × 1026 vocab, delay pattern; 24 kHz output at 25 fps (40 ms/frame) |
| **Context** | 8,192 tokens |
| **Languages** | **102 claimed** — 85 at WER/CER <5, 17 at WER/CER 5–10. **Khmer is in neither tier, but works anyway** (§9.5) |
| **License** | **Boson Higgs Audio v3 Research and Non-Commercial License** — production/hosted/revenue use needs a separate commercial license (a "Creator Use Grant" permits monetized creative content with attribution) |
| **Weights** | [huggingface.co/bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) (also mirrored as [`bosonai/higgs-tts-3-4b`](https://huggingface.co/bosonai/higgs-tts-3-4b)) |
| **Code** | [github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) — v3 needs no local repo code; served via SGLang-Omni or Boson's API |
| **Announcement** | [boson.ai/blog/higgs-tts-3](https://www.boson.ai/blog/higgs-tts-3), [LMSYS/SGLang-Omni blog](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/) |

## 9.1 What the model is for

Where VoxCPM2 (§6) optimizes for *fidelity of read speech* and Fish S2 (§5) for *voice cloning breadth*, Higgs v3 is explicitly designed as the **speech layer of a conversational voice agent**. Two design choices follow from that:

- **Chat-native streaming.** It can begin synthesis *before a full sentence or punctuation mark arrives*, so it can be driven directly by an upstream LLM's token stream rather than waiting for sentence boundaries. This is the headline feature and the main thing v2 could not do.
- **Inline control tokens.** Delivery is steered by tags embedded in the text stream rather than by separate API parameters — 21 emotions, 3 styles (singing / shouting / whispering), 9 sound effects (cough, laughter, sigh, …), and prosody controls (speed, pitch, pauses, expressiveness):

  ```
  <|emotion:amusement|><|prosody:expressive_high|>Wait, that was hilarious.
  ```

It also does zero-shot voice cloning from a reference clip, like the other models in this report.

## 9.2 Architecture notes

The design is a fairly conventional **neural-codec LM** (the family described in [01 §architectures](01-tts-overview.md)) — the opposite bet from VoxCPM2's tokenizer-free semi-discrete approach:

- A Qwen3-derived autoregressive decoder predicts discrete audio tokens, using **8 codebooks with a delay pattern** so the codebooks can be generated in a staggered fashion within a single AR stream instead of requiring 8 sequential passes.
- Output is **24 kHz** — lower than VoxCPM2's 48 kHz, and a deliberate tradeoff: 25 frames/sec keeps the token count per second of audio low, which is what makes sub-second streaming latency achievable.
- Serving is not a plain AR loop. SGLang-Omni splits it into a **multi-stage pipeline** (preprocess → audio encode → TTS generation → vocode), each stage with its own scheduler, because the stages have very different compute profiles.

## 9.3 Reported performance

All self-reported by Boson AI, though the SGLang-Omni team states it reproduced the results — slightly better provenance than the purely vendor-internal numbers in §5/§6.

| Benchmark | Result |
|---|---|
| Seed-TTS (2 languages), multilingual voice clone | **1.11 WER** |
| CV3-eval (13 languages) | **4.41** WER/CER |
| MiniMax-Multilingual (32 languages) | reported as ahead of comparison models |
| Higgs-Multilingual (internal, **111 languages**) | **3.61%** macro-averaged error rate |
| EmergentTTS (LLM-judge, conversational behaviors) | 53.65% overall win-rate vs baseline; 68.57% on paralinguistics, 61.43% on questions |
| Throughput, 1× H100 @ concurrency 16 | 14.74 req/s, **1079 ms mean latency**, **RTF 0.262**, ~61.8 audio-seconds/second |

For scale against the other models in this report: RTF 0.262 is comparable to VoxCPM2's ~0.30 on a 4090, but that Higgs figure is *batched throughput on an H100 at concurrency 16*, which is not the same measurement — do not read it as a like-for-like single-utterance latency comparison.

## 9.4 Practical notes

- **Not usable commercially without a separate license.** Same category as Fish Audio S2 and MMS-TTS-khm, and unlike VoxCPM2's Apache-2.0.
- No local repo install for v3 — either hit Boson's OpenAI-compatible API or self-host with SGLang-Omni. That's simpler than v2 but adds a serving-stack dependency that the other models in this report don't have.
- Boson publishes **no training-data hours or per-language breakdown** at all, and no per-language error table. This is the same transparency gap as Fish Audio, and it is precisely what makes the Khmer question below unresolvable from public sources.

## 9.5 Khmer: undocumented, but real

**Empirically confirmed.** Higgs Audio v3 generates Khmer speech when given Khmer text — verified by direct testing (September 2026), not from any published source. Everything below explains why you won't find that in the documentation, and what it means for using the model.

### 9.5.1 What the documentation says — nothing

No Boson AI source mentions Khmer anywhere:

| Source | Khmer? |
|---|---|
| HF model card `higgs-audio-v3-tts-4b`, full 102-language list (both tiers) | ❌ Absent |
| HF model card `higgs-tts-3-4b` (mirror), same list | ❌ Absent |
| `boson-ai/higgs-audio` GitHub README | ❌ Not mentioned ("100+ languages", no enumeration) |
| boson.ai launch blog | ❌ Not mentioned |
| docs.boson.ai model overview | ❌ Not mentioned |
| SGLang-Omni cookbook / LMSYS blog | ❌ Not mentioned |

The published list's **Southeast Asian coverage is: Thai, Vietnamese, Indonesian, Malay, Tagalog.** No Khmer, no Lao, no Burmese.

Note also that third-party write-ups asserting Khmer support (e.g. the ComfyUI-Wiki release note, whose "Asian languages" list reads *"…Malay, **Burmese, Khmer, Lao**"*) are **not** evidence for it — those three are exactly the ones absent from the card they cite, so the list appears embellished rather than transcribed. The claim happens to be right about Khmer; it isn't right because that source said so.

### 9.5.2 Why it works anyway

The most likely explanation is the **111 vs. 102 gap**: Boson's internal *Higgs-Multilingual* benchmark covers **111 languages**, but only **102** are listed as supported. Nine languages are evaluated and then not claimed — the obvious reason being that they scored worse than the 5–10 WER/CER bar used to define the second tier. Khmer is very plausibly one of those nine.

That is consistent with what a 4B multilingual codec-LM does in general: broad web-scale audio pretraining picks up languages that never get an official quality commitment. The output exists; the vendor just declines to stand behind it.

**The practical consequence: there is no published Khmer error rate for this model, and no promise that a future version won't regress it.** Compare the other three models in this report, all of which have *some* vendor-side Khmer claim (VoxCPM2: 2.05% CER; Fish S2: 75.15% CER; MMS: a dedicated `khm` checkpoint). Higgs v3 is the only candidate whose Khmer quality is entirely unmeasured — and "generates Khmer" and "generates *good* Khmer" are exactly the two things Fish Audio S2 demonstrates are not the same thing (§5: officially listed, 75% CER, effectively unusable).

### 9.5.3 Disposition for this repo

Khmer output exists, so **the quality question is now answerable by the existing harness** — which is precisely what `evaluation/` is for. Two things to weigh before adding a fourth arm:

- **In favor:** it's the only unmeasured candidate that demonstrably produces Khmer, and CER via Whisper-large-v3 is exactly the tool for finding out whether it's VoxCPM2-tier or Fish-S2-tier. An undocumented capability with no vendor number attached is the *strongest* case for independent measurement, not the weakest.
- **Against:** the **research/non-commercial license** caps how useful a good result would be — even a winning CER wouldn't displace Apache-2.0 VoxCPM2 for production without a commercial agreement. And adding a backend is real work: Higgs v3 ships no local repo entrypoint, so `backends/higgs_v3.py` would have to drive either Boson's API (rate-limited, network-dependent, non-reproducible RTF) or a self-hosted SGLang-Omni stack (heavy).

Adding a fourth model does **not** conflict with the fixed-eval-set constraint in `CLAUDE.md` — `eval-set/eval.json` stays untouched and every model reads the same 100 sentences; only the number of synthesis arms changes.

**Status: not yet in the harness — pending a decision on whether the license makes measurement worthwhile.** If it is added, RTF is the one metric to treat carefully: an API-served number is not comparable to the locally-measured RTF for MMS and VoxCPM2, and should be recorded as N/A rather than as a misleading figure.

## References
- [bosonai/higgs-audio-v3-tts-4b — Hugging Face model card](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) (102-language list, architecture, license, throughput)
- [bosonai/higgs-tts-3-4b — Hugging Face](https://huggingface.co/bosonai/higgs-tts-3-4b) (mirror)
- [boson-ai/higgs-audio — GitHub](https://github.com/boson-ai/higgs-audio)
- [Boson AI — "Higgs TTS 3: beyond reading, toward real speech for voice AI"](https://www.boson.ai/blog/higgs-tts-3)
- [LMSYS — "Higgs Audio v3 TTS on SGLang-Omni"](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)
- [Boson AI docs — Higgs TTS overview](https://docs.boson.ai/models/higgs-tts/overview)
- [SGLang-Omni cookbook — Higgs TTS](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- [ComfyUI-Wiki release note](https://comfyui-wiki.com/en/news/2026-06-04-higgs-tts-3-4b-boson) — *cited as the origin of the unsupported Khmer claim, not as evidence for it*
