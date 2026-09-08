# Multilingual Open-Source TTS for Khmer (ខ្មែរ)

A research report surveying **open-source, multilingual text-to-speech (TTS) models that are capable of synthesizing Khmer speech**, even though Khmer was not their primary design target. It covers how TTS systems work, what data they need, how they're benchmarked, and a detailed, sourced breakdown of every candidate model found — including which ones turned out **not** to support Khmer at all (useful negative results for anyone doing this search again).

**TL;DR:** Of the ~15 open-source multilingual TTS systems surveyed, **four** were downloaded and benchmarked head-to-head on a fixed 100-sentence Khmer set. **Two produce usable Khmer.**

| Model | Org | Khmer in official language list? | Real-world Khmer quality | License (commercial use) |
|---|---|---|---|---|
| **[VoxCPM2](docs/research/06-model-voxcpm2.md)** | OpenBMB | ✅ Yes (1 of 30) | **Usable** — confirmed by listening; 2.05% CER claimed on internal benchmark | **Apache-2.0 — free** |
| **[Higgs TTS 3](docs/deep-dives/10-higgs-tts-3-architecture-and-training.md)** | Boson AI | ❌ **No** — `km` is in neither of its 102-language quality tiers | **Usable anyway** — an emergent, undocumented capability, confirmed by listening | Research License — NC only, paid for commercial |
| **[Fish Audio S2 / S2-Pro](docs/research/05-model-fish-audio-s2.md)** | Fish Audio | ✅ Yes (1 of 80+) | **Broken** — 75.15% CER on the same benchmark; observed dropping or padding whole sentences | Research License — NC only, paid for commercial |
| **[Meta MMS-TTS (khm)](docs/research/04-model-meta-mms-tts.md)** | Meta AI | ⚠️ Not "multilingual" in the usual sense — one VITS checkpoint per language, 1,107 languages incl. Khmer | **Faithful but robotic** — reads the text correctly with flat prosody. 36M params, by far the fastest and the only one that fine-tunes on a 12 GB card | CC-BY-NC-4.0 — **non-commercial only** |

> **A language list is evidence, not proof — in either direction.** Fish Audio S2 lists Khmer and cannot speak it. Higgs TTS 3 does not list Khmer and can. The only reliable test is to run the model and listen.

> **The automatic metrics were all wrong.** CER, UTMOS and DNSMOS were run over all 400 clips and **none of them ranked the models correctly**: the highest naturalness score in the entire run went to a clip that omits most of its sentence. The full evidence is in the [evaluation report](evaluation/results_report.html). If you take one thing from this repo, take that.

Everything else checked — Fish Audio S1, XTTS-v2, Chatterbox Multilingual, Qwen3-TTS / Qwen-Audio-3.0-TTS, Higgs Audio v2, CosyVoice2, Zonos, Kokoro, Bark, eSpeak-NG — **does not support Khmer** as of this writing (Sept 2026). See [07 — Models That Don't Support Khmer](docs/research/07-other-models-checked.md) for the full negative-result list, which is worth keeping so this ground doesn't get re-covered.

## Contents

### Research survey — `docs/research/`

1. [How TTS Works](docs/research/01-tts-overview.md) — the pipeline (frontend → acoustic model → vocoder) and the major architecture families (VITS, FastSpeech-style, diffusion, flow-matching, neural-codec LMs like the ones behind Fish Audio and VoxCPM2)
2. [What Data TTS Needs](docs/research/02-training-data-requirements.md) — data requirements from single-speaker fine-tunes to 2M-hour foundation models, and what's actually available for Khmer
3. [How TTS Is Benchmarked](docs/research/03-evaluation-benchmarking.md) — MOS/CMOS/SMOS, WER/CER, speaker similarity (SIM), UTMOS/DNSMOS, RTF, and the standard benchmark suites (Seed-TTS-eval, CV3-eval, TTS-Arena)
4. [Model: Meta MMS-TTS (Khmer checkpoint)](docs/research/04-model-meta-mms-tts.md)
5. [Model: Fish Audio S2 / S2-Pro](docs/research/05-model-fish-audio-s2.md)
6. [Model: VoxCPM2](docs/research/06-model-voxcpm2.md)
7. [Other Models Checked (No Khmer Support)](docs/research/07-other-models-checked.md)
8. [Khmer-Specific Considerations & Recommendation](docs/research/08-khmer-considerations-and-recommendation.md)
9. [Model: Higgs Audio v3](docs/research/09-model-higgs-audio-v3.md) — addendum written after 01–08: v3 does produce Khmer despite not listing it

### Deep dives — `docs/deep-dives/`

- [09 — VoxCPM2: Architecture & Training](docs/deep-dives/09-voxcpm2-architecture-and-training.md) — how it works, and the official fine-tuning path: JSONL manifest format, LoRA config, VRAM, and what Khmer data actually exists
- [10 — Higgs TTS 3: Architecture & Training](docs/deep-dives/10-higgs-tts-3-architecture-and-training.md) — how it works, why Boson ship no trainer, and what writing one would involve
- [11 — VoxCPM2 Style-Control Fine-Tune](docs/deep-dives/11-voxcpm2-style-control-finetune.md) — the style-control experiment, the failure that shaped it, and why the built-in parenthetical prompt made the fine-tune unnecessary

### Reports — `reports/`

Company-facing deliverables in `.docx` / `.pdf`. See [`reports/README.md`](reports/README.md).

| Document | Formats |
|---|---|
| **Smean TTS Literature Review** | [`.docx`](reports/Smean-TTS-Literature-Review.docx) · [`.pdf`](reports/Smean-TTS-Literature-Review.pdf) |
| **Speech Control — VoxCPM2** | [`.docx`](reports/Speech-Control-VoxCPM2.docx) |

Both are generated, not hand-edited — the sources live in `src/docbuild/`:

```
python src/docbuild/build_literature_review.py     # -> reports/Smean-TTS-Literature-Review.docx
python src/docbuild/build_style_control_report.py  # -> reports/Speech-Control-VoxCPM2.docx
```

## The benchmark

`eval-set/` and `evaluation/` hold a reproducible 4-model comparison over a fixed 100-sentence Khmer set (50 pure Khmer, 50 code-switched with English). Runbook in [`evaluation/README.md`](evaluation/README.md):

```
python src/khmer_tts/synthesis/synthesize.py --model {mms,voxcpm2,fish-s2,higgs3}   # audio + RTF
python src/khmer_tts/scoring/score.py      --model {mms,voxcpm2,fish-s2,higgs3}   # CER, UTMOS, DNSMOS
python src/khmer_tts/reporting/report_document.py                                   # the written report
```

All 400 clips are on disk under `evaluation/results/<model>/audio/`. The written report — including embedded audio samples from every model — is `evaluation/results_report.html`.

## Scope & method

This report only covers models that are (a) open-source with downloadable weights, (b) usable outside a paid API (even if commercial use requires a separate license), and (c) multilingual by design rather than built specifically for Khmer. Khmer-only projects (e.g. [KLEA](https://github.com/seanghay/KLEA), the [OpenSLR SLR42](https://www.openslr.org/42/) dataset, community MMS fine-tunes) are mentioned only as supporting context, not as primary subjects.

All facts are sourced inline in each document, pulled from official model cards, GitHub repos, blog posts, and the underlying papers as of **September 2026**. TTS is moving fast — treat version-specific numbers (parameter counts, language counts, benchmark scores) as a snapshot, and check the linked sources for updates.

---
*Compiled September 2026.*
