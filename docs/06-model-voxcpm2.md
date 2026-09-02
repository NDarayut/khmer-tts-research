# 6. Model: VoxCPM2 — the best current option for Khmer

| | |
|---|---|
| **Org** | OpenBMB |
| **Architecture** | Tokenizer-free "Diffusion Autoregressive": LocEnc → TSLM → RALM → LocDiT, built on the **MiniCPM-4** LLM backbone, with an AudioVAE V2 handling 16kHz-in / 48kHz-out audio conversion |
| **Size** | 2B parameters, ~8GB VRAM to run |
| **Languages** | **30**, explicitly including **Khmer**, plus 9 Chinese dialects |
| **License** | **Apache-2.0 — free for commercial use** |
| **Weights** | [huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| **Code** | [github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| **Paper** | [Zhou et al., "VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning," arXiv:2509.24650](https://arxiv.org/abs/2509.24650) |

## Why "tokenizer-free"

Most zero-shot TTS foundation models (Fish Audio included) commit to one of two lossy choices: **discrete audio tokens** (stable to model with an LM, but throw away acoustic detail — the "digitized" sound some codec-LM TTS has) or **continuous mel/latent regression** (keeps acoustic richness, but autoregressive continuous generation accumulates error over long sequences). VoxCPM2's paper frames its whole design around avoiding this tradeoff with a **semi-discrete residual representation**:

- **TSLM** (Text-Semantic Language Model) — an autoregressive model that plans semantic content and prosody from text, working with a "differentiable quantization bottleneck" rather than hard discrete tokens.
- **RALM** (Residual Acoustic [Language] Model) — recovers the fine-grained acoustic detail the TSLM's coarse plan leaves out.
- **LocDiT** — a local diffusion-based decoder that turns the combined representation into high-fidelity speech latents, trained end-to-end under a diffusion objective.
- The whole stack sits on **MiniCPM-4**, giving it a capable pretrained language backbone for text/semantic understanding across languages, rather than training a text encoder from scratch.

## Training data

- **2M+ hours of multilingual speech**, covering the 30 listed languages plus 9 Chinese dialectal varieties.
- OpenBMB does not publish a per-language hour breakdown (same transparency gap as Fish Audio) — but unlike Fish Audio S2, VoxCPM2's own benchmark numbers for Khmer are strong, which is the best evidence available that Khmer got a non-trivial share of the corpus.

## How it performs

From VoxCPM2's own published benchmarks (self-reported, not yet independently reproduced):

| Benchmark | Result |
|---|---|
| Seed-TTS-eval, test-EN | WER 1.84%, speaker similarity 75.3% |
| Seed-TTS-eval, test-ZH | WER 0.97%, speaker similarity 79.5% |
| CV3-eval, 11 languages | CER/WER 3.65%–5.00% for Chinese/English; 4.25%–9.85% for Italian/French |
| Internal 30-language × 500-sample benchmark, average | **1.68%** error rate (English 0.42% WER, Chinese 0.92% CER) |
| **Internal 30-language benchmark, Khmer specifically** | **2.05% CER** — close to the overall 30-language average, and dramatically better than Fish S2-Pro's 75.15% CER on the same test |
| Real-time factor | ~0.30 on an RTX 4090 (standard); ~0.13 with Nano-vLLM acceleration |

A 2.05% CER for Khmer — in the same range as VoxCPM2's English and Chinese numbers, and squarely inside "usable, intelligible synthesis" territory — is, as of this report, the strongest evidence of any open-source multilingual model handling Khmer competently. It's also cited against three separate public benchmark suites (Seed-TTS-eval, CV3-eval, and an internal 30-language set), which is more evaluation rigor than either of the other two models in this report disclose for Khmer.

## Practical notes

- **Apache-2.0** — the only one of the three models here with no commercial-use restriction whatsoever.
- Automatic language detection from input text — no language tag required, which simplifies a Khmer-only pipeline.
- 48kHz output, and a claimed voice-cloning capability ("true-to-life cloning") from reference audio — untested for Khmer specifically in public sources, since the disclosed benchmark is intelligibility (CER), not Khmer speaker-similarity.
- `pip install voxcpm` — needs Python ≥3.10, PyTorch ≥2.5.0, CUDA ≥12.0; 2B params / ~8GB VRAM is a meaningfully heavier deployment than MMS-TTS-khm's 36M params, but far lighter than Fish S2-Pro's ~5B.
- As with every model in this report: self-reported numbers from the vendor should be treated as an upper bound and, ideally, spot-checked against real Khmer text/native-speaker judgment before committing to it for production use.

## References
- [openbmb/VoxCPM2 — Hugging Face model card](https://huggingface.co/openbmb/VoxCPM2)
- [OpenBMB/VoxCPM — GitHub (benchmarks, install instructions)](https://github.com/OpenBMB/VoxCPM)
- [Zhou et al., "VoxCPM: Tokenizer-Free TTS..." — arXiv:2509.24650](https://arxiv.org/abs/2509.24650)
