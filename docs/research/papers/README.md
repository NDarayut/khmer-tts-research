# Research papers

Local PDF copies of the work cited by `finetune/NVV_PLAN.md` and by
`docs/deep-dives/11-voxcpm2-style-control-finetune.md` §3. Downloaded from arXiv
on 2026-09-08 so the plan's citations stay checkable offline.

Grouped by the role each plays in the plan.


## Interface and fine-tuning recipe

| paper | arXiv | why it is here |
|---|---|---|
| [NVSpeech: Integrated and Scalable Pipeline for Human-Like Speech Modeling with Paralinguistic Vocalizations](nvspeech-2025-paralinguistic-pipeline.pdf) | [2508.04195](https://arxiv.org/abs/2508.04195) | interface: inline tokens at event position; 48,430 annotated utts / 18 categories; 573h auto-labelled |
| [ELaTE: Making Flow-Matching-Based Zero-Shot TTS Laugh as You Like (Kanda et al.)](elate-2024-laugh-as-you-like.pdf) | [2402.07383](https://arxiv.org/abs/2402.07383) | the 50:50 conditioned/general mixing ratio; frame-level detector conditioning |
| [SynParaSpeech: Automated Synthesis of Paralinguistic Datasets](synparaspeech-2025-automated-paralinguistic-dataset.pdf) | [2509.14946](https://arxiv.org/abs/2509.14946) | 118.75h, 6 categories, natural conversational speech, precise timestamps |
| [NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations](nonverbaltts-2025-corpus.pdf) | [2507.13155](https://arxiv.org/abs/2507.13155) | 17h / 10 NV types / 8 emotions from VoxCeleb+Expresso; auto-detect + human validation |
| [VocalSound: A Dataset for Improving Human Vocal Sounds Recognition (Gong et al.)](vocalsound-2022-gong.pdf) | [2205.03433](https://arxiv.org/abs/2205.03433) | 21k clips / 3,365 speakers / 6 classes; the borrowed-acoustics source |

## Evaluation

| paper | arXiv | why it is here |
|---|---|---|
| [NVV-SuperBench: Benchmarking Nonverbal Vocalizations in Speech Generation](nvv-superbench-2026-benchmark.pdf) | [2604.16211](https://arxiv.org/abs/2604.16211) | 45-type taxonomy; F1 / NTD / Coverage / PE metric definitions; tag-based baselines |
| [NVMOS: Non-Verbal Vocalization Quality Assessment in Speech](nvmos-2026-nvv-quality-assessment.pdf) | [2606.15888](https://arxiv.org/abs/2606.15888) | 0-5 MOS-like predictor for a specific marked NVV event |
| [InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in TTS](instructtts-eval-2025.pdf) | [2506.16381](https://arxiv.org/abs/2506.16381) | evaluation of the parenthetical/description channel |
| [Beyond Words: Towards Effective Modeling of Non-Verbal Vocalizations in ASR](beyond-words-2026-nvv-in-asr.pdf) | [2607.01563](https://arxiv.org/abs/2607.01563) | detection/recognition side of the same tag inventory |
| [WESR: Scaling and Evaluating Word-level Event-Speech Recognition](wesr-2026-word-level-event-speech-recognition.pdf) | [2601.04508](https://arxiv.org/abs/2601.04508) | detector/recogniser side for placing tags at token positions |

## Explicitly-tag-trained systems (the contrast case)

| paper | arXiv | why it is here |
|---|---|---|
| [CosyVoice 3: Towards In-the-wild Speech Generation](cosyvoice3-2025.pdf) | [2505.17589](https://arxiv.org/abs/2505.17589) | explicitly-trained paralinguistic tags; the contrast case to VoxCPM2's emergent tags |
| [CosyVoice 2: Scalable Streaming Speech Synthesis with LLMs](cosyvoice2-2024.pdf) | [2412.10117](https://arxiv.org/abs/2412.10117) | tag lineage |
| [CosyVoice: Scalable Multilingual Zero-shot TTS with Supervised Semantic Tokens](cosyvoice1-2024.pdf) | [2407.05407](https://arxiv.org/abs/2407.05407) | origin of the [laughter]/[breath] marker format |
| [FunAudioLLM: Voice Understanding and Generation Foundation Models](funaudiollm-2024.pdf) | [2407.04051](https://arxiv.org/abs/2407.04051) | CosyVoice-instruct fine-grained paralinguistic control |
| [PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis](pilottts-2026-modular-recipe.pdf) | [2605.27258](https://arxiv.org/abs/2605.27258) | 4 paralinguistic categories; laughter success rate comparison vs CosyVoice 3 |
| [Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching Zero-Shot TTS](laugh-now-cry-later-2024-hsu.pdf) | [2407.12229](https://arxiv.org/abs/2407.12229) | joint emotion + laughter control |

## Consistency / classifier-free guidance

| paper | arXiv | why it is here |
|---|---|---|
| [Selective Classifier-free Guidance for Zero-shot TTS](selective-cfg-2025.pdf) | [2509.19668](https://arxiv.org/abs/2509.19668) | CFG as an adherence knob; the training-free consistency lever |
| [Joint Residual Reweighting for Classifier Free Guidance in Flow-Matching Zero-Shot TTS](joint-residual-reweighting-cfg-2026.pdf) | [2606.25672](https://arxiv.org/abs/2606.25672) | multi-condition guidance balancing |
| [Towards Flow-Matching-based TTS without Classifier-Free Guidance](fm-tts-without-cfg-2025.pdf) | [2504.20334](https://arxiv.org/abs/2504.20334) | the counter-argument to leaning on CFG |

## Background: description-based style control, and emotion

| paper | arXiv | why it is here |
|---|---|---|
| [PromptTTS: Controllable Text-to-Speech with Text Descriptions](prompttts-2022-guo.pdf) | [2211.12171](https://arxiv.org/abs/2211.12171) | origin of the description-conditioning format |
| [InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt](instructtts-2023-yang.pdf) | [2301.13662](https://arxiv.org/abs/2301.13662) | free-form instruction control |
| [PromptTTS 2: Describing and Generating Voices with Text Prompt](prompttts2-2023-leng.pdf) | [2309.02285](https://arxiv.org/abs/2309.02285) | variation network; LLM-written prompts at 44k hours |
| [Natural Language Guidance of High-Fidelity TTS with Synthetic Annotations (Parler-TTS)](parler-tts-2024-lyth-king.pdf) | [2402.01912](https://arxiv.org/abs/2402.01912) | labels measured not annotated, 45k hours |
| [TextrolSpeech: A Text Style Control Speech Corpus](textrolspeech-2023-ji.pdf) | [2308.14430](https://arxiv.org/abs/2308.14430) | 236h / 33k utts with LLM-generated style descriptions |
| [Emotional Voices Database / Emotional Speech Dataset survey (Zhou et al.)](esd-2022-zhou.pdf) | [2105.14762](https://arxiv.org/abs/2105.14762) | why emotion is blocked on data for Khmer |

## Cited but not archived here

* **Gillick et al. (2021)**, robust frame-level laughter detection and
  segmentation — Interspeech, not on arXiv. Code/model:
  <https://github.com/jrgillick/laughter-detection>.
* **MNV-17 (2025)**, Mandarin performative non-verbal corpus — cited in
  docs/11 §3.3 as the precedent for deliberate recording; no arXiv copy located.
