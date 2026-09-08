# 3. How TTS Is Benchmarked

TTS quality is evaluated along two largely independent axes — **intelligibility** (did it say the right words?) and **naturalness/similarity** (does it sound good, and like the intended speaker?) — using both human and automatic metrics.

## 3.1 Subjective (human-rated) metrics

| Metric | What it measures | How it works |
|---|---|---|
| **MOS** (Mean Opinion Score) | Overall perceived quality/naturalness | Listeners rate samples 1–5; scores averaged. The oldest and still most-cited TTS metric, but expensive and slow to run, and not comparable across studies with different listener pools |
| **CMOS** (Comparative MOS) | Relative quality between two systems | Listeners hear A/B pairs and rate preference/strength on a signed scale (e.g., −3 to +3) |
| **SMOS** (Similarity MOS) | How much a cloned voice sounds like the target speaker | Same 1–5 scale, but the rating question is "does this sound like the reference speaker," not "is this good speech" |
| **Human A/B preference (Arena-style)** | Head-to-head system ranking | Blind pairwise votes are aggregated into an **Elo rating**, the same method Chatbot Arena/LMSYS uses for LLMs. HuggingFace's **TTS-Arena** is the standard public leaderboard of this kind, and is what Fish Audio cites when claiming a "#1 ELO" ranking |

## 3.2 Objective (automatic) metrics

| Metric | Measures | How it's computed |
|---|---|---|
| **WER / CER** (Word/Character Error Rate) | Intelligibility | Run a strong ASR model (commonly **Whisper-large-v3**, or GPT-4o transcription) on the synthesized audio and diff the transcript against the input text. This is the single most-reported number in modern TTS papers, and is what backs every Khmer quality figure cited in this report (e.g., VoxCPM2's 2.05% Khmer CER vs. Fish S2-Pro's 75.15%) — **CER is generally preferred over WER for logographic/non-whitespace-segmented languages like Khmer, Chinese, Thai, and Japanese**, since "word" boundaries are themselves ambiguous |
| **SIM / SECS** (Speaker Similarity) | Voice-cloning fidelity | Cosine similarity between speaker embeddings of the reference audio and the generated audio, typically extracted with a **WavLM-large speaker-verification model** or **ECAPA-TDNN** |
| **UTMOS** | Predicted naturalness | A neural network trained to *predict* human MOS scores from self-supervised speech features — a fast, free proxy for MOS that avoids running a listening study for every experiment |
| **DNSMOS / NISQA** | Predicted perceptual quality (including noise/distortion) | Similar automatic MOS-predictor models, originally built for speech-enhancement/denoising evaluation, now widely reused as a no-reference TTS quality check |
| **MCD** (Mel-Cepstral Distortion) | Acoustic distance to a reference recording | Used mainly when a ground-truth recording of the exact same sentence exists (e.g., Meta's MMS-TTS evaluation) — less common for zero-shot systems, which by definition don't have a single "correct" reference |
| **RTF** (Real-Time Factor) | Inference speed, not quality | Time to synthesize ÷ duration of resulting audio. RTF < 1 means faster than real time. Reported by every model in this survey (VoxCPM2 ~0.13–0.30, Fish S2-Pro ~0.195) since it determines viability for live/streaming use |

## 3.3 Standard benchmark suites

Rather than each lab inventing its own test set, the field has converged on a few shared, public evaluation suites — this is what lets you compare, e.g., VoxCPM2's Khmer score against Fish Audio's on equal footing:

- **Seed-TTS-eval** (from ByteDance's Seed-TTS) — the most widely adopted zero-shot TTS benchmark. Reports **WER** (via Whisper-large-v3 transcription) and **SIM** (via WavLM-large speaker embeddings) on English and Chinese test sets, plus a "hard" subset of linguistically tricky sentences.
- **CV3-eval** — a Common Voice-derived multilingual evaluation set; reports WER/CER, SIM, and **DNSMOS**, and is the main way multilingual coverage (beyond EN/ZH) gets benchmarked — this is the suite VoxCPM2 cites for its 11-language and 30-language internal comparisons, including Khmer.
- **TTS-Arena / TTS-Arena V2** (Hugging Face) — public blind-listening Elo leaderboard; the human-preference equivalent of Seed-TTS-eval, covering naturalness, intelligibility, and similarity in one ranking.
- **InstructTTSEval / MiniMax Multilingual Test** — newer suites for instruction-following ("say this angrily," "whisper this") and broader multilingual robustness, referenced by VoxCPM2's own benchmark claims.

## 3.4 Reading the numbers in this report with appropriate skepticism

A few practical caveats that apply directly to the Khmer numbers cited throughout this report:

- **Self-reported benchmarks should be treated as upper bounds.** VoxCPM2's 2.05% Khmer CER and Fish S2-Pro's 75.15% Khmer CER both come from VoxCPM's own internal 30-language benchmark (a competitor comparison published by OpenBMB, not an independent third party) — directionally credible given the huge gap, but not independently reproduced here.
- **ASR-based CER/WER is only as good as the ASR model used to score it.** For a low-resource language like Khmer, the scoring ASR model itself may have limited accuracy, which adds noise to the reported error rate in both directions.
- **A language appearing on a "supported languages" list is a marketing/tokenizer claim, not a quality claim.** Always look for a benchmark number, not just list membership — this is the exact gap that separates VoxCPM2 (Khmer: listed *and* benchmarked well) from Fish Audio S2-Pro (Khmer: listed but benchmarked poorly).

## References
- [Zilliz — Standard evaluation metrics for TTS quality](https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality)
- [Seed-TTS Eval overview — EvalScope docs](https://evalscope.readthedocs.io/en/latest/benchmarks/seed_tts_eval.html)
- [Artificial Analysis — Text-to-Speech Benchmarking Methodology](https://artificialanalysis.ai/text-to-speech/methodology)
- [Hugging Face TTS-Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)
