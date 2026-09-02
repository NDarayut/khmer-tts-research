# 8. Khmer-Specific Considerations & Recommendation

## 8.1 What makes Khmer hard for TTS specifically

- **No spaces between words.** Khmer script is written without whitespace word boundaries (like Thai, Lao, Japanese, and Chinese), so any pipeline needs word/syllable segmentation before G2P or normalization — a nontrivial NLP step in its own right ([khmertagger](https://github.com/seanghay/khmertagger) exists specifically to handle this for Khmer ASR/TTS).
- **Complex orthography.** Khmer is an abugida with subscript consonants (coeng), two vowel "registers" that change the pronunciation of the same vowel symbol depending on the preceding consonant, and a large inventory of independent vowels — all of which make rule-based G2P substantially harder than for a Latin-script language, and is presumably why eSpeak-NG, despite covering 127+ languages, still doesn't have Khmer.
- **Very little parallel text+audio data exists publicly.** The main open corpus, [OpenSLR SLR42](https://www.openslr.org/42/), is Google's 2016–2018 Khmer dataset; beyond it and whatever religious-text audio backs MMS-lab, there is no large public Khmer speech corpus comparable to what exists for, say, Vietnamese or Thai.
- **No standardized romanization**, unlike many other low-resource languages, which complicates cross-lingual transfer strategies that rely on shared phonetic/romanized representations.

## 8.2 Recommendation

**For anyone building on this today: VoxCPM2 is the clear choice.**

- It is the only one of the three candidate models with **both** an official Khmer language listing **and** a specific, favorable benchmark result (2.05% CER — in line with its English/Chinese numbers, not a "technically supported but broken" long-tail entry).
- **Apache-2.0** removes any licensing friction for commercial use, unlike MMS-TTS-khm (CC-BY-NC-4.0) or Fish Audio S2 (Research License, paid for commercial).
- Runs on a single consumer GPU (~8GB VRAM) at ~0.13–0.30 RTF — practical for real deployment.

**Fish Audio S2/S2-Pro should be avoided for Khmer** on current evidence — its own comparison benchmark implies the language is effectively non-functional (75% CER) despite being on the marketing language list. It remains a strong choice for any of its well-resourced Tier-1/Tier-2 languages.

**Meta's MMS-TTS (`mms-tts-khm`)** is worth keeping as a fallback or comparison point: it is the most "Khmer-first" of the three in spirit (a dedicated checkpoint rather than one line item in an 80-language mix), extremely lightweight (36M params, CPU-capable), and trivial to get running via 🤗 Transformers — but expect lower naturalness (small VITS, single speaker, ~40h of scripture-reading audio) and no commercial-use rights.

## 8.3 Suggested validation before committing

Since every quality number in this report is self-reported by the model vendors (mainly OpenBMB, comparing itself favorably to Fish Audio), the highest-value next step is **independent verification**:

1. Generate a moderate Khmer test set (a few dozen sentences spanning formal/informal register, numbers, loanwords) through VoxCPM2 and MMS-TTS-khm.
2. Get native-speaker judgment (a lightweight MOS-style 1–5 rating) rather than relying on ASR-based CER alone — CER measures intelligibility to a machine, not naturalness to a human ear, and Khmer ASR itself is a low-resource problem, adding scoring noise.
3. If voice cloning is a requirement, specifically test VoxCPM2's cloning claim on Khmer reference audio — this is unbenchmarked in any public source found here.

## References
See reference lists in each linked document ([1](01-tts-overview.md), [2](02-training-data-requirements.md), [3](03-evaluation-benchmarking.md), [4](04-model-meta-mms-tts.md), [5](05-model-fish-audio-s2.md), [6](06-model-voxcpm2.md), [7](07-other-models-checked.md)).
