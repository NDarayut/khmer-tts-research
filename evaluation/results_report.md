# TTS Model Comparison -- Khmer

4 of 4 models scored, over the fixed 100-sentence set in `eval-set/eval.json` (50 `pure_khmer` + 50 `code_switched`). Metrics: CER (correctness, via Whisper-large-v3), UTMOS and DNSMOS (naturalness), RTF (speed).

## Headline comparison

| Model | n | CER mean | CER median | UTMOS | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | P.808 | RTF median | RTF mean |
|---|---|---|---|---|---|---|---|---|---|---|
| `mms` | 100 | 124.14% | 98.82% | 2.95 | 3.23 | 3.44 | 4.16 | 3.71 | 0.010 | 0.011 |
| `voxcpm2` | 100 | 122.31% | 100.68% | 2.49 | 2.98 | 3.34 | 3.87 | 3.50 | 1.382 | 1.377 |
| `fish-s2` | 100 | 123.01% | 100.95% | 3.75 | 3.21 | 3.60 | 3.87 | 3.79 | 2.651 | 2.551 |
| `higgs3` | 100 | 122.06% | 98.22% | 2.98 | 3.14 | 3.47 | 3.92 | 3.84 | 0.745 | 0.750 |

CER lower is better (0% = perfect). UTMOS and DNSMOS are 1-5 MOS scales, higher is better. RTF below 1.0 is faster than real time.

## :warning: CER is not valid in this run

**Do not rank the models on the CER column above.** The scoring ASR failed, so those numbers describe Whisper, not the TTS models.

- `fish-s2` has a median CER of 100.9% -- i.e. essentially every character is wrong, and above 100% the hypothesis is longer than the reference it is meant to match.
- `higgs3` has a median CER of 98.2% -- i.e. essentially every character is wrong, and above 100% the hypothesis is longer than the reference it is meant to match.
- `mms` has a median CER of 98.8% -- i.e. essentially every character is wrong, and above 100% the hypothesis is longer than the reference it is meant to match.
- `voxcpm2` has a median CER of 100.7% -- i.e. essentially every character is wrong, and above 100% the hypothesis is longer than the reference it is meant to match.
- Whisper-large-v3 collapses into a repetition loop on Khmer (`ប្រាប់ប្រាប់ប្រាប់...`). Raw transcripts in each `scores.json` show it directly.

The other three metrics are unaffected -- UTMOS, DNSMOS and RTF never touch the ASR. To restore CER, swap `metrics/cer.py` for a Khmer-capable ASR and re-run `score.py --metrics cer`; the synthesized audio does not need regenerating.

## CER by group

The set is split 50/50 for exactly this comparison -- code-switching is where Khmer TTS is expected to degrade.

| Model | `code_switched` | `pure_khmer` |
|---|---|---|
| `mms` | 105.11% | 143.17% |
| `voxcpm2` | 105.70% | 138.92% |
| `fish-s2` | 105.22% | 140.81% |
| `higgs3` | 105.39% | 138.73% |

## CER by category

Category definitions are in `evaluation/data-promt.md`; counts and examples in `evaluation/dataset_overview.md`.

| Model | `exclamatory` | `long_complex` | `medium` | `mixed_clause` | `numbers_dates` | `numbers_units` | `proper_noun` | `question` | `short` | `short_phrase` | `single_word` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mms` | 232.83% | 79.65% | 88.14% | 89.37% | 95.14% | 113.02% | 107.30% | 161.40% | 242.94% | 99.11% | 118.23% |
| `voxcpm2` | 187.86% | 77.84% | 88.42% | 87.29% | 101.17% | 111.25% | 107.62% | 160.40% | 242.91% | 102.53% | 118.66% |
| `fish-s2` | 201.40% | 84.69% | 91.06% | 90.43% | 97.37% | 113.35% | 112.81% | 159.55% | 238.62% | 96.78% | 118.27% |
| `higgs3` | 242.97% | 79.94% | 89.00% | 88.78% | 96.21% | 110.77% | 108.14% | 156.44% | 217.09% | 98.73% | 120.42% |

## Worst utterances by CER

Read the transcript before blaming the TTS -- a high CER here can be Whisper failing on Khmer rather than the model mispronouncing anything. See the caveat below.

### `mms`

| id | category | CER | ASR transcript |
|---|---|---|---|
| A41 | `exclamatory` | 362.5% | បាន្តាក្បាន់ទាំព្លាំព្លាំព្លាំព្លាំព្លាំព្លាំព្លាំព្លាំព្... |
| A02 | `short` | 329.6% | ខ្លាប់ពីស្សាប់ពីស្សាប់ពីស្សាប់ពីស្សាប់ពីស្សាប់ពីស្សាប់ពីស... |
| A08 | `short` | 277.4% | ខ្លាក្រាប់ខ្លាក្រាប់ខ្លាក់ខ្លាក់ខ្លាក់ខ្លាក់ខ្លាក់ខ្លាក់ខ... |
| A05 | `short` | 271.9% | ខ្លាប់ទាំងស្សាស់ទាំងស្សាស់ទាំងស្សាស់ទាំងស្សាស់ទាំងស្សាស់ទ... |
| A03 | `short` | 263.6% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A43 | `exclamatory` | 248.6% | ខ្លាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រ... |
| A44 | `exclamatory` | 245.9% | ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A07 | `short` | 229.7% | ហើយស្សាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់... |
| A01 | `short` | 227.8% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A06 | `short` | 214.0% | បង្រាប់ក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុង... |

### `voxcpm2`

| id | category | CER | ASR transcript |
|---|---|---|---|
| A02 | `short` | 311.1% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A05 | `short` | 287.5% | បាន្លាប់ក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុ... |
| A08 | `short` | 271.0% | ខ្លាប់បានបានបានប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់... |
| A03 | `short` | 254.5% | ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្លាប់ខ្ល... |
| A43 | `exclamatory` | 251.4% | ល្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្ល... |
| A44 | `exclamatory` | 243.2% | ប្រូប្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប... |
| A07 | `short` | 235.1% | បាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្តាន្ត... |
| A01 | `short` | 227.8% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A04 | `short` | 227.5% | ប្រាប់ក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក្នុងក... |
| A10 | `short` | 210.0% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |

### `fish-s2`

| id | category | CER | ASR transcript |
|---|---|---|---|
| A02 | `short` | 311.1% | ខ្លាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រ... |
| A08 | `short` | 274.2% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A05 | `short` | 262.5% | ប្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម្ម... |
| A44 | `exclamatory` | 259.5% | ប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប្រូប... |
| A03 | `short` | 254.5% | ខ្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្ល... |
| A43 | `exclamatory` | 251.4% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A07 | `short` | 251.4% | ខ្លាប់ក្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត... |
| A01 | `short` | 222.2% | ឬក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្រាប់ក្... |
| A10 | `short` | 210.0% | ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A04 | `short` | 207.5% | ប្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម... |

### `higgs3`

| id | category | CER | ASR transcript |
|---|---|---|---|
| A41 | `exclamatory` | 391.7% | រូវិត្រូវប្រូវប្រូវប្រូវប្រូវប្រូវប្រូវប្រូវប្រូវប្រូវប្រ... |
| A02 | `short` | 337.0% | ខ្លើងស្សាស់ខ្ញុំន្រាប់ការប់ការប់ការប់ការប់ការប់ការប់ការប់... |
| A05 | `short` | 290.6% | ជាម្រាប់ខ្ញុំបាន់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការប... |
| A03 | `short` | 281.8% | ខ្លាប់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការ... |
| A43 | `exclamatory` | 274.3% | សូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូ... |
| A44 | `exclamatory` | 237.8% | ស្លាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម្រាប់កម... |
| A10 | `short` | 235.0% | ជាត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវត្រូវ... |
| A01 | `short` | 227.8% | ខ្លាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រាប់ប្រ... |
| A04 | `short` | 212.5% | បង្រាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្លាប់ព្... |
| A06 | `short` | 207.0% | បង្រាប់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់ការបស់កា... |

## Measurement caveats

These affect what a column *means* for a given row -- read them before comparing across rows.

- `fish-s2`: codec runs on CPU -- RTF not comparable.
- `fish-s2`: 3 utterance(s) ran past 20s (longest 47.5s, median across the set 7.9s) -- the model did not stop on its own. Those clips inflate its RTF and its quality scores reflect the filler, not the sentence.
- `higgs3`: **Khmer is undocumented** -- its model card lists 100 languages and Khmer is not among them, yet it produces Khmer speech (verified by hand before the run). Its scores count, but no published Khmer figure exists to corroborate them.

## Run environment

RTF is hardware-bound and is only comparable between models measured on the same machine -- check that these rows match before comparing the speed column.

| Model | Device | GPU | Synth load (s) | Timestamp |
|---|---|---|---|---|
| `mms` | cuda | NVIDIA GeForce RTX 3060 | 5.297 | 2026-09-04T01:42:45+00:00 |
| `voxcpm2` | cuda | NVIDIA GeForce RTX 3060 | 38.025 | 2026-09-04T01:55:41+00:00 |
| `fish-s2` | cuda | NVIDIA GeForce RTX 3060 | 38.771 | 2026-09-04T08:36:21+00:00 |
| `higgs3` | cuda | NVIDIA GeForce RTX 3060 | 7.273 | 2026-09-04T10:48:17+00:00 |

## Published figures, for comparison

| Model | Vendor/third-party published |
|---|---|
| `mms` | no published Khmer benchmark |
| `voxcpm2` | 2.05% CER (OpenBMB internal, Khmer); RTF 0.13-0.30 on an RTX 4090 |
| `fish-s2` | 75.15% CER (OpenBMB internal, Khmer) |
| `higgs3` | no Khmer figure published; <5 WER/CER on 85 of its 100 languages |

## Reading these numbers

- **The CER floor is not zero.** Per `docs/research/03-evaluation-benchmarking.md` section 3.4, ASR-based CER is only as good as the scoring ASR, and Whisper-large-v3's own Khmer accuracy is limited. Differences between the 3 models are meaningful (same ASR, same sentences); the absolute value is not.
- **UTMOS and DNSMOS have never heard Khmer.** Both were trained on MOS studies of mostly English speech. They rate acoustic quality and prosodic naturalness, which transfers reasonably, but treat them as a relative ranking rather than a calibrated Khmer MOS.
- **DNSMOS BAK is near its ceiling for clean synthesis** and carries little signal here; SIG and OVRL are the informative columns.
- **Published figures are self-reported upper bounds** -- both the VoxCPM2 and Fish S2-Pro Khmer numbers come from OpenBMB's own benchmark, not an independent evaluation.
- **RTF excludes model load time** (reported separately above) and is measured after a discarded warm-up utterance.

_Generated by `src/khmer_tts/reporting/report.py` from `evaluation/results/<model>/scores.json`. See `evaluation/README.md` for the runbook._
