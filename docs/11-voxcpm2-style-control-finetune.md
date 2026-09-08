# 11 — Speech Control for VoxCPM2

What can be controlled in synthetic speech, what VoxCPM2 already controls on
Khmer, what the literature has built for the rest, and where this project goes
next.

**Contents**

1. [Overview](#111-overview) — VoxCPM2, its architecture, and the control it ships with
2. [Speech control](#112-speech-control) — prosody, emotion, non-verbal vocalization
3. [Literature review](#113-literature-review) — what has been built for each, and what we will focus on
4. [Methodology](#114-methodology) — *to be written*

---

## 11.1 Overview

### 11.1.1 What VoxCPM2 is

VoxCPM2 is OpenBMB's **tokenizer-free, diffusion-autoregressive** text-to-speech
model, 2.29 B parameters, Apache-2.0. It is this project's recommended model for
Khmer: on the frozen 100-sentence evaluation set it scores a **2.47% median
character error rate** against Higgs TTS 3's 8.28%, MMS's 25.12% and Fish Audio
S2-Pro's 78.01% (see [`docs/03`](03-evaluation-benchmarking.md) for the metric
and [`CLAUDE.md`](../CLAUDE.md) for the caveat that the Khmer ASR scoring it has
VoxCPM2-synthesized audio in its training pool).

"Tokenizer-free" is the property that matters for a low-resource language. The
model does not quantize speech into a discrete codebook; it predicts *continuous*
latent vectors. There is therefore no audio vocabulary that can be badly fitted
for Khmer — the failure that eliminated Fish Audio S2 has no analogue here. On
the text side it reads raw UTF-8 bytes through a 73,448-entry tokenizer, so
Khmer needs **no G2P, no phonemizer, no lexicon and no word segmenter**, none of
which exist in usable form for the language.

### 11.1.2 Architecture

```
Khmer text
  ↓  byte-level tokenizer (vocab 73,448 — no G2P, no language tag)
┌──────────────────────────────────────────────────┐
│ MiniCPM4 backbone LM        2048 dim × 28 layers │  one latent per audio patch
│ GQA 16 query / 2 KV heads, LongRoPE to 32k       │
└──────────────────────────────────────────────────┘
  ↓  hidden state
┌──────────────────────────────────────────────────┐
│ Residual LM                 8 layers, no RoPE    │  refines the latent
└──────────────────────────────────────────────────┘
  ↓  conditioning vector
┌──────────────────────────────────────────────────┐
│ Local DiT       1024 dim × 12 layers, CFM head   │  diffusion: latent → features
│ euler solver, log-norm schedule, CFG 2.0         │  10 steps at inference
└──────────────────────────────────────────────────┘
  ↓  64-dim features, 4 frames per patch
┌──────────────────────────────────────────────────┐
│ AudioVAE V2     enc 16 kHz  →  dec 48 kHz        │  waveform
└──────────────────────────────────────────────────┘
```

A **Local Encoder** (1024 × 12) sits on the input side and encodes a reference
clip into the same latent space for zero-shot voice cloning.

| property | value | consequence |
|---|---|---|
| `patch_size` | 4 | one LM step covers 4 VAE frames |
| `feat_dim` | 64 | latent width the DiT predicts |
| AudioVAE frame rate | 25 fps | 1 s of audio ≈ 6.25 LM positions |
| encoder / decoder rate | 16 kHz / 48 kHz | training audio must be 16 kHz; super-resolution is free |
| `inference_cfg_rate` | 2.0 | classifier-free guidance, exposed as `--cfg-value` |

Two structural facts govern everything that follows. First, the **Local DiT is a
conditional flow-matching decoder** — the same family as the models the
non-verbal literature fine-tunes, which is why those recipes are applicable here.
Second, the training loss mask (`voxcpm/training/packers.py`, `process_tts_data`)
is **zero across every text position**:

```python
loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])
```

Nothing in the text field is ever a prediction target. The text field is a pure
conditioning channel, and anything can be put in it — a tag, a description, a
marker — without disturbing the objective.

### 11.1.3 The control VoxCPM2 ships with

VoxCPM2 has two built-in control mechanisms, and they operate at different
scopes.

**A parenthetical natural-language prompt** at the start of the text describes
the whole utterance:

```python
model.generate(text="(speaking quickly, a high-pitched voice)ថ្ងៃនេះអាកាសធាតុល្អណាស់។")
```

**Inline square-bracket tags** mark a single event at one position:

```python
model.generate(text="ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…")
```

The documented tag inventory: `[laughing]`, `[laughter]`, `[sigh]`, `[Uhm]`,
`[Shh]`, `[Question-ah/ei/en/oh]`, `[Surprise-wa/yo]`, `[Dissatisfaction-hnn]`.

**The parenthetical prompt works on Khmer.** This was measured on the stock
model, no fine-tuning: 8 sentences drawn from the frozen evaluation set, three
prompt levels per axis, three seeds per cell, median within each cell — 96
generations (`finetune/verify_parenthetical.py`, results in
`finetune/results/parenthetical/`).

| axis | prompt low → high | low | mid | high | Δ | Spearman ρ | p |
|---|---|---|---|---|---|---|---|
| pitch | *(a low-pitched voice)* → *(a high-pitched voice)* | 147.80 | 202.81 | 254.16 Hz | **+106.37 Hz** | **+0.759** | 0.0002 |
| energy | *(speaking softly)* → *(speaking loudly)* | −20.38 | −16.30 | −15.89 dBFS | +4.50 dB | +0.605 | 0.0022 |
| rate | *(speaking slowly)* → *(speaking quickly)* | 14.67 | 15.84 | 16.96 char/s | +2.29 | +0.590 | 0.0035 |
| variation | *(monotone)* → *(lively, expressive)* | 4.55 | 4.06 | 3.92 st | −0.63 | −0.273 | 0.20 |

Read it as three results, not one:

- **Pitch is controlled, strongly.** +106 Hz is nearly **four times** the
  +28.27 Hz that separates the low and high pitch bands of this project's own
  hand-labelled Khmer corpus. Whatever a fine-tune could teach about pitch, the
  base model already exceeds.
- **Energy and rate move in the right direction but are noise-limited.** The
  4.50 dB energy effect sits under a 5.69 dB seed-to-seed standard deviation;
  the rank correlation is real, the per-generation effect is not reliable.
- **Pitch variation fails**, and fails in the wrong direction: asking for a
  lively delivery produces *less* pitch movement than asking for a monotone one
  (ρ −0.273, p 0.20 — indistinguishable from noise).

So one prosodic axis is solved by the base model, two are usable in aggregate,
and one is not addressed at all. Whether the inline non-verbal tags fire on
Khmer has **not** been measured; the tag inventory is documented for Chinese and
English, and nothing establishes that a Khmer text context triggers them.

---

## 11.2 Speech control

"Expressive control" is not one capability. It is a set of separable ones that
differ in what they describe, how long they last, and — the part that decides
engineering cost — whether they are properties of a whole utterance or events at
a single position.

The three that matter for this project are **prosody**, **emotion** and
**non-verbal vocalization**.

### 11.2.1 Prosody

**What it is.** The suprasegmental properties of speech — everything carried
*above* the individual sounds. Prosody is what stays when you strip the words
out: how fast, how high, how loud, how varied, where the pauses fall.

| dimension | acoustic correlate | measured as |
|---|---|---|
| speaking rate | phones or syllables per second | characters per second |
| pitch register | fundamental frequency, F0 | median F0 in Hz |
| pitch variation | F0 range and contour movement | F0 standard deviation in semitones |
| loudness / projection | signal energy | RMS in dBFS |
| phrasing | pause placement and length | inter-pausal unit statistics |

**Examples.** *"I never said she stole my money"* read at 3 syllables per second
versus 6 is a rate change. The same sentence read at 120 Hz versus 240 Hz is a
register change. Read flat, it sounds robotic; read with a 6-semitone F0 range,
it sounds engaged — that is variation. All three leave the words untouched.

**Scope: global.** A prosodic setting is true of the whole utterance (or a long
span of it). It has no onset and no offset.

**Status here.** Largely solved by the parenthetical prompt — §11.1.3.

### 11.2.2 Emotion

**What it is.** The affective state the delivery conveys: neutral, happy, angry,
sad, surprised — the five categories the standard corpora use. Emotion is
*realised through* prosody plus voice quality, but it is not reducible to a
prosody setting: anger and excitement share high energy and high pitch and are
not the same, and the difference lives in voice quality, articulation precision
and timing detail that a rate/pitch/energy vector does not capture.

**Examples.** *"Oh, that's great."* — flat and slow reads as sarcasm; fast, high
and bright reads as delight; slow with creaky voice reads as resignation.
Identical text, three affects.

**Scope: global**, in the usual formulation. One label per utterance is the
convention in essentially every emotional-speech corpus.

**Status here.** Unmeasured. The parenthetical channel plausibly accepts
*(sounding angry)* — the mechanism is the same free-text field that already
carries *(speaking quickly)* — but nothing has been tested on Khmer.

### 11.2.3 Non-verbal vocalization

**What it is.** Sounds a speaker makes that are not words: laughter, sighs,
breaths, filled pauses (*uhm*, *er*), throat-clearing, coughs, gasps, sobs,
hesitation particles. They carry stance, turn-taking cues and affect, and they
are a large part of what separates read speech from conversational speech.

| type | example tag | what it signals |
|---|---|---|
| laughter | `[laughing]` | amusement, affiliation, softening |
| sigh | `[sigh]` | resignation, fatigue, relief |
| filled pause | `[Uhm]` | planning, hesitation, floor-holding |
| breath | `[breath]` | phrasing boundary, effort |
| hesitation / discourse particle | `[Question-ah]`, `[Dissatisfaction-hnn]` | stance, back-channel |
| gasp, sob, cough, throat-clear | — | surprise, distress, physical state |

**Examples.** *"I thought it was fine* `[laughing]` *but apparently not."* The
laugh is at one place, lasts a few hundred milliseconds, and everything before
and after it is ordinary speech. Compare *"So we should* `[Uhm]` *probably wait"*
— a filled pause inserted mid-clause.

**Scope: local.** This is the structural difference from the other two. A
non-verbal vocalization is a bounded event with an onset, a duration and an
offset, and it lives at a specific token position. That makes it a *different
engineering problem*: a global attribute has to compete with the acoustic
context for influence over every frame, while a local event owns the frames it
occupies and nothing else predicts them.

**Status here.** VoxCPM2 ships the tag inventory. Whether the tags fire on Khmer
text is unmeasured, and that measurement is the first thing the methodology will
do.

### 11.2.4 Adjacent categories

Three more exist and are worth naming so they are not confused with the above:

| category | what it is | scope | example |
|---|---|---|---|
| **voice quality** | phonation mode | global | whispered, breathy, creaky, tense |
| **emphasis / focus** | which word carries contrastive stress | local (a span) | *"**I** never said that"* vs *"I never said **that**"* |
| **timing** | pause insertion and length | local | a deliberate beat before a punchline |

Voice quality behaves like emotion (global, prompt-addressable in principle).
Emphasis and timing behave like non-verbal vocalization (local) but need a *span*
syntax rather than a point tag, which VoxCPM2 does not have.

---

## 11.3 Literature review

Four bodies of work bear on this: natural-language prompt control of prosody and
style, emotional speech synthesis, non-verbal vocalization, and the evaluation
problem that runs under all of them.

### 11.3.1 Prosody and natural-language style control

The dominant idea of the last three years is to replace a reference clip with a
*description*, and to get the descriptions by machine rather than by hand.

**PromptTTS** (Guo et al., ICASSP 2023, [arXiv:2211.12171](https://arxiv.org/abs/2211.12171))
established the format: a style prompt in natural language plus a content
prompt, encoded separately and fed to an acoustic model. Its dataset had to be
constructed by hand, which capped it.

**InstructTTS** (Yang et al., 2023, [arXiv:2301.13662](https://arxiv.org/abs/2301.13662))
took free-form instructions rather than attribute lists, using a discrete
diffusion decoder and a cross-modal representation trained to align instruction
text with speech style.

**PromptTTS 2** (Leng et al., ICLR 2024, [arXiv:2309.02285](https://arxiv.org/abs/2309.02285))
addressed the two problems that limit the whole family. First, the *one-to-many*
problem — a description underdetermines the voice — handled with a variation
network that predicts the reference-speech representation from the prompt
representation. Second, the labelling cost: a pipeline where a speech
understanding model recognises attributes (gender, speed, …) and an LLM writes
the prompt sentence. Trained on 44k hours.

**Natural language guidance of high-fidelity TTS with synthetic annotations**
(Lyth & King, 2024, [arXiv:2402.01912](https://arxiv.org/abs/2402.01912)) is the
clearest statement of the annotation argument, and the one released openly as
**Parler-TTS**. Gender, accent, pitch, speaking rate and recording conditions
are labelled *computationally* — classifiers and signal measurements, no human
annotation — across a 45k-hour found-data corpus, then turned into descriptive
sentences the model is conditioned on. The result outperforms prior work on
fidelity while relying entirely on found data. The transferable lesson: **the
labels for prosodic control can be measured, not annotated**, which is exactly
the property this project used when it built its own Khmer corpus by measuring
F0, character rate and RMS per clip.

**TextrolSpeech** (Ji et al., ICASSP 2024, [arXiv:2308.14430](https://arxiv.org/abs/2308.14430))
supplies the corpus side: 236 hours, 33k utterances, with style descriptions
generated by an LLM pipeline over five attribute dimensions.

*Relevance.* This line explains why VoxCPM2's parenthetical prompt exists and
why it works: it is the same conditioning format, trained the same way. It also
sets the bar — these systems control pitch, rate and energy from description,
which is precisely the set our §11.1.3 measurement finds already working on
Khmer.

### 11.3.2 Emotion

**Emotional Voice Conversion: Theory, Databases and ESD** (Zhou, Sisman, Liu &
Li, *Speech Communication* 2022, [arXiv:2105.14762](https://arxiv.org/abs/2105.14762))
is the reference point for data. ESD is 29+ hours, 350 parallel utterances from
10 English and 10 Mandarin speakers across five emotions (neutral, happy, angry,
sad, surprise), designed for multi-speaker and cross-lingual work. The paper also
surveys the emotional voice conversion field around it.

**Laugh Now Cry Later** (Hsu et al., SLT 2024,
[arXiv:2407.12229](https://arxiv.org/abs/2407.12229)) is the useful bridge
between this section and the next: it controls speaker emotion *and* laughter in
one zero-shot system, treating a laugh as a controllable event rather than as an
emotional label. Emotion and non-verbal vocalization are not separate problems in
practice.

*Relevance.* The corpus shape is the obstacle. Every result here rests on
parallel, acted, per-utterance-labelled emotional speech, and no such corpus
exists for Khmer. Whatever emotion work happens here will either ride on the
parenthetical channel for free or need a recording effort, and the second is out
of proportion to the value.

### 11.3.3 Non-verbal vocalization

This is the active area, and the one with recipes that transfer directly.

**NVSpeech** (2025, [arXiv:2508.04195](https://arxiv.org/abs/2508.04195)) is the
closest match to what this project wants. It builds an integrated pipeline across
*recognition and synthesis* of paralinguistic vocalizations: a manually annotated
set of 48,430 utterances covering **18 word-level paralinguistic categories**;
then a paralinguistic-aware ASR that emits the cues as **inline decodable
tokens** — *"You're so funny [Laughter]"* — which is used to auto-label a
573-hour, 174,179-utterance Chinese corpus with word-level alignment; then a
zero-shot TTS fine-tuned on the human- plus auto-labelled data to give explicit
control over the vocalizations at arbitrary token positions. The three ideas that
transfer: **the tag is inline, at the position where the event happens**; **an
ASR-shaped detector can bootstrap the corpus from found audio**; and **a small
human-validated seed set is enough to bootstrap the automatic labeller**.

**ELaTE** (Kanda et al., 2024, [arXiv:2402.07383](https://arxiv.org/abs/2402.07383))
is the closest *method* reference, because it fine-tunes the same class of model
VoxCPM2 uses. It takes a conditional flow-matching zero-shot TTS and adds
frame-level conditioning from a **laughter detector**, giving control over both
when the laugh happens and how it sounds. Two of its results govern our plan: a
comparatively small conditioned dataset suffices, and **mixing the conditioned
data with general data preserves base-model quality** — the fine-tune does not
have to cost intelligibility.

**NonverbalTTS** (2025, SSW, [arXiv:2507.13155](https://arxiv.org/abs/2507.13155))
is the labelling-pipeline reference: 17 hours covering 10 non-verbal types,
built from open sources by automatic detection followed by human validation, and
reported at parity with proprietary systems. It is the template for
corpus-building at a scale one person can actually reach.

**Laughter detection** (Gillick et al., Interspeech 2021,
[ISCA archive](https://www.isca-archive.org/interspeech_2021/gillick21_interspeech.html))
provides the detector those pipelines depend on — robust frame-level laughter
detection and segmentation, trained on found audio. **VocalSound** (Gong et al.,
ICASSP 2022, [arXiv:2205.03433](https://arxiv.org/abs/2205.03433)) covers the
rest of the inventory: 21k crowdsourced recordings of laughter, sighs, coughs,
throat-clearing, sneezes and sniffs from 3,365 speakers, with a classifier
baseline. Between them they supply the automatic detection stage without any
model training on our side.

**MNV-17** (2025, [arXiv:2509.18196](https://arxiv.org/abs/2509.18196)) is a
high-quality performative Mandarin non-verbal vocalization corpus for
recognition, useful as evidence that the "record it deliberately" route is viable
when found audio is thin.

*Relevance.* This is a complete, published recipe: detector → alignment → human
validation → inline-tagged manifest → flow-matching fine-tune with mixed general
data. Every stage has a reference implementation or a released model, and the
target model class matches VoxCPM2's Local DiT.

### 11.3.4 Evaluating naturalness and non-verbal quality

Any control claim needs a measurement, and this project has already found that
the standard naturalness predictors are unusable for Khmer: across 400 clips the
rank correlation between UTMOS and Khmer CER is **ρ = +0.55** — the clips UTMOS
likes best are the ones that get the Khmer most wrong ([`CLAUDE.md`](../CLAUDE.md)).
So the evaluation question has to be answered by instruments specific to the
thing being controlled.

**NVV-SuperBench / NVBench** (2026, [arXiv:2604.16211](https://arxiv.org/abs/2604.16211))
is the benchmark for exactly this. It pairs a unified **45-type taxonomy** of
non-verbal vocalizations with a bilingual English/Chinese dataset, and — the
important part — a multi-axis protocol that **separates general speech
naturalness from NVV-specific controllability, placement and salience**. Fifteen
TTS systems are evaluated under it. Those four axes are the right ones to report
against: *did the event appear*, *was it the right type*, *was it in the right
place*, *did it sound like the thing*.

**NVMOS** (2026, [arXiv:2606.15888](https://arxiv.org/abs/2606.15888)) supplies
the last of those as a model. It predicts a MOS-like 0–5 perceptual quality score
for a *specific marked non-verbal event*, taking as input the audio plus text
containing an explicit tag such as `[laugh]`. The paper also reports that
general-purpose audio LLMs (Gemini among them) disagree measurably with expert
raters on this task, so a multimodal model is not a substitute. Its input format
— tagged text plus audio — is the format our training manifest will already be
in.

*Relevance.* Together these give an evaluation plan that does not depend on
UTMOS: controllability and placement measured automatically with a detector,
event quality with NVMOS, and intelligibility regression with the project's
existing Khmer CTC CER scorer against the frozen `eval-set/eval.json`.

### 11.3.5 What we will focus on, and why

**Non-verbal vocalization.**

The prosodic layer is the one most of the literature targets, and on this model
it is already available: the parenthetical prompt moves Khmer pitch across
+106 Hz at ρ +0.76, four times the separation our own labelled corpus can
express (§11.1.3). Building a prosodic controller would be reproducing a
capability the base model has, in a channel — the same free-text field — that is
already occupied. The residue (pitch variation, reproducibility across seeds) is
real but small.

The emotional layer is blocked on data, not method. Every result in §11.3.2
depends on acted, parallel, per-utterance-labelled emotional speech, and no Khmer
corpus of that description exists.

Non-verbal vocalization is the layer where the cost-to-value ratio is best, for
four reasons:

1. **It is local.** A tagged event owns the frames it occupies. Global attributes
   have to compete for influence over frames that the acoustic context already
   predicts; a laugh at position *k* has no competitor. This is a structural
   advantage, and it is the reason the conditioning problem that dominated the
   prosodic work is not expected to recur.
2. **The model already has the interface.** `[laughing]`, `[sigh]`, `[Uhm]` are
   documented tags in stock VoxCPM2. The work is making them fire on Khmer, not
   inventing a syntax.
3. **The acoustics are substantially language-independent.** A laugh is a laugh;
   a sigh is a sigh. What is language-specific is *where* they go and what they
   mean in context — which is what the tag position supplies. This is why a small
   Khmer corpus can plausibly be enough, and it is the assumption the plan must
   test first.
4. **Every stage has a published reference.** NVSpeech for the pipeline and the
   inline-token format, ELaTE for the flow-matching fine-tune and the data-mixing
   ratio, NonverbalTTS for the human-validation loop, Gillick and VocalSound for
   detection, NVV-SuperBench and NVMOS for evaluation.

---

## 11.4 Methodology

*To be written.*

---

## References

### Prosody and natural-language style control

- Guo, Z. et al. (2023). *PromptTTS: Controllable text-to-speech with text descriptions.* ICASSP. [arXiv:2211.12171](https://arxiv.org/abs/2211.12171)
- Yang, D. et al. (2023). *InstructTTS: Modelling expressive TTS in discrete latent space with natural language style prompt.* [arXiv:2301.13662](https://arxiv.org/abs/2301.13662)
- Leng, Y. et al. (2024). *PromptTTS 2: Describing and generating voices with text prompt.* ICLR. [arXiv:2309.02285](https://arxiv.org/abs/2309.02285)
- Lyth, D. & King, S. (2024). *Natural language guidance of high-fidelity text-to-speech with synthetic annotations.* [arXiv:2402.01912](https://arxiv.org/abs/2402.01912) — released as Parler-TTS.
- Ji, S. et al. (2024). *TextrolSpeech: A text style control speech corpus with codec language text-to-speech models.* ICASSP. [arXiv:2308.14430](https://arxiv.org/abs/2308.14430)

### Emotion

- Zhou, K., Sisman, B., Liu, R. & Li, H. (2022). *Emotional voice conversion: Theory, databases and ESD.* Speech Communication. [arXiv:2105.14762](https://arxiv.org/abs/2105.14762)
- Hsu, C.-C. et al. (2024). *Laugh Now Cry Later: Controlling time-varying emotional states of flow-matching-based zero-shot text-to-speech.* SLT. [arXiv:2407.12229](https://arxiv.org/abs/2407.12229)

### Non-verbal vocalization

- *NVSpeech: An integrated and scalable pipeline for human-like speech modeling with paralinguistic vocalizations.* (2025). [arXiv:2508.04195](https://arxiv.org/abs/2508.04195)
- Kanda, N. et al. (2024). *ELaTE: Making flow-matching-based zero-shot text-to-speech laugh as you like.* [arXiv:2402.07383](https://arxiv.org/abs/2402.07383)
- *NonverbalTTS: A public English corpus of text-aligned nonverbal vocalizations with emotion annotations for text-to-speech.* (2025). SSW. [arXiv:2507.13155](https://arxiv.org/abs/2507.13155)
- Gillick, J. et al. (2021). *Robust laughter detection in noisy environments.* Interspeech. [ISCA archive](https://www.isca-archive.org/interspeech_2021/gillick21_interspeech.html)
- Gong, Y., Yu, J. & Glass, J. (2022). *VocalSound: A dataset for improving human vocal sounds recognition.* ICASSP. [arXiv:2205.03433](https://arxiv.org/abs/2205.03433)
- *MNV-17: A high-quality performative Mandarin dataset for nonverbal vocalization recognition in speech.* (2025). [arXiv:2509.18196](https://arxiv.org/abs/2509.18196)

### Evaluation

- *NVV-SuperBench: Beyond words, beyond quality — benchmarking nonverbal vocalizations in speech generation.* (2026). [arXiv:2604.16211](https://arxiv.org/abs/2604.16211)
- *NVMOS: Non-verbal vocalization quality assessment in speech.* (2026). [arXiv:2606.15888](https://arxiv.org/abs/2606.15888)

### VoxCPM2 and companion documents

- OpenBMB. *VoxCPM2 model card*, [huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — parenthetical voice design, style-guided cloning, and the non-verbal tag inventory.
- Source: `voxcpm/training/packers.py` (the zero text loss mask) and `voxcpm/model/voxcpm2.py`.
- This project: [`docs/09`](09-voxcpm2-architecture-and-training.md) — VoxCPM2 architecture and training; [`docs/03`](03-evaluation-benchmarking.md) — the evaluation metrics; [`finetune/README.md`](../finetune/README.md) — the runbook; [`finetune/results/parenthetical/parenthetical.md`](../finetune/results/parenthetical/parenthetical.md) — the measurement in §11.1.3.
