# 11 — Speech Control in Neural Text-to-Speech

An assessment of VoxCPM2 for Khmer and a review of the literature on
controllable speech synthesis.

Markdown companion to [`Speech-Control-VoxCPM2.docx`](Speech-Control-VoxCPM2.docx),
built by `build_style_control_report.py`. Both carry the same content; the docx
is the formatted version.

**Contents**

1. [Overview](#1-overview)
   · [1.1 Model Description](#11-model-description)
   · [1.2 System Architecture](#12-system-architecture)
   · [1.3 Built-in Control Mechanisms](#13-built-in-control-mechanisms)
   · [1.4 Measurement of Parenthetical Prosodic Control on Khmer](#14-measurement-of-parenthetical-prosodic-control-on-khmer)
2. [Speech Control](#2-speech-control)
   · [2.1 Definition and Scope](#21-definition-and-scope)
   · [2.2 Prosody](#22-prosody)
   · [2.3 Emotion](#23-emotion)
   · [2.4 Non-Verbal Vocalization](#24-non-verbal-vocalization)
   · [2.5 Related Categories](#25-related-categories)
3. [Literature Review](#3-literature-review)
   · [3.1 Natural Language Style Control](#31-natural-language-style-control)
   · [3.2 Emotional Speech Synthesis](#32-emotional-speech-synthesis)
   · [3.3 Non-Verbal Vocalization](#33-non-verbal-vocalization)
   · [3.4 Evaluation of Controlled Speech](#34-evaluation-of-controlled-speech)
   · [3.5 Research Focus](#35-research-focus)
4. [Methodology](#4-methodology)
   · [References](#references)

---

## 1 Overview

This report concerns the control of delivery in synthetic speech: the ability to
specify how an utterance is spoken rather than only what is said. Section 1
describes VoxCPM2, the model this project has adopted for Khmer, and reports a
measurement of the control it already provides. Section 2 defines the categories
of speech control and distinguishes them from one another. Section 3 reviews the
published work addressing each category and identifies the one this project will
pursue. Section 4, the methodology, is reserved.

### 1.1 Model Description

VoxCPM2 is an open-weight text-to-speech model released by OpenBMB under the
Apache 2.0 licence. It has 2.29 billion parameters and is tokenizer-free: rather
than mapping speech onto a discrete codebook, it predicts continuous latent
vectors, one for every four-frame patch of audio.

This property has a direct bearing on low-resource languages. Systems built on
discrete audio codebooks depend on the codebook having been fitted to the target
language during pre-training, and where it has not, synthesis degrades in a way
that fine-tuning does not readily repair. In the four-model comparison conducted
for this project, Fish Audio S2-Pro failed in exactly that manner, returning a
median character error rate of 78.01 per cent on Khmer. Over the same fixed set
of one hundred sentences VoxCPM2 returned 2.47 per cent, against 8.28 per cent
for Higgs TTS 3 and 25.12 per cent for Meta MMS. VoxCPM2 was selected on that
evidence. (See [`docs/03`](03-evaluation-benchmarking.md) for the metric and
[`CLAUDE.md`](../CLAUDE.md) for the scorer's known bias.)

A second property follows from the input side. The model reads raw UTF-8 bytes
through a 73,448-entry tokenizer and is given no language identifier, inferring
the language from the script. Khmer consequently requires no
grapheme-to-phoneme conversion, no pronunciation lexicon and no word segmenter.
None of these components exists for Khmer in a form suitable for production use,
and assembling them ordinarily accounts for the largest share of effort in a
low-resource text-to-speech project.

### 1.2 System Architecture

The generation path comprises the stages set out below. A backbone language
model emits one latent vector per audio patch; a residual language model refines
it; a local diffusion transformer converts the refined latent into acoustic
features under a conditional flow-matching objective; and a variational
autoencoder decodes those features to a waveform, accepting features derived at
16 kHz and emitting audio at 48 kHz. A separate local encoder, not listed, maps
a reference recording into the same latent space and is the mechanism by which
zero-shot voice cloning is performed.

**Table 1.** The VoxCPM2 generation path. Dimensions and layer counts are taken
from `config.json` in the `openbmb/VoxCPM2` release.

| Stage | Configuration | Output |
|---|---|---|
| Byte-level tokenizer | vocabulary 73,448; no grapheme-to-phoneme stage, no language identifier | token sequence |
| MiniCPM4 backbone LM | 2048 dimensions, 28 layers; grouped-query attention, 16 query and 2 key-value heads; LongRoPE to 32k | one latent vector per audio patch |
| Residual LM | 8 layers, no positional encoding | refined latent |
| Local DiT | 1024 dimensions, 12 layers; conditional flow matching; Euler solver, guidance scale 2.0, 10 steps at inference | 64-dimensional acoustic features, 4 frames per patch |
| AudioVAE V2 | encoder at 16 kHz, decoder at 48 kHz | waveform |

**Table 2.** Architecture parameters bearing on training and control.

| Parameter | Value | Consequence |
|---|---|---|
| `patch_size` | 4 | One language-model step spans four autoencoder frames. |
| `feat_dim` | 64 | Width of the latent the diffusion transformer predicts. |
| Frame rate | 25 fps | One second of audio occupies 6.25 language-model positions. |
| Encoder rate | 16 kHz | Training audio must be supplied at 16 kHz; the validator rejects other rates. |
| Decoder rate | 48 kHz | Bandwidth extension is internal; 48 kHz training data is not required. |
| `inference_cfg_rate` | 2.0 | Default classifier-free guidance scale, exposed as `--cfg-value`. |

Two of these properties bear on the remainder of the report.

The first is the identity of the acoustic decoder. The local diffusion
transformer is a conditional flow-matching model. The methods reviewed in §3.3
for controlling non-verbal vocalization were developed for, and evaluated on,
models of this class. They are therefore applicable to VoxCPM2 without
alteration of the underlying training objective, which is not true of methods
developed for discrete-codec systems.

The second concerns the treatment of the text field during training. The loss
mask constructed in the data packer is zero at every text position:

```python
# voxcpm/training/packers.py, process_tts_data
loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])
#                ^^^^^^^^^^^^^^^^^^ zero at every text position
```

No token in the text field is ever a prediction target. The field operates
purely as a conditioning channel, and its contents may therefore be extended
with tags, markers or descriptive text without perturbing the objective the
model is trained under. Both control mechanisms described below exploit this,
and any mechanism added later would do the same.

### 1.3 Built-in Control Mechanisms

VoxCPM2 provides two mechanisms for influencing delivery. They differ in the
interval over which they apply, and that difference is developed in §2.

The first is a parenthetical description placed before the text, which
characterises the utterance as a whole:

```python
model.generate(
    text="(speaking quickly, a high-pitched voice)"
         "ថ្ងៃនេះអាកាសធាតុល្អណាស់។")
```

The second is a bracketed tag placed inline, which marks a single event at one
position in the text:

```python
model.generate(text="ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…")
```

The documented tag inventory is:

```
[laughing]   [laughter]   [sigh]   [Uhm]   [Shh]
[Question-ah] [Question-ei] [Question-en] [Question-oh]
[Surprise-wa] [Surprise-yo] [Dissatisfaction-hnn]
```

The model documentation describes both mechanisms with reference to Chinese and
English. It makes no statement about their behaviour in other languages, and the
training corpus composition is not published in sufficient detail to infer one.
Their efficacy on Khmer is therefore an empirical question, which the next
section addresses for the first mechanism.

### 1.4 Measurement of Parenthetical Prosodic Control on Khmer

The parenthetical mechanism was evaluated on the unmodified model, using eight
sentences drawn from the project's fixed Khmer evaluation set. For each of four
prosodic axes, three prompts were written to span the axis (Table 3). Every
sentence was synthesised under every prompt at three random seeds, giving 96
generations, and the median taken within each cell of the design. Acoustic
measurement follows the definitions used elsewhere in the project: median
fundamental frequency for pitch, its standard deviation in semitones for
variation, root-mean-square level for energy, and Khmer characters per second
for rate.

Script: `finetune/verify_parenthetical.py`. Results:
`finetune/results/parenthetical/`.

**Table 3.** Prompts used to span each prosodic axis.

| Axis | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| Pitch | *(a low-pitched voice)* | *(a normal-pitched voice)* | *(a high-pitched voice)* |
| Energy | *(speaking softly, quietly)* | *(speaking at a normal volume)* | *(speaking loudly)* |
| Rate | *(speaking slowly)* | *(speaking at a normal pace)* | *(speaking quickly)* |
| Variation | *(a flat, monotone delivery)* | *(a normal delivery)* | *(a lively, expressive delivery)* |

**Table 4.** Response of the unmodified model to a parenthetical prompt on
Khmer. The corpus bound is the separation between the low and high bands of this
project's hand-labelled Khmer corpus, and represents an upper limit on what
fine-tuning against that corpus could teach.

| Axis | Unit | Level 0 | Level 1 | Level 2 | Change | Seed s.d. | Corpus bound | ρ | p |
|---|---|---|---|---|---|---|---|---|---|
| Pitch | Hz | 147.80 | 202.81 | 254.16 | +106.37 | 28.78 | 28.27 | +0.759 | 0.0002 |
| Energy | dBFS | −20.38 | −16.30 | −15.89 | +4.50 | 5.69 | 5.81 | +0.605 | 0.0022 |
| Rate | char/s | 14.67 | 15.84 | 16.96 | +2.29 | 1.58 | 6.57 | +0.590 | 0.0035 |
| Variation | st | 4.55 | 4.06 | 3.92 | −0.63 | 0.68 | 1.71 | −0.273 | 0.2039 |

ρ is the Spearman rank correlation between prompt level and measured value; p is
a permutation test over 10,000 relabellings. Seed s.d. is the mean run-to-run
standard deviation within a cell.

Three findings follow.

**Pitch is controlled reliably.** The separation between the extreme prompts is
106.37 Hz with a rank correlation of +0.759 (p = 0.0002). This exceeds the
corpus bound of 28.27 Hz by a factor of approximately four. The capability a
fine-tune on the project's own labelled data could add to this axis is therefore
negative.

**Energy and speaking rate are controlled in aggregate but not per generation.**
Both show significant rank correlations (+0.605 and +0.590), but the energy
effect of 4.50 dB is smaller than the 5.69 dB standard deviation observed across
seeds within a single cell. The ordering of the levels is dependable; the value
of any individual synthesis is not.

**Pitch variation is not controlled.** The correlation is negative (−0.273) and
not significant (p = 0.20). Prompts requesting a lively delivery produced
marginally less pitch movement than prompts requesting a monotone one, which is
consistent with the axis being unaddressed rather than inverted.

The behaviour of the inline tags on Khmer was not measured, and no claim is made
about it here. Establishing whether they fire at all on Khmer text is a
prerequisite for the work §3.5 selects.

---

## 2 Speech Control

### 2.1 Definition and Scope

The term *expressive control* is used loosely in the literature to denote any
influence over delivery not exercised through the choice of words. It subsumes
several capabilities that differ in what they describe, over what interval they
apply, and in the effort each requires to implement. This section separates the
three that concern this project, and names three further categories that are
adjacent to them.

One distinction cuts across all of them and is used throughout what follows. A
**global attribute** is a property of a whole utterance, or of a long span of
one: it has no onset and no offset, and it is realised in every frame. A **local
event** is bounded: it begins, occupies a short interval, and ends, and it is
associated with a specific position in the text. The two are not merely
different in duration. A global attribute must influence frames whose acoustic
content is already largely determined by the surrounding speech, whereas a local
event is the only thing that determines the frames it occupies. The consequences
for training are taken up in §3.5.

### 2.2 Prosody

Prosody comprises the suprasegmental properties of speech, that is, those
carried above the level of the individual sound. It is what remains when the
identity of the words is set aside: the rate at which they are spoken, the pitch
at which they are set, the loudness with which they are projected, the degree of
pitch movement, and the placement of pauses.

**Table 5.** Prosodic dimensions and their acoustic correlates.

| Dimension | Acoustic correlate | Measurement used here |
|---|---|---|
| Speaking rate | phones or syllables per unit time | Khmer characters per second |
| Pitch register | fundamental frequency | median F0, hertz |
| Pitch variation | F0 range and contour movement | F0 standard deviation, semitones |
| Loudness | signal energy | root-mean-square level, dBFS |
| Phrasing | pause placement and duration | inter-pausal unit statistics |

The sentence *I never said she stole my money* spoken at three syllables per
second and at six differs in rate alone. Spoken with a median fundamental
frequency of 120 hertz and of 240 hertz, it differs in register. Spoken with an
F0 standard deviation near zero it is heard as mechanical, and with a range of
six semitones as engaged; that is variation. In each case the words, and the
meaning they carry, are unchanged.

Prosody is a global attribute. It is the category most extensively treated in
the literature, and, as §1.4 established, the one VoxCPM2 already addresses on
Khmer.

### 2.3 Emotion

Emotion denotes the affective state conveyed by the delivery. The standard
corpora encode it as a small closed set of categories, most commonly neutral,
happy, angry, sad and surprised.

Emotion is realised through prosody together with voice quality, but it does not
reduce to a prosodic setting. Anger and excitement share elevated pitch and
elevated energy and are not perceptually similar; the distinction resides in
phonation, articulatory precision and fine timing, none of which a
rate-pitch-energy specification captures. The sentence *Oh, that's great* spoken
flatly and slowly is heard as sarcastic, spoken quickly and brightly as pleased,
and spoken slowly with creaky phonation as resigned. The three readings differ
in affect, not in prosodic setting alone.

Emotion is treated as a global attribute in essentially all of the literature:
one label per utterance is the near-universal convention. Its status in VoxCPM2
on Khmer is unmeasured. The parenthetical channel would plausibly accept a
description such as *(sounding angry)*, since it is the same free-text field
that already carries *(speaking quickly)*, but this has not been tested.

### 2.4 Non-Verbal Vocalization

A non-verbal vocalization is a sound produced by a speaker that is not a word:
laughter, a sigh, an audible breath, a filled pause, a cough, a gasp, a sob, a
hesitation particle. Such sounds carry stance, regulate turn-taking and convey
affect, and their presence is a substantial part of what distinguishes
conversational speech from read speech.

**Table 6.** Non-verbal vocalization types. Tags in the first five rows are
documented in VoxCPM2; the sixth row is not.

| Type | Tag | Communicative function |
|---|---|---|
| Laughter | `[laughing]` | amusement, affiliation, mitigation |
| Sigh | `[sigh]` | resignation, fatigue, relief |
| Filled pause | `[Uhm]` | planning, hesitation, floor-holding |
| Attention marker | `[Shh]` | silencing, conspiratorial framing |
| Discourse particle | `[Question-ah]`, `[Surprise-wa]`, `[Dissatisfaction-hnn]` | stance, back-channelling, question marking |
| Breath, gasp, cough, sob | — | phrasing, surprise, distress, physical state |

In the utterance *I thought it was fine* `[laughing]` *but apparently not*, the
laugh occurs at one place, occupies a few hundred milliseconds, and leaves the
speech before and after it unaffected. In *So we should* `[Uhm]` *probably
wait*, a filled pause is inserted mid-clause and has the same bounded character.

Non-verbal vocalization is therefore a local event, and this is its significant
property for present purposes. A global attribute competes for influence over
frames that the surrounding acoustic context already predicts. A tagged event
has no such competitor: the frames it occupies are predicted by nothing else,
and the conditioning signal is the only available explanation for them. The
engineering problem is correspondingly different, and, as §3.5 argues, easier.

VoxCPM2 supplies the tag inventory. Whether the tags are realised on Khmer text
is unmeasured.

### 2.5 Related Categories

Three further categories are named here so that they are not conflated with the
preceding three.

**Table 7.** Categories adjacent to prosody, emotion and non-verbal vocalization.

| Category | Definition | Scope | Example |
|---|---|---|---|
| Voice quality | mode of phonation | global | whispered, breathy, creaky, tense |
| Emphasis | placement of contrastive stress | local span | *I* never said that; I never said *that* |
| Timing | insertion and duration of pauses | local | a deliberate pause before a resolution |

Voice quality behaves as emotion does: it is global, and in principle
addressable through the same descriptive channel. Emphasis and timing are local,
as non-verbal vocalization is, but they require a syntax that delimits a span
rather than marking a point. VoxCPM2 provides no such syntax, and adding one is
a larger undertaking than adding a tag.

---

## 3 Literature Review

Four bodies of work bear on the categories set out above: natural language
control of prosody and style, emotional speech synthesis, non-verbal
vocalization, and the evaluation of controlled speech. Each is reviewed in turn,
and §3.5 states which category this project will pursue and on what grounds.

### 3.1 Natural Language Style Control

The organising idea of this literature is to replace the reference recording
with a description, and to obtain the descriptions by automatic means rather
than by annotation.

Guo et al. (2023) established the format with **PromptTTS**, which takes a style
prompt and a content prompt, encodes them separately, and conditions an acoustic
model on both. The approach was constrained by its data: the style-annotated
corpus had to be constructed manually, which limited its scale. Yang et al.
(2023) relaxed the input format in **InstructTTS**, accepting free-form
instructions rather than attribute lists and learning a cross-modal
representation that aligns instruction text with speech style, decoded in a
discrete latent space.

Leng et al. (2024) addressed the two limitations that constrain the family as a
whole. The first is that a description underdetermines a voice: many distinct
voices satisfy *a young woman speaking quickly*, and a model trained to map
descriptions to speech must resolve that ambiguity somehow. **PromptTTS 2**
introduces a variation network that predicts, from the prompt representation,
the reference-speech representation that would otherwise have been supplied. The
second is annotation cost, addressed by a pipeline in which a speech
understanding model recognises attributes and a large language model writes the
corresponding prompt sentence. The system was trained on 44,000 hours.

Lyth and King (2024) give the clearest statement of the annotation argument and
the one released without restriction, as **Parler-TTS**. Gender, accent, pitch,
speaking rate and recording conditions are labelled computationally across a
45,000-hour corpus of found data using classifiers and signal measurements, and
the resulting attributes are rendered as descriptive sentences on which the
model is conditioned. The system outperforms prior work on fidelity while using
no manually annotated data. The transferable result is that the labels required
for prosodic control can be measured rather than annotated, which is the
property this project relied upon when it labelled a Khmer corpus by measuring
fundamental frequency, character rate and root-mean-square level per clip. Ji et
al. (2024) supply the corpus counterpart in **TextrolSpeech**: 236 hours and
33,000 utterances with style descriptions generated by a language-model pipeline
over five attribute dimensions.

This literature accounts for the parenthetical mechanism in VoxCPM2, which uses
the same conditioning format and was presumably trained in the same manner. It
also sets the expectation against which §1.4 should be read: these systems
control pitch, rate and energy from description, and those are precisely the
axes the measurement finds already functioning on Khmer.

### 3.2 Emotional Speech Synthesis

Work on emotion is organised around acted, parallel corpora. Zhou et al. (2022)
survey the field and introduce the **Emotional Speech Dataset**, which remains
the standard reference: 350 parallel utterances from ten English and ten
Mandarin speakers across five emotion categories, exceeding 29 hours recorded
under controlled acoustic conditions, and designed to support multi-speaker and
cross-lingual conversion.

Hsu et al. (2024) are the bridge between this section and the next. Their system
controls speaker emotion and laughter within a single flow-matching zero-shot
model, and in doing so treats a laugh as a controllable event rather than as an
emotional label. The separation between emotion and non-verbal vocalization,
clear enough as a definition, is not clean in practice.

The obstacle in this category is the shape of the data rather than the adequacy
of the method. Every result rests on parallel, acted, utterance-labelled
emotional speech. No Khmer corpus of that description exists, and commissioning
one is not proportionate to the value it would return.

### 3.3 Non-Verbal Vocalization

This is the most active of the four areas and the one whose methods transfer
most directly to VoxCPM2.

**NVSpeech** (2025) is the closest match to the problem stated in §2.4. It
treats recognition and synthesis as one pipeline. A manually annotated set of
48,430 utterances covering eighteen word-level paralinguistic categories is used
to train a paralinguistic-aware speech recogniser that emits the cues as inline
decodable tokens, so that a transcript reads *You're so funny [Laughter]*. That
recogniser then labels a corpus of 174,179 Chinese utterances, 573 hours, with
word-level alignment. A zero-shot synthesiser is finally fine-tuned on the
combined human- and machine-labelled data, yielding explicit control over
vocalizations inserted at arbitrary token positions. Three elements transfer:
the tag is placed inline at the position of the event rather than in a header; a
recogniser-shaped detector can bootstrap a corpus from found audio; and a modest
human-validated seed set suffices to bootstrap the automatic labeller.

Kanda et al. (2024) provide the closest methodological reference. **ELaTE**
fine-tunes a conditional flow-matching zero-shot synthesiser using frame-level
conditioning derived from a laughter detector, obtaining control over both the
timing of a laugh and its acoustic character. Two of its results constrain any
plan built on it. A comparatively small conditioned dataset is sufficient; and
mixing the conditioned data with general training data preserves the quality of
the base model, so the fine-tune need not be paid for in intelligibility. The
relevance is direct: the decoder ELaTE modifies belongs to the same class as the
VoxCPM2 local diffusion transformer.

**NonverbalTTS** (2025) is the reference for corpus construction at a tractable
scale. Seventeen hours covering ten non-verbal types were assembled from open
sources by automatic detection followed by human validation, and the resulting
system is reported at parity with proprietary alternatives. The detectors on
which such pipelines depend are themselves available: Gillick et al. (2021)
release robust frame-level laughter detection and segmentation trained on found
audio, and Gong et al. (2022) release **VocalSound**, 21,000 crowdsourced
recordings of laughter, sighs, coughs, throat-clearing, sneezes and sniffs from
3,365 speakers, together with a classifier baseline. Between them the automatic
detection stage requires no model training. Where found audio is too thin to
support detection, deliberate recording remains viable; **MNV-17** (2025)
demonstrates this for Mandarin.

**Table 8.** Principal references for non-verbal vocalization, by pipeline stage.

| Stage | Reference | Contribution |
|---|---|---|
| Detection | Gillick et al. (2021); Gong et al. (2022) | Released frame-level laughter detector; released classifier over six vocalization types. |
| Corpus construction | NVSpeech (2025); NonverbalTTS (2025) | Recogniser-driven auto-labelling at 573 hours; detection plus human validation at 17 hours. |
| Synthesis | Kanda et al. (2024); Hsu et al. (2024) | Flow-matching fine-tuning with detector conditioning; joint emotion and laughter control. |
| Deliberate recording | MNV-17 (2025) | Performative corpus where found audio is insufficient. |

Taken together these constitute a complete and published procedure: detection,
alignment, human validation, an inline-tagged manifest, and a flow-matching
fine-tune with general data mixed in. Every stage has either a reference
implementation or a released model.

### 3.4 Evaluation of Controlled Speech

A control claim requires an instrument, and this project has already established
that the usual instruments are unsuitable for Khmer. Across 400 clips the rank
correlation between UTMOS, a learned mean-opinion-score predictor, and Khmer
character error rate is +0.55: the recordings the predictor scores highest are
those that render the Khmer least correctly. The inversion is between models
rather than within any one model's output, which is precisely the comparison the
predictor was being used to make. Naturalness predictors trained without Khmer
in view cannot be relied upon here.

Two recent contributions address the evaluation of non-verbal vocalization
specifically. **NVV-SuperBench** (2026) pairs a unified taxonomy of 45
vocalization types with a bilingual English and Chinese dataset and, more
usefully, defines a protocol that separates general speech naturalness from
vocalization-specific controllability, placement and salience; fifteen systems
are evaluated under it. Those four axes are the appropriate ones to report
against, since they decompose the question into whether the event occurred,
whether it was of the requested type, whether it occurred in the requested
place, and whether it was acoustically convincing.

**NVMOS** (2026) supplies the last of these as a model rather than as a
listening panel. It predicts a mean-opinion-score-like value between zero and
five for a specific marked non-verbal event, taking as input the audio together
with text containing an explicit tag such as `[laugh]`. The authors additionally
report that general-purpose audio-capable multimodal models disagree measurably
with expert raters on this task, so a multimodal model is not an acceptable
substitute. Its input format, tagged text paired with audio, is the format in
which a training manifest for this work would already exist.

An evaluation plan that avoids the inverted predictors therefore exists:
controllability and placement measured automatically with a detector, acoustic
quality of the event measured with NVMOS, and intelligibility regression
measured with the project's existing Khmer connectionist temporal classification
scorer against the frozen evaluation set.

### 3.5 Research Focus

**This project will pursue non-verbal vocalization.** The reasoning proceeds by
elimination and is then stated positively.

*Prosody is already provided.* The parenthetical mechanism moves Khmer pitch
across 106.37 hertz at a rank correlation of +0.759, roughly four times the
28.27 hertz separation the project's own labelled corpus is able to express
(§1.4). A prosodic controller trained on that corpus would reproduce a
capability the base model possesses, in a conditioning channel that is already
occupied. The residue, principally pitch variation and reproducibility across
random seeds, is genuine but small.

*Emotion is blocked on data rather than on method.* The results reviewed in §3.2
depend without exception on acted, parallel, utterance-labelled emotional
speech, and no Khmer corpus of that description exists.

Non-verbal vocalization is the remaining category, and four considerations
recommend it.

1. **It is a local event.** A tagged vocalization is the only predictor of the
   frames it occupies, whereas a global attribute must compete for influence
   over frames the surrounding context already determines. The conditioning
   difficulty that attends global-attribute training is therefore not expected
   to arise.
2. **The interface exists.** The tags `[laughing]`, `[sigh]` and `[Uhm]` are
   documented in the unmodified model. The work is to make them operate on
   Khmer, not to design a syntax and persuade the model to read it.
3. **The acoustics are substantially language-independent.** Laughter and
   sighing are not language-specific gestures; what is language-specific is
   where they are placed and what they signal in context, and placement is
   supplied by the tag position. This is why a small Khmer corpus may be
   sufficient, and it is the assumption that any methodology must test before
   committing resources.
4. **Every stage has a published reference:** NVSpeech for the pipeline and the
   inline-token format, ELaTE for the flow-matching fine-tune and the
   data-mixing ratio, NonverbalTTS for the human-validation loop, Gillick et al.
   and VocalSound for detection, and NVV-SuperBench and NVMOS for evaluation.

---

## 4 Methodology

Reserved.

---

## References

Gillick, J., Deng, W., Ryokai, K. and Bamman, D. (2021). Robust laughter
detection in noisy environments. *Interspeech 2021*, 2481–2485.
[ISCA archive](https://www.isca-archive.org/interspeech_2021/gillick21_interspeech.html)

Gong, Y., Yu, J. and Glass, J. (2022). VocalSound: a dataset for improving human
vocal sounds recognition. *ICASSP 2022*.
[arXiv:2205.03433](https://arxiv.org/abs/2205.03433)

Guo, Z., Leng, Y., Wu, Y., Zhao, S. and Tan, X. (2023). PromptTTS: controllable
text-to-speech with text descriptions. *ICASSP 2023*.
[arXiv:2211.12171](https://arxiv.org/abs/2211.12171)

Hsu, C.-C., Kanda, N., Zhu, Y. et al. (2024). Laugh Now Cry Later: controlling
time-varying emotional states of flow-matching-based zero-shot text-to-speech.
*IEEE SLT 2024*. [arXiv:2407.12229](https://arxiv.org/abs/2407.12229)

Ji, S., Zuo, J., Fang, M. et al. (2024). TextrolSpeech: a text style control
speech corpus with codec language text-to-speech models. *ICASSP 2024*.
[arXiv:2308.14430](https://arxiv.org/abs/2308.14430)

Kanda, N., Wang, X., Eskimez, S. E. et al. (2024). Making flow-matching-based
zero-shot text-to-speech laugh as you like.
[arXiv:2402.07383](https://arxiv.org/abs/2402.07383)

Leng, Y., Guo, Z., Shen, K. et al. (2024). PromptTTS 2: describing and
generating voices with text prompt. *ICLR 2024*.
[arXiv:2309.02285](https://arxiv.org/abs/2309.02285)

Lyth, D. and King, S. (2024). Natural language guidance of high-fidelity
text-to-speech with synthetic annotations.
[arXiv:2402.01912](https://arxiv.org/abs/2402.01912). Released as Parler-TTS.

MNV-17: a high-quality performative Mandarin dataset for nonverbal vocalization
recognition in speech (2025).
[arXiv:2509.18196](https://arxiv.org/abs/2509.18196)

NonverbalTTS: a public English corpus of text-aligned nonverbal vocalizations
with emotion annotations for text-to-speech (2025). *Speech Synthesis Workshop
2025*. [arXiv:2507.13155](https://arxiv.org/abs/2507.13155)

NVMOS: non-verbal vocalization quality assessment in speech (2026).
[arXiv:2606.15888](https://arxiv.org/abs/2606.15888)

NVSpeech: an integrated and scalable pipeline for human-like speech modeling
with paralinguistic vocalizations (2025).
[arXiv:2508.04195](https://arxiv.org/abs/2508.04195)

NVV-SuperBench: beyond words, beyond quality — benchmarking nonverbal
vocalizations in speech generation (2026).
[arXiv:2604.16211](https://arxiv.org/abs/2604.16211)

OpenBMB (2025). VoxCPM2 model card.
[huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
Source referenced in §1.2: `voxcpm/training/packers.py` and
`voxcpm/model/voxcpm2.py`.

Yang, D., Liu, S., Huang, R. et al. (2023). InstructTTS: modelling expressive
TTS in discrete latent space with natural language style prompt.
[arXiv:2301.13662](https://arxiv.org/abs/2301.13662)

Zhou, K., Sisman, B., Liu, R. and Li, H. (2022). Emotional voice conversion:
theory, databases and ESD. *Speech Communication*, 137, 1–18.
[arXiv:2105.14762](https://arxiv.org/abs/2105.14762)

---

**Project companions.** [`docs/03`](03-evaluation-benchmarking.md) — evaluation
metrics · [`docs/09`](09-voxcpm2-architecture-and-training.md) — VoxCPM2
architecture and training · [`finetune/README.md`](../finetune/README.md) — the
runbook · [`finetune/results/diagnosis.md`](../finetune/results/diagnosis.md) —
the measurement record of the prosodic fine-tune and its conditioning failure.
