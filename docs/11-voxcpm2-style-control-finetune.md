# 11 — Speech Control for VoxCPM2

[Document 9](09-voxcpm2-architecture-and-training.md) covers VoxCPM2's architecture and what fine-tuning it involves. This document covers **expressive control**: what kinds of it exist, which ones VoxCPM2 already has, what was measured on Khmer, and what it would take to add the ones it does not.

The motivating gap: [Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md) accepts inline control tokens — `<|emotion:sadness|>`, `<|speed:fast|>` — so a caller can request a delivery. VoxCPM2 appeared to have no equivalent, and this project's recommendation rests on VoxCPM2 (2.47% Khmer CER against Higgs's 8.28%, Apache-2.0).

---

## 11.1 Findings

**1. VoxCPM2 already has a prosodic control channel, and on Khmer it mostly works.** A natural-language description in parentheses before the text — `(speaking quickly)`, `(a high-pitched voice)` — steers the delivery. Measured on the base model, 8 sentences × 3 levels × 3 seeds:

| axis | low | mid | high | low→high | corpus ceiling | rho | p | run-to-run sd | effect ÷ noise |
|---|---|---|---|---|---|---|---|---|---|
| `pitch` | 147.80 | 202.81 | 254.16 | **+106.37 Hz** | +28.27 Hz | **+0.759** | 0.0002 | 28.78 | **3.7** |
| `energy` | −20.38 | −16.30 | −15.89 | +4.50 dB | +5.81 dB | +0.605 | 0.0022 | 5.69 | 0.8 |
| `rate` | 14.67 | 15.84 | 16.96 | +2.29 ch/s | +6.57 ch/s | +0.590 | 0.0035 | 1.58 | 1.4 |
| `var` | 4.55 | 4.06 | 3.92 | −0.63 st | +1.71 st | −0.273 | 0.2039 | 0.68 | — |

Pitch spans nearly four times what a hand-labelled corpus of the 20 available Khmer speakers can express. Rate and energy correlate with the prompt but are noise-limited for a single utterance: re-rolling the seed moves energy further than changing the command does. Pitch *variation* fails outright and in the wrong direction — asking for "a lively, expressive delivery" produced less pitch movement than asking for a flat one.

*Instrument:* `finetune/verify_parenthetical.py`. *Results:* `finetune/results/parenthetical/`. *Caveat:* the prompt wordings are extrapolated from OpenBMB's two documented examples, so the `var` failure could be a vocabulary mismatch rather than a model limit.

**2. A global attribute placed in the text field earns almost no gradient under teacher forcing.** Measured on the training objective directly, with a positive control:

| condition | mean loss/diff | delta | rows worse | sign test |
|---|---|---|---|---|
| correct speaker tag | 0.86908 | — | — | — |
| swapped speaker tag | 0.86930 | +0.03% | 22/40 | p = 0.32 |
| scrambled transcript | 0.93348 | +7.4% | 35/40 | — |

A 290× ratio. Cause: every audio patch is predicted with the ground-truth preceding patches visible, and pitch is trivially readable off those, so the tag supplies nothing new. Restricting the loss to the first K patches confirms it — the swap penalty is +0.00318 at K=1 and decays monotonically to +0.00022 over the full clip.

**3. That is a cost, not a wall.** The parenthetical prompt of finding 1 is *also* a global attribute in the text field, and it works. The teacher-forcing shortcut makes global conditioning **data-expensive**: OpenBMB paid the cost at pre-training scale, 11.7 hours of Khmer through a LoRA did not.

**4. Two changes make a small run learn a global tag anyway.** Isolated one at a time on a two-speaker probe against a +30 Hz bar fixed in advance:

| arm | change | separation | verdict |
|---|---|---|---|
| `end` | tag adjacent to `audio_start` | +5.8 Hz | fail |
| `proj` | + `enable_proj: true` | +19.5 Hz | fail |
| `onset` | + 8× onset-weighted loss, τ=4 | **+39.9 Hz** | **pass** |

Confirmed at the objective level: the swap penalty went from +0.00022 (p = 0.32) to +0.00086 (p = 0.008).

**5. VoxCPM2 LoRA fine-tuning fits in 12 GB.** The documented figure is ~20 GB. Casting frozen weights to bf16 and checkpointing all 36 transformer layers brings peak usage to 11.3 GiB on an RTX 3060.

**6. Local events are cheap where global attributes are expensive.** Nothing in the acoustic prefix predicts that a laugh is coming, so a local tag faces none of the competition finding 2 measures. This is the argument for prioritising the non-verbal layer (§11.4–11.5) over emotion or voice quality.

**Status of the prosodic fine-tune.** Stopped at step 1,980 of 4,000; the checkpoint is on disk and resumable. Findings 2–5 are from that work and stand independently of it.

---

## 11.2 The taxonomy of speech control

"Expressive control" is not one capability. The distinction that decides everything downstream is **global attributes** — true of every frame of the utterance — versus **local events**, which occupy one bounded position. Finding 2 shows why: a global attribute competes with the teacher-forced acoustic prefix and loses; a local event has no such competitor.

| # | layer | controls | scope | mechanism in VoxCPM2 | Khmer status | literature |
|---|---|---|---|---|---|---|
| 1 | **Prosodic** | pitch register, rate, loudness | global | parenthetical prompt `(…)`, built in | **measured, works** (§11.3) | PromptTTS, PromptTTS 2, InstructTTS, Parler-TTS, TextrolSpeech |
| 1b | **Prosodic residue** | pitch *variation*; named speaker; reproducibility | global | would need a fine-tune | gaps confirmed (§11.3.4) | Parler-TTS, `dataspeech` |
| 2 | **Non-verbal vocalization** | laughter, sighs, hesitation, breath | **local** | inline `[laughing]` tags, built in | **untested** (§11.4) | ELaTE, NonverbalTTS, VocalSound, Gillick et al. |
| 3 | **Affective** | emotion — sadness, joy, anger | global | plausibly the same `(…)` prompt | unmeasured | Laugh Now Cry Later, EmoKnob-style prompt control, ESD |
| 4 | **Voice quality** | whisper, breathy, creaky | global | plausibly the same `(…)` prompt | unmeasured | wTIMIT, whisper voice-conversion |
| 5 | **Discourse** | word emphasis, contrastive focus | **local** | none — needs a span syntax | not designed | Controllable Emphasis with zero data |
| 6 | **Timing** | pause insertion, phrase breaks | local | punctuation only, implicit | not designed | — |

**Layers 3 and 4 are unmeasured, not blocked.** They are global attributes that the parenthetical channel plausibly already carries — OpenBMB's own documented example is `(slightly faster, cheerful tone)`, which is layer 3. Extending `finetune/verify_parenthetical.py` to them is a dictionary edit and about twenty minutes per layer. Only if that returns negative does the corpus problem apply, and then finding 3 sets the price.

**Layers 5 and 6 need a different syntax.** A header tag cannot say *which word* to emphasise. That requires span marking in the text — `<em>` around a word, or a parallel per-token flag — and the reference approach is Amazon's zero-data method: lengthen the predicted duration of the target word rather than train an emphasis label at all.

---

## 11.3 Layer 1 — Prosody

### 11.3.1 The built-in channel

```python
wav = model.generate(text="(speaking quickly)ថ្ងៃនេះអាកាសធាតុល្អណាស់។")
```

OpenBMB document this for voice design (`(A young woman, gentle and sweet voice)`) and style-guided cloning (`(slightly faster, cheerful tone)`), listing gender, age, tone, emotion and pace as what it steers. The measurement in finding 1 is the first on Khmer.

Prompts used: `(speaking slowly / at a normal pace / quickly)`, `(a low- / normal- / high-pitched voice)`, `(a flat, monotone / a normal / a lively, expressive delivery)`, `(speaking softly, quietly / at a normal volume / loudly)`. Each cell was generated three times because the model card warns results vary between runs; that repeat is what separates *a measurable effect* from *control*, and it is the reason energy is reported as unusable despite p = 0.002.

This places VoxCPM2 in the natural-language-prompt family: PromptTTS, InstructTTS, PromptTTS 2 and Parler-TTS all condition on a free-text style description rather than on categorical labels. TextrolSpeech and SpeechCraft are the corpora that made that family trainable.

### 11.3.2 The fine-tuned tag: mechanism

Prefix a control tag to the text field:

```
<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។
```

There is no architectural change. `voxcpm/training/packers.py::process_tts_data()` builds each sequence as `[text tokens] [audio start] [audio patches] [audio end]` and sets:

```python
loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])
```

Zero across every text position — the model is never asked to reproduce the text, only to use it in predicting the audio latents. Anything in the text field is pure conditioning, costing only sequence length: 25 tokens against roughly 140 for a typical Khmer sentence. Parler-TTS uses the same trick with a separate description encoder, which VoxCPM2 does not need because its tokenizer is byte-level over raw text.

The base model ignores the tag entirely — identical transcripts and identical 2.08 s duration with and without it — which makes it a clean control condition for verification.

### 11.3.3 The labels

There is no expressive or emotional Khmer corpus, so the axes are defined as quantities measurable on a waveform. This is the Parler-TTS recipe: annotate automatically, bin the measurements, name the bins.

| axis | levels | measured as |
|---|---|---|
| `spk` | `f1`…`f9`, `m1`…`m11` | speaker identity |
| `rate` | `slow` `mid` `fast` | Khmer characters per second |
| `pitch` | `low` `mid` `high` | median F0 |
| `var` | `flat` `mid` `lively` | F0 standard deviation, semitones |
| `energy` | `soft` `mid` `loud` | RMS level |

Two design choices, both of which `dataspeech` also makes: bin edges come from the 15/85 tails rather than tertiles (a 40–54% wider control range), and `pitch`/`var`/`energy` are ranked **within speaker** so the axis describes *how* something is said rather than *who* says it. `rate` is ranked globally.

Corpus: 5,646 train / 174 val clips, 20 speakers, 11.73 h of Khmer read speech. Per-slot dropout leaves partial tags valid at inference.

### 11.3.4 What a fine-tune would still add

| | built-in prompt | fine-tuned tag |
|---|---|---|
| pitch | +106 Hz, robust | ceiling +28 Hz — strictly worse |
| rate | +2.29 ch/s, 1.4× noise | comparable at best |
| energy | effect below the noise | possible improvement, unproven |
| pitch variation | fails, wrong direction | possible improvement, unproven |
| named speaker, no reference clip | not available | **only the tag does this** |
| reproducible for one utterance | no | yes, deterministic given a seed |
| enumerable and sweepable | no | yes |

Three residues: pitch variation, named-speaker selection, reproducibility.

### 11.3.5 Configuration

```yaml
lora: {enable_lm: true, enable_dit: true, enable_proj: true, r: 64, alpha: 64}
batch_size: 1
grad_accum_steps: 8          # effective batch 8
learning_rate: 0.0001
max_batch_tokens: 896
```

Run with `--onset-weight 8.0 --onset-tau 4.0`. `enable_proj` adapts `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj` and `fusion_concat_proj` — the only route text-derived information reaches the DiT. The onset weighting gives position *i* of the audio span weight `1 + (W−1)·exp(−i/τ)`, which is a true per-position weight because `unified_cfm.compute_loss` computes `(mask*losses).sum()/sum(mask)`; upstream casts the mask to `int32`, so it is rebuilt as float in `finetune/train.py` PATCH 4.

The 12 GB result comes from casting frozen weights to bf16 while keeping LoRA parameters in fp32, and checkpointing all 36 transformer layers. Upstream leaves 2.29 B frozen parameters in float32, which is 8.6 GiB before a single activation.

---

## 11.4 Layer 2 — Non-verbal vocalization

VoxCPM2 ships with this layer. Inline square-bracket tags are placed in the text at the point where the vocalization should occur, on the stock checkpoint, with no training:

| category | tags | effect |
|---|---|---|
| laughter and breath | `[laughing]`, `[sigh]` | audible laugh or exhalation |
| hesitation | `[Uhm]`, `[Shh]` | filled pause; shushing sound |
| question particles | `[Question-ah]`, `[Question-ei]`, `[Question-en]`, `[Question-oh]` | interrogative interjections |
| emotional interjections | `[Surprise-wa]`, `[Surprise-yo]`, `[Dissatisfaction-hnn]` | surprise; dissatisfaction |

```python
# stock VoxCPM2 -- no adapter
wav = model.generate(text="ខ្ញុំគិតថា... [laughing] ...")

# the mechanisms compose
text = ("(a young woman, warm voice)"
        "<|spk:any|rate:slow|pitch:any|var:lively|energy:any|>"
        "ខ្ញុំ... [sigh] ...")
```

OpenBMB's guidance: use them sparingly, prefer lowercase `[laughing]` over `[Laughter]`, do not stack several in one sentence. The model card lists instability on "very long or highly expressive inputs".

**All of this is documented for VoxCPM2 in general and none of it for Khmer.** Three open questions:

| question | why it is open |
|---|---|
| Do the tags fire in Khmer text at all? | They are English strings inside a Khmer sentence, and Khmer tokenises to UTF-8 byte tokens while the tag does not. |
| Are they spoken instead of performed? | The failure to check for is the model pronouncing "laughing" as a word. |
| Does the prosodic adapter erode them? | It was trained on read speech containing no laughter, and LoRA erodes capabilities absent from the fine-tuning distribution. |

Non-verbal vocalizations are largely language-independent in their acoustics — a laugh is a laugh — which is why this layer is tractable for Khmer where an emotion corpus is not. It is also why detectors trained on English data (VocalSound, AudioSet) can be pointed at Khmer audio to build labels.

---

## 11.5 Training plan: non-verbal vocalization by tag

The goal is not to add the capability — VoxCPM2 has it — but to make it fire reliably on Khmer text with Khmer-appropriate vocalization. Two published recipes bound the approach:

- **ELaTE** ([Kanda et al., 2024](https://arxiv.org/abs/2402.07383)) fine-tunes a **conditional-flow-matching zero-shot TTS** — the same family as VoxCPM2's Local DiT — for laughter control, using frame-level output from a laughter detector as extra conditioning. Two results transfer directly: a *small* laughter-conditioned set suffices, and mixing it with general pre-training data preserves base quality.
- **NonverbalTTS** ([Borisov et al., SSW 2025](https://arxiv.org/abs/2507.13155)) builds a 17-hour corpus covering 10 non-verbal types by **automatic detection followed by human validation** over VoxCeleb and Expresso, then shows open TTS models fine-tuned on it reach parity with CosyVoice2. That is the labelling pipeline to copy.

### Phase 0 — measure the base model (no data, no training)

Go/no-go, and it may end the project. Synthesize a fixed set of Khmer sentences three ways — base model, adapter enabled, adapter disabled — with and without each tag. Metrics, all automatic:

| signal | what it answers | tool |
|---|---|---|
| duration delta | did anything get inserted? | `verify_control.py`'s measurement layer |
| Khmer CTC ASR transcript | is the tag being *spoken*? | `evaluation/score_cer_khmer.py` |
| laughter-detector probability on the output | is what was inserted actually a laugh? | [`jrgillick/laughter-detection`](https://github.com/jrgillick/laughter-detection) ([Gillick et al., 2021](https://www.isca-archive.org/interspeech_2021/gillick21_interspeech.html)) |

The third is the load-bearing one, and it is the same instrument that will later produce labels. If the tags already fire on Khmer, no training is needed for those tags and the plan reduces to whatever subset failed.

### Phase 1 — source audio

Read-speech corpora contain no non-verbal events; the 11.73 h prosody corpus has none. The material has to be conversational: Khmer podcasts, interviews, broadcast talk. Pipeline per NonverbalTTS: VAD → diarization → ASR → non-verbal detection → human validation of a sample.

Budget, extrapolating from the two references: NonverbalTTS is 17 h for 10 types across many speakers, and ELaTE shows a single event type needs far less. **Target ~500–1,000 validated events per tag** for `[laughing]`, `[sigh]` and `[Uhm]` first; the interjection tags are lower value and can wait.

### Phase 2 — detection and labelling

- **Laughter:** Gillick et al.'s detector gives start/end timestamps directly.
- **Other events:** a classifier trained on [VocalSound](https://arxiv.org/abs/2205.03433) (21k crowdsourced clips of laughter, sighs, coughs, sneezes, sniffs, throat-clearing) or AudioSet's corresponding classes.
- **Alignment:** force-align the ASR transcript to get word timings, then insert the tag string at the word boundary nearest the detected event onset.
- **Validation:** listen to a sample. NonverbalTTS's precision came from human validation over automatic detection, not from the detector alone.

Output rows look exactly like the existing prosody manifest, with the tag inline rather than in a header:

```json
{"audio": "/…/clip_00412.wav", "text": "ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…", "duration": 4.21}
```

### Phase 3 — training

Same harness as the prosody run. Differences that follow from the layer being local:

| setting | prosody run | non-verbal run | why |
|---|---|---|---|
| tag position | header, before the text | **inline, at the event** | the model must learn *when*, not only *whether* |
| `--onset-weight` | 8.0 | **1.0 (off)** | onset weighting exists to beat the prefix shortcut for global attributes; a local event has no prefix competitor |
| `enable_proj` | true | true | still the only text→DiT route |
| data mixing | n/a | **~1:4 tagged to untagged** | ELaTE's finding; prevents unprompted laughter and preserves base quality |
| negative examples | n/a | **required** | clips with no tag and no event, so absence is trained too |

The data mixing and negative examples are the two things most likely to be skipped and most likely to cause the characteristic failure: a model that laughs everywhere.

### Phase 3b — frame-level conditioning, only if timing is poor

If the inline tag places the event but with unreliable timing, ELaTE's actual method is the escalation: condition the DiT on a per-frame laughter-probability channel from the detector rather than on a discrete text token. This is invasive — it adds an input to the flow-matching head — and should not be attempted before the cheap version is measured.

### Phase 4 — evaluation

| axis | metric | pass condition |
|---|---|---|
| does the event occur | detector fires on the output | precision and recall against the commanded tag |
| is it in the right place | detector onset vs commanded position | within a stated tolerance |
| is the tag silent | Khmer CTC ASR transcript | no tag text in the transcript |
| no intelligibility regression | CER on the frozen 100-sentence eval set | within noise of the base model's 2.47% |
| does it sound right | `evaluation/listening_test.py` | blind A/B against base |

The CER regression check is the one that must not be skipped: this project already has the harness, and `eval-set/eval.json` is fixed across all models precisely so that a fine-tune can be checked against it.

---

## 11.6 Layers 3–6

**Layer 3 — affective.** Measure the parenthetical channel first (`(a sad tone)`, `(an angry voice)`, `(cheerful)`), since OpenBMB's own documented example is already an emotion prompt. If it fails, the corpus problem is real: no Khmer emotion corpus exists, and the English reference points are ESD and the emotion subset of NonverbalTTS. [Laugh Now Cry Later](https://arxiv.org/abs/2407.12229) is the closest method reference — time-varying emotional state control on the same flow-matching zero-shot TTS family as ELaTE.

**Layer 4 — voice quality.** Same measurement first. Whisper is the tractable case in the literature — wTIMIT provides parallel normal/whispered recordings, and Amazon's [voice-conversion approach](https://arxiv.org/abs/1912.05289) shipped in Alexa's Whisper Mode — but no Khmer equivalent exists, so a positive result from the prompt channel is the only cheap path.

**Layer 5 — discourse and emphasis.** Needs span marking, not a header or inline tag. The reference is [Controllable Emphasis with zero data](https://arxiv.org/abs/2307.07062): rather than training an emphasis label, increase the predicted duration of the target word. That is a decoder-side intervention and does not obviously port to VoxCPM2's patch-level diffusion decoder, which has no explicit per-word duration predictor to reach into — so this layer is genuinely undesigned, not merely unbuilt.

**Layer 6 — timing.** Currently implicit in punctuation. No work done.

---

## 11.7 Using it today

**Prosody, base model, no training:**

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
wav = model.generate(text="(a high-pitched voice, speaking quickly)ថ្ងៃនេះអាកាសធាតុល្អណាស់។")
```

Generate one to three times; pitch is reliable, rate and loudness are not reliable per-utterance.

**Non-verbal, base model, no training:**

```python
wav = model.generate(text="ខ្ញុំគិតថា... [laughing] ...")
```

Unverified on Khmer — run Phase 0 before depending on it.

**The prosodic adapter:**

```python
model.load_lora("finetune/checkpoints/khmer_style/step_0001980")
wav = model.generate(text="<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>…")
```

```bash
.venv/bin/python finetune/synthesize_styled.py \
    --ckpt finetune/checkpoints/khmer_style/step_0001980 \
    --text "…" --spk f2 --rate fast --out hello.wav
```

`model.set_lora_enabled(False)` restores the base model bit for bit, so the other 29 languages cannot be damaged by an adapter that can be switched off.

---

## 11.8 What is standard here, and what is not

**Standard.** Text-side style descriptions as conditioning (the PromptTTS / InstructTTS / Parler-TTS line). Manufacturing labels by measurement rather than annotation, including dropping the extremes when computing bin edges and judging pitch relative to comparable speakers (`dataspeech`). LoRA with fp32 adapters over frozen bf16 weights. Condition dropout for classifier-free guidance, already in VoxCPM2 as `training_cfg_rate`.

**A known problem in an unexpected place.** Finding 2 is the information-preference / posterior-collapse failure of conditional autoregressive models — [Chen et al. (2017)](https://arxiv.org/abs/1611.02731), [Bowman et al. (2016)](https://arxiv.org/abs/1511.06349) — combined with the teacher-forcing mismatch of [Bengio et al. (2015)](https://arxiv.org/abs/1506.03099). It is not documented for this class of TTS system.

**Local to this repo.** `enable_proj: true` is an architectural fact about VoxCPM2, not a general principle; docs/09's advice to freeze those layers is correct for speaker cloning. The onset-weighted loss is mine — the literature's fix is to corrupt the shortcut (word dropout, prefix corruption) rather than reweight around it, and prefix corruption is the comparison that would have to be run if this were written up as research.

---

## References

### Prompt- and description-based control (layer 1, and layers 3–4 by extension)

- Guo et al., [*PromptTTS: Controllable Text-to-Speech with Text Descriptions*](https://arxiv.org/abs/2211.12171) (ICASSP 2023)
- Leng et al., [*PromptTTS 2: Describing and Generating Voices with Text Prompt*](https://arxiv.org/abs/2309.02285) (ICLR 2024) — diffusion variation network for the one-to-many problem
- Yang et al., [*InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt*](https://arxiv.org/abs/2301.13662)
- Lyth & King, [*Natural language guidance of high-fidelity text-to-speech with synthetic annotations*](https://arxiv.org/abs/2402.01912) (2024) — Parler-TTS, and [`dataspeech`](https://github.com/huggingface/dataspeech), the measure-and-bin pipeline
- Ji et al., [*TextrolSpeech: A Text Style Control Speech Corpus*](https://arxiv.org/abs/2308.14430) (ICASSP 2024)

### Non-verbal vocalization (layer 2)

- Kanda et al., [*Making Flow-Matching-Based Zero-Shot Text-to-Speech Laugh as You Like*](https://arxiv.org/abs/2402.07383) (ELaTE, 2024) — the closest method reference; same CFM decoder family as VoxCPM2
- Borisov et al., [*NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations*](https://arxiv.org/abs/2507.13155) (SSW 2025) — the labelling pipeline, and [the dataset](https://huggingface.co/datasets/deepvk/NonverbalTTS)
- Gillick et al., [*Robust Laughter Detection in Noisy Environments*](https://www.isca-archive.org/interspeech_2021/gillick21_interspeech.html) (Interspeech 2021) — [code](https://github.com/jrgillick/laughter-detection)
- Gong et al., [*VocalSound: A Dataset for Improving Human Vocal Sounds Recognition*](https://arxiv.org/abs/2205.03433) (ICASSP 2022) — 21k clips, six non-verbal classes

### Affective and voice quality (layers 3–4)

- Kanda et al., [*Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching-Based Zero-Shot Text-to-Speech*](https://arxiv.org/abs/2407.12229) (2024)
- Zhou et al., *Emotional voice conversion: Theory, databases and ESD* (Speech Communication, 2022)
- [*Voice Conversion for Whispered Speech Synthesis*](https://arxiv.org/abs/1912.05289) — wTIMIT, and the method behind Alexa's Whisper Mode

### Emphasis (layer 5)

- [*Controllable Emphasis with zero data for text-to-speech*](https://arxiv.org/abs/2307.07062) (2023) — lengthen the target word's predicted duration; +40% correct identification of the emphasised word

### Conditioning failure and training method

- Chen et al., [*Variational Lossy Autoencoder*](https://arxiv.org/abs/1611.02731) (ICLR 2017) — why an expressive autoregressive decoder ignores its conditioning
- Bowman et al., [*Generating Sentences from a Continuous Space*](https://arxiv.org/abs/1511.06349) (2016) — the same collapse in text VAEs; word dropout as the cure
- Bengio et al., [*Scheduled Sampling for Sequence Prediction with RNNs*](https://arxiv.org/abs/1506.03099) (2015) — teacher forcing and the train/inference mismatch
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021)
- Ho & Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) (2022)

### VoxCPM2 sources

- `voxcpm/training/packers.py` — the zero text loss-mask
- `voxcpm/model/voxcpm2.py` — `from_local`, and the fp32-in-training line behind the VRAM gap
- [Fine-Tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) · [FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html) · [cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html) — the non-verbal tag inventory · [model card](https://huggingface.co/openbmb/VoxCPM2) — parenthetical control, and the stated variability between runs
- Companion documents: [09 — VoxCPM2 architecture and training](09-voxcpm2-architecture-and-training.md), [10 — Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md)
- Runbook and code: [`finetune/README.md`](../finetune/README.md)
