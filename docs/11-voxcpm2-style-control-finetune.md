# 11 — Style Control for VoxCPM2: What the Model Already Does, and What a Fine-Tune Adds

[Document 9](09-voxcpm2-architecture-and-training.md) explains what fine-tuning VoxCPM2 would involve. This document is the record of doing it for expressive control, and of a course correction partway through that is more useful than the fine-tune itself.

The gap looked easy to state. [Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md) accepts inline control tokens — `<|emotion:sadness|>`, `<|speed:fast|>` — so a caller can ask for a delivery rather than accept whatever the model produces. VoxCPM2 appeared to have no equivalent, and this project's recommendation rests on VoxCPM2 (2.47% Khmer CER against Higgs's 8.28%, Apache-2.0). So: add the control rather than trade the model.

**The correction: VoxCPM2 already has a prosodic control channel, and on Khmer it mostly works.** A natural-language description in parentheses at the start of the text — `(speaking quickly)`, `(a high-pitched voice)` — steers the delivery. This is documented by OpenBMB for the model in general, was undocumented for Khmer, and had never been measured by anyone. §11.3 measures it. Pitch moves **106 Hz** across the prompt range, nearly four times what this project's hand-labelled corpus can even express, at rho +0.76.

That finding arrived after a LoRA adapter had been built and one 6.5-hour training run had failed in an instructive way. The work was stopped at that point rather than carried to completion, and the honest accounting is:

- **§11.3 — what the built-in prompt already does**, measured against the same instrument used for everything else. Three of four prosodic axes respond. This is the finding that should have come first.
- **§11.4 — what it does not do**: pitch *variation* fails outright and in the wrong direction, energy is dominated by run-to-run noise, and nothing selects a specific speaker reproducibly. That residue is what a fine-tune would be for, and it is much smaller than the one this document set out to build.
- **§11.7–11.7 — why the first fine-tune ignored its own control tag.** This survives the course correction intact, because it is not about prosody. It is about why *any* global attribute supplied through the text field earns no gradient under teacher forcing, and it applies to layers 3 and 4 (emotion, voice quality) exactly as it applied here. It is the transferable result.

A reader who wants only the practical answer needs §11.0 and §11.3. A reader who intends to fine-tune this model for any text-derived conditioning needs §11.7.

---

## 11.0 The control layers, and which one this is

"Expressive control" is not one capability. It is several, they behave differently, and treating them as one is how a roadmap goes wrong. The distinction that matters is between **attributes that hold over a whole utterance** and **events that happen at a point in it**. A speaker's pitch register is true of every frame; a laugh occupies 400 ms and nothing else. §11.7 shows this is not cosmetic — it decides how expensive a control signal is to teach.

| layer | controls | scope | where it comes from | status |
|---|---|---|---|---|
| **1. Prosodic** | pitch register, rate, level | **global** | **already in the base model**, via the parenthetical prompt | **works on Khmer — §11.1** |
| **1b. Prosodic residue** | pitch *variation*; naming a specific speaker; per-utterance reproducibility | global | would need a fine-tune | gaps confirmed, §11.2; work stopped |
| **2. Non-verbal vocalization** | laughter, sighs, hesitation, breath | **local** | ships with VoxCPM2, no training | untested on Khmer — §11.9 |
| **3. Affective** | emotion — sadness, joy, anger | global | partly in the base model already (`(cheerful tone)`); unmeasured on Khmer | **measure before building** |
| **4. Voice quality** | whisper, breathy, creaky | global | same — plausibly already there | **measure before building** |
| **5. Discourse** | emphasis, contrastive focus, question contour | local | needs a span-marking syntax, not a header tag | not designed |

**The rule this document paid to learn: measure the base model before training anything.** Layers 3 and 4 are written above as "measure first" rather than "blocked on corpus", which is how an earlier draft had them. The instrument is `finetune/verify_parenthetical.py` and it costs about twenty minutes per layer.

**Why the scope column still matters.** §11.7 measures that a global attribute supplied through the text field earns almost no gradient: swapping the speaker tag costs 0.03% of training loss, while corrupting the transcript costs 7.4%. Teacher forcing is why — the model can read pitch off the ground-truth acoustic prefix, so the tag tells it nothing it does not already have. A local event has no such competition: nothing in the prefix predicts a laugh is coming.

**But note what §11.1 proves about that argument.** The parenthetical prompt is *also* a global attribute in the text field, and it works — so the teacher-forcing shortcut does not make global conditioning impossible. It makes it **data-expensive**. A signal worth 0.03% of the loss is still learnable given enough data and a full training run; OpenBMB paid that cost at pre-training scale, and 11.7 hours of Khmer through a LoRA could not. That is the sharper form of the lesson, and it generalises: *the cost of teaching a global attribute is high enough that re-deriving one the base model already has is close to the worst use of a small corpus.*

---

## 11.1 The built-in parenthetical control, measured on Khmer

VoxCPM2 accepts a natural-language description in parentheses before the text:

```python
wav = model.generate(text="(speaking quickly)ថ្ងៃនេះអាកាសធាតុល្អណាស់។")
```

OpenBMB document this for voice design (`(A young woman, gentle and sweet voice)`) and for style-guided cloning (`(slightly faster, cheerful tone)`), listing "gender, age, tone, emotion, pace" as what it steers. Nothing in their documentation covers Khmer, and this project had already been burned once by assuming a control surface transfers to an undocumented language — docs/10 carries exactly that caveat for Higgs.

So it was measured, with the same instrument used to test the fine-tuned adapter: the same 8 sentences from the frozen eval set, the same four measurements, the same Spearman correlation against a commanded level, and the same corpus ceilings for scale. One addition — **each cell is generated three times at different seeds**, because OpenBMB's model card states results "may vary between runs; generating 1–3 times is recommended". That warning turns out to matter more than the effect sizes.

| axis | low | mid | high | low→high | corpus ceiling | rho | p | run-to-run sd | effect ÷ noise |
|---|---|---|---|---|---|---|---|---|---|
| `pitch` | 147.80 | 202.81 | 254.16 | **+106.37 Hz** | +28.27 Hz | **+0.759** | 0.0002 | 28.78 | **3.7** |
| `energy` | −20.38 | −16.30 | −15.89 | **+4.50 dB** | +5.81 dB | +0.605 | 0.0022 | 5.69 | 0.8 |
| `rate` | 14.67 | 15.84 | 16.96 | **+2.29 ch/s** | +6.57 ch/s | +0.590 | 0.0035 | 1.58 | 1.4 |
| `var` | 4.55 | 4.06 | 3.92 | **−0.63 st** | +1.71 st | −0.273 | 0.2039 | 0.68 | — |

Prompts used: `(speaking slowly / at a normal pace / quickly)`, `(a low- / normal- / high-pitched voice)`, `(a flat, monotone / a normal / a lively, expressive delivery)`, `(speaking softly, quietly / at a normal volume / loudly)`.

**Pitch is the headline.** A 106 Hz range at rho +0.76, against a corpus ceiling of 28 Hz — the built-in prompt spans nearly four times what the hand-labelled corpus can express, because the labels are bounded by the 20 speakers who happened to be recorded while the prompt is not. On this axis a fine-tune is not merely unnecessary, it is *worse than the thing it would replace.*

**Rate and energy respond, but read the last column.** `run-to-run sd` is the standard deviation across repeated generations of the *same* prompt on the *same* sentence. For energy it is 5.69 dB against a 4.50 dB effect: re-rolling the seed moves the output further than changing the command does. For rate the effect is 1.4× the noise. Both correlations are real and statistically solid across many sentences; neither gives a caller a delivery they can rely on for one utterance. This is the distinction between *a measurable effect* and *control*, and averaging over sentences hides it — which is why the repeats were built in.

**Pitch variation fails.** rho −0.273 at p = 0.20: not significant, and pointing the wrong way. Asking for "a lively, expressive delivery" produced *less* pitch movement than asking for a flat one. This is the axis closest to what "expressive TTS" ordinarily means, and it is the one axis the built-in mechanism does not deliver on Khmer.

### A caveat on this measurement

The prompt wordings are mine, not OpenBMB's. Only `(slightly faster, cheerful tone)` and `(A young woman, gentle and sweet voice)` appear in their documentation, so the phrasings above are a reasonable extrapolation rather than a sanctioned API. A negative result on one axis could in principle be a wording problem rather than a model limitation — `var` is the axis where this doubt should be taken most seriously, since "lively" and "expressive" may simply not be the vocabulary the model was trained against. The positive results do not carry this doubt in the same way: they establish that the channel works, whatever better wording might add.

---

## 11.2 What is left for a fine-tune, and why the work was stopped

Given §11.3, the case for the adapter this document describes shrinks to a residue:

| | built-in prompt | fine-tuned tag |
|---|---|---|
| pitch | **+106 Hz, robust** | ceiling +28 Hz — strictly worse |
| rate | +2.29 ch/s, 1.4× noise | comparable at best |
| energy | effect below the noise | *possible* improvement, unproven |
| pitch variation | **fails, wrong direction** | *possible* improvement, unproven |
| named speaker, no reference clip | not available | **only the tag does this** |
| reproducible for one utterance | no — see the sd column | yes, deterministic given a seed |
| enumerable / sweepable / verifiable | no | yes |

Two genuine gaps — pitch variation, and selecting one of twenty specific voices by name — plus reproducibility. That is a real but much narrower product than "add expressive control to VoxCPM2", and it does not justify the remaining GPU time on the terms the project originally assumed.

**Training was therefore stopped at step 1,980 of 4,000.** The checkpoint is kept and the run resumes from it if the residue above later proves worth closing. What follows in §11.5 onward is the record of the work up to that point: it stands on its own for the corpus method, the 12 GB result, and — most importantly — the conditioning failure in §11.7, which is not about prosody and does not go away.

---

## 11.3 The fine-tune's mechanism: nearly free, and not sufficient

Prefix a control tag to the text field:

```
<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។
```

That is the entire architectural change. There is none.

The reason this works rather than merely being convenient is in the training packer. `voxcpm/training/packers.py`, `process_tts_data()` builds each training sequence as `[text tokens] [audio start] [audio patches] [audio end]`, and sets:

```python
loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])
```

**Zero across every text position.** The model is never asked to reproduce the text, only to use it in predicting the audio latents that follow. Anything placed in the text field is therefore pure conditioning, and a tag costs exactly what it costs in sequence length — 25 tokens against roughly 140 for a typical Khmer sentence — and nothing else. This is the same trick Parler-TTS uses, minus Parler's separate description encoder, which VoxCPM2 does not need because its tokenizer is byte-level over raw text.

**A prior worry, checked and dismissed.** Would the model read the tag aloud? Before training anything, the base model was asked to synthesize a Khmer sentence with and without a tag, and both clips were transcribed with the project's Khmer CTC ASR:

| input | transcript |
|---|---|
| `សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។` | សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា |
| `<\|spk:f2\|rate:fast\|…\|>` + same | សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា |

Identical, and both 2.08 s. The base model silently ignores the tag. That settles the risk, and it also hands the verification pass a free gift: the base model is a **clean control condition**, because whatever the adapter does with the tag, the base model demonstrably does nothing with it.

**What this section does not establish.** That the tag *can* condition the model is not the same as the model *learning* to use it. The loss-mask argument shows the channel is free and lossless; it says nothing about whether gradient descent has any reason to send anything down it. It does not, by default — see §11.7. Two further changes are needed, and both are in the run that produced the numbers here: `enable_proj: true`, and a loss weighted toward the audio onset.

---

## 11.4 Where the prosodic labels come from

There is no expressive Khmer speech corpus. There is no Khmer emotion corpus. Trying to copy Higgs's 21-emotion catalogue would mean inventing labels for data that does not carry them, and the result would be unfalsifiable.

So the axes are defined as **things that can be measured on a waveform**. This is the Parler-TTS recipe ([Lyth & King, 2024](https://arxiv.org/abs/2402.01912)), which exists precisely because "reliance on human-labeled descriptions prevents scaling": they annotate speaking rate, pitch, SNR and reverberation automatically over 45k hours and render the results into a text description the model conditions on. The accompanying [`dataspeech`](https://github.com/huggingface/dataspeech) tooling bins each continuous measurement into categorical descriptors — seven for speaking rate, from `very slowly` to `very fast`. The tag here is the same idea in a more compact syntax:

| axis | levels | measured as |
|---|---|---|
| `spk` | `f1`…`f9`, `m1`…`m11` | speaker identity, from the corpus |
| `rate` | `slow` `mid` `fast` | Khmer characters per second |
| `pitch` | `low` `mid` `high` | median F0 |
| `var` | `flat` `mid` `lively` | F0 standard deviation, in semitones |
| `energy` | `soft` `mid` `loud` | RMS level |

This is a narrower promise than Higgs makes, and a deliberate one. It buys something Higgs's Khmer control does not have: **the claim is checkable.** "Did asking for `rate:fast` produce faster speech?" is arithmetic. Document 10 notes that Higgs's own control tokens are untested on Khmer — they were trained on the documented languages, and Khmer is not one of them. Here, every axis is verified on Khmer by construction, against a control condition, in §11.7.

It also sidesteps the trap this project has already documented at length. CLAUDE.md, "Metrics", establishes that no learned metric ranks Khmer TTS correctly — UTMOS is *inverted*, rho = +0.55 against CER. An axis defined as a measurable acoustic quantity needs no learned metric to score it.

### Within-speaker or global?

`rate` is ranked globally: characters per second means the same thing from any voice.

`pitch`, `var` and `energy` are ranked **within speaker**. Ranking them globally measures better — `var` widens from about 1.7 to 3.0 semitones — but it widens by letting the axis select a *speaker* rather than a delivery. `var:lively` would come to mean "use one of the animated voices", and the verification sweep, which runs with `spk:any`, would then report a large effect for entirely the wrong reason. Within-speaker ranking keeps each axis about *how* something is said rather than *who* says it, and pays for that with a narrower range. For a control channel that is the right way round.

### Levels come from the tails, not from tertiles

Both of the choices below have precedent in `dataspeech`, which derives its bins from histograms "from which the extreme values have been eliminated" and compares a speaker's pitch against *others of the same gender* rather than against the whole corpus. The reasoning that follows was arrived at independently and then found to agree, which is mild evidence it is the right call rather than a local quirk.

The first build used balanced tertiles and produced a control range too narrow to ship. Measured on the same 6000 clips:

| axis | tertiles | 15/85 tails | gain |
|---|---|---|---|
| `rate` | +4.49 chars/s | **+6.57** | +46% |
| `pitch` | +20.3 Hz | **+28.3 Hz** | +40% |
| `var` | +1.16 st | **+1.71 st** | +47% |
| `energy` | +3.78 dB | **+5.81 dB** | +54% |

Taking the outer classes from the bottom and top 15% leaves 880 exemplars of each extreme instead of 2000, and puts the remaining 70% in `mid`. For teaching a *direction*, unambiguous examples beat numerous borderline ones, and the gain is roughly 1.5× on every axis. This was caught by measuring the label separation before training rather than after — the spread the labels themselves achieve is the ceiling on anything the model can learn, and it costs seconds to compute.

### Partial tags: per-slot dropout

Each slot is independently replaced by `any` with probability 0.15 during training. Without this the model only ever sees fully-specified tags and generalises badly to partial ones. With it, `<|spk:any|rate:any|pitch:any|var:lively|energy:any|>` genuinely means "expressive, everything else free" rather than "expressive plus four accidental defaults" — which is also what makes the single-axis verification sweep meaningful.

---

## 11.5 The corpus

| | |
|---|---|
| Source | `/run/media/pc/disk1/streaming_asr/data/dataset` — 1067 h, 20 speakers, 16 kHz |
| Selected | 5,820 clips, **11.7 hours** |
| Duration | 3.0–10.0 s, median 7.4 s |
| Speakers | 20 (9 female, 11 male), per-speaker cap to keep the tail voices represented |
| Silence | trimmed to 80 ms either side |
| Level | normalised **per speaker**, not per clip |

Two of those choices are load-bearing.

**Silence trimming is not cosmetic.** The official VoxCPM FAQ names trailing silence over 0.5 s as the most common cause of runaway generation — clips that end in a long silent tail teach the model that utterances do not end. This is the exact failure that made `fish-s2` unusable in the evaluation.

**Per-speaker level normalisation, not per-clip.** Normalising each clip individually would erase precisely the level variation the `energy` axis is supposed to control. Normalising per speaker removes the between-speaker gain differences while leaving within-speaker dynamics intact.

The corpus is real human speech throughout. Document 9 flags `Panhapich/khmer-english-codeswitch-tts` as tagged `synthetic` / `tts-generated` and warns that training on it distils another model's artefacts into yours; none of it is used here.

**The honest limitation:** this is read speech from a studio corpus, all one register. The axes move within the range of read Khmer. They do not reach shouting, whispering, or grief, and no amount of fine-tuning on this data will make them — that would need expressive recordings that, for Khmer, do not exist. What this delivers is the prosody half of Higgs's control surface, measured; not its emotion catalogue.

---

## 11.6 Fitting a 20 GB job into 12 GB

Document 9 §9.5 records OpenBMB's figure for VoxCPM2 LoRA: **~20 GB**, with a hardware reality check noting that this project's RTX 3060 has 12 GB and is "below the LoRA minimum", and recommending a rented 24 GB card.

That recommendation turns out to be unnecessary, and the reason is a single line in `voxcpm/model/voxcpm2.py`:

```python
if not training:
    model = model.to(get_dtype(model.config.dtype))   # bfloat16
else:   # training mode -- weights stay float32
```

In training mode the model is left in float32. That is 2.29 B frozen parameters at **8.6 GiB**, before a single activation is allocated — while the forward pass runs under `autocast(bfloat16)` anyway. The fp32 master copy buys nothing for parameters that are frozen and never receive an update.

`finetune/train.py` wraps upstream's trainer, leaving it byte-for-byte unmodified, and applies three patches:

1. **Frozen weights → bf16, LoRA parameters kept fp32.** 2.29 B frozen at 4.27 GiB; 36.2 M trainable stay in fp32 because AdamW at lr 1e-4 produces updates about 1e-2 the size of the weights, and bf16's ~3 significant decimal digits would quantise a meaningful share of them away.
2. **Gradient checkpointing** on all 36 transformer layers (28 backbone + 8 residual).
3. AudioVAE kept fp32 on GPU, encode under `no_grad`.

| | measured |
|---|---|
| Model + AudioVAE resident | 5.01 GiB |
| Peak, training only | 10.3–10.9 GiB |
| Peak, incl. validation audio generation | 11.1 GiB |
| Card total | 12.3 GiB |

It fits, with `batch_size: 1` and `grad_accum_steps: 8`.

One diagnostic worth recording: dropping `max_batch_tokens` from 1024 to 512 moved the peak by only 0.6 GiB. **Peak memory here is essentially independent of sequence length** — it is resident weights plus the AudioVAE encode, not activations. Anyone tuning this for a smaller card should attack the weights, not the batch.

### Configuration

`r: 64`, not 32 — Document 9 §9.5: r=32 clones a speaker, r=64 is for style and language work, and five control axes across 20 voices is style work. Both `enable_lm` and `enable_dit` on: the backbone carries the text→prosody mapping, the DiT carries acoustic detail, and this needs both. lr 1e-4, cosine with 200 warmup steps, 4000 iterations at effective batch 8 — about 5.5 epochs over the corpus, ~6.8 h on the 3060.

---

## 11.7 It did not work, and why

The first full run trained without incident. 4000 steps, 6.5 hours, validation `loss/diff` 0.8819 → 0.8076, ten checkpoints. The adapter was real: median `‖ΔW‖/‖W‖` of **3.03e-02** across 192 adapted layers, and all 192 `lora_B` tensors non-zero, which matters because they initialise to exactly zero. It audibly changed the model — the base model's F0 wanders between 105.9 and 231.5 Hz from sentence to sentence, the adapter clamps output to 201.3–261.6 Hz, and speaking rate moved 17.61 → 14.07 chars/s.

It ignored the control tag completely.

| axis | measured | commanded low → high | rho | p | what the labels themselves achieve |
|---|---|---|---|---|---|
| `rate` | chars/s | +0.27 | +0.141 | 0.46 | **+6.57** |
| `pitch` | F0 median | −2.81 Hz | +0.019 | 0.93 | **+28.27 Hz** |
| `var` | F0 std | +0.02 st | −0.057 | 0.77 | **+1.71 st** |
| `energy` | RMS | −0.37 dB | −0.038 | 0.85 | **+5.81 dB** |

Null on every axis, with movement indistinguishable from zero against ceilings measured from the training labels before training started. Ranking all nine checkpoints by control response — not by loss, per docs/09 §9.5 — found no trend across training and a best mean rho of +0.026, so this was not a matter of picking the wrong one.

The decisive axis is `spk`. Six speaker tags spanning 105 → 280 Hz of real corpus pitch produced generated medians of 198.5, 209.3, 210.3, 222.9, 208.6 and 196.7 Hz — **26 Hz of spread against 175 Hz of corpus spread, rho = −0.200.** Speaker identity is the one attribute the model cannot infer from the text, so if that axis does not land, nothing is landing. It also explains the voice collapse above: a model that cannot read the tag can only average its twenty speakers.

### Eliminating the easy explanations

Out-of-distribution tag shape (per-slot dropout had made all-`any` tags 0.27% of rows) — ruled out, because the in-distribution sweep above pins the unswept slots to `mid` and is equally null. Inert adapter — ruled out by the weight deltas. Tokenization — the tags survive the model's wrapped tokenizer, differ in ten token positions between extremes and produce zero UNK. Train/inference mismatch — both paths call the same `base_model.text_tokenizer`, and inference passes the text through unsplit. Text normalization — `normalize` defaults to `False` and is never passed. Tag placement, on the theory that Khmer's ~330 UTF-8 byte tokens bury a prefix 350 positions from the audio — a probe moving the tag adjacent to `audio_start` produced **5.8 Hz** of speaker separation. Not it either.

### The measurement that actually settled it

Every result above measures generated audio, and generated audio cannot distinguish *never learned the tag* from *learned it, but sampling washes it out*. So the question went back to the training objective. For each held-out clip: one teacher-forced forward pass with the true speaker tag, one with the other speaker's, with the diffusion timestep and the noise held identical by reseeding from the row index — the CFM loss is stochastic, and unseeded this measures nothing but noise.

| condition | mean `loss/diff` | delta |
|---|---|---|
| correct tag | 0.86908 | — |
| swapped tag | 0.86930 | **+0.03%** — 22/40 rows, sign-test p = 0.32 |
| scrambled transcript *(positive control)* | 0.93348 | **+7.4%** — 35/40 rows |

The positive control is what makes this readable. Corrupting the transcript, which the model certainly uses, moves the loss by 7.4%; corrupting the tag moves it by 0.03%, a **290× difference** indistinguishable from chance. The model reads the text and does not read the tag *in its own objective*. No inference-side change could have rescued that.

**The cause is teacher forcing**, and it is a known phenomenon wearing unfamiliar clothes. Every audio patch is predicted with the preceding *ground-truth* patches visible, and a speaker's pitch is trivially readable off those. The tag is therefore redundant with the acoustic prefix everywhere except the very beginning of the clip, and redundant information receives no gradient.

This is the **information-preference problem** of [Chen et al., *Variational Lossy Autoencoder* (ICLR 2017)](https://arxiv.org/abs/1611.02731): a sufficiently expressive autoregressive decoder ignores a conditioning signal entirely, because it is cheaper in coding terms to predict the next step from its own history than to route information through the conditioning path. Their remedies are all forms of *taking the shortcut away* — weakening the decoder, dropout, limiting its receptive field. The same failure is [Bowman et al. (2016)](https://arxiv.org/abs/1511.06349)'s KL-vanishing in text VAEs, fixed there by word dropout: replace some fraction of the decoder's history with `UNK` so it cannot recover the next word without consulting the latent. The connection to teacher forcing generally is [Bengio et al. (2015)](https://arxiv.org/abs/1506.03099) on scheduled sampling — the mismatch between a training signal that is available and an inference-time one that is not.

None of this literature is about TTS, and I did not reach the diagnosis through it; the measurement came first and the family was recognised afterwards. But it means the finding is not exotic. It is the standard failure of conditional autoregressive generation, arriving in a place the VoxCPM2 documentation does not warn about. The prediction that follows is sharp: restrict the loss to the first K patches, where no prefix exists yet, and the tag must start to matter.

| loss restricted to first K patches | 1 | 2 | 4 | 8 | all |
|---|---|---|---|---|---|
| penalty for the wrong tag | **+0.00318** | +0.00214 | +0.00143 | +0.00082 | +0.00022 |

Monotone, and **14× larger on the first patch than over the whole clip**. The tag matters exactly where the prefix cannot answer for it, and is drowned everywhere else.

This is a property of the training objective, not of VoxCPM2 and not of the tag format. Any global attribute — speaker, rate, register, and by extension emotion — supplied through the text field competes against a teacher-forced acoustic prefix that already encodes it, and loses.

### The fix

Two changes, each isolated by its own 400-step probe on the easiest discrimination the corpus offers: two speakers an octave apart, one binary tag. **The pass bar — more than 30 Hz of separation, in the right direction — was fixed before the first probe ran**, against 175 Hz of real separation. Each arm changes exactly one thing from the one above it.

| arm | change | separation | |
|---|---|---|---|
| `end` | tag adjacent to `audio_start` | +5.8 Hz | FAIL |
| `proj` | `end` + `enable_proj: true` | +19.5 Hz | FAIL |
| `onset` | `proj` + diffusion loss weighted 8× at the onset | **+39.9 Hz** | **PASS** |

**`enable_proj: false` was wrong here.** `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj` and `fusion_concat_proj` are the linear bottleneck through which everything the LM knows reaches the diffusion transformer that makes the acoustics. docs/09 recommends freezing them, and for speaker *cloning* that is right — the voice arrives through the reference-audio encoder and never crosses that bridge. For conditioning that arrives only as text there is no other route across. Worth +13.7 Hz.

**Weighting the loss toward the onset** puts the gradient where the tag is the only available cue. This is *not* the textbook fix, and it is worth being explicit about that. The remedies in the literature above all attack the shortcut itself — corrupt the decoder's history so it cannot lean on it (Bowman's word dropout), or restrict what it can see (VLAE). The equivalent here would be randomly noising or masking the ground-truth audio prefix during training, and VoxCPM2 already contains machinery of exactly that shape: `unified_cfm.compute_loss` noises the DiT's previous-patch conditioning with a probability that ramps over training (`noise_cond_prob_range`), and drops the LM conditioning entirely at rate `training_cfg_rate` — the standard condition-dropout that enables classifier-free guidance at inference. That the designers included both suggests they knew about the problem.

The reason for not doing it that way: the shortcut is the *language model's* view of the ground-truth prefix, not only the DiT's, and removing that properly means surgery inside the library rather than a wrapper around it. Onset weighting attacks the same target from the other side — instead of making the prefix less useful everywhere, it makes the one position where the prefix does not exist count for more. It required no library change at all. Whether prefix corruption would work better is untested and is the more principled experiment; this one had the advantage of being runnable in an afternoon.

Concretely: position *i* of the audio span gets weight `1 + 7·exp(-i/4)`, 8× on the first patch decaying to ~1 by the twentieth. Nothing downstream needs modifying — `unified_cfm.compute_loss` computes `(mask·losses).sum() / sum(mask)` and `adaptive_loss_weighting` at `p=0` returns the mask unchanged, so a non-binary mask is a self-renormalising weighted mean. Upstream casts the mask to `int32`, which would truncate the weights, so `train.py` PATCH 4 rebuilds it as float. Worth a further +20.4 Hz, and the change that clears the bar. And if the onset is set correctly, autoregression carries it: the rest of the utterance is generated conditioned on that first patch.

The diagnostic that found the failure confirms the repair at the same level:

| checkpoint | correct tag | swapped tag | delta | sign-test p |
|---|---|---|---|---|
| `proj` | 0.86908 | 0.86930 | +0.00022 | 0.32 |
| `onset` | 0.87204 | 0.87289 | **+0.00086** | **0.008** |

Four times the penalty for the wrong tag, significant where it was not. Full measurement record in [`finetune/results/diagnosis.md`](../finetune/results/diagnosis.md).

### The two runs

| | run 1 | run 2 |
|---|---|---|
| corpus | 5,646 train / 174 val, 20 speakers, 11.73 h | same |
| steps | 4000 (≈5.5 epochs), 6.5 h | 4000, ≈7 h |
| LoRA | r=64, α=64, `enable_lm`+`enable_dit` | same, **plus `enable_proj`** |
| loss | uniform over audio positions | **8× at onset, τ=4 patches** |
| peak VRAM | 10.3–11.7 GiB | 11.3 GiB |
| control response | **none** — rho ≈ 0 on all axes | *in progress* |

Run 1's checkpoints and sweep are kept under `finetune/checkpoints/khmer_style_run1_failed/` and `finetune/results/select_run1_failed/` rather than deleted: they are the control condition for run 2, and the evidence for everything in this section.

**Run 2 is still training as of this commit.** Its sweep will land in `finetune/results/control_sweep.md`, and until it does, nothing in this document claims the five-axis control works — only that the probe shows the model can now learn to read a tag at all, on the easiest discrimination the corpus offers.

---

## 11.8 Using the fine-tuned tag

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="finetune/checkpoints/khmer_style/latest",
)
tag = "<|spk:f2|rate:fast|pitch:any|var:lively|energy:any|>"
wav = model.generate(text=tag + "សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។")
```

or from the CLI wrapper, which builds the tag for you and fills unspecified slots with `any`:

```bash
python finetune/synthesize_styled.py \
    --lora finetune/checkpoints/khmer_style/latest \
    --text "សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។" \
    --style var=lively,rate=fast,spk=f2 \
    --out hello.wav
```

The adapter hot-swaps: `model.set_lora_enabled(False)` restores the base model bit for bit, and the base model's other 29 languages cannot be damaged by a Khmer adapter that can be switched off. A full fine-tune would have no such escape hatch.

---

## 11.9 Layer 2 — non-verbal vocalization

This layer needs no fine-tuning. VoxCPM2 ships with it: inline square-bracket tags placed in the text at the point where the vocalization should occur, part of the base model's training and available today on the stock checkpoint.

| category | tags | effect |
|---|---|---|
| laughter and breath | `[laughing]`, `[sigh]` | audible laugh or exhalation |
| hesitation | `[Uhm]`, `[Shh]` | filled pause; a shushing sound |
| question particles | `[Question-ah]`, `[Question-ei]`, `[Question-en]`, `[Question-oh]` | interrogative interjections |
| emotional interjections | `[Surprise-wa]`, `[Surprise-yo]`, `[Dissatisfaction-hnn]` | surprise; dissatisfaction |

```python
# stock VoxCPM2 -- no adapter, no fine-tuning
wav = model.generate(text="ខ្ញុំគិតថា... [laughing] ...")

# the mechanisms compose: parenthetical voice design, our prosodic header
# tag, and an inline non-verbal tag in one call
text = ("(a young woman, warm voice)"
        "<|spk:any|rate:slow|pitch:any|var:lively|energy:any|>"
        "ខ្ញុំ... [sigh] ...")
```

OpenBMB's guidance is unusually specific and worth repeating: use them sparingly, prefer the lowercase `[laughing]` over variants like `[Laughter]`, and do not stack several into one sentence. The model card lists instability on "very long or highly expressive inputs" among its limitations, which is the same caution from the other side.

### A separate mechanism, easily confused with it

VoxCPM2 also accepts a natural-language description in parentheses at the start of the text — `(A young woman, gentle and sweet voice)` for voice design from nothing, or `(slightly faster, cheerful tone)` over a reference clip for style-guided cloning.

This overlaps layer 1 and does not replace it. It is free text: not enumerable, not reproducible across runs — the model card recommends generating one to three times to get what you want — and it offers no way to sweep an axis or check that a request was honoured. The tag in §11.3 trades expressiveness for exactly those properties: 32 slot values, deterministic given a seed, measurable. The two compose, one being a parenthetical prefix and the other a bracketed header.

### What is not known

All of the above is documented for VoxCPM2 in general and **none of it for Khmer**. This project has been bitten by that assumption once already — docs/10 records the same caveat for Higgs TTS 3's control tokens, untested on Khmer because Khmer is not among its documented languages.

| open question | why it is genuinely open |
|---|---|
| Do the tags fire in Khmer text at all? | They are English strings inside a Khmer sentence. Khmer tokenises to UTF-8 byte tokens; the tag does not. Whether the model recognises the pattern in that context is empirical. |
| Are they read aloud instead? | The failure to check for is the model pronouncing "laughing" rather than laughing — the same check §11.3 ran on the prosodic tag, which passed. |
| Does the prosodic adapter damage them? | The adapter was trained on read speech containing no laughter, and LoRA can erode capabilities absent from the fine-tuning distribution. Testable against the same clip with the adapter disabled. |

**The cheapest useful experiment.** Synthesize a fixed set of Khmer sentences with and without each tag, three ways: base model, adapter enabled, adapter disabled. Then measure rather than listen — duration delta, since a real laugh lengthens the clip, and the project's Khmer CTC ASR on the transcript, which shows immediately whether "laughing" is being spoken as a word. This reuses `verify_control.py`'s measurement layer and `score_cer.py` unchanged, and needs no corpus, no labels and no training. That is what makes layer 2 the right next step rather than layers 3 or 4, where the blocker is data that does not exist.

---

## 11.10 What this does and does not settle

**The result that matters most, and it is not the fine-tune.** VoxCPM2's built-in parenthetical prompt already controls Khmer pitch, rate and level (§11.1). Pitch spans 106 Hz at rho +0.76 — nearly four times what this project's hand-labelled corpus can even express. The fine-tune was building something the model largely already had, and nobody checked first. **Measure the base model before training anything** is the cheapest lesson in this document and it cost the most to learn: about twenty minutes of measurement would have saved two training runs.

**Settled about the fine-tune.** VoxCPM2 can be given a control surface without touching its architecture; the packer's zero text loss-mask makes the text field a free conditioning channel. It trains on 12 GB, not the documented 20. And the failure mode that makes the obvious construction quietly not work is identified and measured: a text-side global attribute is redundant with the teacher-forced acoustic prefix, so it earns a gradient roughly 290× smaller than the transcript does. Weighting the loss toward the onset and adapting the LM→DiT projections took a two-speaker probe from +5.8 Hz to +39.9 Hz of separation, over a bar fixed in advance.

**Settled about method, and more durable than any number here.** Two habits did the real work. Generated-audio metrics could not distinguish "never learned it" from "learned it, sampling lost it"; going back to the training objective with a positive control could, in minutes. And measuring the label separation *before* training bounds what a model can possibly learn from those labels. Both are ordinary practice. Neither was applied to the base model's own capabilities until far too late, which is precisely the mistake §11.1 corrects.

**Not settled.** Naturalness, still — for the same reason CLAUDE.md gives at length. No automatic metric ranks Khmer TTS, and whether any of this *sounds* better is a listening-test question. The apparatus exists (`evaluation/listening_test.py`) and has not been run with real listeners.

**Open, and cheap to close.** Whether the parenthetical prompt also covers layers 3 and 4 — emotion, voice quality — on Khmer. An earlier draft of this document called those "blocked on corpus"; §11.1 is the reason that phrasing was wrong. They are blocked on a measurement nobody has run, and only on a corpus if that measurement comes back negative. `finetune/verify_parenthetical.py` extends to them by editing one dictionary.

**Genuinely needs data, if it is wanted at all.** Pitch *variation* — the one prosodic axis the built-in prompt fails (rho −0.27, wrong direction) — and reproducible selection of a named speaker. Note that `laughter` is **not** in this category — as a local event it belongs to layer 2 (§11.9), which ships with the model and needs no corpus at all.

---

## 11.11 What here is standard, and what is not

Worth stating plainly, because a method document that does not separate borrowed
ideas from local inventions is hard to trust and harder to build on.

**Standard, and deliberately so.**

- *Conditioning TTS on text-side style descriptions.* Parler-TTS ([Lyth & King, 2024](https://arxiv.org/abs/2402.01912)) and the prompt-controlled TTS line generally. The compact `<|slot:value|>` syntax is closer to Higgs's inline control tokens (Document 10) than to Parler's prose descriptions, but the mechanism is the same.
- *Manufacturing labels by measurement rather than annotation.* Parler-TTS again, and its [`dataspeech`](https://github.com/huggingface/dataspeech) pipeline: measure continuous attributes, bin them, name the bins. Including two details this document arrived at independently — dropping the extremes when computing bin edges, and judging pitch relative to comparable speakers rather than to the whole corpus.
- *LoRA for adaptation* ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), and the mixed-precision arrangement of fp32 adapters over frozen bf16 weights.
- *Condition dropout for classifier-free guidance* ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)) — already in VoxCPM2 as `training_cfg_rate`.

**A known problem, met in an unexpected place.** The tag being ignored is the information-preference / posterior-collapse failure of conditional autoregressive models: [Chen et al. (2017)](https://arxiv.org/abs/1611.02731), [Bowman et al. (2016)](https://arxiv.org/abs/1511.06349), and the teacher-forcing mismatch of [Bengio et al. (2015)](https://arxiv.org/abs/1506.03099). §11.7 gives the argument. The measurement came first; the literature was recognised afterwards, which is worth admitting because it is why the search took as long as it did. Anyone who saw the connection earlier would have gone straight to the fix.

**Local to this repo, with no citation behind it.**

- *`enable_proj: true`.* An architectural fact about VoxCPM2 — four projection layers are the only route from the LM into the acoustic generator — not a general principle. docs/09's advice to freeze them is correct for the case it was written for.
- *Onset-weighted loss.* Mine. The literature's fix is to corrupt the shortcut, not to reweight around it; §11.7 says why the cheaper version was chosen and what the more principled experiment would be. If this were being written up as research rather than as a build log, prefix corruption is the comparison that would have to be run.

**Method, rather than result.** Two habits did more work than any single idea and neither is novel: measuring the label separation *before* training, because it bounds what the model can possibly learn; and taking a null result back to the training objective with a positive control, because generated-audio metrics cannot distinguish "never learned it" from "learned it and lost it in sampling". The second is ordinary ablation practice. It answered in two minutes a question that a 6.5-hour run had left ambiguous.

---

## Sources

- `voxcpm/training/packers.py` — the zero text loss-mask that makes the whole approach work
- `voxcpm/model/voxcpm2.py` — `from_local`, and the fp32-in-training line behind the VRAM gap
- [VoxCPM Fine-Tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) and [FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html)
- [VoxCPM2 cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html) — the non-verbal tag inventory in §11.9, and the guidance to use them sparingly and prefer lowercase forms
- [VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2) — parenthetical voice design and style-guided cloning, and the stated variability between runs
- Companion documents: [09 — VoxCPM2 architecture and training](09-voxcpm2-architecture-and-training.md), [10 — Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md)

**External:**

- Lyth & King, [*Natural language guidance of high-fidelity text-to-speech with synthetic annotations*](https://arxiv.org/abs/2402.01912) (2024) — Parler-TTS; measured attributes as text conditioning, and [`dataspeech`](https://github.com/huggingface/dataspeech), the binning pipeline
- Chen et al., [*Variational Lossy Autoencoder*](https://arxiv.org/abs/1611.02731) (ICLR 2017) — why an expressive autoregressive decoder ignores its conditioning, and what taking the shortcut away looks like
- Bowman et al., [*Generating Sentences from a Continuous Space*](https://arxiv.org/abs/1511.06349) (2016) — the same collapse in text VAEs; word dropout as the cure
- Bengio et al., [*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*](https://arxiv.org/abs/1506.03099) (2015) — teacher forcing and the train/inference mismatch
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021)
- Ho & Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) (2022) — the condition dropout VoxCPM2 implements as `training_cfg_rate`
- Runbook and code: [`finetune/README.md`](../finetune/README.md)
