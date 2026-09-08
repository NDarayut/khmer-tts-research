# 11 — Giving VoxCPM2 Style Control: A Khmer Fine-Tune

[Document 9](09-voxcpm2-architecture-and-training.md) explains what fine-tuning VoxCPM2 would involve. This document is the record of actually doing it, and of the one capability the model comparison found VoxCPM2 lacking.

The gap is easy to state. [Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md) accepts inline control tokens — `<|emotion:sadness|>`, `<|speed:fast|>`, `<|pitch_high|>` — so a caller can ask for a delivery rather than accept whatever the model produces. VoxCPM2 has no equivalent. Its text path is a single `text_tokenizer(text)` call: no language tag, no style channel, no speaker channel. You get one voice and one delivery, and the only lever is a reference clip for zero-shot cloning.

But VoxCPM2 is the model this project recommends, on the strength of a 2.47% Khmer CER against Higgs's 8.28% and an Apache-2.0 licence you can ship (see CLAUDE.md, "The verdict"). Losing that to gain expressive control would be a bad trade. The question this document answers is whether the control can be added instead.

**It can, but not the way it first appears.** The result is a LoRA adapter, trained on a single 12 GB consumer GPU, that gives VoxCPM2 five control axes — voice, speaking rate, pitch register, pitch variation, and level — driven by a tag prefixed to the text.

The reason for the qualifier is the most useful thing in this document. The obvious construction — put a tag in the text field, train a LoRA on labelled speech — produces an adapter that trains cleanly, converges, audibly changes the model, and **ignores the tag completely**. It took a 6.5-hour run and four falsifying experiments to establish that, and the cause turned out to be neither the tag format nor the architecture but the training objective itself. §11.5 is that story, because a reader who copies the design in §11.1 without it will reproduce the failure exactly.

---

## 11.1 The mechanism, and why it is nearly free — and not sufficient

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

**What this section does not establish.** That the tag *can* condition the model is not the same as the model *learning* to use it. The loss-mask argument shows the channel is free and lossless; it says nothing about whether gradient descent has any reason to send anything down it. It does not, by default — see §11.5. Two further changes are needed, and both are in the run that produced the numbers here: `enable_proj: true`, and a loss weighted toward the audio onset.

---

## 11.2 Where the labels come from

There is no expressive Khmer speech corpus. There is no Khmer emotion corpus. Trying to copy Higgs's 21-emotion catalogue would mean inventing labels for data that does not carry them, and the result would be unfalsifiable.

So the axes are defined as **things that can be measured on a waveform**. This is the Parler-TTS recipe ([Lyth & King, 2024](https://arxiv.org/abs/2402.01912)), which exists precisely because "reliance on human-labeled descriptions prevents scaling": they annotate speaking rate, pitch, SNR and reverberation automatically over 45k hours and render the results into a text description the model conditions on. The accompanying [`dataspeech`](https://github.com/huggingface/dataspeech) tooling bins each continuous measurement into categorical descriptors — seven for speaking rate, from `very slowly` to `very fast`. The tag here is the same idea in a more compact syntax:

| axis | levels | measured as |
|---|---|---|
| `spk` | `f1`…`f9`, `m1`…`m11` | speaker identity, from the corpus |
| `rate` | `slow` `mid` `fast` | Khmer characters per second |
| `pitch` | `low` `mid` `high` | median F0 |
| `var` | `flat` `mid` `lively` | F0 standard deviation, in semitones |
| `energy` | `soft` `mid` `loud` | RMS level |

This is a narrower promise than Higgs makes, and a deliberate one. It buys something Higgs's Khmer control does not have: **the claim is checkable.** "Did asking for `rate:fast` produce faster speech?" is arithmetic. Document 10 notes that Higgs's own control tokens are untested on Khmer — they were trained on the documented languages, and Khmer is not one of them. Here, every axis is verified on Khmer by construction, against a control condition, in §11.5.

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

## 11.3 The corpus

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

## 11.4 Fitting a 20 GB job into 12 GB

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

## 11.5 It did not work, and why

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

## 11.6 Using it

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

## 11.7 What this does and does not settle

**Settled.** VoxCPM2 can be given a control surface without touching its architecture; the packer's zero text loss-mask makes the text field a free conditioning channel. It trains on 12 GB, not the documented 20. And the failure mode that makes the obvious construction quietly not work is identified, measured and fixed: a text-side control tag is redundant with the teacher-forced acoustic prefix, so it earns no gradient unless the loss is weighted toward the onset and the LM→DiT projections are adapted.

**Settled about method, and more durable than the numbers.** Generated-audio metrics could not tell "never learned it" from "learned it, sampling lost it"; going back to the training objective with a positive control could, in minutes. Any future conditioning work on this model should run that test first — it is far cheaper than a 6.5-hour run that answers nothing.

**Not settled.** Naturalness, still — for the same reason CLAUDE.md gives at length. Nothing here changes the fact that no automatic metric ranks Khmer TTS, and whether the styled output *sounds* better is a listening-test question. The apparatus exists (`evaluation/listening_test.py`) and has not been run with real listeners.

**Out of reach with this data.** Emotion and non-prosodic style — Higgs's `sadness`, `whispering`, `laughter`. Those need expressive Khmer recordings. The mechanism in §11.1 would carry them, but §11.5 is the correction to the obvious next thought: the mechanism alone is not enough, and an emotion tag would face exactly the same competition against the acoustic prefix that the speaker tag lost. The corpus is what is missing; the onset weighting would still be required.

---

## 11.8 What here is standard, and what is not

Worth stating plainly, because a method document that does not separate borrowed
ideas from local inventions is hard to trust and harder to build on.

**Standard, and deliberately so.**

- *Conditioning TTS on text-side style descriptions.* Parler-TTS ([Lyth & King, 2024](https://arxiv.org/abs/2402.01912)) and the prompt-controlled TTS line generally. The compact `<|slot:value|>` syntax is closer to Higgs's inline control tokens (Document 10) than to Parler's prose descriptions, but the mechanism is the same.
- *Manufacturing labels by measurement rather than annotation.* Parler-TTS again, and its [`dataspeech`](https://github.com/huggingface/dataspeech) pipeline: measure continuous attributes, bin them, name the bins. Including two details this document arrived at independently — dropping the extremes when computing bin edges, and judging pitch relative to comparable speakers rather than to the whole corpus.
- *LoRA for adaptation* ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), and the mixed-precision arrangement of fp32 adapters over frozen bf16 weights.
- *Condition dropout for classifier-free guidance* ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)) — already in VoxCPM2 as `training_cfg_rate`.

**A known problem, met in an unexpected place.** The tag being ignored is the information-preference / posterior-collapse failure of conditional autoregressive models: [Chen et al. (2017)](https://arxiv.org/abs/1611.02731), [Bowman et al. (2016)](https://arxiv.org/abs/1511.06349), and the teacher-forcing mismatch of [Bengio et al. (2015)](https://arxiv.org/abs/1506.03099). §11.5 gives the argument. The measurement came first; the literature was recognised afterwards, which is worth admitting because it is why the search took as long as it did. Anyone who saw the connection earlier would have gone straight to the fix.

**Local to this repo, with no citation behind it.**

- *`enable_proj: true`.* An architectural fact about VoxCPM2 — four projection layers are the only route from the LM into the acoustic generator — not a general principle. docs/09's advice to freeze them is correct for the case it was written for.
- *Onset-weighted loss.* Mine. The literature's fix is to corrupt the shortcut, not to reweight around it; §11.5 says why the cheaper version was chosen and what the more principled experiment would be. If this were being written up as research rather than as a build log, prefix corruption is the comparison that would have to be run.

**Method, rather than result.** Two habits did more work than any single idea and neither is novel: measuring the label separation *before* training, because it bounds what the model can possibly learn; and taking a null result back to the training objective with a positive control, because generated-audio metrics cannot distinguish "never learned it" from "learned it and lost it in sampling". The second is ordinary ablation practice. It answered in two minutes a question that a 6.5-hour run had left ambiguous.

---

## Sources

- `voxcpm/training/packers.py` — the zero text loss-mask that makes the whole approach work
- `voxcpm/model/voxcpm2.py` — `from_local`, and the fp32-in-training line behind the VRAM gap
- [VoxCPM Fine-Tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) and [FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html)
- Companion documents: [09 — VoxCPM2 architecture and training](09-voxcpm2-architecture-and-training.md), [10 — Higgs TTS 3](10-higgs-tts-3-architecture-and-training.md)

**External:**

- Lyth & King, [*Natural language guidance of high-fidelity text-to-speech with synthetic annotations*](https://arxiv.org/abs/2402.01912) (2024) — Parler-TTS; measured attributes as text conditioning, and [`dataspeech`](https://github.com/huggingface/dataspeech), the binning pipeline
- Chen et al., [*Variational Lossy Autoencoder*](https://arxiv.org/abs/1611.02731) (ICLR 2017) — why an expressive autoregressive decoder ignores its conditioning, and what taking the shortcut away looks like
- Bowman et al., [*Generating Sentences from a Continuous Space*](https://arxiv.org/abs/1511.06349) (2016) — the same collapse in text VAEs; word dropout as the cure
- Bengio et al., [*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*](https://arxiv.org/abs/1506.03099) (2015) — teacher forcing and the train/inference mismatch
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021)
- Ho & Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) (2022) — the condition dropout VoxCPM2 implements as `training_cfg_rate`
- Runbook and code: [`finetune/README.md`](../finetune/README.md)
