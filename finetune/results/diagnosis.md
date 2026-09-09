# Why the control tag was ignored — the measurement record

The adapter trained. It changed the model. It does not read the control tag.
This file is the evidence for that claim and for the cause, in the order the
evidence was collected. The design and the verdict are in
[`docs/11`](../../docs/deep-dives/11-voxcpm2-style-control-finetune.md) §2.1.

## 1. The adapter is real, and it did change the model

| check | result |
|---|---|
| LoRA weight delta, median `‖ΔW‖/‖W‖` over 192 adapted layers | **3.03e-02** (range 2.78e-03 … 8.31e-02) |
| `lora_B` tensors that are non-zero (they initialise to zero) | **192 / 192** |
| validation `loss/diff`, step 0 → 4000 | 0.8819 → 0.8076 |
| generated F0, base model, across sentences | wanders 105.9 – 231.5 Hz |
| generated F0, adapter, across sentences | clamped 201.3 – 261.6 Hz (median delta **+66.5 Hz**) |
| generated speaking rate, base → adapter | 17.61 → 14.07 chars/s |

So the fine-tune itself succeeded. The adapter has a large, audible effect —
it just is not a *controllable* one.

## 2. It does not respond to the tag

Ten eval-set sentences per level, every axis swept, non-swept slots pinned to
`mid` so the tag stays in-distribution. `rho` is Spearman's correlation between
commanded level and measured quantity; `p` is a 10,000-shuffle permutation test.

| axis | measured | low → high | rho | p | corpus ceiling |
|---|---|---|---|---|---|
| `rate` | chars/s | +0.27 | +0.141 | 0.46 | **+6.57** |
| `pitch` | F0 median | −2.81 Hz | +0.019 | 0.93 | **+28.27 Hz** |
| `var` | F0 std | +0.02 st | −0.057 | 0.77 | **+1.71 st** |
| `energy` | RMS | −0.37 dB | −0.038 | 0.85 | **+5.81 dB** |

Every axis is null, and the movement is ~0 against ceilings the labels
themselves achieve. Ranking all nine checkpoints by control response found no
trend across training and a best mean rho of **+0.026** — the failure is not a
matter of picking the wrong checkpoint.

The decisive axis is `spk`. Six speaker tags spanning 105 → 280 Hz of real
corpus pitch produced generated medians of 198.5, 209.3, 210.3, 222.9, 208.6,
196.7 Hz: **26 Hz of spread against 175 Hz of corpus spread, rho = −0.200.**
Speaker identity cannot be inferred from the text, so if that axis does not
land, nothing is landing. It also explains the voice collapse in §1 — a model
that cannot read the tag can only average its twenty speakers.

## 3. What was ruled out

| hypothesis | how it was ruled out |
|---|---|
| tag shape out of distribution | per-slot dropout made all-`any` tags 0.27% of rows — but the in-distribution `--hold mid` sweep above is also null |
| adapter inert | §1: 3% weight delta, all 192 `lora_B` non-zero |
| tokenization | tags survive the model's wrapped tokenizer, differ in 10 token positions between extremes, zero UNK, present in the training rows |
| train/inference tokenizer mismatch | both call the same `base_model.text_tokenizer`; inference passes the text through unsplit and un-chunked |
| text normalization eating the tag | `normalize: bool = False` by default in `core.py::_generate`, never passed |
| tag placement (350 byte-tokens from the audio) | probe arm `end`, tag moved adjacent to `audio_start`: **+5.8 Hz** separation |
| LM→DiT projection bottleneck not adapted | probe arm `proj`, `enable_proj: true`: **+19.5 Hz** — 3× better, still under the +30 Hz bar |

## 4. The cause

Everything above measures *generated audio*, which cannot distinguish "never
learned the tag" from "learned it, but sampling washes it out". So the question
was taken back to the training objective. For each held-out clip, one
teacher-forced forward pass with the true speaker tag and one with the other
speaker's tag, with the diffusion timestep and noise held identical by
reseeding from the row index (the CFM loss is stochastic; unseeded, this
measures noise).

| condition | mean `loss/diff` | delta | rows worse |
|---|---|---|---|
| correct tag | 0.86908 | — | — |
| **swapped tag** | 0.86930 | **+0.00022** (+0.03%) | 22/40, sign-test p = 0.32 |
| **scrambled transcript** (positive control) | 0.93348 | **+0.06439** (+7.4%) | 35/40 |

The instrument works — corrupting the transcript, which the model certainly
uses, moves the loss by 7.4%. Corrupting the tag moves it by 0.03%, a **290×**
difference indistinguishable from chance. The model reads the text and does not
read the tag, in its own objective. No change to the sampler could have rescued
this.

**Why.** Training is teacher-forced: every audio patch is predicted with the
preceding *ground-truth* patches visible, and a speaker's pitch is trivially
readable off those. The tag is therefore redundant with the acoustic prefix
everywhere except the very start of the clip, and carries no gradient. The
prediction that follows is that the tag must matter more when the loss is
restricted to the first few patches, where no prefix exists yet:

| loss restricted to first K patches | correct | swapped | delta |
|---|---|---|---|
| 1 | 1.09809 | 1.10126 | **+0.00318** |
| 2 | 0.96984 | 0.97198 | +0.00214 |
| 4 | 0.91577 | 0.91720 | +0.00143 |
| 8 | 0.89910 | 0.89991 | +0.00082 |
| all | 0.86908 | 0.86930 | +0.00022 |

Monotone, and **14× larger on the first patch than over the whole clip.** The
tag matters exactly where the prefix cannot answer for it, and is drowned
everywhere else. That is the mechanism.

This is a property of the *training objective*, not of VoxCPM2's architecture
and not of the tag format. Any global attribute — speaker, rate, register —
supplied through the text field faces the same competition against a
teacher-forced acoustic prefix that already encodes it.

## 5. The fix, and the probe that isolated it

Each arm below is 400 steps on the easiest discrimination the corpus offers:
two speakers an octave apart (m5 ~105 Hz, f4 ~280 Hz), 765 training rows, one
binary tag `<|spk:m5|>` / `<|spk:f4|>`. **The pass bar — more than 30 Hz of
separation between the two tags, in the correct direction — was fixed before
the first arm ran.** 30 Hz is a modest ask against 175 Hz of real separation.

Each arm changes exactly one thing from the arm above it, so a result attaches
to a cause rather than to a bundle of changes.

| arm | what it changes | m5 | f4 | separation | |
|---|---|---|---|---|---|
| `start` | tag prefixed (the failed run's layout) | — | — | — | *not run separately; the 4000-step run is its evidence* |
| `end` | tag moved adjacent to `audio_start` | 266.5 | 272.3 | **+5.8 Hz** | FAIL |
| `proj` | `end` + `enable_proj: true` | 269.7 | 289.2 | **+19.5 Hz** | FAIL |
| `onset` | `proj` + diffusion loss weighted 8× at the onset | 272.3 | 312.2 | **+39.9 Hz** | **PASS** |

Placement was not the problem. The two things that were:

1. **`enable_proj: false`.** `enc_to_lm_proj`, `lm_to_dit_proj`,
   `res_to_dit_proj` and `fusion_concat_proj` are the linear bottleneck through
   which everything the LM knows reaches the diffusion transformer that
   produces the acoustics. docs/09 recommends leaving them frozen, which is
   sound for speaker *cloning* — the voice arrives through the reference-audio
   encoder and never needs to cross that bridge — and wrong for conditioning
   that arrives only as text and has no other route across. Worth +13.7 Hz.

2. **The teacher-forcing shortcut of §4.** Weighting the diffusion loss toward
   the audio onset — position *i* of the audio span gets weight
   `1 + 7·exp(-i/4)`, so 8× on the first patch decaying to ~1 by the twentieth
   — puts the gradient where the tag is the only available cue. Worth a further
   +20.4 Hz, and it is the change that clears the bar. Implementation and the
   reason a non-binary mask works unmodified are in `finetune/train.py`, PATCH 4.

The same measurement that diagnosed the failure confirms the fix at the level
of the objective, not just of the audio:

| checkpoint | correct tag | swapped tag | delta | rows worse | sign-test p |
|---|---|---|---|---|---|
| `proj` | 0.86908 | 0.86930 | +0.00022 | 22/40 | 0.32 |
| `onset` | 0.87204 | 0.87289 | **+0.00086** | 28/40 | **0.008** |

Four times the penalty for the wrong tag, and significant where it was not.
The model now reads the tag in its own loss, which is the thing that was
actually missing.

## 6. Where this sits in the literature

The failure in §4 is not specific to VoxCPM2 or to TTS. It is the
**information-preference problem**: a sufficiently expressive autoregressive
decoder ignores a conditioning signal, because predicting the next step from its
own history costs fewer bits than routing information through the conditioning
path.

- [Chen et al., *Variational Lossy Autoencoder*](https://arxiv.org/abs/1611.02731)
  (ICLR 2017) states it directly and lists the remedies: weaken the decoder, add
  dropout, limit its receptive field — all forms of removing the shortcut.
- [Bowman et al.](https://arxiv.org/abs/1511.06349) (2016) hit the same collapse
  in text VAEs and fixed it with word dropout: replace part of the decoder's
  history with `UNK` so it must consult the latent.
- [Bengio et al.](https://arxiv.org/abs/1506.03099) (2015) frame the general
  train/inference mismatch that teacher forcing creates.

The label design in `build_corpus.py` follows Parler-TTS
([Lyth & King, 2024](https://arxiv.org/abs/2402.01912)) and its
[`dataspeech`](https://github.com/huggingface/dataspeech) pipeline: measure
continuous attributes, drop the extremes, bin the rest, name the bins, and judge
pitch against comparable speakers rather than the whole corpus.

The onset-weighted loss of §5 has no such precedent — the literature's fix is to
corrupt the shortcut, not to reweight around it. See docs/11 §2.1 for why the
cheaper intervention was chosen and what the more principled experiment would be.

---

## Note on artefacts (2026-09-09)

The checkpoints for the three ablation arms (`experiments/ckpt_end`,
`ckpt_proj`, `ckpt_onset`) and for `checkpoints/khmer_style_run1_failed` were
deleted in a disk cleanup. The measurements above are the record; the weights
were only ever the means of obtaining them. Every surviving run keeps its
`latest/lora_weights.safetensors`, with optimizer state stripped — the adapters
load, but training cannot be resumed from them.
