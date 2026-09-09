# Can VoxCPM2 be conditioned on tags it was never given?

**Result: yes.** Allowed to memorize 60 recordings, the model learned to place a
cough where `[cough]` is written and not to produce one when the word is absent
— for four bracketed strings that appear nowhere in VoxCPM2's documentation.
Deleting the tag raised the loss on 59 of 60 clips (p = 5.3 × 10⁻¹⁷) and
*relocating* it, without deleting it, raised the loss on 43 of 60
(p = 5.3 × 10⁻⁴). The mechanism works. What had failed earlier was the data.

Listening page (four undocumented tags, ground truth alongside):
<https://claude.ai/code/artifact/95f1b9f5-f6b6-4148-adca-6b72fc17406b>

---

## 1. The question this run exists to answer

A normal fine-tune had already been run: 3.51 hours of tagged English, 2,500
steps, LoRA on lm + dit + proj. It produced almost nothing. On 150 clips
*drawn from its own training set*, deleting the inline tag cost **5.8%** of what
scrambling the transcript cost (p = 0.14, i.e. not distinguishable from zero),
and moving the tag to the wrong end of the sentence cost **nothing** (p = 0.28).
On the base model the same two figures are 2.3% (p = 0.23) and p = 0.98.

Those 150 clips come from `train.jsonl`, not from the held-out split. This
matters: the fine-tune failed to bind the tag on sentences it had been shown
2,500 steps' worth of times. See §7 for the held-out measurement, run later.

That null has two completely different explanations, and they lead to opposite
decisions:

| | what it would mean | what to do |
|---|---|---|
| **A. mechanism** | VoxCPM2 cannot bind a text tag to a *local* acoustic event. Text conditioning reaches the DiT as an utterance-level vector; nothing carries "here, at this moment". | abandon inline tags; fall back to global style headers, or a different model |
| **B. signal** | It can, but roughly 100 coughs spread over 3.5 hours is too weak a gradient to find it. | keep the approach, fix the corpus |

Nothing in the generated audio separates A from B, because audio conflates
"never learned it" with "learned it and the sampler washed it out". The
separation has to happen at the training objective.

**The overfitting test is the cheapest decisive experiment.** Take a handful of
clips, train until the model has memorized them, and ask whether the tag still
does nothing. A model that cannot learn the tag *even when handed the answer*
has problem A. One that can has problem B — and problem B is solvable.

## 2. Design

Every choice here is the opposite of a normal run, deliberately.

* **60 clips, 5.4 minutes**, 12 each for `[cough]`, `[groan]`,
  `[throat-clear]`, `[sniff]`, `[laughing]`. Source is
  [deepvk/NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS),
  English, emoji annotations converted to inline `[tag]` by
  `build_corpus_nvv.py`.
* **Exactly one tag type per clip, occurring once, 2–8 s.** A clip with two
  events teaches less per gradient step.
* **No untagged mix.** ELaTE's 50:50 ratio exists to stop the base model
  regressing, which is a generalization concern. This run wants the opposite of
  regularization.
* **5× the learning rate** (5e-4 against the documented 1e-4), weight decay 0,
  50-step warmup.
* **2,000 steps** at effective batch 8 — about **265 passes** over the same
  handful of recordings. Roughly an hour on one 12 GB card.
* **val == train**, so the validation loss reads memorization, not
  generalization.

Config: `finetune/conf/nvv_overfit.yaml`. Corpus: `finetune/build_overfit_set.py`.

Four of the five tags — `[cough]`, `[groan]`, `[throat-clear]`, `[sniff]` — are
strings OpenBMB never documented. Only `[laughing]` is in the published
inventory, and it is excluded from the listening page for exactly that reason:
the base model already produces laughter, so it can only flatter the result.

## 3. The measurement

`finetune/experiments/nvv_tag_sensitivity.py`. One teacher-forced forward pass
per clip per condition, comparing `loss/diff`:

| condition | text | what it tests |
|---|---|---|
| `true` | tag inline where the event happens | reference |
| `removed` | tag deleted | does the tag carry information at all? |
| `moved` | tag pushed to the far end of the sentence | does its **position** matter? |
| `scrambled` | word order destroyed, tags left in place | **positive control** |

Two design points make the result readable:

`moved` is the condition that distinguishes a local event from an utterance-level
flag. A model that has learned "this clip contains a cough" scores identically
here. Only a model that has learned "the cough happens *at this point*" gets
worse.

`scrambled` is what makes a null result meaningful. It must cost a lot. If even
a destroyed transcript does not move the loss, the probe is broken and "no
effect" would mean nothing.

The diffusion timestep and the noise are reseeded from the row index before
every forward pass, so all four conditions for a given clip see identical
sampling. The CFM loss is stochastic; an unseeded comparison would measure
noise.

## 4. Result

Validation `loss/diff` fell **0.975 → 0.234** over the run — the memorization
worked, as intended.

n = 60, reference `true` mean = 0.23070:

| condition | mean | Δ vs true | worse on | sign test |
|---|---:|---:|---:|---:|
| `removed` | 0.34955 | **+0.11885** | 59/60 | p = 5.3 × 10⁻¹⁷ |
| `moved` | 0.29089 | **+0.06019** | 43/60 | p = 5.3 × 10⁻⁴ |
| `scrambled` | 0.84472 | +0.61402 | 59/60 | p = 5.3 × 10⁻¹⁷ |

The positive control passed decisively, so the other two rows can be read.

**Removing the tag now costs 19.4% of what destroying the entire transcript
costs.** Against 5.8% for the normal fine-tune and 2.3% for the base model,
that is **3.3× the trained model and 8.3× the base model** in how much the tag
matters.

**Moving the tag costs more than half of what deleting it costs** — and on the
base model that same condition was p = 0.98, indistinguishable from doing
nothing. This is the more important of the two rows. The model is not
classifying the utterance; it is reading *where* in the sentence the string sits
and putting the event there.

Per-tag `removed` means, showing the effect is not carried by one tag:

| tag | documented by OpenBMB? | `removed` mean |
|---|---|---:|
| `[throat-clear]` | no | 0.3745 |
| `[sniff]` | no | 0.3619 |
| `[cough]` | no | 0.3618 |
| `[laughing]` | **yes** | 0.3269 |
| `[groan]` | no | 0.3226 |

The four undocumented tags are not weaker than the documented one. Whatever the
model is doing, it is not retrieving prior knowledge about the word `laughing`.

By ear on the listening page, `with tag` carries the event that is in the
original recording and `no tag` does not, on the same sentence and the same
seed.

## 5. What this does and does not establish

**It establishes** that VoxCPM2's architecture can bind an arbitrary inline
bracketed string to a specific non-verbal event at a specific position in the
sentence, and that a LoRA on lm + dit + proj is sufficient to install that
binding. Tag expansion is not blocked by the model. The earlier null was a data
problem.

**It does not establish** that the tags work. The evaluation is on the exact
sentences the model was trained on, hundreds of times over. That is what
overfitting means, and it is the point — quoting this as evidence that the tags
generalize would be dishonest. Generalization is a separate experiment with a
held-out set; it is reported in §7, and its result is negative.

Two supporting facts, verified separately:

* Adding a tag is **architecturally free**. VoxCPM2 is tokenizer-free at the
  input; `[cough]` already tokenizes to four ordinary tokens with no `<unk>`.
  There is no vocabulary to resize, no embedding to initialize, no cold start.
  (`probe_tag_representation.py`, 4/4 checks pass on voxcpm 2.0.3.)
* The base model already emits 0.8–1.9 s of non-lexical audio for *any*
  bracketed string, including nonsense like `[xyzzy]`, and never speaks the
  bracketed word aloud. It appears to have learned a generic
  "brackets = instruction" convention rather than knowledge of specific tag
  names — which is consistent with the four undocumented tags training as
  readily as the documented one.

## 6. What follows

The bottleneck is event density and positional precision in the corpus, not the
model. The two changes that follow directly:

1. **Event-span-weighted loss.** Upweight the loss over the frames where the
   event actually occurs, so a 300 ms cough is not averaged away against 6 s of
   speech. This is what the overfitting run achieved by brute repetition.
2. **A corpus with precise event timestamps** — SynParaSpeech (118.75 h) is the
   candidate — instead of NonverbalTTS's ~100 coughs.

Khmer stays out of this until tag expansion generalizes on English.

## 7. The held-out test: no generalization

§5 said generalization was a separate experiment that had not been run. It has
now been run, and the fine-tune does not survive it.

The held-out split (`val.jsonl`) was never shown to the model. Of its 87 clips,
28 carry a tag, and those 28 are the whole available sample — small, but
the only honest one. The same four-condition probe, on the fine-tuned model and
on the base model, over the identical clips:

| model | reference loss | tag-deletion Δ | worse | p | ratio |
|---|---:|---:|---:|---:|---:|
| base (untrained) | 0.87491 | +0.01603 | 19/28 | 0.044 | 0.519 |
| fine-tuned | 0.85221 | +0.01639 | 18/28 | 0.092 | 0.619 |

**The two rows are the same.** The tag-deletion cost differs by 0.0004 — well
inside noise at n = 28 — and the sign-test counts differ by one clip in the
*wrong* direction. Whatever tag sensitivity these clips show, the base model
already had it; 2,500 steps of fine-tuning on 3.5 hours added nothing that
transfers to unseen text.

Two cautions on reading the numbers:

* **n = 28 is underpowered.** A two-sided sign test needs about 20 of 28 to
  reach p < 0.05. This test could not have detected a small true effect. What it
  rules out is a large one — and after the overfitting run, a large effect is
  exactly what a working method would have produced.
* **The ratio is not comparable across clip sets.** The 0.5–0.6 figures here are
  far above the 0.023 the base model scored on 150 training clips, because these
  are different sentences, not because anything improved. That is precisely why
  the base model was re-run on *these* clips: within one clip set, the comparison
  is valid, and within this one the fine-tune is inert.

Set against §4, the picture is consistent and unflattering to the current
recipe: the mechanism exists (the overfit run proves it), and the fine-tune as
configured does not install it in any form that reaches new sentences. The
changes in §6 are not optional refinements — they are the difference between a
method that works and one that does not.

Reproducing:

```bash
.venv/bin/python finetune/experiments/nvv_tag_sensitivity.py \
    --lora finetune/checkpoints/nvv/latest \
    --manifest finetune/data-nvv/manifests/val.jsonl --rows 28 \
    --out finetune/results/nvv/tag_sensitivity_heldout.json
.venv/bin/python finetune/experiments/nvv_tag_sensitivity.py \
    --manifest finetune/data-nvv/manifests/val.jsonl --rows 28 \
    --out finetune/results/nvv/tag_sensitivity_base_heldout.json
```

---

## Reproducing the overfitting run

```bash
.venv/bin/python finetune/build_overfit_set.py --per-tag 12
.venv/bin/python finetune/train.py --config finetune/conf/nvv_overfit.yaml
.venv/bin/python finetune/experiments/nvv_tag_sensitivity.py \
    --lora finetune/checkpoints/nvv_overfit/latest \
    --manifest finetune/data-nvv/overfit/val.jsonl --rows 60 \
    --out finetune/results/nvv/tag_sensitivity_overfit.json
.venv/bin/python finetune/nvv_infer_overfit.py \
    --lora finetune/checkpoints/nvv_overfit/latest \
    --out-dir finetune/results/nvv/overfit_lora
.venv/bin/python finetune/nvv_infer_overfit.py \
    --out-dir finetune/results/nvv/overfit_base
.venv/bin/python finetune/build_overfit_artifact.py \
    --lora finetune/results/nvv/overfit_lora \
    --base finetune/results/nvv/overfit_base \
    --out finetune/results/nvv/overfit.html
```

**The trained adapter no longer exists.** It and the generated wavs were deleted
by mistake after the run (see commit `5316bb7`). What remained in
`finetune/checkpoints/nvv_overfit/` was an untrained step-0 checkpoint and an
empty directory — nothing trained — so the directory has since been removed
during a disk cleanup. The `--lora` path above therefore does not exist; the
commands must be run from `build_overfit_set.py` onward.

The numbers in §4 survive in
`finetune/results/nvv/tag_sensitivity_overfit.json` and the audio survives
embedded in the published listening page. Regenerating new audio from this
adapter requires rerunning the ~1 h training above.
