# `finetune/` — runbook

Style- and voice-controlled VoxCPM2 for Khmer. The *why*, the design reasoning
and the results are in [`docs/11-voxcpm2-style-control-finetune.md`](../docs/11-voxcpm2-style-control-finetune.md).
This file is the operational sequence.

Nothing here writes to `eval-set/eval.json`. `verify_control.py` reads it.

## What you get

A LoRA adapter that makes VoxCPM2 respond to a control tag prefixed to the text:

```
<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។
```

| slot | levels | what it moves |
|---|---|---|
| `spk` | `f1`…`f9`, `m1`…`m11` | which of the corpus voices, no reference clip needed |
| `rate` | `slow` `mid` `fast` | speaking rate |
| `pitch` | `low` `mid` `high` | pitch register, *relative to that voice* |
| `var` | `flat` `mid` `lively` | pitch variation — monotone vs expressive |
| `energy` | `soft` `mid` `loud` | level / projection |

Any slot may be set to `any` to leave it unspecified; the adapter is trained
with per-slot dropout precisely so partial tags work.

## Prerequisites

* This repo's `.venv` (torch 2.6 + cu124, `voxcpm` 2.0.3).
* The upstream trainer, which is **not** in the pip package:
  ```bash
  git clone --depth 1 https://github.com/OpenBMB/VoxCPM VoxCPM-src
  ```
* The `openbmb/VoxCPM2` checkpoint in the HF cache (the evaluation harness
  already downloads it).
* Source speech: `/run/media/pc/disk1/streaming_asr/data/dataset` — 1067 h of
  16 kHz read Khmer across 20 speakers, from this project's sibling ASR repo.
* For the CER regression check only: the ASR repo's own interpreter,
  `/run/media/pc/disk1/streaming_asr/venv/bin/python`.

## Sequence

```bash
# 1. corpus: extract clips, measure prosody, label, write manifests   (~50 min)
python finetune/build_corpus.py --clips 6000 --min-dur 3.0 --max-dur 10.0

# 2. sanity-check the manifest with the package's own validator
voxcpm validate --manifest finetune/data/train.jsonl --sample-rate 16000

# 3. train                                                        (~7 h, 12 GB)
#    --onset-weight is NOT optional: without it the adapter trains fine and
#    ignores the control tag. See docs/11 §11.5.
python finetune/train.py --config finetune/conf/khmer_style_lora.yaml \
    --onset-weight 8.0 --onset-tau 4.0

# 4. pick the best checkpoint, then verify it fully                     (~2 h)
#    Ranks every saved checkpoint by how strongly it responds to the control
#    tag, NOT by validation loss -- docs/09 §9.5 warns the best checkpoint is
#    often not the last, and loss cannot see the failure it warns about.
bash finetune/select_checkpoint.sh

#    or verify one checkpoint directly                                (~1.5 h)
python finetune/verify_control.py --lora finetune/checkpoints/khmer_style/latest

# 5. did it cost intelligibility?
/run/media/pc/disk1/streaming_asr/venv/bin/python finetune/score_cer.py \
    --dir base=finetune/results/audio/base \
    --dir lora=finetune/results/audio/lora

# 6. use it
python finetune/synthesize_styled.py \
    --lora finetune/checkpoints/khmer_style/latest \
    --text "សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។" \
    --style var=lively,rate=fast,spk=f2 \
    --out /tmp/hello.wav
```

## The one thing not to skip

The obvious version of this — tag in the text, LoRA on labelled speech — trains
cleanly and produces an adapter that ignores the tag. Two settings prevent that,
and both are cheap to get wrong: `enable_proj: true` in the YAML (the LM→DiT
projections are the only route text-side information has into the acoustic
generator) and `--onset-weight 8.0` on the command line (teacher forcing makes
the tag redundant with the ground-truth acoustic prefix everywhere but the start
of the clip). Each was isolated by a 400-step probe against a pass bar fixed in
advance; together they took speaker separation from 5.8 Hz to 39.9 Hz. The
reasoning is in docs/11 §11.5, the numbers in `results/diagnosis.md`.

If you change the conditioning scheme, verify it with
`experiments/tag_sensitivity.py` before committing to a long run. It asks
whether the tag changes the training loss at all, carries a positive control so
a null result is interpretable, and takes about two minutes.

Training resumes automatically from `checkpoints/khmer_style/latest` — kill it
and restart with the same command. `save_interval: 500` in the YAML, and
SIGTERM/SIGINT also force a save.

Watch it with `tensorboard --logdir finetune/logs/khmer_style`. Per docs/09
§9.5, **stop when it sounds right, not when the loss bottoms out**; checkpoints
are kept every 500 steps for exactly that reason.

## Files

| file | role |
|---|---|
| `build_corpus.py` | extract + measure + label + write manifests. Defines the tag format; everything else imports it from here. |
| `train.py` | wraps upstream's `train_voxcpm_finetune.py` with the 12 GB memory patches. Upstream stays unmodified. |
| `conf/khmer_style_lora.yaml` | the real run |
| `conf/khmer_style_pilot.yaml` | 20 steps, for proving the pipeline and measuring VRAM |
| `synthesize_styled.py` | tag + text → wav. Also the engine for the sweep. |
| `verify_control.py` | the falsifiable check: sweep each axis, measure, compare against the base model as control |
| `select_checkpoint.sh` | ranks checkpoints by control response rather than by loss, then runs the full verification on the winner |
| `score_cer.py` | CER regression vs the base model, using the project's Khmer CTC scorer |
| `experiments/probe_conditioning.py` | 400-step falsifiable probe: can the model learn to read a tag at all? One arm per hypothesis, pass bar fixed in advance |
| `experiments/tag_sensitivity.py` | asks the *training objective* whether the tag matters, with a scrambled-transcript positive control. Minutes, not hours — run this before any long conditioning run |
| `results/diagnosis.md` | the full measurement record of the first run's failure and the two fixes, and where it sits in the literature |

## Why 12 GB works when the docs say 20

`docs/09` records OpenBMB's figure of ~20 GB for VoxCPM2 LoRA, and this card has
12. The gap is one line: upstream's `from_local` leaves the model in float32 in
training mode — 2.29 B frozen parameters at 8.6 GiB — while running the forward
pass under `autocast(bfloat16)` anyway. `train.py` casts frozen weights to bf16
(4.27 GiB), keeps the ~36 M LoRA parameters in fp32 so AdamW updates are not
quantised away, and checkpoints all 36 transformer layers. Measured peak on this
machine: **10.3–10.9 GiB of 12.3**, desktop included. Batch size is 1 with
8-step accumulation.

Peak memory turns out to be nearly independent of sequence length here — dropping
`max_batch_tokens` from 1024 to 512 moved it by only 0.6 GiB — so the ceiling is
resident weights and the AudioVAE encode, not activations.

## Not gitignored, and why

`finetune/data/wavs/` and `finetune/checkpoints/` are excluded (audio and
weights, regenerable and large). The manifests are excluded too, because they
carry absolute paths into a corpus that lives outside this repo. The metadata
(`corpus_meta.json`), the results and this documentation are tracked.
