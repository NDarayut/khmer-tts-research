"""
Why did the control tag get ignored? A fast, falsifiable probe.

THE SITUATION
-------------
The first 4000-step run produced an adapter that demonstrably changed the model
(median |dW|/|W| = 3.0%, and generated F0 moved from the base model's wandering
106-232 Hz to a tight 201-262 Hz) but that ignores the control tag completely.
The `spk` axis is the proof: tags spanning 105-280 Hz of real speaker pitch
produced 26 Hz of output spread, rho = -0.20. Speaker identity is the axis with
the most information -- the model *cannot* infer it from the text -- so if that
one does not land, nothing is landing.

Tokenisation was ruled out first: the tags survive the model's own wrapped
tokenizer, differ in 10 positions between extremes, and contain no UNK. Text
normalisation was ruled out too (`normalize` defaults to False).

THE HYPOTHESIS THIS PROBE TESTS
-------------------------------
Khmer tokenises to UTF-8 *byte* tokens -- roughly three per character, ~330 for
a typical sentence. The training sequence is therefore

    [tag ~24 tokens][Khmer ~330 tokens][audio_start][audio patches...]

so every audio position must carry information from a prefix ~350 positions
back, across a stretch of byte-level noise. Moving the tag to sit immediately
before `audio_start` puts it adjacent to the audio it is supposed to condition.

WHY A PROBE RATHER THAN JUST RETRAINING WITH THE FIX
----------------------------------------------------
A full run is 6.5 h. This probe reduces the task to the easiest possible version
of itself -- two speakers an octave apart, one binary tag, a few hundred steps --
so that "can this model learn to read a tag at all?" gets answered in ~35
minutes per arm. It also runs both arms (tag at start, tag at end), so the
answer distinguishes between "placement was the problem" and "something deeper
is wrong", instead of confounding the fix with the retrain.

Success criterion, fixed in advance: the two speaker tags must produce generated
F0 medians separated by more than 30 Hz, in the correct direction. The real
speakers are 105 Hz and 280 Hz apart, so 30 Hz is a modest ask; the failed run
managed 26 Hz across six tags in no particular order.

    python finetune/experiments/probe_conditioning.py --stage data
    python finetune/experiments/probe_conditioning.py --stage train --arm end
    python finetune/experiments/probe_conditioning.py --stage test  --arm end
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "finetune" / "experiments"
SNAPSHOT = ("/home/pc/.cache/huggingface/hub/models--openbmb--VoxCPM2/"
            "snapshots/32279effe8c19989596f05d353d1447f51d9e915")

# Two voices roughly an octave apart -- the easiest discrimination the corpus
# offers. If a tag cannot separate these, it cannot separate anything.
VOICE_LOW = "m5"      # corpus median ~105 Hz
VOICE_HIGH = "f4"     # corpus median ~280 Hz
N_PER_VOICE = 700
STEPS = 400


def tag_for(voice):
    return f"<|spk:{voice}|>"


def build_data(args):
    meta = json.loads((ROOT / "finetune" / "data" / "corpus_meta.json")
                      .read_text(encoding="utf-8"))
    by_voice = {VOICE_LOW: [], VOICE_HIGH: []}
    for c in meta["clips"]:
        v = c["lab"]["spk"]
        if v in by_voice:
            by_voice[v].append(c)
    rng = random.Random(0)
    rows = []
    for v, cs in by_voice.items():
        rng.shuffle(cs)
        print(f"  {v}: {len(cs)} available, taking {min(N_PER_VOICE, len(cs))}",
              file=sys.stderr)
        rows.extend(cs[:N_PER_VOICE])
    rng.shuffle(rows)

    out = EXP / "data"
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("start", "end"):
        for split, sel in (("val", rows[:40]), ("train", rows[40:])):
            lines = []
            for c in sel:
                tag = tag_for(c["lab"]["spk"])
                text = tag + c["text"] if arm == "start" else c["text"] + tag
                lines.append(json.dumps(
                    {"audio": c["wav"], "text": text,
                     "duration": round(c["duration"], 3)}, ensure_ascii=False))
            p = out / f"{arm}_{split}.jsonl"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"wrote {p} ({len(lines)} rows)", file=sys.stderr)
    print(f"\n{VOICE_LOW} vs {VOICE_HIGH}; corpus F0 "
          f"{np.median([c['f0_median_hz'] for c in by_voice[VOICE_LOW]]):.0f} Hz "
          f"vs {np.median([c['f0_median_hz'] for c in by_voice[VOICE_HIGH]]):.0f} Hz",
          file=sys.stderr)


ARM_DATA = {"start": "start", "end": "end", "proj": "end", "onset": "end"}


def write_config(arm):
    """Arm `proj` differs from `end` in exactly one thing: enable_proj.

    `enable_proj` adapts enc_to_lm_proj / lm_to_dit_proj / res_to_dit_proj /
    fusion_concat_proj -- the linear bottleneck through which everything the LM
    knows reaches the diffusion transformer that actually produces the acoustics.
    The main run left it false, following docs/09's recommendation, which is
    sound advice for *speaker cloning* (where the voice arrives through the
    reference-audio encoder) and possibly wrong for text-derived conditioning,
    which has no other route across.

    Arm `onset` keeps everything `proj` had and adds the onset-weighted loss
    (see `finetune/train.py`, PATCH 4). It is the fix implied by the measured
    diagnosis: the tag is redundant with the teacher-forced acoustic prefix
    everywhere except the very start of the clip, so the gradient is moved
    there.
    """
    enable_proj = "true" if arm in ("proj", "onset") else "false"
    data = ARM_DATA[arm]
    cfg = EXP / f"probe_{arm}.yaml"
    cfg.write_text(f"""# Probe arm: tag at {arm} of text. Deliberately small and short.
pretrained_path: {SNAPSHOT}
train_manifest: {EXP}/data/{data}_train.jsonl
val_manifest: {EXP}/data/{data}_val.jsonl
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 8
num_workers: 4
num_iters: {STEPS}
log_interval: 25
valid_interval: 100000     # no val audio generation; it only costs time here
save_interval: {STEPS}
learning_rate: 0.0002      # 2x the main run: a short probe needs to move fast
weight_decay: 0.01
warmup_steps: 40
max_steps: {STEPS}
max_batch_tokens: 1024
max_grad_norm: 1.0
save_path: {EXP}/ckpt_{arm}
tensorboard: {EXP}/logs_{arm}
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: {enable_proj}
  r: 64
  alpha: 64
  dropout: 0.0
""", encoding="utf-8")
    return cfg


def train(args):
    cfg = write_config(args.arm)
    print(f"training arm={args.arm} -> {cfg}", file=sys.stderr)
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "finetune/train.py"),
           "--config", str(cfg)]
    if args.arm == "onset":
        cmd += ["--onset-weight", "8.0", "--onset-tau", "4.0"]
    subprocess.run(cmd, check=True)


def test(args):
    import torch
    from finetune.synthesize_styled import load_model
    from finetune.verify_control import measure_clip, load_sentences

    ckpt = EXP / f"ckpt_{args.arm}" / "latest"
    if not ckpt.exists():
        sys.exit(f"no checkpoint at {ckpt}")
    sents = load_sentences(args.sentences, 0)
    model = load_model(ckpt)
    res = {}
    for voice in (VOICE_LOW, VOICE_HIGH):
        tag = tag_for(voice)
        f0s = []
        for e in sents:
            torch.manual_seed(0)
            place = ARM_DATA[args.arm]
            text = tag + e["sentence"] if place == "start" else e["sentence"] + tag
            r = model.generate(text=text)
            audio, sr = (r[0], int(r[1])) if isinstance(r, tuple) else (r, 48000)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            m = measure_clip(audio, sr, e["sentence"])
            if m:
                f0s.append(m["f0_median_hz"])
        res[voice] = f0s
        print(f"  {voice}: median F0 {np.median(f0s):.1f} Hz  (n={len(f0s)})",
              file=sys.stderr)

    lo, hi = np.median(res[VOICE_LOW]), np.median(res[VOICE_HIGH])
    sep = hi - lo
    verdict = "PASS" if sep > 30 else "FAIL"
    print(f"\narm={args.arm}: {VOICE_LOW}={lo:.1f} Hz  {VOICE_HIGH}={hi:.1f} Hz  "
          f"separation={sep:+.1f} Hz  -> {verdict} (threshold +30 Hz)")
    (EXP / f"result_{args.arm}.json").write_text(json.dumps(
        {"arm": args.arm, "low": lo, "high": hi, "separation": sep,
         "verdict": verdict, "f0": res}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("data", "train", "test"))
    ap.add_argument("--arm", default="end", choices=("start", "end", "proj", "onset"))
    ap.add_argument("--sentences", type=int, default=6)
    args = ap.parse_args()
    {"data": build_data, "train": train, "test": test}[args.stage](args)


if __name__ == "__main__":
    main()
