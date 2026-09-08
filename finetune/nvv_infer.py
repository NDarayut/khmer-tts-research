"""
Synthesize a fixed set of tagged English sentences, with and without the tag.

WHY THIS EXISTS
---------------
This is the instrument for the tag-expansion experiment. The claim under test
is that VoxCPM2 can be taught a *new* NVV tag -- one it was never trained on
and does not document, such as `[cough]` -- cheaply, because the model is
tokenizer-free and so already reads the string.

Proving that needs a before and an after, generated identically. This script is
both. Run it against the base model to get the "before" column, then against a
LoRA adapter to get the "after", with the same sentences and the same seeds so
the only difference is the adapter.

Every sentence is synthesized twice: once with the tag inline at the event
position, and once with the tag removed. The paired version is the control --
if a "laugh" appears in the untagged version too, the tag is not what caused
it. Both are kept for listening.

    python finetune/nvv_infer.py --out-dir finetune/results/nvv/base
    python finetune/nvv_infer.py --lora <path> --out-dir finetune/results/nvv/lora

English, deliberately. Khmer is a later question; this experiment isolates
whether the tag mechanism can be extended at all.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Three groups, and the third is what makes this an experiment rather than a
# demo.
#
#   DOCUMENTED  -- in VoxCPM2's published inventory, and also in the training
#                  corpus. The reference for what a working tag sounds like.
#   NOVEL       -- never documented by OpenBMB, but present in NonverbalTTS, so
#                  the fine-tune sees them. These carry the claim.
#   HELD_OUT    -- never documented AND absent from the training corpus, which
#                  was verified by scanning its symbol inventory. The model
#                  cannot have learned these, so whatever they do after training
#                  is what a *trained* tag must beat to count. Without this arm,
#                  an adapter that simply got more expressive everywhere would
#                  look like it had learned the tags.
DOCUMENTED_TAGS = ["[laughing]", "[sigh]"]
# [breath] is deliberately absent: its symbol is stripped from the training
# text (see build_corpus_nvv.DROP_SYMBOLS) so breath-only clips can serve as
# the untagged half of the mix. That makes it neither trained nor held out,
# and a muddy third category is worse than no category.
NOVEL_TAGS = ["[cough]", "[sniff]", "[throat-clear]", "[groan]"]
HELD_OUT_TAGS = ["[gasp]", "[yawn]"]

# Carrier sentences. Each has a marked slot where the tag goes, so placement is
# controlled rather than left to the end of the utterance. Kept short and plain
# so the event is easy to hear and CER is easy to score.
CARRIERS = [
    "I thought it was fine {tag} but apparently not.",
    "Well {tag} I suppose we should get started.",
    "She looked at the letter {tag} and put it down again.",
    "{tag} Sorry, could you say that one more time?",
    "It has been a very long day {tag} honestly.",
]


def build_cases(tags, carriers, seeds):
    """One case per (tag, carrier, seed, tagged/untagged)."""
    cases = []
    for tag in tags:
        for ci, carrier in enumerate(carriers):
            tagged = carrier.format(tag=tag)
            # The control: same sentence, tag removed, whitespace tidied.
            untagged = " ".join(carrier.format(tag="").split())
            for seed in seeds:
                for cond, text in (("tagged", tagged), ("untagged", untagged)):
                    cases.append({
                        "tag": tag,
                        "carrier_index": ci,
                        "seed": seed,
                        "condition": cond,
                        "text": text,
                        "tag_char_offset": carrier.index("{tag}") if cond == "tagged" else None,
                        "id": f"{tag.strip('[]')}_{ci}_{seed}_{cond}",
                    })
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", default=None,
                    help="adapter to load; omit for the base model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="a tag that fires on one seed and not the next is the "
                         "failure this experiment is about, so never use one seed")
    ap.add_argument("--tags", default=None,
                    help="comma-separated; default is documented + novel")
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--inference-timesteps", type=int, default=10)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model

    seeds = [int(s) for s in args.seeds.split(",")]
    tags = (args.tags.split(",") if args.tags
            else DOCUMENTED_TAGS + NOVEL_TAGS + HELD_OUT_TAGS)
    cases = build_cases(tags, CARRIERS, seeds)

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.lora or 'BASE (no adapter)'}")
    print(f"{len(tags)} tags x {len(CARRIERS)} carriers x {len(seeds)} seeds "
          f"x 2 conditions = {len(cases)} clips")
    model = load_model(args.lora)

    records = []
    t_start = time.time()
    for i, case in enumerate(cases, 1):
        # Resumable: a full sweep is long enough to outlive a shell timeout,
        # and re-synthesizing a clip that already exists would also re-roll it,
        # breaking the seed-for-seed comparability the experiment depends on.
        existing = audio_dir / f"{case['id']}.wav"
        if existing.exists() and existing.stat().st_size > 0:
            info = sf.info(existing)
            records.append({**case, "wav": str(existing.relative_to(out_dir)),
                            "sample_rate": info.samplerate,
                            "duration": info.frames / info.samplerate,
                            "synth_seconds": None, "resumed": True})
            continue
        torch.manual_seed(case["seed"])
        np.random.seed(case["seed"])
        t0 = time.time()
        result = model.generate(
            text=case["text"],
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )
        elapsed = time.time() - t0
        # generate() returns either a waveform or (waveform, sample_rate).
        # The decoder runs at 48 kHz, so never assume 16 -- writing the wrong
        # rate would make every clip play at the wrong speed and pitch, which
        # is exactly the thing this experiment is listening for.
        if isinstance(result, tuple):
            wav, sr = result[0], int(result[1])
        else:
            wav, sr = result, 48000
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype="float32").squeeze()
        path = audio_dir / f"{case['id']}.wav"
        sf.write(path, wav, sr)
        records.append({**case,
                        "wav": str(path.relative_to(out_dir)),
                        "sample_rate": sr,
                        "duration": len(wav) / sr,
                        "synth_seconds": round(elapsed, 2)})
        if i % 10 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}  ({time.time() - t_start:.0f}s elapsed)")

    manifest = {
        "model": args.lora or "base",
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "seeds": seeds,
        "documented_tags": [t for t in tags if t in DOCUMENTED_TAGS],
        "novel_tags": [t for t in tags if t in NOVEL_TAGS],
        "held_out_tags": [t for t in tags if t in HELD_OUT_TAGS],
        "carriers": CARRIERS,
        "clips": records,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} clips and {mpath}")


if __name__ == "__main__":
    main()
