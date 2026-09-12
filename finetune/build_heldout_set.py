"""
Pick clips for a listening page from BOTH splits, with their ground-truth audio.

WHY THIS EXISTS
---------------
The probe says the tag is inert on held-out text: deleting it moves the loss no
more than chance (185/359, p=0.30, against the base model's 180/359). That is a
statement about the training objective, and a reasonable person may want to
check it by ear before accepting it.

WHY BOTH SPLITS
---------------
A page of held-out clips alone can only show that something failed; it cannot
say what. Sampling the training split alongside it separates two very different
failures, and the listener can hear which one this is:

  * train carries the event, held-out does not
        -> the model learned the tag and did not generalize it. A data and
           objective problem, and the overfitting run already showed the
           mechanism exists.
  * neither carries the event
        -> the fine-tune installed nothing at all, on sentences it saw 21 times
           over. A much worse result, and one the loss curves hint at.

Both rows are generated identically and shuffled together on the page, so the
comparison is between splits rather than between page sections.

Four things play per sentence:

    original     the real NonverbalTTS recording -- what the event sounds like
    with tag     generated from the sentence, tag inline
    no tag       the same sentence, tag deleted -- the control
    before       the same tagged sentence on the untrained model

If `with tag` and `no tag` are indistinguishable in both splits, the ear agrees
with the probe. If `with tag` carries the event anywhere, the probe is missing
something and that matters more than the p-value.

    .venv/bin/python finetune/build_heldout_set.py --per-tag 4
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z-]*\]")

# The tags with enough events in both splits to be worth listening to. sneeze
# (11 train / 4 val), snore (2/6) and grunt (2/0) are omitted: too few clips to
# form an impression, and too few training events to have had a chance.
TAGS = ["[cough]", "[sniff]", "[throat-clear]", "[groan]", "[laughing]", "[sigh]"]

SPLITS = ("train", "val")


def pick(path, tags, per_tag, min_dur, max_dur, rng, split):
    rows = [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    cand = defaultdict(list)
    for r in rows:
        present = [t for t in tags if t in r["text"]]
        # One tag type, once, so the listener knows which sound to listen for.
        if len(present) == 1 and r["text"].count(present[0]) == 1 \
                and min_dur <= r["duration"] <= max_dur:
            cand[present[0]].append(r)

    out = []
    for t in tags:
        pool = sorted(cand[t], key=lambda r: r["audio"])
        rng.shuffle(pool)
        take = pool[:per_tag]
        for r in take:
            r = dict(r)
            r["tag"] = t
            r["split"] = split
            out.append(r)
        print(f"  {split:5} {t:16} {len(take):3d} of {len(pool)} available")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir",
                    default=str(ROOT / "finetune" / "data-nvv" / "manifests_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "data-nvv" / "heldout"))
    ap.add_argument("--per-tag", type=int, default=4,
                    help="clips per tag PER SPLIT")
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    picked = []
    for split in SPLITS:
        picked += pick(Path(args.manifest_dir) / f"{split}.jsonl", TAGS,
                       args.per_tag, args.min_dur, args.max_dur, rng, split)

    out_dir = Path(args.out_dir)
    orig_dir = out_dir / "originals"
    orig_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(picked):
        dst = orig_dir / f"{i:03d}_{r['split']}_{r['tag'].strip('[]')}.wav"
        shutil.copyfile(r["audio"], dst)
        r["original"] = str(dst)
        # The inference script keys on this name; it means "row index".
        r["overfit_index"] = i

    n_tr = sum(1 for r in picked if r["split"] == "train")
    (out_dir / "heldout_meta.json").write_text(json.dumps(
        {"per_tag": args.per_tag, "tags": TAGS, "n": len(picked),
         "n_train": n_tr, "n_val": len(picked) - n_tr,
         "source": args.manifest_dir,
         "note": "train rows were seen ~21 times during training; val rows were "
                 "never seen and their speakers are disjoint from train",
         "clips": picked}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(picked)} clips ({n_tr} train, {len(picked)-n_tr} held out), "
          f"{sum(r['duration'] for r in picked)/60:.1f} min")
    print(f"  {out_dir/'heldout_meta.json'}")


if __name__ == "__main__":
    main()
