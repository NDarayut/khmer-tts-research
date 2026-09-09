"""
Carve a tiny, deliberately memorizable subset out of the NVV corpus.

WHY THIS EXISTS
---------------
`nvv_tag_sensitivity.py` found the inline tag nearly inert: removing it costs
2-6% of what scrambling the transcript costs, and moving it to the wrong end of
the sentence costs nothing (p=0.98 on the base model). That is a statement
about a 2,500-step run on 3.5 hours. It does not distinguish two very different
situations:

  A. the architecture cannot bind a text tag to a local acoustic event at all,
     in which case the whole approach is dead and no corpus rescues it; or
  B. it can, but 78 coughs spread over 3.5 hours is far too weak a signal.

An overfitting test separates them, and it is the cheapest decisive experiment
available. Take a handful of clips, train until the model has memorized them,
then ask whether the tag still does nothing. A model that cannot learn the tag
*even when allowed to memorize the answer* has a mechanism problem. One that
can has a data problem, and data problems are solvable.

This is explicitly NOT a generalization test. Success here proves capability,
not usefulness -- the model is expected to have memorized these exact
sentences, and the evaluation is on those same sentences. Saying so up front is
the point: an overfit result quoted as if it were a generalization result would
be a lie, and this file is where that distinction is recorded.

DESIGN
------
* Few clips per tag, so memorization is reachable in minutes not hours.
* Single tag per clip, occurring once, 2-8 s. Ambiguity is the enemy here: a
  clip with two events teaches the model less per gradient step.
* No untagged mix. ELaTE's 50:50 exists to protect the base model from
  regression, which is a generalization concern. This run wants the opposite of
  regularization.
* The originals are copied out alongside, because the ground-truth clip is the
  target the model is being asked to reproduce and the right thing to compare
  generated audio against.

    python finetune/build_overfit_set.py --per-tag 12
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TAGS = ["[cough]", "[groan]", "[throat-clear]", "[sniff]", "[laughing]"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",
                    default=str(ROOT / "finetune" / "data-nvv" / "manifests" / "train.jsonl"))
    ap.add_argument("--out-dir",
                    default=str(ROOT / "finetune" / "data-nvv" / "overfit"))
    ap.add_argument("--per-tag", type=int, default=12)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]

    cand = defaultdict(list)
    for r in rows:
        present = [t for t in TAGS if t in r["text"]]
        # Exactly one tag type, appearing exactly once: the cleanest possible
        # mapping from one string to one event.
        if len(present) == 1 and r["text"].count(present[0]) == 1 \
                and args.min_dur <= r["duration"] <= args.max_dur:
            cand[present[0]].append(r)

    rng = random.Random(args.seed)
    picked = []
    for t in TAGS:
        pool = sorted(cand[t], key=lambda r: r["audio"])
        rng.shuffle(pool)
        take = pool[: args.per_tag]
        for r in take:
            r = dict(r)
            r["tag"] = t
            picked.append(r)
        print(f"  {t:16} {len(take):3d} of {len(pool)} available")

    out_dir = Path(args.out_dir)
    orig_dir = out_dir / "originals"
    orig_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground-truth audio next to the manifest. The originals are what
    # the generated clips get compared against, and keeping them here means the
    # listening page does not depend on the full corpus staying on disk.
    for i, r in enumerate(picked):
        dst = orig_dir / f"{i:03d}_{r['tag'].strip('[]')}.wav"
        shutil.copyfile(r["audio"], dst)
        r["original"] = str(dst)
        r["overfit_index"] = i

    train = out_dir / "train.jsonl"
    with train.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps({"audio": r["audio"], "text": r["text"],
                                 "duration": r["duration"]},
                                ensure_ascii=False) + "\n")
    # The trainer wants a val manifest; there is no meaningful held-out set for
    # an overfitting run, so val is the train set and its loss is a memorization
    # readout, not a generalization one.
    shutil.copyfile(train, out_dir / "val.jsonl")

    (out_dir / "overfit_meta.json").write_text(json.dumps(
        {"per_tag": args.per_tag, "tags": TAGS, "n": len(picked),
         "hours": round(sum(r["duration"] for r in picked) / 3600, 3),
         "note": "val == train by design; this run measures memorization",
         "clips": picked}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(picked)} clips, {sum(r['duration'] for r in picked)/60:.1f} min")
    print(f"  {train}")
    print(f"  originals -> {orig_dir}")


if __name__ == "__main__":
    main()
