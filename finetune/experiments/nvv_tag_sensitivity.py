"""
Does an inline NVV tag change the training loss at all?

WHY THIS EXISTS
---------------
The tag-expansion fine-tune ran 2,500 steps and `loss/diff` never came down
(0.768 at step 0, 1.045 at step 2000). That is the same smell as the Khmer
layer-1b run, where the control tag turned out to earn a gradient ~290x smaller
than the transcript and the failure was invisible in the loss *curve*.

Generated audio cannot tell those apart, because it conflates two failures:

  A. the model never learned any dependence on the tag, or
  B. it learned one and sampling washes it out (CFG, the diffusion head, the
     Euler solver), so nothing survives into the waveform.

A costs data or a reshaped loss. B costs a sampler setting and is free. This
goes back to the training objective to separate them.

HOW
---
For each held-out clip, one teacher-forced forward pass per condition, and
compare `loss/diff`:

  true      -- the clip's real text, tag inline where the event happens
  removed   -- the same text with the tag deleted
  moved     -- the tag relocated to the far end of the sentence
  scrambled -- POSITIVE CONTROL: word order destroyed, tag kept

`removed` tests whether the tag carries information at all. `moved` tests
something `removed` cannot: whether the tag's *position* matters, which is the
whole claim for an inline local event as against a global header tag. If
`moved` costs nothing, the model is at best treating the tag as an utterance-
level flag.

`scrambled` is what makes a null result readable. It must cost a lot. If even a
destroyed transcript does not move the loss, the probe is broken and "no
effect" would mean nothing.

Everything else is held identical -- the diffusion timestep and the noise are
reseeded from the row index before every forward, because the CFM loss is
stochastic and an unseeded comparison would measure noise.

    .venv/bin/python finetune/experiments/nvv_tag_sensitivity.py \
        --lora finetune/checkpoints/nvv/latest
"""

import argparse
import json
import random
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VoxCPM-src" / "scripts"))

SNAPSHOT = ("/home/pc/.cache/huggingface/hub/models--openbmb--VoxCPM2/"
            "snapshots/32279effe8c19989596f05d353d1447f51d9e915")

TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z-]*\]")


def strip_tags(text):
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()


def make_removed(text):
    return strip_tags(text)


def make_moved(text):
    """Same tags, wrong place: all of them pushed to the end of the sentence.

    This is the condition that distinguishes a local event from a global flag.
    A model that has learned 'this clip contains a cough' scores the same here;
    one that has learned 'the cough happens *at this point*' does not.
    """
    tags = TAG_RE.findall(text)
    if not tags:
        return text
    return (strip_tags(text) + " " + " ".join(tags)).strip()


def make_scrambled(text, seed):
    """Positive control: destroy word order, keep the tags in place."""
    rng = random.Random(seed)
    parts = text.split()
    words = [i for i, w in enumerate(parts) if not TAG_RE.fullmatch(w)]
    shuffled = [parts[i] for i in words]
    rng.shuffle(shuffled)
    it = iter(shuffled)
    return " ".join(parts[i] if TAG_RE.fullmatch(parts[i]) else next(it)
                    for i in range(len(parts)))


CONDITIONS = {
    "true": lambda t, s: t,
    "removed": lambda t, s: make_removed(t),
    "moved": lambda t, s: make_moved(t),
    "scrambled": lambda t, s: make_scrambled(t, s),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", default=None,
                    help="adapter dir; omit to probe the base model")
    ap.add_argument("--manifest",
                    default=str(ROOT / "finetune" / "data-nvv" / "manifests" / "val.jsonl"))
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "finetune" / "results" / "nvv"
                                        / "tag_sensitivity.json"))
    args = ap.parse_args()

    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model
    from voxcpm.training.data import (BatchProcessor, HFVoxCPMDataset,
                                      load_audio_text_datasets)

    lora_cfg = None
    if args.lora:
        cfg_file = Path(args.lora) / "lora_config.json"
        lora_cfg = LoRAConfig(**json.loads(cfg_file.read_text())["lora_config"])
    model = VoxCPM2Model.from_local(SNAPSHOT, optimize=False, training=True,
                                    lora_config=lora_cfg)
    if args.lora:
        loaded, skipped = model.load_lora_weights(str(args.lora))
        print(f"loaded {len(loaded)} lora params, skipped {len(skipped)}",
              file=sys.stderr)
    # Nothing here is trained, so freeze everything first. Without an adapter
    # every parameter is still trainable and the cast below would skip the whole
    # model, leaving 2.29 B parameters in fp32 -- which OOMs a 12 GB card.
    for prm in model.parameters():
        prm.requires_grad_(False)
    # Same bf16 cast the trainer uses, so the loss is comparable to training.
    for n, prm in model.named_parameters():
        if not n.startswith("audio_vae."):
            prm.data = prm.data.to(torch.bfloat16)
    model.cuda().eval()

    src = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in src if l.strip()]
    # Only clips that actually carry a tag: a row with nothing to remove would
    # make every condition identical and dilute the comparison toward zero.
    rows = [r for r in rows if TAG_RE.search(r["text"])][: args.rows]
    if not rows:
        sys.exit("no tagged rows in the manifest")
    print(f"{len(rows)} tagged held-out clips", file=sys.stderr)

    tmp = Path(args.out).parent / "_sens"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cond, fn in CONDITIONS.items():
        p = tmp / f"{cond}.jsonl"
        p.write_text("\n".join(
            json.dumps({**r, "text": fn(r["text"], i)}, ensure_ascii=False)
            for i, r in enumerate(rows)) + "\n", encoding="utf-8")
        paths[cond] = p
    for cond in CONDITIONS:
        print(f"  {cond:10} {CONDITIONS[cond](rows[0]['text'], 0)[:96]}",
              file=sys.stderr)

    tok = model.text_tokenizer
    losses = {}
    for cond, path in paths.items():
        ds, _ = load_audio_text_datasets(train_manifest=str(path), val_manifest="",
                                         sample_rate=16000)
        ds = ds.map(lambda b: {"text_ids": [tok(t) for t in b["text"]]},
                    batched=True, remove_columns=["text"])
        wrapped = HFVoxCPMDataset(ds)
        bp = BatchProcessor(config=model.config, audio_vae=model.audio_vae,
                            dataset_cnt=1, device=torch.device("cuda"))
        vals = []
        with torch.no_grad():
            for i in range(len(wrapped)):
                batch = HFVoxCPMDataset.collate_fn([wrapped[i]])
                p = bp(batch)
                # Identical timestep and noise across conditions for this row.
                torch.manual_seed(1000 + i)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(p["text_tokens"], p["text_mask"], p["audio_feats"],
                                p["audio_mask"], p["loss_mask"], p["position_ids"],
                                p["labels"], progress=0.0, sample_generate=False)
                vals.append(float(out["loss/diff"]))
        losses[cond] = np.array(vals)
        print(f"{cond:10} mean loss/diff {np.mean(vals):.5f}", file=sys.stderr)

    base = losses["true"]
    n = len(base)
    report = {"model": args.lora or "base", "n": n,
              "true_mean": float(base.mean()), "conditions": {}}
    print(f"\nmodel: {args.lora or 'BASE'}   n={n}")
    print(f"  correct text          {base.mean():.5f}")
    print(f"\n  {'condition':12}{'mean':>10}{'delta':>10}{'worse':>9}{'p':>10}")
    print("  " + "-" * 51)
    for cond in ("removed", "moved", "scrambled"):
        d = losses[cond] - base
        worse = int((d > 0).sum())
        p = sum(comb(n, k) for k in range(worse, n + 1)) / 2 ** n
        report["conditions"][cond] = {
            "mean": float(losses[cond].mean()), "delta_mean": float(d.mean()),
            "worse": worse, "p": p, "values": losses[cond].tolist()}
        print(f"  {cond:12}{losses[cond].mean():>10.5f}{d.mean():>+10.5f}"
              f"{worse:>6}/{n}{p:>10.4g}")

    sc = report["conditions"]["scrambled"]["delta_mean"]
    rm = report["conditions"]["removed"]["delta_mean"]
    ratio = (rm / sc) if sc > 0 else float("nan")
    report["removed_over_scrambled"] = ratio
    print(f"\n  removing the tag costs {ratio:.4g}x what scrambling the "
          f"transcript costs")
    if sc <= 0:
        print("  ** POSITIVE CONTROL FAILED: scrambling the transcript did not "
              "raise the loss. The probe is not measuring what it should, and a "
              "null result above means nothing. **")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
