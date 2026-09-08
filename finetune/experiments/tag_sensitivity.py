"""
Did the model learn to *read* the tag, or only fail to *act* on it?

Every result so far measures generated audio. That conflates two very
different failures:

  A. the adapter never learned any dependence on the tag, or
  B. it learned one, but sampling washes it out (CFG, the diffusion head,
     the euler solver) so nothing survives into the waveform.

This script separates them by going back to the training objective itself.
For each held-out clip it runs one teacher-forced forward pass twice --
once with the clip's true speaker tag, once with the *other* speaker's tag --
and compares `loss/diff`. Everything else, including the diffusion timestep
and the noise, is held identical by reseeding from the row index before each
forward, because the CFM loss is stochastic and an unseeded comparison would
measure noise.

If the model reads the tag at all, the wrong tag must cost it: swapped loss
> correct loss, consistently, over rows. If the two are indistinguishable the
tag is genuinely inert in the model's own objective, and no change to the
sampler could have rescued it.

    python finetune/experiments/tag_sensitivity.py --arm proj
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VoxCPM-src" / "scripts"))

EXP = ROOT / "finetune" / "experiments"
SNAPSHOT = ("/home/pc/.cache/huggingface/hub/models--openbmb--VoxCPM2/"
            "snapshots/32279effe8c19989596f05d353d1447f51d9e915")

from finetune.experiments.probe_conditioning import (  # noqa: E402
    ARM_DATA, VOICE_HIGH, VOICE_LOW, tag_for)


def swap_tag(text, arm):
    """Replace the row's speaker tag with the other speaker's."""
    lo, hi = tag_for(VOICE_LOW), tag_for(VOICE_HIGH)
    other = hi if lo in text else lo
    return text.replace(lo, "").replace(hi, "") + other if ARM_DATA[arm] == "end" \
        else other + text.replace(lo, "").replace(hi, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="proj", choices=("start", "end", "proj", "onset"))
    ap.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()

    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model
    from voxcpm.training.data import HFVoxCPMDataset, BatchProcessor, load_audio_text_datasets

    ckpt = EXP / f"ckpt_{args.arm}" / "latest"
    lora_cfg = LoRAConfig(**json.loads((ckpt / "lora_config.json")
                                       .read_text())["lora_config"])
    model = VoxCPM2Model.from_local(SNAPSHOT, optimize=False, training=True,
                                    lora_config=lora_cfg)
    loaded, skipped = model.load_lora_weights(str(ckpt))
    print(f"loaded {len(loaded)} lora params, skipped {len(skipped)}", file=sys.stderr)
    for n, p in model.named_parameters():
        if not n.startswith("audio_vae.") and not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    model.cuda().eval()

    # Two manifests over the same audio: true tags, and tags swapped.
    src = (EXP / "data" / f"{ARM_DATA[args.arm]}_val.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in src][: args.rows]
    paths = {}
    for cond in ("true", "swap"):
        out = EXP / f"sens_{cond}.jsonl"
        out.write_text("\n".join(
            json.dumps({**r, "text": r["text"] if cond == "true"
                        else swap_tag(r["text"], args.arm)}, ensure_ascii=False)
            for r in rows) + "\n", encoding="utf-8")
        paths[cond] = out
    print(f"true: ...{rows[0]['text'][-24:]}\n"
          f"swap: ...{swap_tag(rows[0]['text'], args.arm)[-24:]}", file=sys.stderr)

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
                torch.manual_seed(1000 + i)          # same t and noise both conds
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(p["text_tokens"], p["text_mask"], p["audio_feats"],
                                p["audio_mask"], p["loss_mask"], p["position_ids"],
                                p["labels"], progress=0.0, sample_generate=False)
                vals.append(float(out["loss/diff"]))
        losses[cond] = np.array(vals)
        print(f"{cond}: mean loss/diff {np.mean(vals):.5f}", file=sys.stderr)

    d = losses["swap"] - losses["true"]
    worse = int((d > 0).sum())
    n = len(d)
    # Sign test against "the tag makes no difference": p = P(>= worse heads).
    from math import comb
    p = sum(comb(n, k) for k in range(worse, n + 1)) / 2 ** n
    print(f"\narm={args.arm}  n={n}")
    print(f"  correct tag : {losses['true'].mean():.5f}")
    print(f"  swapped tag : {losses['swap'].mean():.5f}")
    print(f"  delta       : {d.mean():+.5f}  (swapped minus correct; positive = "
          f"the model uses the tag)")
    print(f"  swapped worse on {worse}/{n} rows, sign-test p = {p:.4g}")
    (EXP / f"sensitivity_{args.arm}.json").write_text(json.dumps(
        {"arm": args.arm, "n": n, "true": losses["true"].tolist(),
         "swap": losses["swap"].tolist(), "delta_mean": float(d.mean()),
         "worse": worse, "p": p}, indent=1))


if __name__ == "__main__":
    main()
