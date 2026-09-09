"""
Re-synthesize the exact clips the overfit run memorized, with and without the tag.

WHY THIS EXISTS
---------------
The overfitting run asks whether VoxCPM2 can bind an inline tag to a local event
at all. The test has to be on the training sentences themselves -- that is what
overfitting means -- and it has to include the real recording, because the real
recording is the target the model was asked to reproduce.

So each row here produces four things:

  original    the ground-truth clip from NonverbalTTS. The answer key.
  with tag    the memorized sentence, tag inline where the event is.
  no tag      the same sentence with the tag deleted. The control.
  base model  the same tagged sentence before any training.

If the model learned the tag, "with tag" should carry the event and "no tag"
should not, on sentences it has seen hundreds of times. If those two are still
indistinguishable after memorizing 60 clips, the mechanism does not work and no
corpus fixes it.

    python finetune/nvv_infer_overfit.py --lora finetune/checkpoints/nvv_overfit/latest
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z-]*\]")


def strip_tags(text):
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--meta", default=str(ROOT / "finetune" / "data-nvv" /
                                          "overfit" / "overfit_meta.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--cfg-value", type=float, default=2.0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    clips = meta["clips"]
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.lora or 'BASE'}   {len(clips)} memorized clips "
          f"x {len(seeds)} seeds x 2 conditions")
    model = load_model(args.lora)

    records = []
    t0 = time.time()
    n = 0
    for c in clips:
        tagged = c["text"]
        untagged = strip_tags(tagged)
        for seed in seeds:
            for cond, text in (("tagged", tagged), ("untagged", untagged)):
                cid = f"{c['overfit_index']:03d}_{c['tag'].strip('[]')}_{seed}_{cond}"
                path = audio_dir / f"{cid}.wav"
                n += 1
                if path.exists() and path.stat().st_size > 0:
                    info = sf.info(path)
                    records.append({**_row(c, seed, cond, text, cid, path, out_dir),
                                    "sample_rate": info.samplerate,
                                    "duration": info.frames / info.samplerate})
                    continue
                torch.manual_seed(seed)
                np.random.seed(seed)
                res = model.generate(text=text, cfg_value=args.cfg_value,
                                     inference_timesteps=10)
                wav, sr = (res if isinstance(res, tuple) else (res, 48000))
                if hasattr(wav, "detach"):
                    wav = wav.detach().cpu().numpy()
                wav = np.asarray(wav, dtype="float32").squeeze()
                sf.write(path, wav, sr)
                records.append({**_row(c, seed, cond, text, cid, path, out_dir),
                                "sample_rate": sr, "duration": len(wav) / sr})
                if n % 20 == 0:
                    print(f"  {n}/{len(clips)*len(seeds)*2} "
                          f"({time.time()-t0:.0f}s)")

    manifest = {"model": args.lora or "base", "cfg_value": args.cfg_value,
                "seeds": seeds, "overfit": True,
                "note": "evaluated on the training sentences; this measures "
                        "memorization, never generalization",
                "clips": records}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} clips and {out_dir/'manifest.json'}")


def _row(c, seed, cond, text, cid, path, out_dir):
    return {"id": cid, "tag": c["tag"], "overfit_index": c["overfit_index"],
            "seed": seed, "condition": cond, "text": text,
            "original": c["original"],
            "wav": str(Path(path).relative_to(out_dir))}


if __name__ == "__main__":
    main()
