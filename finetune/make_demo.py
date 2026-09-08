"""
Generate a listening set for the style-control adapter, as audio you can play.

Verification (`verify_control.py`) answers "did the axis move?" with a
correlation. This answers the different and equally necessary question: does it
sound like what was asked for? The two are not the same -- a statistically
significant 3 Hz shift is a real effect and an inaudible one -- so this writes
clips a person can actually listen to, always in matched sets.

Every axis is swept across its three levels on the SAME sentence with the SAME
seed, so the only difference between the clips in a row is the commanded level.
The base model is included on the same sentences as a control: it has never seen
the tag, so if a difference is audible there, the difference is not the adapter.

    python finetune/make_demo.py --lora finetune/checkpoints/khmer_style/latest
    python finetune/make_demo.py --lora ... --sentences 3 --voices f2,m5

Writes wavs under finetune/results/demo/ plus demo.json describing every clip,
which build_demo_artifact.py turns into a playable page.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.build_corpus import ANY, LEVELS, SLOTS, make_tag  # noqa: E402
from finetune.synthesize_styled import load_model, synthesize  # noqa: E402

AXES = ("rate", "pitch", "var", "energy")
# Held at mid rather than left unspecified: per-slot dropout makes an all-`any`
# tag 0.27% of training rows, so `any` tests generalisation rather than control.
HOLD = "mid"


def load_sentences(n):
    data = json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))
    rows = data["sentences"] if isinstance(data, dict) else data
    pure = [r for r in rows if r.get("group") == "pure_khmer" or r["id"].startswith("A")]
    return (pure or rows)[:n]


def write_wav(path, audio, sr):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:                      # guard only; never rescale between clips
        audio = audio / peak            # of one comparison, or the energy axis
    sf.write(str(path), audio, sr)      # would be normalised away
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--sentences", type=int, default=2)
    ap.add_argument("--voices", default="f2,m5",
                    help="comma-separated spk tags for the voice comparison")
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "results" / "demo"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    out = Path(args.out_dir)
    sents = load_sentences(args.sentences)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    clips = []

    def gen(model, tag_values, sentence, name):
        torch.manual_seed(args.seed)
        audio, sr = synthesize(model, sentence["sentence"], tag_values)
        p = write_wav(out / f"{name}.wav", audio, sr)
        clips.append({"file": p.name, "sentence_id": sentence["id"],
                      "sentence": sentence["sentence"], "tag": make_tag(tag_values),
                      "duration": round(len(audio) / sr, 2), **tag_values})
        print(f"  {p.name}  ({len(audio)/sr:.2f}s)", file=sys.stderr)

    print("== adapter ==", file=sys.stderr)
    model = load_model(args.lora)
    for e in sents:
        for axis in AXES:
            for level in LEVELS[axis]:
                v = {s: (HOLD if s in AXES else ANY) for s in SLOTS}
                v[axis] = level
                gen(model, v, e, f"lora_{axis}_{level}_{e['id']}")
        for voice in voices:
            v = {s: (HOLD if s in AXES else ANY) for s in SLOTS}
            v["spk"] = voice
            gen(model, v, e, f"lora_spk_{voice}_{e['id']}")

    print("== base model (control: has never seen the tag) ==", file=sys.stderr)
    del model
    torch.cuda.empty_cache()
    base = load_model(None)
    for e in sents:
        for axis in AXES:
            for level in (LEVELS[axis][0], LEVELS[axis][2]):   # extremes only
                v = {s: (HOLD if s in AXES else ANY) for s in SLOTS}
                v[axis] = level
                gen(base, v, e, f"base_{axis}_{level}_{e['id']}")

    meta = {"lora": str(args.lora), "hold": HOLD, "seed": args.seed,
            "axes": list(AXES), "levels": {a: list(LEVELS[a]) for a in AXES},
            "voices": voices, "sentences": sents, "clips": clips}
    (out / "demo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\nwrote {len(clips)} clips and {out/'demo.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
