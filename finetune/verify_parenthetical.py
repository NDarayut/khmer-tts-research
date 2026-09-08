"""
Does VoxCPM2's built-in parenthetical prompt already control Khmer prosody?

WHY THIS EXISTS
---------------
OpenBMB document a natural-language control channel -- a description in
parentheses at the start of the text, e.g.

    (slightly faster, cheerful tone)your text here

-- and list "gender, age, tone, emotion, pace" as things it steers. If that
already works on Khmer, then the LoRA adapter in this directory is largely
redundant and the honest thing to do is say so and stop. Nobody has measured it:
the mechanism is documented for the model in general and for none of its
languages specifically, and OpenBMB's own model card says results "may vary
between runs", which is a statement that it is not reliable even where supported.

This is the same experiment as verify_control.py, with the tag swapped for a
prompt, so the two produce directly comparable numbers: the same sentences, the
same seed, the same four measurements, the same Spearman correlation against the
commanded level, and the same corpus ceilings for scale.

READING THE RESULT
------------------
rho near +1 means the prompt moves the axis in the direction asked. The number
to compare against is not zero, it is the adapter's rho on the same axis, and
the spread in physical units against the corpus ceiling -- a real but inaudible
1 Hz shift is not control.

Because the model card warns about run-to-run variance, each cell is generated
`--repeats` times with different seeds and the median is taken. A mechanism that
only works one run in three is a different product from one that always works,
and averaging is what exposes that.

    python finetune/verify_parenthetical.py --sentences 10 --repeats 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.verify_control import (  # noqa: E402
    AXES, MEASURE_OF, corpus_ceilings, load_sentences, measure_clip, permutation_p,
    spearman)

# Three levels per axis, worded the way OpenBMB's own examples are worded.
# The middle level is deliberately a real phrase rather than an empty string:
# an empty prompt would change the sequence length as well as the content, and
# the comparison has to isolate the wording.
PROMPTS = {
    "rate":   ["(speaking slowly)", "(speaking at a normal pace)", "(speaking quickly)"],
    "pitch":  ["(a low-pitched voice)", "(a normal-pitched voice)", "(a high-pitched voice)"],
    "var":    ["(a flat, monotone delivery)", "(a normal delivery)",
               "(a lively, expressive delivery)"],
    "energy": ["(speaking softly, quietly)", "(speaking at a normal volume)",
               "(speaking loudly)"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3,
                    help="generations per cell; the median is taken, because the "
                         "model card warns results vary between runs")
    ap.add_argument("--axes", default=",".join(AXES))
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "results" / "parenthetical"))
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sents = load_sentences(args.sentences)
    model = load_model(None)          # base model: no adapter anywhere in this test

    rows = []
    for axis in axes:
        for li, prompt in enumerate(PROMPTS[axis]):
            for ent in sents:
                vals = []
                for rep in range(args.repeats):
                    torch.manual_seed(rep)
                    try:
                        r = model.generate(text=prompt + ent["sentence"],
                                           cfg_value=args.cfg_value)
                    except Exception as exc:                       # noqa: BLE001
                        print(f"  ! {ent['id']} {axis}[{li}] rep{rep}: {exc}",
                              file=sys.stderr)
                        continue
                    audio, sr = (r[0], int(r[1])) if isinstance(r, tuple) else (r, 48000)
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    if rep == 0:
                        w = out / "audio" / axis / f"{ent['id']}_{li}.wav"
                        w.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(w, audio, sr)
                    m = measure_clip(audio, sr, ent["sentence"])
                    if m:
                        vals.append(m)
                if not vals:
                    continue
                med = {k: float(np.median([v[k] for v in vals]))
                       for k in vals[0] if isinstance(vals[0][k], (int, float))}
                spread = float(np.std([v[MEASURE_OF[axis]] for v in vals]))
                rows.append({"axis": axis, "level_ord": li, "prompt": prompt,
                             "id": ent["id"], "n_reps": len(vals),
                             "run_to_run_sd": spread, **med})
                print(f"  {axis}[{li}] {ent['id']}: "
                      f"{med[MEASURE_OF[axis]]:.2f} (sd {spread:.2f} over {len(vals)})",
                      file=sys.stderr)

    # corpus_ceilings returns (speaker F0 map, per-axis ceilings)
    summary = {}
    _, ceilings = corpus_ceilings(axes)
    for axis in axes:
        rs = [r for r in rows if r["axis"] == axis]
        if not rs:
            continue
        key = MEASURE_OF[axis]
        rho = spearman([r["level_ord"] for r in rs], [r[key] for r in rs])
        per = {li: float(np.median([r[key] for r in rs if r["level_ord"] == li]))
               for li in (0, 1, 2)}
        summary[axis] = {
            "measure": key, "spearman_rho": rho,
            "p": permutation_p([r["level_ord"] for r in rs], [r[key] for r in rs]),
            "median_by_level": per,
            "low_to_high": per[2] - per[0],
            "mean_run_to_run_sd": float(np.mean([r["run_to_run_sd"] for r in rs])),
        }

    (out / "parenthetical.json").write_text(json.dumps(
        {"prompts": PROMPTS, "n_sentences": len(sents), "repeats": args.repeats,
         "summary": summary, "corpus_ceilings": ceilings, "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# Does the built-in parenthetical prompt control Khmer prosody?", "",
         f"Base VoxCPM2, no adapter. {len(sents)} sentences x 3 levels x "
         f"{args.repeats} generations, median per cell.", "",
         "| axis | measure | low | mid | high | low->high | corpus ceiling | rho | p | "
         "run-to-run sd |", "|---|---|---|---|---|---|---|---|---|---|"]
    for axis, d in summary.items():
        c = ceilings.get(axis, {})
        L.append(f"| `{axis}` | {d['measure']} | {d['median_by_level'][0]:.2f} | "
                 f"{d['median_by_level'][1]:.2f} | {d['median_by_level'][2]:.2f} | "
                 f"**{d['low_to_high']:+.2f}** | {c.get('spread', float('nan')):+.2f} | "
                 f"{d['spearman_rho']:+.3f} | {d['p']:.4f} | {d['mean_run_to_run_sd']:.2f} |")
    L += ["", "`run-to-run sd` is the standard deviation across repeated generations of "
              "the *same* cell. Where it is comparable to `low->high`, the mechanism is "
              "not reliably controllable even if rho looks positive."]
    (out / "parenthetical.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
