"""
How many levels of pitch, rate and energy can a prompt actually address?

WHY THIS EXISTS
---------------
`sweep_prosody_prompts.py` established that three axes respond and produced a
usable prompt per direction. It did not answer the next question: can you ask
for a *little* slower as distinct from slower as distinct from much slower, and
have the model deliver three separable things?

That is not the same question as "does this wording work", and the first sweep
already hinted the answer is not free. `(speaking slightly quickly)` moved rate
by +0.18 char/s against a seed-to-seed spread of 1.1 -- a level nobody could
hear, wedged between two that work.

THE CEILING IS SET BY NOISE, NOT BY VOCABULARY
----------------------------------------------
Repeated generations of the *same* prompt on the *same* sentence differ by
about 1.6 char/s, 27 Hz and 5.5 dB. A ladder spanning 4 char/s therefore holds
two or three distinguishable rungs however many phrases are written for it, and
writing nine is a way of hiding that rather than fixing it. So the headline
output here is not a table of nine levels: it is the subset of rungs that stand
at least one noise-width apart, which is the ladder that can actually be used.

FOUR WAYS OF ASKING, NOT ONE
----------------------------
Granularity might live in any of these, and they are tested against each other:

    adverb      extremely / very / (plain) / a little ...
    multiplier  "at 0.6x speed", "3 semitones lower", "at 30% volume"
    scale       "a pace of 1 out of 5"
    framing     a described situation that implies the level -- "from across
                the room", "a child's voice" -- rather than naming it

The multiplier family is the interesting one: if the model reads "0.8x" as a
number it is a genuinely continuous control, and if it does not, the adverb
ladder is the only route and its resolution is whatever it is.

WHAT IS REPORTED
----------------
Per rung, the paired change against the unprompted sentence. Then three things
the per-rung table cannot show:

    monotonic     does the measure follow the ladder's own order (Spearman)
    adjacent      does rung k+1 beat rung k on the same sentence and seed --
                  the only evidence that a step is real rather than nominal
    resolved      the greedily-chosen rungs standing >= 1 seed-sd apart

Rows share `finetune/results/prompt_sweep/rows.jsonl` with the first sweep, so
the unprompted references and the wordings both banks have in common are reused
rather than regenerated.

    .venv/bin/python finetune/sweep_prosody_levels.py --axes rate
    .venv/bin/python finetune/sweep_prosody_levels.py --report-only
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.sweep_prosody_prompts import (  # noqa: E402
    BARE, PRIMARY, measure_clip, sign_test_p)
from finetune.verify_control import load_sentences, spearman  # noqa: E402

UNIT = {"char_rate": "char/s", "f0_median_hz": "Hz", "rms_dbfs": "dB"}

# (name, prompt, family, level) -- level is an ordinal on that family's own
# ladder, 0 at the middle, negative downward. Names that already exist in
# sweep_prosody_prompts.BANK are deliberately reused: those cells are on disk
# and must not be regenerated with different text.
LADDERS = {
    "rate": [
        ("lv_r_adv_m4", "(speaking extremely slowly, drawing every word out)", "adverb", -4),
        ("very_slow",   "(speaking very slowly and deliberately)", "adverb", -3),
        ("slow",        "(speaking slowly)", "adverb", -2),
        ("lv_r_adv_m1", "(speaking a little slowly)", "adverb", -1),
        ("normal_pace", "(speaking at a normal pace)", "adverb", 0),
        ("lv_r_adv_p1", "(speaking a little quickly)", "adverb", 1),
        ("quick",       "(speaking quickly)", "adverb", 2),
        ("very_quick",  "(speaking very quickly, rushed)", "adverb", 3),
        ("lv_r_adv_p4", "(speaking extremely quickly, racing through the words)", "adverb", 4),

        ("lv_r_mul_06", "(speaking at 0.6x speed)", "multiplier", -2),
        ("lv_r_mul_08", "(speaking at 0.8x speed)", "multiplier", -1),
        ("lv_r_mul_10", "(speaking at 1.0x speed)", "multiplier", 0),
        ("lv_r_mul_13", "(speaking at 1.3x speed)", "multiplier", 1),
        ("lv_r_mul_16", "(speaking at 1.6x speed)", "multiplier", 2),

        ("lv_r_sc_1",   "(speaking at a pace of 1 out of 5, where 5 is fastest)", "scale", -2),
        ("lv_r_sc_3",   "(speaking at a pace of 3 out of 5, where 5 is fastest)", "scale", 0),
        ("lv_r_sc_5",   "(speaking at a pace of 5 out of 5, where 5 is fastest)", "scale", 2),
    ],
    "pitch": [
        ("lv_p_adv_m4", "(an extremely low, deep voice)", "adverb", -4),
        ("lv_p_adv_m3", "(a very low-pitched voice)", "adverb", -3),
        ("low",         "(a low-pitched voice)", "adverb", -2),
        ("lv_p_adv_m1", "(a slightly low-pitched voice)", "adverb", -1),
        ("normal_pitch", "(a normal-pitched voice)", "adverb", 0),
        ("lv_p_adv_p1", "(a slightly high-pitched voice)", "adverb", 1),
        ("high",        "(a high-pitched voice)", "adverb", 2),
        ("very_high",   "(a very high-pitched voice)", "adverb", 3),
        ("lv_p_adv_p4", "(an extremely high-pitched voice)", "adverb", 4),

        ("lv_p_st_m6",  "(a voice six semitones lower than normal)", "multiplier", -2),
        ("lv_p_st_m3",  "(a voice three semitones lower than normal)", "multiplier", -1),
        ("lv_p_st_p3",  "(a voice three semitones higher than normal)", "multiplier", 1),
        ("lv_p_st_p6",  "(a voice six semitones higher than normal)", "multiplier", 2),

        ("lv_p_fr_m2",  "(an elderly man speaking)", "framing", -2),
        ("deep",        "(a deep, low male voice)", "framing", -1),
        ("lv_p_fr_p1",  "(a young woman speaking)", "framing", 1),
        ("lv_p_fr_p2",  "(a small child speaking)", "framing", 2),
    ],
    "energy": [
        ("lv_e_adv_m4", "(speaking extremely quietly, almost inaudibly)", "adverb", -4),
        ("lv_e_adv_m3", "(speaking very quietly)", "adverb", -3),
        ("soft",        "(speaking softly, quietly)", "adverb", -2),
        ("lv_e_adv_m1", "(speaking a little quietly)", "adverb", -1),
        ("normal_vol",  "(speaking at a normal volume)", "adverb", 0),
        ("lv_e_adv_p1", "(speaking a little loudly)", "adverb", 1),
        ("loud",        "(speaking loudly)", "adverb", 2),
        ("lv_e_adv_p3", "(speaking very loudly)", "adverb", 3),
        ("lv_e_adv_p4", "(speaking extremely loudly, at full volume)", "adverb", 4),

        ("lv_e_pc_20",  "(speaking at 20% volume)", "multiplier", -2),
        ("lv_e_pc_50",  "(speaking at 50% volume)", "multiplier", -1),
        ("lv_e_pc_100", "(speaking at 100% volume)", "multiplier", 2),

        ("lv_e_fr_m2",  "(speaking from the far side of a large room)", "framing", -2),
        ("lv_e_fr_m1",  "(speaking quietly so as not to wake anyone)", "framing", -1),
        ("lv_e_fr_p2",  "(calling out to someone far away)", "framing", 2),
    ],
}


def resolve_ladder(rungs, sd):
    """Greedily keep rungs that stand at least one seed-sd from the last kept
    one. This is the ladder that can be used: two rungs closer together than
    the run-to-run noise are the same rung wearing two names.

    Sorted by MEASURED delta, not by the rung's nominal position. Greedy
    selection is only optimal along a chain that is ordered by the quantity
    being spaced, and several of these ladders are not monotonic -- ordering by
    the label would understate how many distinct levels the wordings reach.
    """
    keep = []
    for name, prompt, lvl, med in sorted(rungs, key=lambda r: r[3]):
        if not keep or med - keep[-1][3] >= sd:
            keep.append((name, prompt, lvl, med))
    return keep


def generate(args, todo, out_dir, rows_path):
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model

    model = load_model(None)
    t0 = time.time()
    with rows_path.open("a", encoding="utf-8") as fh:
        for k, (axis, name, prompt, ent, seed) in enumerate(todo, 1):
            torch.manual_seed(seed)
            try:
                r = model.generate(text=prompt + ent["sentence"],
                                   cfg_value=args.cfg_value)
            except Exception as exc:                              # noqa: BLE001
                print(f"  ! {axis}/{name} {ent['id']} s{seed}: {exc}", file=sys.stderr)
                continue
            audio, sr = (r[0], int(r[1])) if isinstance(r, tuple) else (r, 48000)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            if seed == 0:
                w = out_dir / "audio" / axis / f"{ent['id']}_{name}.wav"
                w.parent.mkdir(parents=True, exist_ok=True)
                sf.write(w, audio, sr)
            m = measure_clip(audio, sr, ent["sentence"])
            if m is None:
                print(f"  ! {axis}/{name} {ent['id']} s{seed}: unmeasurable",
                      file=sys.stderr)
                continue
            fh.write(json.dumps({"axis": axis, "prompt_name": name,
                                 "prompt": prompt, "id": ent["id"],
                                 "seed": seed, **m}, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el/k:.1f}s/clip  "
                      f"eta {(len(todo)-k)*el/k/60:.0f}m", file=sys.stderr, flush=True)


def report(rows, axes, out_dir):
    by = {(r["axis"], r["prompt_name"], r["id"], r["seed"]): r for r in rows}
    L = ["# How many prosody levels can a prompt address?", "",
         "Base VoxCPM2, no adapter. Every rung is paired against the same sentence "
         "at the same seed with no parenthetical, so `delta` is a within-sentence "
         "change. `sd` is the spread across seeds of one sentence in one cell -- "
         "the noise any step has to clear to be audible.", ""]
    doc = {}

    for axis in axes:
        mk = PRIMARY[axis]
        unit = UNIT[mk]
        refs = {(r["id"], r["seed"]): r for r in rows
                if r["axis"] == axis and r["prompt_name"] == BARE}
        if not refs:
            continue
        fams = defaultdict(list)
        for name, prompt, fam, lvl in LADDERS[axis]:
            fams[fam].append((name, prompt, lvl))

        L += [f"## `{axis}` &rarr; {mk} ({unit})", ""]
        axdoc = {}
        for fam, rungs in fams.items():
            cells = {}
            for name, prompt, lvl in rungs:
                vals = [(by[(axis, name, i, s)], ref)
                        for (i, s), ref in refs.items()
                        if (axis, name, i, s) in by]
                if not vals:
                    continue
                d = np.array([c[mk] - r[mk] for c, r in vals], float)
                per = defaultdict(list)
                for c, _ in vals:
                    per[c["id"]].append(c[mk])
                sd = float(np.mean([np.std(v) for v in per.values() if len(v) > 1])) \
                    if any(len(v) > 1 for v in per.values()) else float("nan")
                cells[name] = {"prompt": prompt, "level": lvl,
                               "median_delta": float(np.median(d)),
                               "n": int(d.size), "sd": sd,
                               "raw": {(c["id"], c["seed"]): c[mk] for c, _ in vals}}
            if not cells:
                continue

            # ordering across the whole family
            ords, vals_ = [], []
            for name, c in cells.items():
                for (i, s), v in c["raw"].items():
                    ords.append(c["level"]); vals_.append(v)
            rho = spearman(ords, vals_)

            # adjacent steps: does rung k+1 beat rung k on the same sentence+seed?
            order = sorted(cells.items(), key=lambda kv: kv[1]["level"])
            adj = []
            for (n0, c0), (n1, c1) in zip(order, order[1:]):
                shared = set(c0["raw"]) & set(c1["raw"])
                if not shared:
                    continue
                wins = sum(1 for k_ in shared if c1["raw"][k_] > c0["raw"][k_])
                adj.append({"from": n0, "to": n1, "wins": wins, "n": len(shared),
                            "p": sign_test_p(wins, len(shared)),
                            "gap": c1["median_delta"] - c0["median_delta"]})

            sd_typ = float(np.nanmean([c["sd"] for c in cells.values()]))
            kept = resolve_ladder([(n, c["prompt"], c["level"], c["median_delta"])
                                   for n, c in cells.items()], sd_typ)
            span = (max(c["median_delta"] for c in cells.values())
                    - min(c["median_delta"] for c in cells.values()))
            axdoc[fam] = {"rho": rho, "sd": sd_typ, "span": span,
                          "resolved": [k[0] for k in kept],
                          "n_resolved": len(kept),
                          "rungs": {n: {kk: vv for kk, vv in c.items() if kk != "raw"}
                                    for n, c in cells.items()},
                          "adjacent": adj}

            L += [f"### family `{fam}` &mdash; rho {rho:+.3f}, "
                  f"span {span:.2f} {unit}, noise {sd_typ:.2f} {unit} "
                  f"({span/sd_typ:.1f} noise-widths)", "",
                  "| level | prompt | delta | vs previous rung |",
                  "|---|---|---|---|"]
            prev = None
            for name, c in order:
                step = ""
                for aj in adj:
                    if aj["to"] == name:
                        p = aj["p"]
                        step = (f"{aj['gap']:+.2f} &middot; {aj['wins']}/{aj['n']} "
                                f"({'<1e-4' if p is not None and p < 1e-4 else f'p {p:.3f}'})")
                L.append(f"| {c['level']:+d} | `{c['prompt']}` | "
                         f"{c['median_delta']:+.2f} {unit} | {step or '&mdash;'} |")
                prev = name
            L += ["",
                  f"**Resolves {len(kept)} of {len(cells)} rungs** at one noise-width "
                  f"apart: " + ", ".join(f"`{k[1]}`" for k in kept), ""]
        doc[axis] = axdoc

    (out_dir / "prompt_levels.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "prompt_levels.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="rate,pitch,energy")
    ap.add_argument("--sentences", type=int, default=8,
                    help="must match the first sweep for its cells to be reused")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out-dir",
                    default=str(ROOT / "finetune" / "results" / "prompt_sweep"))
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in LADDERS:
            sys.exit(f"unknown axis {a!r}; have {', '.join(LADDERS)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    done, rows = set(), []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["axis"], r["prompt_name"], r["id"], r["seed"]))

    if not args.report_only:
        sents = load_sentences(args.sentences)
        todo = []
        for axis in axes:
            for name, prompt, _fam, _lvl in [(BARE, None, None, None)] + LADDERS[axis]:
                if prompt is None:
                    continue
                for ent in sents:
                    for seed in range(args.repeats):
                        if (axis, name, ent["id"], seed) not in done:
                            todo.append((axis, name, prompt, ent, seed))
        print(f"{len(todo)} generations to do ({len(done)} already on disk)",
              file=sys.stderr)
        if todo:
            generate(args, todo, out_dir, rows_path)
            rows = [json.loads(l) for l in
                    rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    report(rows, axes, out_dir)


if __name__ == "__main__":
    main()
