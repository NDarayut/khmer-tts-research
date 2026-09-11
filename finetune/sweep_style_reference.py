"""
Does a reference RECORDING carry speaking style, where a description does not?

WHY THIS EXISTS
---------------
`sweep_prosody_prompts.py` swept 28 parenthetical wordings and found no control
over pitch modulation: expressive vocabulary is routed into the pitch control
instead, and once that component is removed nothing is left. But that is a
result about ONE of VoxCPM2's three conditioning channels. `_generate` also
takes two audio inputs that experiment never touched:

    prompt_wav_path + prompt_text   continuation mode
    reference_wav_path              voice cloning, "structurally isolated
                                    via ref_audio tokens"

Style transfer by example is how most modern TTS actually does speaking style,
so a null on descriptions says very little about it.

THE DESIGN, AND WHY IT HOLDS THE SPEAKER FIXED
----------------------------------------------
The obvious experiment -- clone from an expressive speaker, clone from a flat
speaker, compare -- cannot answer the question. Those two references differ in
voice as well as style, and voice transfer is a capability the model
unambiguously has, so any difference would be attributable to it.

This project's hand-labelled Khmer corpus makes the clean version possible:
all 20 speakers carry clips in both the `flat` and `lively` variation bands,
with 1-3 semitones of within-speaker separation. So each cell pins ONE speaker
and changes only which band the reference clip is drawn from. A difference in
the output is then style transfer, because the voice was held constant by
construction.

Every generation is paired against the same sentence at the same seed with the
same speaker, so the unit of evidence is a within-speaker within-sentence delta
-- the same pairing the prompt sweep used, for the same reason.

TWO THINGS ARE MEASURED, NOT ONE
--------------------------------
The binary question (does `lively` beat `flat`) and the dose-response question
(does the reference's OWN pitch variation predict the output's) are different
strengths of evidence. A dose-response over many references is much harder to
get by chance than a two-cell difference, so both are reported.

READ THE RESULT WITH THE PROJECT'S OWN WARNING IN HAND
------------------------------------------------------
CLAUDE.md records that global F0 statistics do not track perceived
expressiveness: prosody_stats.py called `mms` the most expressive model when it
was eliminated by ear for flat, robotic prosody. So `f0_std_st` moving is
evidence the channel does SOMETHING, and is not evidence that the output sounds
more expressive. The listening page built from these clips is what settles that,
and it outranks this table.

    .venv/bin/python finetune/sweep_style_reference.py --speakers 6
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.sweep_prosody_prompts import measure_clip, sign_test_p  # noqa: E402
from finetune.verify_control import load_sentences, spearman  # noqa: E402

BANDS = ("flat", "lively")
# "ref"  -> reference_wav_path alone (the documented cloning channel)
# "cont" -> prompt_wav_path + prompt_text (continuation mode)
CHANNELS = ("ref", "cont")


def pick_references(meta_path, n_speakers, per_band, rng, min_dur, max_dur):
    """One speaker per cell, `per_band` clips from each band. Clips are chosen
    from the extremes of each speaker's own distribution rather than at random:
    a `lively` clip that happens to sit near that speaker's median is a weak
    stimulus, and the point is to give the channel the best chance it has."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    by = defaultdict(lambda: defaultdict(list))
    for c in meta["clips"]:
        if min_dur <= c["duration"] <= max_dur:
            by[c["speaker_id"]][c["lab"]["var"]].append(c)

    ok = [s for s, d in by.items()
          if len(d["flat"]) >= per_band and len(d["lively"]) >= per_band]
    ok.sort()
    rng.shuffle(ok)
    chosen = ok[:n_speakers]

    refs = []
    for spk in chosen:
        for band in BANDS:
            pool = sorted(by[spk][band], key=lambda c: c["f0_std_st"])
            # flat -> the flattest; lively -> the liveliest
            take = pool[:per_band] if band == "flat" else pool[-per_band:]
            for j, c in enumerate(take):
                refs.append({"speaker_id": spk, "band": band, "slot": j,
                             "wav": c["wav"], "text": c["text"],
                             "ref_f0_std_st": c["f0_std_st"],
                             "ref_f0_median_hz": c["f0_median_hz"],
                             "ref_duration": c["duration"]})
    return chosen, refs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(ROOT / "finetune" / "data" / "corpus_meta.json"))
    ap.add_argument("--speakers", type=int, default=6)
    ap.add_argument("--per-band", type=int, default=2,
                    help="reference clips per speaker per band")
    ap.add_argument("--sentences", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out-dir",
                    default=str(ROOT / "finetune" / "results" / "style_reference"))
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    out = Path(args.out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"

    rng = random.Random(args.seed)
    speakers, refs = pick_references(args.meta, args.speakers, args.per_band,
                                     rng, args.min_dur, args.max_dur)
    (out / "references.json").write_text(
        json.dumps({"speakers": speakers, "refs": refs}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    sents = load_sentences(args.sentences)

    done, rows = set(), []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["channel"], r["speaker_id"], r["band"], r["slot"],
                          r["id"], r["seed"]))

    if not args.report_only:
        todo = [(ch, rf, ent, sd) for ch in channels for rf in refs
                for ent in sents for sd in range(args.repeats)
                if (ch, rf["speaker_id"], rf["band"], rf["slot"],
                    ent["id"], sd) not in done]
        print(f"{len(speakers)} speakers, {len(refs)} references, "
              f"{len(todo)} generations to do ({len(done)} on disk)",
              file=sys.stderr)

        if todo:
            import soundfile as sf
            import torch
            from finetune.synthesize_styled import load_model

            model = load_model(None)      # base model; this tests a built-in channel
            t0 = time.time()
            with rows_path.open("a", encoding="utf-8") as fh:
                for k, (ch, rf, ent, sd) in enumerate(todo, 1):
                    kw = ({"reference_wav_path": rf["wav"]} if ch == "ref"
                          else {"prompt_wav_path": rf["wav"],
                                "prompt_text": rf["text"]})
                    torch.manual_seed(sd)
                    try:
                        r = model.generate(text=ent["sentence"],
                                           cfg_value=args.cfg_value, **kw)
                    except Exception as exc:                       # noqa: BLE001
                        print(f"  ! {ch}/{rf['speaker_id']}/{rf['band']} "
                              f"{ent['id']} s{sd}: {exc}", file=sys.stderr)
                        continue
                    audio, sr = (r[0], int(r[1])) if isinstance(r, tuple) else (r, 48000)
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    if sd == 0 and rf["slot"] == 0:
                        w = (out / "audio" / ch /
                             f"{ent['id']}_{rf['speaker_id']}_{rf['band']}.wav")
                        w.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(w, audio, sr)
                    m = measure_clip(audio, sr, ent["sentence"])
                    if m is None:
                        continue
                    fh.write(json.dumps(
                        {"channel": ch, "speaker_id": rf["speaker_id"],
                         "band": rf["band"], "slot": rf["slot"], "id": ent["id"],
                         "seed": sd, "ref_f0_std_st": rf["ref_f0_std_st"],
                         "ref_f0_median_hz": rf["ref_f0_median_hz"], **m},
                        ensure_ascii=False) + "\n")
                    fh.flush()
                    if k % 20 == 0 or k == len(todo):
                        el = time.time() - t0
                        print(f"  {k}/{len(todo)}  {el/k:.1f}s/clip  "
                              f"eta {(len(todo)-k)*el/k/60:.0f}m",
                              file=sys.stderr, flush=True)
            rows = [json.loads(l) for l in
                    rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    report(rows, channels, out)


def report(rows, channels, out):
    L = ["# Does a reference recording carry speaking style?", "",
         "Base VoxCPM2, no adapter. Each cell pins one speaker and changes only "
         "which variation band the reference clip is drawn from, so voice is held "
         "constant by construction and any difference is style transfer.", ""]
    doc = {}

    for ch in channels:
        rs = [r for r in rows if r["channel"] == ch]
        if not rs:
            continue
        # Pair flat against lively within (speaker, sentence, seed, slot).
        flat = {(r["speaker_id"], r["id"], r["seed"], r["slot"]): r
                for r in rs if r["band"] == "flat"}
        pairs = [(flat[k], r) for r in rs if r["band"] == "lively"
                 for k in [(r["speaker_id"], r["id"], r["seed"], r["slot"])]
                 if k in flat]
        if not pairs:
            continue
        name = ("reference_wav_path (cloning)" if ch == "ref"
                else "prompt_wav_path + prompt_text (continuation)")
        L += [f"## `{ch}` -- {name}", "",
              "| measure | median delta | lively > flat | p |",
              "|---|---|---|---|"]
        cell = {}
        for mk in ("f0_std_st", "f0_range_st", "f0_delta_st",
                   "f0_median_hz", "char_rate", "rms_dbfs"):
            d = [b[mk] - a[mk] for a, b in pairs
                 if a.get(mk) is not None and b.get(mk) is not None]
            if not d:
                continue
            d = np.asarray(d, float)
            h = int((d > 0).sum())
            p = sign_test_p(h, d.size)
            cell[mk] = {"median_delta": float(np.median(d)), "hits": h,
                        "n": int(d.size), "p": p}
            L.append(f"| `{mk}` | {np.median(d):+.3f} | {h}/{d.size} "
                     f"({h/d.size*100:.0f}%) | "
                     f"{'<1e-4' if p is not None and p < 1e-4 else f'{p:.4f}'} |")

        # Dose-response: does the reference's own variation predict the output's?
        rho = spearman([r["ref_f0_std_st"] for r in rs],
                       [r["f0_std_st"] for r in rs])
        # ...and does it survive holding the speaker fixed? Between-speaker
        # differences could carry the whole correlation otherwise.
        within = []
        by_spk = defaultdict(list)
        for r in rs:
            by_spk[r["speaker_id"]].append(r)
        for spk, v in by_spk.items():
            rr = spearman([x["ref_f0_std_st"] for x in v],
                          [x["f0_std_st"] for x in v])
            if rr is not None:
                within.append(rr)
        cell["_dose"] = {"rho_pooled": rho, "n": len(rs),
                         "rho_within_speaker_median":
                             (float(np.median(within)) if within else None),
                         "n_speakers": len(within)}
        L += ["",
              f"Dose-response: the reference clip's own `f0_std_st` against the "
              f"generated clip's, rho = **{rho:+.3f}** over {len(rs)} generations"
              + (f"; median rho within a single speaker "
                 f"{np.median(within):+.3f} over {len(within)} speakers."
                 if within else "."),
              ""]
        doc[ch] = cell

    L += ["Global F0 statistics do not track perceived expressiveness -- "
          "CLAUDE.md records `prosody_stats.py` calling `mms` the most expressive "
          "model when it was eliminated by ear for robotic prosody. A moving "
          "number here shows the channel does something; only listening shows "
          "what.", ""]
    (out / "style_reference.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "style_reference.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
