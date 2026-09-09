"""
Find parenthetical prompts that steer Khmer prosody reliably enough to reuse.

WHY THIS EXISTS
---------------
`verify_parenthetical.py` tested one wording per level and found three of the
four axes respond (rate rho +0.59, pitch +0.76, energy +0.61) and one does not:
`var` came out at rho -0.27, p 0.20, and the "flat, monotone" prompt produced
*more* pitch movement than the neutral one. That is a result about four
sentences of English, not about the model's capability. A single wording
failing is the weakest possible evidence that an axis is uncontrollable.

So this script separates the two claims. It sweeps a BANK of wordings per axis
and scores each one on its own, which can distinguish "the model cannot do this"
from "we asked badly".

THE DESIGN, AND WHY IT IS PAIRED
--------------------------------
Sentence-to-sentence variance in every one of these measures is larger than the
effect being looked for, so pooling absolute values across sentences -- what the
first script did -- throws away most of the power. Here every prompt is compared
against the SAME sentence generated at the SAME seed with no parenthetical at
all. The unit of evidence is a within-sentence delta, and the headline number is
not a correlation but a hit rate: on how many of the (sentence, seed) pairs did
this wording move the measure in the direction it asked for?

That hit rate is the thing the user actually wants. A prompt you can "plug into
any text and it works consistently" is precisely a prompt with a high hit rate
and a delta larger than the run-to-run noise -- not a prompt with a good p-value.

THE BARE CONDITION IS THE REFERENCE, NOT A NEUTRAL PHRASE
---------------------------------------------------------
`verify_parenthetical.py` used "(a normal delivery)" as its middle level to keep
the prefix length comparable. That is right for a three-level ordinal sweep and
wrong here: it silently assumes the neutral phrase is itself neutral. The bank
includes the neutral wordings as ordinary entries with `dir = 0`, so how far
they move the measure is a result rather than an assumption, and everything is
referenced to the unprompted sentence.

WHAT IS MEASURED
----------------
Four variation measures are recorded, not one, because `var` is the axis in
doubt and `f0_std_st` is only one operationalisation of "expressive":

    f0_std_st     sd of semitone deviation from the clip's own median
    f0_range_st   5th-95th percentile semitone spread -- robust version
    f0_delta_st   mean absolute semitone step between adjacent voiced frames
                  -- contour SPEED. A voice can be wide but dull.
    energy_cv     coefficient of variation of frame energy -- loudness dynamics

`docs`/CLAUDE.md already record that these statistics cannot be read as a
naturalness score (they ranked `mms` as the most expressive model, which is
backwards). That warning is about using them to judge quality. Using them to
verify that a command moved the quantity it names is exactly what they are for.

RESULTS ARE APPENDED AS THEY ARE PRODUCED
-----------------------------------------
One row per generation, streamed to `rows.jsonl`, and a completed (prompt,
sentence, seed) triple is never regenerated. The sweep takes hours and dies for
ordinary reasons; re-running the command resumes it.

    .venv/bin/python finetune/sweep_prosody_prompts.py --axes var
    .venv/bin/python finetune/sweep_prosody_prompts.py --report-only
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

from finetune.verify_control import (  # noqa: E402
    FMAX, FMIN, FRAME, HOP, KHMER_RE, SILENCE_DB, SR, load_sentences)

# Which measure each axis is *defined* as. `var` additionally reports the three
# alternates above, because that is the axis whose operationalisation is in doubt.
PRIMARY = {"var": "f0_std_st", "rate": "char_rate",
           "pitch": "f0_median_hz", "energy": "rms_dbfs"}
VAR_MEASURES = ("f0_std_st", "f0_range_st", "f0_delta_st", "energy_cv")
ALL_MEASURES = ("char_rate", "f0_median_hz", "rms_dbfs") + VAR_MEASURES

# dir: which way the wording asks the primary measure to move. 0 = "neutral",
# which is a hypothesis about the wording, not an exemption from measurement.
BANK = {
    "var": [
        # -- the wordings the first script used, kept so the two runs join up
        ("flat_monotone",     "(a flat, monotone delivery)", -1),
        ("normal_delivery",   "(a normal delivery)", 0),
        ("lively_expressive", "(a lively, expressive delivery)", +1),

        # -- graded intensity on one adjective pair. If the axis works at all,
        #    these four should come out ordered, and that ordering is a much
        #    stronger result than any single cell.
        ("anim_slight",  "(slightly expressive and animated)", +1),
        ("anim_fairly",  "(fairly expressive and animated)", +1),
        ("anim_very",    "(very expressive and animated)", +1),
        ("anim_extreme", "(extremely expressive and animated)", +1),

        # -- naming the acoustic quantity outright rather than a personality
        ("wide_pitch",   "(with wide pitch variation)", +1),
        ("lots_of_into", "(with a lot of intonation, rising and falling)", +1),
        ("narrow_pitch", "(with very little pitch variation)", -1),
        ("level_tone",   "(in a level, unchanging tone)", -1),

        # -- register/role framing. Prompt-conditioned TTS often responds to a
        #    described SPEAKER where it ignores a described PARAMETER.
        ("storyteller",  "(reading a story to a child, warm and animated)", +1),
        ("sportscaster", "(an excited sports commentator)", +1),
        ("newsreader",   "(a news anchor reading calmly and evenly)", -1),
        ("robot",        "(a robotic, emotionless machine voice)", -1),

        # -- emotion as a proxy. Emotion is on OpenBMB's own documented list;
        #    "expressiveness" is not, so this may be the supported route in.
        ("excited",      "(an excited, enthusiastic tone)", +1),
        ("dramatic",     "(a dramatic, theatrical tone)", +1),
        ("bored",        "(a bored, deadpan tone)", -1),

        # -- Khmer-language wording. Every other prompt here is English text in
        #    front of Khmer text; whether the control channel is language-bound
        #    is a separate question and one line of code to answer.
        ("kh_lively",    "(និយាយដោយមានអារម្មណ៍រំភើប)", +1),
        ("kh_flat",      "(និយាយរាបស្មើគ្មានអារម្មណ៍)", -1),

        # -- ROUND TWO. The first round found no wording that raises pitch
        #    variation, and found why: "expressive" vocabulary is routed into
        #    the PITCH control instead. Animated wordings raised median F0 by
        #    40-77 Hz, and a 50 Hz rise mechanically costs 0.35 st of semitone
        #    spread, which accounted for about half of the apparent inversion.
        #
        #    So this block stops asking for the same thing in more words and
        #    attacks the mechanism. Two strategies:
        #
        #    (a) Pin the pitch inside the prompt, so an expressive request has
        #        nowhere to collapse into. If variation is genuinely absent the
        #        pinned prompts change nothing; if it was only being masked by
        #        the pitch rise, this is where it appears.
        ("pin_normal_wide", "(a normal-pitched voice with wide pitch variation)", +1),
        ("pin_low_expr",    "(a low-pitched voice speaking very expressively)", +1),
        ("pin_same_emph",   "(keeping the same pitch, but with strong emphasis "
                            "on important words)", +1),

        #    (b) Ask for the CORRELATES of expressive speech -- emphasis, stress,
        #        pausing, contrast -- without using expressive vocabulary at all,
        #        so nothing in the wording resembles a pitch command.
        ("emphasis",     "(with strong emphasis and clear stress on key words)", +1),
        ("pauses",       "(reading with dramatic pauses between phrases)", +1),
        ("loud_soft",    "(alternating between loud and soft)", +1),
        ("melody",       "(an audiobook narrator giving each phrase its own melody)", +1),
        ("even_stress",  "(giving every word exactly the same stress and length)", -1),
    ],
    "rate": [
        ("slow",         "(speaking slowly)", -1),
        ("normal_pace",  "(speaking at a normal pace)", 0),
        ("quick",        "(speaking quickly)", +1),
        ("very_slow",    "(speaking very slowly and deliberately)", -1),
        ("very_quick",   "(speaking very quickly, rushed)", +1),
        ("slight_slow",  "(speaking slightly slowly)", -1),
        ("slight_quick", "(speaking slightly quickly)", +1),
    ],
    "pitch": [
        ("low",          "(a low-pitched voice)", -1),
        ("normal_pitch", "(a normal-pitched voice)", 0),
        ("high",         "(a high-pitched voice)", +1),
        ("deep",         "(a deep, low male voice)", -1),
        ("very_high",    "(a very high-pitched voice)", +1),
    ],
    "energy": [
        ("soft",         "(speaking softly, quietly)", -1),
        ("normal_vol",   "(speaking at a normal volume)", 0),
        ("loud",         "(speaking loudly)", +1),
        ("whisper",      "(whispering)", -1),
        ("shout",        "(shouting, projecting the voice)", +1),
    ],
}

BARE = "__bare__"   # the reference condition: sentence with no parenthetical


def measure_clip(audio, sr, text):
    """Superset of verify_control.measure_clip -- same trimming and voicing
    decisions, three extra variation statistics. Kept as its own function
    rather than editing the original, because `verify_control` results are
    already reported in the .docx and its numbers must not move."""
    import librosa

    if sr != SR:
        audio = librosa.resample(np.asarray(audio, np.float32), orig_sr=sr, target_sr=SR)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size < FRAME:
        return None
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        thr = peak * (10.0 ** (SILENCE_DB / 20.0))
        loud = np.flatnonzero(np.abs(audio) > thr)
        if loud.size:
            audio = audio[loud[0]:loud[-1] + 1]
    if audio.size < FRAME:
        return None

    dur = audio.size / SR
    f0 = librosa.yin(audio, fmin=FMIN, fmax=FMAX, sr=SR,
                     frame_length=FRAME, hop_length=HOP)
    frames = audio[: (audio.size // HOP) * HOP].reshape(-1, HOP)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    n = min(len(f0), len(energy))
    f0, energy = f0[:n], energy[:n]
    voiced = (energy > np.max(energy) * (10.0 ** (SILENCE_DB / 20.0))) & \
             (f0 > FMIN * 1.05) & (f0 < FMAX * 0.95)
    if voiced.sum() < 10:
        return None
    vf0 = f0[voiced]
    med = float(np.median(vf0))
    st = 12.0 * np.log2(vf0 / med)
    # Contour speed is only meaningful between frames that are *adjacent* in
    # time; stepping across an unvoiced gap measures the gap, not the contour.
    idx = np.flatnonzero(voiced)
    adj = np.diff(idx) == 1
    steps = np.abs(np.diff(st))[adj] if adj.any() else np.array([])
    loud_e = energy[energy > np.max(energy) * (10.0 ** (SILENCE_DB / 20.0))]
    return {
        "duration": dur,
        "char_rate": len(KHMER_RE.findall(text)) / dur,
        "f0_median_hz": med,
        "f0_std_st": float(np.std(st)),
        "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "f0_delta_st": float(np.mean(steps)) if steps.size else None,
        "energy_cv": float(np.std(loud_e) / (np.mean(loud_e) + 1e-12)),
        "rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(audio ** 2) + 1e-12) + 1e-12)),
    }


def sign_test_p(hits, n):
    """Two-sided exact binomial against p=0.5. Stdlib only -- the same test the
    tag-sensitivity probe uses, so the two report comparable numbers."""
    from math import comb
    if not n:
        return None
    k = max(hits, n - hits)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def generate(args, todo, out_dir, rows_path):
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model

    model = load_model(None)          # base model: this tests the built-in channel
    t0 = time.time()
    with rows_path.open("a", encoding="utf-8") as fh:
        for k, (axis, name, prompt, ent, seed) in enumerate(todo, 1):
            text = ent["sentence"] if prompt is None else prompt + ent["sentence"]
            torch.manual_seed(seed)
            try:
                r = model.generate(text=text, cfg_value=args.cfg_value)
            except Exception as exc:                              # noqa: BLE001
                print(f"  ! {axis}/{name} {ent['id']} s{seed}: {exc}", file=sys.stderr)
                continue
            audio, sr = (r[0], int(r[1])) if isinstance(r, tuple) else (r, 48000)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            if seed == 0 and args.keep_audio:
                w = out_dir / "audio" / axis / f"{ent['id']}_{name}.wav"
                w.parent.mkdir(parents=True, exist_ok=True)
                sf.write(w, audio, sr)
            m = measure_clip(audio, sr, ent["sentence"])
            if m is None:
                print(f"  ! {axis}/{name} {ent['id']} s{seed}: unmeasurable",
                      file=sys.stderr)
                continue
            fh.write(json.dumps({"axis": axis, "prompt_name": name,
                                 "prompt": prompt or "", "id": ent["id"],
                                 "seed": seed, **m}, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el/k:.1f}s/clip  "
                      f"eta {(len(todo)-k)*el/k/60:.0f}m", file=sys.stderr, flush=True)


def report(rows, axes, out_dir):
    by = {(r["axis"], r["prompt_name"], r["id"], r["seed"]): r for r in rows}
    dirs = {(a, n): d for a, bank in BANK.items() for n, _, d in bank}
    prompts = {(a, n): p for a, bank in BANK.items() for n, p, _ in bank}

    doc = {}
    for axis in axes:
        # every (sentence, seed) that has a bare reference to pair against
        refs = {(r["id"], r["seed"]): r for r in rows
                if r["axis"] == axis and r["prompt_name"] == BARE}
        # For `var`, carry the other three axes' measures too. A prompt that
        # moves pitch variation by switching to a different-sounding VOICE is a
        # different result from one that changes delivery, and the two are
        # indistinguishable without looking at what else moved.
        measures = (VAR_MEASURES + ("f0_median_hz", "char_rate", "rms_dbfs")
                    if axis == "var" else (PRIMARY[axis],))
        entries = []
        for name, prompt, d in BANK[axis]:
            cells = [(by[(axis, name, i, s)], ref)
                     for (i, s), ref in refs.items() if (axis, name, i, s) in by]
            if not cells:
                continue
            rec = {"prompt_name": name, "prompt": prompt, "dir": d,
                   "n_pairs": len(cells), "measures": {}}
            for mk in measures:
                deltas = [c[mk] - ref[mk] for c, ref in cells
                          if c.get(mk) is not None and ref.get(mk) is not None]
                if not deltas:
                    continue
                deltas = np.asarray(deltas, float)
                # Hit = moved the way the wording asked. A dir-0 prompt has no
                # asked-for direction, so it is scored on |delta| only.
                hits = (int(np.sum(deltas * d > 0)) if d else None)
                # run-to-run sd: spread across seeds within one sentence, then
                # averaged. This is the noise the effect has to clear.
                per_sent = defaultdict(list)
                for c, _ in cells:
                    if c.get(mk) is not None:
                        per_sent[c["id"]].append(c[mk])
                sd = float(np.mean([np.std(v) for v in per_sent.values() if len(v) > 1])) \
                    if any(len(v) > 1 for v in per_sent.values()) else None
                rec["measures"][mk] = {
                    "median_delta": float(np.median(deltas)),
                    "mean_delta": float(np.mean(deltas)),
                    "hits": hits, "n": int(deltas.size),
                    "hit_rate": (hits / deltas.size if hits is not None else None),
                    "p": (sign_test_p(hits, int(deltas.size)) if hits is not None else None),
                    "run_to_run_sd": sd,
                    # the number that decides usability: effect over noise
                    "snr": (abs(float(np.median(deltas))) / sd if sd else None),
                }
            entries.append(rec)
        doc[axis] = entries

    (out_dir / "prompt_sweep.json").write_text(
        json.dumps({"bank": {a: [{"name": n, "prompt": p, "dir": d}
                                 for n, p, d in BANK[a]] for a in axes},
                    "axes": doc}, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# Which parenthetical prompts actually steer Khmer prosody?", "",
         "Base VoxCPM2, no adapter. Every prompt is paired against the *same* "
         "sentence at the *same* seed with no parenthetical at all, so `delta` is "
         "a within-sentence change and `hit rate` is the fraction of "
         "(sentence, seed) pairs that moved the way the wording asked.", "",
         "`sd` is the spread across seeds of one sentence in one condition -- the "
         "run-to-run noise. `snr` is |delta| / sd. **A prompt is usable when the "
         "hit rate is high and snr exceeds about 1**; a large delta with snr below "
         "1 is a prompt that sometimes works, which is what we are trying to "
         "avoid.", ""]
    for axis in axes:
        measures = (VAR_MEASURES if axis == "var" else (PRIMARY[axis],))
        for mk in measures:
            star = " *(primary)*" if mk == PRIMARY[axis] else ""
            L += [f"## `{axis}` -> {mk}{star}", "",
                  "| prompt | wording | asks | delta | hit rate | p | sd | snr |",
                  "|---|---|---|---|---|---|---|---|"]
            es = [e for e in doc[axis] if mk in e["measures"]]
            es.sort(key=lambda e: -(e["measures"][mk]["median_delta"]))
            for e in es:
                m = e["measures"][mk]
                arrow = {1: "up", -1: "down", 0: "--"}[e["dir"]]
                hr = ("--" if m["hit_rate"] is None
                      else f"{m['hits']}/{m['n']} ({m['hit_rate']*100:.0f}%)")
                p = "--" if m["p"] is None else (f"{m['p']:.4f}" if m["p"] >= 1e-4 else "<1e-4")
                sd = "--" if m["run_to_run_sd"] is None else f"{m['run_to_run_sd']:.2f}"
                snr = "--" if m["snr"] is None else f"{m['snr']:.2f}"
                L.append(f"| `{e['prompt_name']}` | {e['prompt']} | {arrow} | "
                         f"{m['median_delta']:+.2f} | {hr} | {p} | {sd} | {snr} |")
            L.append("")
        if axis == "var":
            L += ["### `var` -> leakage onto the other axes", "",
                  "What else each wording moved. A prompt that changes pitch "
                  "variation by selecting a different-sounding voice, or by "
                  "slowing the speech down, is not a variation control.", "",
                  "| prompt | asks | d f0 median (Hz) | d rate (char/s) | d level (dB) |",
                  "|---|---|---|---|---|"]
            for e in sorted(doc[axis], key=lambda e: e["prompt_name"]):
                m = e["measures"]
                arrow = {1: "up", -1: "down", 0: "--"}[e["dir"]]
                def g(k):
                    return ("--" if k not in m
                            else f"{m[k]['median_delta']:+.2f}")
                L.append(f"| `{e['prompt_name']}` | {arrow} | "
                         f"{g('f0_median_hz')} | {g('char_rate')} | {g('rms_dbfs')} |")
            L.append("")
    (out_dir / "prompt_sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="var")
    ap.add_argument("--sentences", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3,
                    help="seeds per (prompt, sentence); the model card warns "
                         "results vary between runs, so one seed says nothing")
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--keep-audio", action="store_true", default=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out-dir",
                    default=str(ROOT / "finetune" / "results" / "prompt_sweep"))
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in BANK:
            sys.exit(f"unknown axis {a!r}; have {', '.join(BANK)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    done = set()
    rows = []
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
            # the bare reference first: without it nothing in this axis can be scored
            for name, prompt in [(BARE, None)] + [(n, p) for n, p, _ in BANK[axis]]:
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
