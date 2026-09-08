"""
Does the control tag actually control anything?

This is the falsifiable check the whole fine-tune stands or falls on, and it is
built to be able to say "no". For each style axis it sweeps the three levels
across a set of fixed Khmer sentences, measures the acoustic quantity that axis
is *defined* as, and asks whether the measurement moved in the commanded
direction.

    axis    command                 measured
    rate    slow  / mid / fast      Khmer characters per second
    pitch   low   / mid / high      median F0 (Hz)
    var     flat  / mid / lively    F0 standard deviation (semitones)
    energy  soft  / mid / loud      RMS level (dBFS)

THE CONTROL CONDITION
---------------------
Every sweep is also run on the BASE model with the identical tags. The base
model has never seen the tag format, so it should show no systematic response.
If the base model "responds" too, the effect is coming from the tag's presence
as text -- longer prefix, different tokenisation -- and not from anything the
adapter learned. Reporting the fine-tuned numbers without this control would be
the same mistake this project already documented for UTMOS: a number that moves
for a reason other than the one claimed.

The statistic is Spearman's rho between commanded level (ordinal: 0, 1, 2) and
the measured value, pooled over sentences. Reported per axis, per model. A
working axis is a clearly positive rho on the adapter and a rho near zero on
the base model, and the base column carries a permutation p-value because at
this sample size a null rho of +/-0.4 is unremarkable.

THIS SCRIPT ALONE CANNOT VALIDATE THE `rate` AXIS
-------------------------------------------------
Speaking rate is measured as Khmer characters per second over the clip's own
duration -- so a model that simply *drops the end of the sentence* scores as
faster. That is not a hypothetical failure: `audio_stats.py` caught exactly it
in `fish-s2`, where 8 of 100 utterances were truncated and UTMOS rewarded them
for it. `finetune/score_cer.py` is the guard, and it must be run alongside this
sweep: a `rate:fast` group with a healthy CER is genuinely faster speech, and
one with a CER blow-up is a truncation artefact wearing a control tag. Read the
two results together or neither.

    python finetune/verify_control.py --lora finetune/checkpoints/khmer_style/latest
    python finetune/verify_control.py --lora ... --sentences 24 --axes rate,var

Writes finetune/results/control_sweep.json and a markdown summary.
Audio is kept under finetune/results/audio/ so it can be listened to.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.build_corpus import ANY, LEVELS, SLOTS, make_tag  # noqa: E402

KHMER_RE = re.compile(r"[ក-៝០-៩]")
SR = 16000
FMIN, FMAX = 65.0, 400.0
FRAME, HOP = 1024, 160
SILENCE_DB = -40.0

AXES = ("rate", "pitch", "var", "energy")
MIN_CLIPS_PER_VOICE = 50   # below this a `spk` tag is undertrained, not a voice
# which measured quantity each axis is defined as
MEASURE_OF = {
    "rate": "char_rate",
    "pitch": "f0_median_hz",
    "var": "f0_std_st",
    "energy": "rms_dbfs",
}
UNITS = {
    "char_rate": "khmer chars/s",
    "f0_median_hz": "Hz",
    "f0_std_st": "semitones",
    "rms_dbfs": "dBFS",
}


def measure_clip(audio, sr, text):
    """Same quantities the corpus builder labelled on, recomputed on generated
    audio. Deliberately the same code path shape so the comparison is apples
    to apples."""
    import librosa

    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size < FRAME:
        return None

    # trim leading/trailing silence before rate and level, or padding skews both
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
    return {
        "duration": dur,
        "char_rate": len(KHMER_RE.findall(text)) / dur,
        "f0_median_hz": med,
        "f0_std_st": float(np.std(12.0 * np.log2(vf0 / med))),
        "rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(audio ** 2) + 1e-12) + 1e-12)),
    }


def spearman(x, y):
    """Rank correlation, stdlib-plus-numpy only -- no scipy dependency, and the
    tie handling is explicit rather than inherited."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, float)
        r[order] = np.arange(v.size, dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x, y, n_perm=10000, seed=0):
    """Two-sided permutation p-value for Spearman's rho.

    With ~16 sentences per level, a rank correlation of +/-0.4 arises from noise
    often enough that eyeballing the base-model column is not good enough: the
    control condition's whole job is to support the claim "no response here",
    and that claim needs a number. Permutation rather than a t-approximation
    because the sample is small and the measures are not normal.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 6:
        return None
    obs = spearman(x, y)
    if obs is None:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        r = spearman(x, rng.permutation(y))
        if r is not None and abs(r) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def load_sentences(n, seed=0):
    """Pure-Khmer entries from the project's fixed eval set. Using the frozen
    set means these results sit alongside the existing model comparison instead
    of on a private set of sentences chosen after the fact."""
    entries = json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))
    pure = [e for e in entries if e.get("group") == "pure_khmer"] or entries
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pure))[:n]
    return [pure[int(i)] for i in sorted(idx)]


def held_values(hold):
    """Values for the slots that are NOT being swept.

    `any` looks like the neutral choice, and it is what the first sweep used --
    but per-slot dropout at p=0.15 makes a four-`any` tag vanishingly rare in
    training (15 of 5646 rows, 0.27%), so sweeping one axis with the rest set to
    `any` queries the model with a tag shape it has effectively never seen. That
    is a question about out-of-distribution generalisation, not about whether
    the control was learned.

    `--hold mid` pins the other style slots to their middle level and the
    speaker to a fixed voice, producing a fully-specified tag -- 44% of the
    training set. That is the in-distribution question, and it has to be asked
    first: if the control does not work here it does not work at all.
    """
    if hold == "any":
        return {s: ANY for s in SLOTS}
    values = {s: LEVELS[s][1] for s in SLOTS if s != "spk"}
    values["spk"] = hold if hold != "mid" else ANY
    return values


def run_model(label, lora, sentences, axes, out_dir, seed, cfg_value, hold="any"):
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model, synthesize

    print(f"\n=== {label} ===", file=sys.stderr, flush=True)
    model = load_model(lora)
    rows = []
    total = len(sentences) * sum(len(LEVELS[a]) for a in axes)
    k = 0
    for axis in axes:
        for li, level in enumerate(LEVELS[axis]):
            values = held_values(hold)
            values[axis] = level
            for ent in sentences:
                k += 1
                torch.manual_seed(seed)
                try:
                    audio, sr = synthesize(model, ent["sentence"], values,
                                           cfg_value=cfg_value)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  ! {ent['id']} {axis}={level}: {exc}",
                          file=sys.stderr)
                    continue
                wav = out_dir / label / axis / f"{ent['id']}_{level}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, audio, sr)
                m = measure_clip(audio, sr, ent["sentence"])
                if m is None:
                    continue
                rows.append({"model": label, "axis": axis, "level": level,
                             "level_ord": li, "id": ent["id"],
                             "tag": make_tag(values), "wav": str(wav), **m})
                if k % 10 == 0:
                    print(f"  {k}/{total}", file=sys.stderr, flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def summarise(rows, axes):
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["axis"])].append(r)
    for (model, axis), rs in sorted(by.items()):
        key = MEASURE_OF[axis]
        levels = LEVELS[axis]
        per_level = {lv: [r[key] for r in rs if r["level"] == lv] for lv in levels}
        medians = {lv: (float(np.median(v)) if v else None)
                   for lv, v in per_level.items()}
        ords = [r["level_ord"] for r in rs]
        vals = [r[key] for r in rs]
        rho = spearman(ords, vals)
        pval = permutation_p(ords, vals)
        lo, hi = medians[levels[0]], medians[levels[-1]]
        # Leakage: how much does commanding THIS axis move the OTHER axes'
        # quantities? An axis that only works by dragging everything else with
        # it is not a control, it is a preset. Reported as rho on each of the
        # other measures, on the same clips.
        leak = {other: spearman([r["level_ord"] for r in rs],
                                [r[MEASURE_OF[other]] for r in rs])
                for other in AXES if other != axis}
        out[f"{model}/{axis}"] = {
            "model": model, "axis": axis, "measure": key, "unit": UNITS[key],
            "n": len(rs), "median_by_level": medians,
            "spread_low_to_high": (None if lo is None or hi is None else hi - lo),
            "spearman_rho": rho,
            "p_value": pval,
            "leakage_rho": leak,
            "monotonic": all(
                medians[a] is not None and medians[b] is not None and medians[a] <= medians[b]
                for a, b in zip(levels, levels[1:])
            ),
        }
    return out


def run_speaker_sweep(label, lora, sentences, out_dir, seed, cfg_value, spk_tags):
    """Separate sweep for the `spk` axis.

    Voice selection without a reference clip is the one capability here the base
    model simply does not have, so it deserves its own check. The question is
    not "did a number go up" but "are these actually different voices, and are
    they the voices asked for" -- so it is scored two ways: between-voice spread
    of median F0 (do the tags separate at all?) and correlation against each
    tag's F0 in the training corpus (are they the RIGHT voices?).
    """
    import soundfile as sf
    import torch
    from finetune.synthesize_styled import load_model, synthesize

    print(f"\n=== {label} / spk ({len(spk_tags)} voices) ===", file=sys.stderr,
          flush=True)
    model = load_model(lora)
    rows = []
    for tag in spk_tags:
        values = {s: ANY for s in SLOTS}
        values["spk"] = tag
        for ent in sentences:
            torch.manual_seed(seed)
            try:
                audio, sr = synthesize(model, ent["sentence"], values,
                                       cfg_value=cfg_value)
            except Exception as exc:                          # noqa: BLE001
                print(f"  ! {ent['id']} spk={tag}: {exc}", file=sys.stderr)
                continue
            wav = out_dir / label / "spk" / f"{ent['id']}_{tag}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, audio, sr)
            m = measure_clip(audio, sr, ent["sentence"])
            if m:
                rows.append({"model": label, "axis": "spk", "level": tag,
                             "level_ord": -1, "id": ent["id"],
                             "tag": make_tag(values), "wav": str(wav), **m})
        print(f"  {tag}: done", file=sys.stderr, flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def summarise_speaker(rows, corpus_f0):
    """corpus_f0: {spk_tag: median F0 of that speaker in the training corpus}"""
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    for model, rs in by.items():
        per_tag = defaultdict(list)
        for r in rs:
            per_tag[r["level"]].append(r["f0_median_hz"])
        med = {t: float(np.median(v)) for t, v in per_tag.items() if v}
        if not med:
            continue
        shared = [t for t in med if t in corpus_f0]
        out[model] = {
            "n_voices": len(med),
            "median_f0_by_tag": med,
            "between_voice_spread_hz": max(med.values()) - min(med.values()),
            # does the generated voice track the real speaker it names?
            "rho_vs_corpus_f0": spearman([corpus_f0[t] for t in shared],
                                         [med[t] for t in shared]),
            "n_matched": len(shared),
        }
    return out


def markdown(summary, axes, n_sent, spk_summary=None, corpus_ceiling=None):
    L = ["# Style-control verification", "",
         f"{n_sent} sentences from the frozen eval set, each synthesized at every "
         "level of every axis, with all other slots left `any`.", "",
         "`rho` is Spearman's correlation between the commanded level (0,1,2) and "
         "the measured quantity. The **base** rows are the control: the base model "
         "has never seen these tags, so a non-zero rho there would mean the effect "
         "comes from the prefix as text rather than from the adapter. A working "
         "axis is a clearly positive rho on `lora` and a rho near zero on `base`. "
         "`p` is a two-sided permutation test (10,000 shuffles) against rho = 0; "
         "with this many sentences a null rho of +/-0.4 is not rare, so the base "
         "column needs the p-value, not just the eye.",
         ""]
    if corpus_ceiling:
        L += ["`corpus` is the separation the training labels themselves achieve "
              "-- the ceiling on what the adapter could have learned.", ""]
    for axis in axes:
        key = MEASURE_OF[axis]
        L += [f"## `{axis}` -> {key} ({UNITS[key]})", "",
              "| model | " + " | ".join(LEVELS[axis]) + " | low->high | rho | p |",
              "|---|" + "---|" * (len(LEVELS[axis]) + 3)]
        for model in ("lora", "base"):
            s = summary.get(f"{model}/{axis}")
            if not s:
                continue
            cells = []
            for lv in LEVELS[axis]:
                v = s["median_by_level"][lv]
                cells.append("--" if v is None else f"{v:.2f}")
            sp = s["spread_low_to_high"]
            rho = s["spearman_rho"]
            pv = s.get("p_value")
            L.append(f"| {model} | " + " | ".join(cells) +
                     f" | {'--' if sp is None else f'{sp:+.2f}'}"
                     f" | {'--' if rho is None else f'{rho:+.3f}'}"
                     f" | {'--' if pv is None else (f'{pv:.4f}' if pv >= 1e-4 else '<1e-4')} |")
        if corpus_ceiling and axis in corpus_ceiling:
            c = corpus_ceiling[axis]
            L.append("| *corpus (ceiling)* | " +
                     " | ".join(f"*{c['medians'][lv]:.2f}*" for lv in LEVELS[axis]) +
                     f" | *{c['spread']:+.2f}* | -- | -- |")
        s = summary.get(f"lora/{axis}")
        if s and s.get("leakage_rho"):
            leaks = ", ".join(
                f"{k} {v:+.2f}" for k, v in sorted(s["leakage_rho"].items())
                if v is not None)
            L += ["", f"Leakage into the other axes (lora): {leaks}. "
                      "Near zero means this axis moves what it claims to and "
                      "little else."]
        L.append("")

    if spk_summary:
        L += ["## `spk` -> voice selection", "",
              "The base model has no way to select a voice without a reference "
              "clip, so there is nothing to compare against here; the question is "
              "whether the tags produce *distinct* voices and whether they are the "
              "*right* ones. `rho vs corpus` correlates each tag's generated median "
              "F0 against that speaker's median F0 in the training corpus.", "",
              "| model | voices | F0 spread across tags | rho vs corpus F0 |",
              "|---|---|---|---|"]
        for model, s in sorted(spk_summary.items()):
            rho = s["rho_vs_corpus_f0"]
            L.append(f"| {model} | {s['n_voices']} | "
                     f"{s['between_voice_spread_hz']:.1f} Hz | "
                     f"{'--' if rho is None else f'{rho:+.3f}'} |")
        L.append("")
    return "\n".join(L)


def corpus_ceilings(axes):
    """What the training labels themselves separate. Without this the sweep has
    no scale: a +3 Hz pitch response reads very differently against a +28 Hz
    ceiling than against a +5 Hz one."""
    meta_path = ROOT / "finetune" / "data" / "corpus_meta.json"
    if not meta_path.exists():
        return None, {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = {}
    for axis in axes:
        key = MEASURE_OF[axis]
        g = defaultdict(list)
        for c in meta["clips"]:
            g[c["lab"][axis]].append(c[key])
        levels = LEVELS[axis]
        if not all(g[lv] for lv in levels):
            continue
        meds = {lv: float(np.median(g[lv])) for lv in levels}
        out[axis] = {"medians": meds,
                     "spread": meds[levels[-1]] - meds[levels[0]]}
    spk_f0 = defaultdict(list)
    for c in meta["clips"]:
        spk_f0[c["lab"]["spk"]].append(c["f0_median_hz"])
    # Drop the tail voices. The source corpus has a few speakers contributing
    # only a handful of utterances; asking the adapter to reproduce a voice it
    # saw four times and then scoring it would be testing the corpus, not the
    # method.
    spk_f0 = {t: v for t, v in spk_f0.items() if len(v) >= MIN_CLIPS_PER_VOICE}
    return {t: float(np.median(v)) for t, v in spk_f0.items()}, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--sentences", type=int, default=16)
    ap.add_argument("--axes", default=",".join(AXES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the control condition (not recommended)")
    ap.add_argument("--spk-voices", type=int, default=6,
                    help="how many voices to sweep for the spk axis (0 = skip). "
                         "Chosen to span the corpus F0 range, not at random.")
    ap.add_argument("--spk-sentences", type=int, default=4)
    ap.add_argument("--hold", default="mid",
                    help="what the non-swept slots are set to. 'mid' pins them "
                         "to their middle level (an in-distribution, fully "
                         "specified tag -- 44%% of training); 'any' leaves them "
                         "unspecified, which per-slot dropout makes rare (0.27%% "
                         "of training) and therefore tests generalisation rather "
                         "than whether the control was learned. A speaker tag "
                         "such as f7 pins spk too.")
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "results"))
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    bad = [a for a in axes if a not in AXES]
    if bad:
        sys.exit(f"unknown axes: {bad}; valid: {', '.join(AXES)}")

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences(args.sentences, args.seed)
    print(f"{len(sentences)} sentences x {sum(len(LEVELS[a]) for a in axes)} tags "
          f"x {1 if args.skip_base else 2} models", file=sys.stderr)

    corpus_f0, ceilings = corpus_ceilings(axes)

    spk_tags = []
    if args.spk_voices and corpus_f0:
        # Span the corpus F0 range rather than sampling at random: a voice sweep
        # over six similar voices would prove nothing either way.
        ordered = sorted(corpus_f0, key=lambda t: corpus_f0[t])
        idx = np.linspace(0, len(ordered) - 1, args.spk_voices).round().astype(int)
        spk_tags = [ordered[i] for i in sorted(set(idx.tolist()))]
        print(f"spk sweep over {spk_tags} "
              f"({corpus_f0[spk_tags[0]]:.0f}-{corpus_f0[spk_tags[-1]]:.0f} Hz "
              f"in the corpus)", file=sys.stderr)

    print(f"non-swept slots held at: {args.hold} -> "
          f"{make_tag(held_values(args.hold))}", file=sys.stderr)
    rows = run_model("lora", args.lora, sentences, axes, audio_dir,
                     args.seed, args.cfg_value, args.hold)
    spk_rows = []
    if spk_tags:
        spk_rows += run_speaker_sweep("lora", args.lora,
                                      sentences[:args.spk_sentences], audio_dir,
                                      args.seed, args.cfg_value, spk_tags)
    if not args.skip_base:
        rows += run_model("base", None, sentences, axes, audio_dir,
                          args.seed, args.cfg_value, args.hold)
        if spk_tags:
            spk_rows += run_speaker_sweep("base", None,
                                          sentences[:args.spk_sentences],
                                          audio_dir, args.seed, args.cfg_value,
                                          spk_tags)

    summary = summarise(rows, axes)
    spk_summary = summarise_speaker(spk_rows, corpus_f0) if spk_rows else None
    (out_dir / "control_sweep.json").write_text(json.dumps(
        {"n_sentences": len(sentences), "axes": axes, "lora": args.lora,
         "hold": args.hold,
         "corpus_ceilings": ceilings, "corpus_speaker_f0": corpus_f0,
         "summary": summary, "speaker_summary": spk_summary,
         "rows": rows, "speaker_rows": spk_rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    md = markdown(summary, axes, len(sentences), spk_summary, ceilings)
    (out_dir / "control_sweep.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nwrote {out_dir/'control_sweep.json'} and control_sweep.md",
          file=sys.stderr)


if __name__ == "__main__":
    main()
