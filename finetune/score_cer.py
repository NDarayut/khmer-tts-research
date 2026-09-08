"""
CER regression check for the style adapter: did control cost intelligibility?

Adding a control channel is only worth having if the speech is still correct.
This re-uses the project's Khmer CTC scorer -- the same model, the same CER
definition, the same caveats -- but points it at arbitrary directories of wavs
instead of `evaluation/results/<model>/audio`, so the adapter's sweep output
can be scored against the base model's on identical sentences.

The bias notice in evaluation/score_cer_khmer.py applies here in FULL and in a
sharper form: that ASR was trained on ~47 h of VoxCPM2-synthesized Khmer, and
both sides of this comparison are VoxCPM2. That is actually the one situation
where the bias is least harmful -- it lands equally on base and adapter, so a
*difference* between them is more trustworthy than either absolute number. Read
the delta, not the level.

RUN IT WITH THE ASR's INTERPRETER:

    /run/media/pc/disk1/streaming_asr/venv/bin/python finetune/score_cer.py \\
        --dir base=finetune/results/audio/base \\
        --dir lora=finetune/results/audio/lora

Each --dir is scanned recursively for `<eval-id>_<suffix>.wav`; the eval id
picks the reference sentence out of eval-set/eval.json. Results are broken down
`by_axis` as well as overall.

THE `rate` AXIS DEPENDS ON THIS SCRIPT TO MEAN ANYTHING
-------------------------------------------------------
verify_control.py measures speaking rate as characters per second over the
clip's duration, which a model can "improve" simply by truncating the sentence
-- the failure audio_stats.py caught in fish-s2, where UTMOS rewarded 8 clipped
utterances. The per-axis CER here is the guard: if the `rate` group's CER is in
line with the others, the speed-up is real speech; if it spikes, the tag is
buying truncation, not pace.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation.score_cer_khmer as base  # noqa: E402

ID_RE = re.compile(r"^([AB]\d{2})(?:_|\.)")


def summarize(rows):
    return base.summarize(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", action="append", dest="dirs", required=True,
                    metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--asr-repo", default=None)
    ap.add_argument("--checkpoint", default="hub_export")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "finetune" / "results" / "cer.json"))
    args = ap.parse_args()

    entries = {e["id"]: e for e in json.loads(
        (ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}

    targets = {}
    for spec in args.dirs:
        if "=" not in spec:
            sys.exit(f"--dir expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"no such directory: {p}")
        targets[label] = p

    repo = base.asr_repo(args)
    bundle = base.load_asr(repo, args.checkpoint, args.device)

    out = {}
    for label, root in targets.items():
        wavs = sorted(root.rglob("*.wav"))
        print(f"{label}: {len(wavs)} wavs under {root}", file=sys.stderr)
        rows = []
        for i, wav in enumerate(wavs, 1):
            m = ID_RE.match(wav.name)
            if not m or m.group(1) not in entries:
                continue
            ent = entries[m.group(1)]
            audio, native_sr = base.load_wav_16k(wav)
            hyp = base.transcribe(bundle, audio)
            rows.append({
                # repo-relative when it can be, absolute otherwise -- --dir
                # accepts any path, including scratch directories outside ROOT
                "wav": str(wav.relative_to(ROOT)) if wav.is_relative_to(ROOT)
                       else str(wav),
                "id": ent["id"],
                # the sweep encodes the commanded level in the filename suffix
                "variant": wav.stem[len(ent["id"]) + 1:] or None,
                "axis": wav.parent.name,
                "cer": base.char_error_rate(ent["sentence"], hyp),
                "transcript": hyp,
                "reference": ent["sentence"],
                "audio_seconds": len(audio) / float(base.SAMPLE_RATE),
                "native_sample_rate": native_sr,
            })
            if i % 25 == 0:
                print(f"  {label}: {i}/{len(wavs)}", file=sys.stderr, flush=True)
        out[label] = {
            "n_wavs": len(wavs), "summary": summarize(rows),
            "by_axis": {ax: summarize([r for r in rows if r["axis"] == ax])
                        for ax in sorted({r["axis"] for r in rows})},
            # Per-level, which is the breakdown that actually answers the
            # truncation question: `rate/fast` is the cell to look at.
            "by_axis_level": {
                f"{ax}/{lv}": summarize([r for r in rows
                                         if r["axis"] == ax and r["variant"] == lv])
                for ax, lv in sorted({(r["axis"], r["variant"]) for r in rows}
                                     if rows else set())
            },
            "rows": rows,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# CER: does the control tag cost intelligibility?", "",
             "Both columns are VoxCPM2, so the ASR's known bias toward VoxCPM2 "
             "lands equally on each. Read the delta, not the level.", "",
             "| set | n | median CER | mean CER |", "|---|---|---|---|"]
    for label, d in out.items():
        s = d["summary"]
        if s.get("n"):
            lines.append(f"| {label} | {s['n']} | {s['cer_median']*100:.2f}% | "
                         f"{s['cer_mean']*100:.2f}% |")
    cells = sorted({k for d in out.values() for k in d["by_axis_level"]})
    if cells:
        lines += ["", "## Per commanded level", "",
                  "`rate/fast` is the truncation check: a CER spike there means "
                  "the tag bought a dropped ending, not a faster delivery.", "",
                  "| axis/level | " + " | ".join(f"{l} median" for l in out) + " |",
                  "|---|" + "---|" * len(out)]
        for cell in cells:
            vals = []
            for d in out.values():
                s = d["by_axis_level"].get(cell, {})
                vals.append(f"{s['cer_median']*100:.2f}%" if s.get("n") else "--")
            lines.append(f"| `{cell}` | " + " | ".join(vals) + " |")
    md = "\n".join(lines) + "\n"
    (out_path.parent / "cer.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {out_path} and {out_path.parent/'cer.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
