"""
CER per prompt, so a "faster" prompt can be told apart from a truncating one.

WHY THIS EXISTS
---------------
`sweep_prosody_prompts.py` measures speaking rate as Khmer characters divided by
clip duration. A generation that drops the end of the sentence therefore scores
as *faster*, and this project has already been burned by exactly that failure:
CLAUDE.md records `audio_stats.py` catching 8 of 100 `fish-s2` utterances
truncated, with UTMOS rewarding them for it. `verify_control.py` carries the
same warning in its docstring and says the rate axis cannot be validated without
a CER guard beside it.

So before any rate prompt goes into a recommended list, it has to be shown that
the text is still all there. A prompt with a large positive rate delta and a
healthy CER is genuinely faster speech; the same delta with a CER blow-up is a
truncation artefact wearing a prompt.

This also guards a second failure the sweep cannot see: a prompt that is *read
aloud* rather than obeyed. The Khmer-language prompts in the bank added about a
second of audio each, which is consistent with the model speaking the
parenthetical. If it is speaking it, the transcript will contain it, and CER
against the Khmer sentence alone will rise sharply.

Only seed 0 is kept as audio by the sweep, so this scores 8 clips per prompt --
enough to separate "fine" from "broken", not enough for a precise CER.

RUN IT WITH THE ASR's INTERPRETER, not this repo's .venv (fairseq2 ships
compiled extensions and pins a different torch):

    /run/media/pc/disk1/streaming_asr/venv/bin/python \
        finetune/score_prompt_sweep_cer.py --axes rate,var
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from khmer_tts.scoring.metrics.khmer_text import cer  # noqa: E402
from khmer_tts.scoring.score_cer_khmer import (  # noqa: E402
    asr_repo, load_asr, load_wav_16k, transcribe)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="rate,var")
    ap.add_argument("--sweep-dir",
                    default=str(ROOT / "finetune" / "results" / "prompt_sweep"))
    ap.add_argument("--asr-repo", default="/run/media/pc/disk1/streaming_asr")
    ap.add_argument("--checkpoint", default="hub_export",
                    help="checkpoint dir under the ASR repo; matches "
                         "score_cer_khmer.py's own default")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    entries = {e["id"]: e["sentence"] for e in
               json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}

    jobs = []
    for axis in (a.strip() for a in args.axes.split(",") if a.strip()):
        for wav in sorted((sweep / "audio" / axis).glob("*.wav")):
            sid, name = wav.stem.split("_", 1)
            if sid in entries:
                jobs.append((axis, name, sid, wav))
    if not jobs:
        sys.exit(f"no audio under {sweep/'audio'}; run the sweep with --keep-audio")
    print(f"{len(jobs)} clips", file=sys.stderr)

    bundle = load_asr(asr_repo(args), args.checkpoint, args.device)

    rows = []
    for k, (axis, name, sid, wav) in enumerate(jobs, 1):
        # load_wav_16k returns (audio, sr); transcribe wants the array alone.
        hyp = transcribe(bundle, load_wav_16k(wav)[0])
        rows.append({"axis": axis, "prompt_name": name, "id": sid,
                     "cer": cer(entries[sid], hyp), "hyp": hyp,
                     "wav": str(wav.relative_to(ROOT))})
        if k % 25 == 0:
            print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)

    out = sweep / "prompt_cer.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    import statistics as st
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["axis"], r["prompt_name"])].append(r["cer"])
    print(f"\n{'axis':8}{'prompt':22}{'n':>4}{'median CER':>12}{'max':>9}")
    for (axis, name), v in sorted(by.items(), key=lambda kv: (kv[0][0], st.median(kv[1]))):
        print(f"{axis:8}{name:22}{len(v):>4}{st.median(v)*100:>11.1f}%{max(v)*100:>8.1f}%")
    print("\nThe Khmer ASR floor on out-of-domain natural speech is 12-15% CER "
          "(CLAUDE.md), and\nthe bare/neutral prompts in the same table are the "
          "within-run reference. Read a prompt\nas broken when it sits well above "
          "its own axis's neutral row, not above zero.")
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
