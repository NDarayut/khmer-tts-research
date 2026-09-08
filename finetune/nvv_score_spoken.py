"""
Is the model *performing* the tag, or just reading it out loud?

WHY THIS EXISTS
---------------
The base-model sweep shows every tagged clip running about a second longer than
its untagged control -- for invented tags as much as documented ones. That looks
like a result and is not one. A model that simply pronounces the word "cough"
where the tag sits also produces a longer clip, and duration cannot tell the two
apart.

An English ASR can. If the tag is being spoken, its word appears in the
transcript of the tagged clip and not in the control's. If the tag is being
performed as a vocalization, the transcript stays close to the carrier sentence
and the extra time carries no words.

That gives three outcomes per take, which is the measurement this experiment
actually needs:

  spoken     -- the tag word is in the transcript. The tag is being read.
  performed  -- the transcript matches the control, but the clip is longer.
                Something non-lexical was added.
  inert      -- transcript and duration both match the control. Nothing happened.

This is English-only and that is deliberate: Whisper is unusable for Khmer in
this project (median CER ~100%, documented in the top-level CLAUDE.md), but the
tag-expansion experiment is in English precisely so that instruments like this
one work.

    .venv/bin/python finetune/nvv_score_spoken.py --dir finetune/results/nvv/base
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What a spoken tag would sound like. Whisper will not write "[cough]", it will
# write "cough" or a near neighbour, so match on stems and common variants.
SPOKEN_FORMS = {
    "[laughing]": ["laughing", "laughter", "laugh", "haha", "ha ha"],
    "[sigh]": ["sigh", "sighing", "sighs"],
    "[cough]": ["cough", "coughing", "coughs"],
    "[gasp]": ["gasp", "gasping", "gasps"],
    "[breath]": ["breath", "breathing", "breathe"],
    "[sniff]": ["sniff", "sniffing", "sniffs"],
    "[yawn]": ["yawn", "yawning", "yawns"],
}

# How much longer than its control a clip must run before the extra time counts
# as an event rather than ordinary generation jitter.
PERFORM_THRESHOLD_S = 0.3


def norm(s):
    return re.sub(r"[^a-z ]+", " ", (s or "").lower()).strip()


def contains_form(text, forms):
    t = f" {norm(text)} "
    return any(f" {f} " in t for f in forms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="an nvv_infer.py output dir")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--threshold", type=float, default=PERFORM_THRESHOLD_S)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    # Not transformers.pipeline: constructing an ASR pipeline imports torchcodec,
    # whose CUDA extension is broken in this venv (libnvrtc.so.13 missing). The
    # processor plus the model directly avoids that import entirely.
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    doc = set(man["documented_tags"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype).to(dev).eval()

    print(f"transcribing {len(man['clips'])} clips with {args.model}")
    tx = {}
    for i, c in enumerate(man["clips"], 1):
        # Decode with soundfile and hand the pipeline an array rather than a
        # path: passing a path routes through torchcodec, whose CUDA extension
        # is broken in this venv (libnvrtc.so.13 missing). Whisper wants 16 kHz
        # mono and the clips are 48 kHz.
        wav, sr = sf.read(str(d / c["wav"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        feats = proc(wav, sampling_rate=16000,
                     return_tensors="pt").input_features.to(dev, dtype)
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe",
                                 max_new_tokens=128)
        tx[c["id"]] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if i % 25 == 0:
            print(f"  {i}/{len(man['clips'])}")

    cells = defaultdict(dict)
    for c in man["clips"]:
        cells[(c["tag"], c["carrier_index"], c["seed"])][c["condition"]] = c

    rows = []
    for (tag, ci, seed), v in cells.items():
        if "tagged" not in v or "untagged" not in v:
            continue
        t, u = v["tagged"], v["untagged"]
        t_txt, u_txt = tx[t["id"]], tx[u["id"]]
        forms = SPOKEN_FORMS.get(tag, [tag.strip("[]").lower()])
        # The control's transcript is the reference for "was this word already
        # going to be said" -- the carrier sentences contain no tag words, but
        # checking the control rather than assuming keeps it honest.
        spoken = contains_form(t_txt, forms) and not contains_form(u_txt, forms)
        d_dur = t["duration"] - u["duration"]
        if spoken:
            verdict = "spoken"
        elif d_dur > args.threshold:
            verdict = "performed"
        else:
            verdict = "inert"
        rows.append({"tag": tag, "carrier_index": ci, "seed": seed,
                     "verdict": verdict, "delta_duration": round(d_dur, 3),
                     "tagged_text": t_txt, "untagged_text": u_txt})

    out = {"model": man["model"], "asr": args.model,
           "threshold_s": args.threshold, "takes": rows}
    p = d / "spoken_vs_performed.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    by_tag = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_tag[r["tag"]][r["verdict"]] += 1
    print(f"\n{'tag':16} {'kind':11} {'spoken':>7} {'performed':>10} {'inert':>6}")
    print("-" * 56)
    for tag in list(doc) + man["novel_tags"]:
        b = by_tag.get(tag)
        if not b:
            continue
        kind = "documented" if tag in doc else "INVENTED"
        print(f"{tag:16} {kind:11} {b['spoken']:>7} {b['performed']:>10} "
              f"{b['inert']:>6}")
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
