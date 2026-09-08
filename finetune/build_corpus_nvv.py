"""
Turn NonverbalTTS into a VoxCPM2 training manifest with inline `[tag]` markers.

WHY THIS EXISTS
---------------
The experiment this feeds is narrow and deliberately not in Khmer: can VoxCPM2
be taught non-verbal tags it never shipped with? NonverbalTTS is the right test
bed because its annotations are already *text-aligned* -- each event is marked
by an emoji sitting at the position in the transcript where the event occurs,
which is the same inline placement NVSpeech argues for and the same placement
our tags need. The conversion is therefore a rename, not an alignment problem:
emoji -> `[cough]`.

Most of the ten NV types have no counterpart in VoxCPM2's documented inventory
(`[laughing] [laughter] [sigh] [Uhm] [Shh] [Question-*] [Surprise-*]
[Dissatisfaction-hnn]`), which is exactly what makes them a test of *adding*
a tag rather than strengthening one.

THE 50:50 MIX
-------------
ELaTE's result is that fine-tuning on conditioned data alone degrades the base
model, and that mixing conditioned with unconditioned data 50:50 prevents it.
The untagged half here comes from the same corpus -- utterances with no NV
annotation at all -- so speaker and channel statistics match and the only
difference is the presence of events.

    python finetune/build_corpus_nvv.py --out-dir finetune/data-nvv/manifests
"""

import argparse
import io
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Emoji -> tag. Discovered from the corpus (see --report-symbols) and matched
# against the ten types the dataset card lists: breathing, laughter, sighing,
# sneezing, coughing, throat clearing, groaning, grunting, snoring, sniffing.
#
# The tag spellings are ours. Only laughter and sigh overlap VoxCPM2's
# documented inventory; the rest are new strings the model has never been
# trained to associate with anything, which is the point.
SYMBOL_TO_TAG = {
    "\U0001F32C": "[breath]",         # wind blowing face
    "\U0001F923": "[laughing]",       # rolling on the floor laughing
    "\U0001F637": "[cough]",          # face with medical mask
    "\U0001F624": "[sigh]",           # face with look of triumph (huffing)
    "\U0001F927": "[sneeze]",         # sneezing face
    "\U0001F62B": "[groan]",          # tired face
    "\U0001F634": "[snore]",          # sleeping face
    "\U0001F44A": "[grunt]",          # oncoming fist
    "\U0001F444": "[throat-clear]",   # mouth
    "\U0001F443": "[sniff]",          # nose
}

# Which of the above VoxCPM2 already documents. Everything else is novel and
# carries the experimental claim.
DOCUMENTED = {"[laughing]", "[sigh]"}

VARIATION_SELECTOR = "️"
ZWJ = "‍"


def strip_selectors(text):
    return text.replace(VARIATION_SELECTOR, "").replace(ZWJ, "")


def find_symbols(text):
    """Every non-ASCII pictograph in the string, selectors removed."""
    out = []
    for ch in strip_selectors(text or ""):
        if ord(ch) > 0x2000 and not ch.isspace() and not ch.isalnum():
            if unicodedata.category(ch) in ("So", "Sk"):
                out.append(ch)
    return out


def convert_text(text):
    """Emoji -> [tag], preserving position. Returns (text, tags, unknown)."""
    text = strip_selectors(text or "")
    tags, unknown = [], []
    out = []
    for ch in text:
        syms = find_symbols(ch)
        if syms:
            tag = SYMBOL_TO_TAG.get(ch)
            if tag:
                tags.append(tag)
                # Space around the tag so it never fuses with an adjacent word
                # into a different subword sequence than it had in training.
                out.append(f" {tag} ")
            else:
                unknown.append(ch)
        else:
            out.append(ch)
    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    return cleaned, tags, unknown


def iter_rows(parquet_files, columns):
    import pyarrow.parquet as pq

    for f in parquet_files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=columns).to_pylist():
                yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",
                    default=str(ROOT / "finetune" / "data-nvv" / "NonverbalTTS"))
    ap.add_argument("--out-dir",
                    default=str(ROOT / "finetune" / "data-nvv" / "manifests"))
    ap.add_argument("--wav-dir",
                    default=str(ROOT / "finetune" / "data-nvv" / "wavs"))
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=14.0)
    ap.add_argument("--min-dnsmos", type=float, default=3.0,
                    help="the corpus ships a quality score; poor audio teaches "
                         "the adapter noise as readily as it teaches events")
    ap.add_argument("--untagged-ratio", type=float, default=1.0,
                    help="untagged clips per tagged clip. 1.0 is ELaTE's 50:50")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--report-symbols", action="store_true",
                    help="scan and print the symbol inventory, write nothing")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("default/train/*.parquet"))
    if not files:
        sys.exit(f"no train parquets under {data_dir}/default/train/")
    print(f"{len(files)} parquet shards")

    if args.report_symbols:
        counts = Counter()
        for row in iter_rows(files, ["Result"]):
            counts.update(find_symbols(row.get("Result")))
        for ch, n in counts.most_common():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "?"
            mapped = SYMBOL_TO_TAG.get(ch, "** UNMAPPED **")
            print(f"  U+{ord(ch):05X} {ch}  {n:6d}  {mapped:16} {name}")
        return

    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["index", "audio", "Result", "dnsmos", "duration", "speaker_id",
            "data_name", "gender"]
    tagged, untagged = [], []
    stats = Counter()
    unknown_syms = Counter()

    for row in iter_rows(files, cols):
        stats["seen"] += 1
        dur = row.get("duration") or 0.0
        if not (args.min_dur <= dur <= args.max_dur):
            stats["drop_duration"] += 1
            continue
        if (row.get("dnsmos") or 0.0) < args.min_dnsmos:
            stats["drop_dnsmos"] += 1
            continue
        text, tags, unknown = convert_text(row.get("Result"))
        if unknown:
            unknown_syms.update(unknown)
            stats["drop_unknown_symbol"] += 1
            continue
        if not text or len(text) < 8:
            stats["drop_empty_text"] += 1
            continue

        audio = row.get("audio") or {}
        raw = audio.get("bytes")
        if not raw:
            stats["drop_no_audio"] += 1
            continue
        try:
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception:
            stats["drop_unreadable"] += 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != args.sample_rate:
            # The corpus mixes 16 kHz (VoxCeleb) and 48 kHz (Expresso); the
            # AudioVAE encoder takes 16 kHz, so normalise here rather than
            # letting the trainer see two rates.
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sample_rate)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak < 1e-4:
            stats["drop_silent"] += 1
            continue

        name = re.sub(r"[^A-Za-z0-9]+", "_", row["index"]).strip("_")
        path = wav_dir / f"{name}.wav"
        sf.write(path, wav, args.sample_rate)

        rec = {
            "audio": str(path.resolve()),
            "text": text,
            "duration": round(len(wav) / args.sample_rate, 3),
            "tags": tags,
            "speaker_id": row.get("speaker_id"),
            "data_name": row.get("data_name"),
            "dnsmos": row.get("dnsmos"),
        }
        if tags:
            tagged.append(rec)
            stats["tagged"] += 1
            for t in tags:
                stats[f"tag{t}"] += 1
        else:
            untagged.append(rec)
            stats["untagged"] += 1

    # ELaTE's mixing ratio. Untagged clips are shuffled deterministically so a
    # rebuild produces the same manifest.
    import random

    rng = random.Random(0)
    rng.shuffle(untagged)
    keep = int(len(tagged) * args.untagged_ratio)
    untagged_kept = untagged[:keep]

    rows = tagged + untagged_kept
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    for split, items in (("train", train), ("val", val)):
        p = out_dir / f"{split}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(
                    {"audio": r["audio"], "text": r["text"],
                     "duration": r["duration"]}, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(items)} rows -> {p}")

    tag_counts = {k[3:]: v for k, v in stats.items() if k.startswith("tag[")}
    meta = {
        "source": "deepvk/NonverbalTTS",
        "symbol_to_tag": SYMBOL_TO_TAG,
        "documented_by_voxcpm2": sorted(DOCUMENTED),
        "novel_tags": sorted(set(SYMBOL_TO_TAG.values()) - DOCUMENTED),
        "filters": {"min_dur": args.min_dur, "max_dur": args.max_dur,
                    "min_dnsmos": args.min_dnsmos,
                    "untagged_ratio": args.untagged_ratio},
        "counts": dict(stats),
        "tag_counts": tag_counts,
        "n_train": len(train), "n_val": len(val),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
        "unknown_symbols": {f"U+{ord(c):05X}": n for c, n in unknown_syms.items()},
    }
    (out_dir / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{meta['hours']} h over {len(rows)} clips "
          f"({stats['tagged']} tagged / {len(untagged_kept)} untagged)")
    print("events per tag:")
    for t, n in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        mark = "documented" if t in DOCUMENTED else "NOVEL"
        print(f"  {t:16} {n:6d}  {mark}")
    if unknown_syms:
        print(f"\nunmapped symbols dropped {stats['drop_unknown_symbol']} clips:")
        for c, n in unknown_syms.most_common(10):
            print(f"  U+{ord(c):05X} {c} x{n}")


if __name__ == "__main__":
    main()
