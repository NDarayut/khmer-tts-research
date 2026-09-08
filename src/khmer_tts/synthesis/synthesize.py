"""
Synthesize eval-set/eval.json with one TTS model, and measure RTF.

RTF (real-time factor = synthesis wall time / duration of the resulting audio)
can only be measured while generating, never recovered from a .wav afterwards
-- so speed is captured here, alongside the audio, and score.py carries it
through untouched.

Usage:
    python evaluation/synthesize.py --model mms
    python evaluation/synthesize.py --model voxcpm2 --limit 3
    python evaluation/synthesize.py --model fish-s2 --fish-repo D:/src/fish-speech

Writes: evaluation/results/<model>/audio/<id>.wav   (native sample rate)
        evaluation/results/<model>/synthesis.json

Reads eval.json; never writes to it -- the set is fixed (CLAUDE.md).
"""

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from khmer_tts.synthesis.backends import BACKEND_KEYS, get_backend
from khmer_tts.common import (
    audio_seconds,
    load_entries,
    median,
    read_json,
    rel_to_root,
    select_entries,
    synthesis_path,
    wav_path,
    write_json,
    write_wav,
)

# Discarded first call: the very first inference pays for cuDNN autotune, lazy
# CUDA context creation and graph compilation. Charging that to sentence A01
# would inflate its RTF by an order of magnitude.
WARMUP_TEXT = "សួស្តី។"


def build_env_block(backend, device, seed, load_seconds):
    """An RTF number is meaningless without the hardware it was measured on."""
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "seed": seed,
        "load_seconds": round(load_seconds, 3),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    env.update(backend.describe())
    try:
        import torch

        env["torch_version"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def synthesize_entry(backend, entry, model, overwrite):
    """-> a per-utterance record. Never raises: a model that fails on one
    sentence should not throw away the other 99."""
    entry_id = entry["id"]
    path = wav_path(model, entry_id)
    record = {
        "id": entry_id,
        "group": entry.get("group"),
        "category": entry.get("category"),
        "wav": rel_to_root(path),
    }

    if path.is_file() and not overwrite:
        record["skipped"] = "already synthesized"
        return record, True

    try:
        start = time.perf_counter()
        audio, sample_rate = backend.synthesize(entry["sentence"])
        synth_seconds = time.perf_counter() - start
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, False

    duration = audio_seconds(audio, sample_rate)
    write_wav(path, audio, sample_rate)

    record.update(
        {
            "sample_rate": int(sample_rate),
            "synth_seconds": round(synth_seconds, 4),
            "audio_seconds": round(duration, 4),
            "rtf": round(synth_seconds / duration, 4) if duration > 0 else None,
        }
    )
    if duration <= 0:
        record["error"] = "model returned empty audio"
    return record, False


def merge_previous(model, records):
    """Carry forward timings for ids we skipped, so a resumed run still writes
    a complete synthesis.json."""
    path = synthesis_path(model)
    if not path.is_file():
        return records
    previous = {r["id"]: r for r in read_json(path).get("utterances", [])}
    merged = []
    for record in records:
        if "skipped" in record and record["id"] in previous:
            merged.append(previous[record["id"]])
        else:
            merged.append(record)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", required=True, choices=BACKEND_KEYS)
    parser.add_argument("--ids", help="comma-separated eval ids, e.g. A01,B07")
    parser.add_argument("--group", choices=("pure_khmer", "code_switched"))
    parser.add_argument("--limit", type=int, help="first N entries (smoke runs)")
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-synthesize ids that already have a wav")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the discarded warm-up utterance")
    # backend-specific passthrough
    parser.add_argument("--ref-audio", help="voxcpm2: reference wav for cloning")
    parser.add_argument("--ref-text", help="voxcpm2: transcript of --ref-audio")
    parser.add_argument("--fish-repo", help="fish-s2: path to a fish-speech checkout")
    parser.add_argument("--fish-checkpoint", help="fish-s2: path to s2-pro weights")
    args = parser.parse_args()

    entries = select_entries(
        load_entries(),
        ids=[i.strip() for i in args.ids.split(",")] if args.ids else None,
        groups=[args.group] if args.group else None,
        limit=args.limit,
    )
    if not entries:
        parser.error("no entries selected")

    backend = get_backend(
        args.model,
        device=args.device,
        seed=args.seed,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        fish_repo=args.fish_repo,
        fish_checkpoint=args.fish_checkpoint,
    )

    print(f"Loading {args.model} ...")
    start = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - start
    device = backend.device or "cpu"
    print(f"  loaded in {load_seconds:.1f}s on {device} (excluded from RTF)")

    if not args.no_warmup:
        try:
            backend.synthesize(WARMUP_TEXT)
        except Exception as exc:
            print(f"  ! warm-up failed ({type(exc).__name__}: {exc}) -- continuing")

    records = []
    skipped = 0
    for index, entry in enumerate(entries, start=1):
        record, was_skipped = synthesize_entry(
            backend, entry, args.model, args.overwrite
        )
        records.append(record)
        skipped += was_skipped
        if was_skipped:
            continue
        if "error" in record:
            print(f"  [{index}/{len(entries)}] {record['id']}  FAILED: {record['error']}")
        else:
            print(
                f"  [{index}/{len(entries)}] {record['id']}  "
                f"{record['audio_seconds']:.2f}s audio  RTF {record['rtf']:.3f}"
            )

    records = merge_previous(args.model, records)
    rtfs = [r.get("rtf") for r in records if r.get("rtf") is not None]
    payload = {
        "model": args.model,
        "env": build_env_block(backend, device, args.seed, load_seconds),
        "summary": {
            "count": len(records),
            "failed": sum(1 for r in records if "error" in r),
            "rtf_median": round(median(rtfs), 4) if rtfs else None,
            "rtf_mean": round(sum(rtfs) / len(rtfs), 4) if rtfs else None,
        },
        "utterances": records,
    }
    write_json(synthesis_path(args.model), payload)

    print(
        f"\n{args.model}: {len(records)} utterances "
        f"({skipped} already present, {payload['summary']['failed']} failed), "
        f"median RTF {payload['summary']['rtf_median']}"
    )
    print(f"Wrote {synthesis_path(args.model)}")


if __name__ == "__main__":
    main()
