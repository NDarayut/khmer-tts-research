"""
Signal-level diagnostics for the synthesized clips.

Everything here is arithmetic on the waveform: duration, speaking rate, level,
clipping, silence padding, and high-band energy. No ASR, no MOS predictor, no
learned model of any kind -- so unlike CER (whose scorer has heard VoxCPM2,
see score_cer_khmer.py) and unlike UTMOS/DNSMOS (trained on English and
Japanese MOS studies), nothing in this file can be biased toward a model.

Two jobs:

1. **Catch truncation.** A model that silently drops half a sentence still
   sounds clean, and the naturalness predictors reward it for that -- this is
   how fish-s2 earned the best UTMOS in the run while being unusable
   (CLAUDE.md). Characters-per-second against the model's own median is a
   cheap, reliable detector: it needs no reference audio and no transcript.

2. **Rule confounds in or out of the UTMOS comparison.** UTMOS is sensitive to
   level and to leading/trailing silence. Before concluding that it disagrees
   with a listener about *naturalness*, check that it is not simply responding
   to loudness or padding, which are trivially fixable and not a property of
   the synthesis.

    python evaluation/audio_stats.py
    python evaluation/audio_stats.py --model voxcpm2 --model higgs3

Writes evaluation/results/audio_stats.json and prints a summary table.
Reads eval-set/eval.json for reference text; never writes to it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from khmer_tts.common import read_wav, resample  # noqa: E402
from khmer_tts.scoring.metrics.khmer_text import normalize  # noqa: E402

DEFAULT_MODELS = ("mms", "voxcpm2", "fish-s2", "higgs3")

SILENCE_DB = -40.0      # below this, relative to peak, counts as silence
FRAME = 512
TRUNCATION_RATIO = 1.6  # chars/sec this far above a model's own median = suspect


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-10))


def silence_edges(audio, threshold_db=SILENCE_DB):
    """-> (leading, trailing) seconds-worth of samples below the threshold."""
    if audio.size == 0:
        return 0, 0
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio.size, audio.size
    loud = np.abs(audio) > peak * (10.0 ** (threshold_db / 20.0))
    if not loud.any():
        return audio.size, audio.size
    idx = np.flatnonzero(loud)
    return int(idx[0]), int(audio.size - 1 - idx[-1])


def high_band_ratio(audio, sample_rate, cutoff=6000.0):
    """Fraction of spectral energy above `cutoff`. A model whose output is
    band-limited (or upsampled from a lower rate) shows a small value here."""
    if audio.size < FRAME:
        return None
    window = np.hanning(FRAME)
    hops = range(0, audio.size - FRAME, FRAME // 2)
    total = high = 0.0
    freqs = np.fft.rfftfreq(FRAME, d=1.0 / sample_rate)
    mask = freqs >= cutoff
    for start in hops:
        spectrum = np.abs(np.fft.rfft(audio[start:start + FRAME] * window)) ** 2
        total += spectrum.sum()
        high += spectrum[mask].sum()
    return float(high / total) if total > 0 else None


def clip_fraction(audio, ceiling=0.999):
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= ceiling))


def measure(path, reference):
    audio, sample_rate = read_wav(path)
    seconds = len(audio) / float(sample_rate)
    lead, trail = silence_edges(audio)
    chars = len(normalize(reference))
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    speech = audio[lead: audio.size - trail] if audio.size - trail > lead else audio
    speech_seconds = len(speech) / float(sample_rate)
    return {
        "seconds": seconds,
        "speech_seconds": speech_seconds,
        "lead_silence_s": lead / float(sample_rate),
        "trail_silence_s": trail / float(sample_rate),
        "ref_chars": chars,
        "chars_per_sec": chars / speech_seconds if speech_seconds > 0 else None,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "peak_dbfs": float(db(np.max(np.abs(audio)))) if audio.size else None,
        "rms": rms,
        "rms_dbfs": float(db(rms)) if rms > 0 else None,
        "clip_fraction": clip_fraction(audio),
        "dc_offset": float(np.mean(audio)) if audio.size else 0.0,
        # measured at a common rate so models with different native rates are
        # compared over the same band
        "high_band_ratio_16k": high_band_ratio(resample(audio, sample_rate, 16000), 16000),
        "native_sample_rate": sample_rate,
    }


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", dest="models")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    models = args.models or list(DEFAULT_MODELS)
    entries = json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    out = {}
    for model_key in models:
        audio_dir = ROOT / "evaluation" / "results" / model_key / "audio"
        rows = []
        for entry in entries:
            wav = audio_dir / f"{entry['id']}.wav"
            if not wav.exists():
                continue
            row = measure(wav, entry["sentence"])
            row.update(id=entry["id"], group=entry.get("group"),
                       category=entry.get("category"))
            rows.append(row)

        rate_median = median([r["chars_per_sec"] for r in rows])
        for row in rows:
            cps = row["chars_per_sec"]
            row["rate_vs_median"] = cps / rate_median if cps and rate_median else None
            row["suspect_truncation"] = bool(
                row["rate_vs_median"] and row["rate_vs_median"] >= TRUNCATION_RATIO
            )

        suspects = [r for r in rows if r["suspect_truncation"]]
        out[model_key] = {
            "n": len(rows),
            "median_seconds": median([r["seconds"] for r in rows]),
            "median_chars_per_sec": rate_median,
            "median_rms_dbfs": median([r["rms_dbfs"] for r in rows]),
            "median_peak_dbfs": median([r["peak_dbfs"] for r in rows]),
            "median_lead_silence_s": median([r["lead_silence_s"] for r in rows]),
            "median_trail_silence_s": median([r["trail_silence_s"] for r in rows]),
            "median_high_band_ratio_16k": median([r["high_band_ratio_16k"] for r in rows]),
            "max_clip_fraction": max([r["clip_fraction"] for r in rows], default=0.0),
            "n_suspect_truncation": len(suspects),
            "suspect_ids": [r["id"] for r in suspects],
            "utterances": rows,
        }

    path = ROOT / "evaluation" / "results" / "audio_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    head = (f"{'model':9s} {'dur':>6s} {'ch/s':>6s} {'RMS dB':>7s} {'peak dB':>8s} "
            f"{'lead s':>7s} {'trail s':>8s} {'HF>6k':>7s} {'clip':>6s} {'trunc?':>7s}")
    print(head)
    print("-" * len(head))
    for model_key, s in out.items():
        print(f"{model_key:9s} {s['median_seconds']:6.2f} {s['median_chars_per_sec']:6.2f} "
              f"{s['median_rms_dbfs']:7.1f} {s['median_peak_dbfs']:8.1f} "
              f"{s['median_lead_silence_s']:7.3f} {s['median_trail_silence_s']:8.3f} "
              f"{s['median_high_band_ratio_16k']:7.4f} {s['max_clip_fraction']:6.4f} "
              f"{s['n_suspect_truncation']:7d}")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
