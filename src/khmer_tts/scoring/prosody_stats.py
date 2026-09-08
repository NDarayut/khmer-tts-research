"""
Prosody measurement -- what actually differs between these voices.

This does NOT score naturalness. There is no trustworthy automatic naturalness
metric for Khmer (score_naturalness.py demonstrates that UTMOS is inverted
here), and nothing in this file should be read as one. What it does is
*describe* the acoustic properties a listener is responding to when they say a
voice sounds flat, or lively, or rushed -- so that a listening verdict can be
stated in measurable terms instead of only as an impression.

The measures, and why each one:

  f0_std_st        Pitch variation, in SEMITONES. The single best-known
                   correlate of "flat" vs "expressive" delivery. Semitones,
                   not Hz, because pitch perception is logarithmic and these
                   models have different base voices -- a male voice varying
                   20 Hz and a female voice varying 30 Hz can be equally
                   expressive, and only the log scale says so.
  f0_range_st      5th-95th percentile spread. Robust version of the above.
  f0_delta_st      Mean absolute semitone change between adjacent voiced
                   frames -- how *fast* the contour moves, not just how far.
                   A voice can have wide range but move in dull steps.
  voiced_fraction  Share of frames with detectable pitch. Very low values
                   suggest breathiness, creak, or synthesis artifacts.
  energy_cv        Coefficient of variation of frame energy: dynamic range of
                   loudness across the utterance. Monotone delivery is flat in
                   amplitude as well as pitch.
  n_pauses,        Internal silences (excluding leading/trailing). Phrasing.
  pause_fraction   A model that never pauses reads like a list; one that
                   pauses in the wrong places reads worse.

VALIDATION BUILT IN: `mms` was eliminated by ear for flat, robotic prosody.
If these measures work, mms must come out as the flattest -- lowest f0_std_st
and f0_delta_st. That is a falsifiable prediction about a model whose verdict
was reached independently, so it is a real check on the measure rather than a
post-hoc story. Read the mms row first; if it is not flattest, distrust the
rest of the table.

    python evaluation/prosody_stats.py
    python evaluation/prosody_stats.py --model voxcpm2 --model higgs3

Writes evaluation/results/prosody_stats.json. Slow -- pYIN is the bottleneck,
roughly 15-20 minutes for all 400 clips.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from khmer_tts.common import read_wav, resample  # noqa: E402

DEFAULT_MODELS = ("mms", "voxcpm2", "fish-s2", "higgs3")

SR = 16000
FMIN, FMAX = 65.0, 400.0     # covers male and female speech
FRAME_LENGTH = 1024
HOP = 160                    # 10 ms
SILENCE_DB = -40.0
MIN_PAUSE_S = 0.15           # shorter gaps are stop closures, not phrasing


def to_semitones(f0_hz, reference_hz):
    """Semitones relative to the utterance's own median pitch. Referencing to
    the utterance rather than a fixed value is what makes male and female
    voices comparable on the same scale."""
    return 12.0 * np.log2(f0_hz / reference_hz)


def pauses(audio, sr):
    """Internal silences only -- leading and trailing are padding, not phrasing."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return 0, 0.0
    thresh = peak * (10.0 ** (SILENCE_DB / 20.0))
    loud = np.abs(audio) > thresh
    if not loud.any():
        return 0, 0.0
    idx = np.flatnonzero(loud)
    inner = loud[idx[0]: idx[-1] + 1]

    count = 0
    total = 0
    run = 0
    for is_loud in inner:
        if is_loud:
            if run >= MIN_PAUSE_S * sr:
                count += 1
                total += run
            run = 0
        else:
            run += 1
    speech_len = max(len(inner), 1)
    return count, total / float(speech_len)


def measure(path):
    audio, sr = read_wav(path)
    if sr != SR:
        audio = resample(audio, sr, SR)
    if audio.size < FRAME_LENGTH:
        return None

    f0, voiced_flag, _ = __import__("librosa").pyin(
        audio, fmin=FMIN, fmax=FMAX, sr=SR,
        frame_length=FRAME_LENGTH, hop_length=HOP,
    )
    voiced = f0[np.isfinite(f0)]
    row = {"voiced_fraction": float(np.mean(voiced_flag)) if voiced_flag.size else 0.0}

    if voiced.size >= 10:
        ref = float(np.median(voiced))
        st = to_semitones(voiced, ref)
        row["f0_median_hz"] = ref
        row["f0_std_st"] = float(np.std(st))
        row["f0_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        # contour speed: only across frames that are voiced back-to-back
        contig = np.isfinite(f0)
        both = contig[:-1] & contig[1:]
        if both.any():
            steps = 12.0 * np.abs(np.log2(f0[1:][both] / f0[:-1][both]))
            row["f0_delta_st"] = float(np.mean(steps))
        else:
            row["f0_delta_st"] = None
    else:
        row.update(f0_median_hz=None, f0_std_st=None, f0_range_st=None, f0_delta_st=None)

    frames = audio[: (audio.size // HOP) * HOP].reshape(-1, HOP)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    loud = energy > np.max(energy) * (10.0 ** (SILENCE_DB / 20.0))
    speech_energy = energy[loud]
    row["energy_cv"] = (
        float(np.std(speech_energy) / np.mean(speech_energy))
        if speech_energy.size and np.mean(speech_energy) > 0 else None
    )

    n_pause, pause_frac = pauses(audio, SR)
    row["n_pauses"] = n_pause
    row["pause_fraction"] = float(pause_frac)
    return row


def med(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def main():
    ap = argparse.ArgumentParser(description="Prosody description, not a naturalness score")
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
        for i, entry in enumerate(entries, start=1):
            wav = audio_dir / f"{entry['id']}.wav"
            if not wav.exists():
                continue
            row = measure(wav)
            if row is None:
                continue
            row["id"] = entry["id"]
            rows.append(row)
            if i % 25 == 0:
                print(f"  {model_key}: {i}/{len(entries)}", file=sys.stderr, flush=True)
        out[model_key] = {
            "n": len(rows),
            "f0_median_hz": med([r["f0_median_hz"] for r in rows]),
            "f0_std_st": med([r["f0_std_st"] for r in rows]),
            "f0_range_st": med([r["f0_range_st"] for r in rows]),
            "f0_delta_st": med([r["f0_delta_st"] for r in rows]),
            "voiced_fraction": med([r["voiced_fraction"] for r in rows]),
            "energy_cv": med([r["energy_cv"] for r in rows]),
            "n_pauses": med([float(r["n_pauses"]) for r in rows]),
            "pause_fraction": med([r["pause_fraction"] for r in rows]),
            "utterances": rows,
        }
        print(f"  {model_key} done ({len(rows)} clips)", file=sys.stderr)

    path = ROOT / "evaluation" / "results" / "prosody_stats.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'model':9s} {'F0 med Hz':>10s} {'F0 std st':>10s} {'F0 rng st':>10s} "
          f"{'F0 dlt st':>10s} {'voiced':>7s} {'energyCV':>9s} {'pauses':>7s}")
    print("-" * 78)
    for model_key, s in out.items():
        def f(key, w=10, d=2):
            v = s[key]
            return f"{v:{w}.{d}f}" if v is not None else " " * (w - 3) + "n/a"
        print(f"{model_key:9s} {f('f0_median_hz')} {f('f0_std_st')} {f('f0_range_st')} "
              f"{f('f0_delta_st')} {f('voiced_fraction', 7)} {f('energy_cv', 9)} "
              f"{f('n_pauses', 7, 1)}")
    print("\nSanity check: mms was eliminated by ear for flat prosody -- it should have "
          "the lowest f0_std_st / f0_delta_st. If it does not, distrust this table.")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
