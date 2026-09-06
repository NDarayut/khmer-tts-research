"""
Naturalness, re-measured under controlled conditions.

WHY THIS EXISTS

The original run scored fish-s2 best on UTMOS (3.75) while it was unusable,
and voxcpm2 worst (2.49) while it was a contender -- which is why CLAUDE.md
records that UTMOS and DNSMOS "measure the wrong thing" for Khmer. That
conclusion is right about fish-s2, whose audio is clean *because* it drops
half the sentence. But it leaves a second question unanswered: between the two
usable models, higgs3 (2.98) outscored voxcpm2 (2.49), and a native Khmer
listener hears the opposite.

Before accepting "the predictor is deaf to Khmer" as the explanation, rule out
the boring ones. audio_stats.py found a candidate: the two models are not
delivered at the same level. voxcpm2 sits at -15.6 dBFS RMS and peaks at
-0.2 dBFS; higgs3 sits at -24.5 dBFS RMS and peaks at -6.8 dBFS -- a ~9 dB
gap. UTMOS and DNSMOS both take a raw waveform and are known to respond to
level and to leading/trailing silence. If the ranking moves when those are
equalized, it was never a judgement about naturalness.

So each model is scored under four conditions, same clips throughout:

    raw        exactly as synthesized -- reproduces the original run
    peak       peak-normalized to -1 dBFS
    loudness   ITU-R BS.1770 loudness-normalized to -23 LUFS
    trimmed    loudness-normalized, then leading/trailing silence removed

A ranking that survives all four is a real (if Khmer-blind) preference. A
ranking that flips is an artifact of delivery level, and the original number
should not be quoted.

Also reports the **paired** comparison, which a mean hides: over the 100
sentences, how often does each model win head-to-head, and by how much. Two
models can have distant means because of a few outliers while being tied on
most utterances.

    python evaluation/score_naturalness.py
    python evaluation/score_naturalness.py --model voxcpm2 --model higgs3

Writes evaluation/results/naturalness.json. Needs the DNSMOS .onnx files in
evaluation/dnsmos_models/ (see metrics/dnsmos.py); UTMOS is fetched from
torch.hub on first use and cached.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# imported as `evaluation.*`, not `metrics.*` -- metrics/dnsmos.py reaches back
# up with `from ..common import`, which needs the package root on the path.
sys.path.insert(0, str(ROOT))

from evaluation.common import median, read_wav, resample  # noqa: E402
from evaluation.metrics import dnsmos as dnsmos_metric  # noqa: E402
from evaluation.metrics import utmos as utmos_metric  # noqa: E402

DEFAULT_MODELS = ("mms", "voxcpm2", "fish-s2", "higgs3")
CONDITIONS = ("raw", "peak", "loudness", "trimmed")

METRIC_SR = 16000
PEAK_TARGET_DBFS = -1.0
LOUDNESS_TARGET_LUFS = -23.0
SILENCE_DB = -40.0


def peak_normalize(audio, target_dbfs=PEAK_TARGET_DBFS):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    return audio * (10.0 ** (target_dbfs / 20.0)) / peak


def loudness_normalize(audio, sample_rate, target_lufs=LOUDNESS_TARGET_LUFS):
    """ITU-R BS.1770 via pyloudnorm, with a peak guard: normalizing quiet
    audio up can push it past full scale, and clipping would be a new artifact
    invented by this script rather than a property of the model."""
    import pyloudnorm

    # BS.1770 needs at least one 400 ms block.
    if audio.size < int(0.4 * sample_rate):
        return peak_normalize(audio)
    meter = pyloudnorm.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return peak_normalize(audio)
    if not np.isfinite(loudness):
        return peak_normalize(audio)
    out = audio * (10.0 ** ((target_lufs - loudness) / 20.0))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out.astype(np.float32)


def trim_silence(audio, threshold_db=SILENCE_DB):
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    loud = np.abs(audio) > peak * (10.0 ** (threshold_db / 20.0))
    if not loud.any():
        return audio
    idx = np.flatnonzero(loud)
    return np.ascontiguousarray(audio[idx[0]: idx[-1] + 1])


def condition_audio(audio, sample_rate, condition):
    if condition == "raw":
        return audio
    if condition == "peak":
        return peak_normalize(audio)
    if condition == "loudness":
        return loudness_normalize(audio, sample_rate)
    if condition == "trimmed":
        return trim_silence(loudness_normalize(audio, sample_rate))
    raise ValueError(condition)


def paired_comparison(rows_a, rows_b, key):
    """Head-to-head over the utterances both models scored."""
    by_id_b = {r["id"]: r for r in rows_b}
    deltas, a_wins, b_wins, ties = [], 0, 0, 0
    for row in rows_a:
        other = by_id_b.get(row["id"])
        if not other or row.get(key) is None or other.get(key) is None:
            continue
        delta = row[key] - other[key]
        deltas.append(delta)
        if abs(delta) < 1e-6:
            ties += 1
        elif delta > 0:
            a_wins += 1
        else:
            b_wins += 1
    if not deltas:
        return None
    return {
        "n": len(deltas),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_rate": a_wins / len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", dest="models")
    ap.add_argument("--condition", action="append", dest="conditions",
                    choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-dnsmos", action="store_true")
    args = ap.parse_args()

    models = args.models or list(DEFAULT_MODELS)
    conditions = args.conditions or list(CONDITIONS)
    entries = json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    print("loading UTMOS...", file=sys.stderr)
    utmos_model = utmos_metric.load_utmos(device=args.device)
    dnsmos_sessions = None
    if not args.no_dnsmos:
        print("loading DNSMOS...", file=sys.stderr)
        dnsmos_sessions = dnsmos_metric.load_dnsmos()

    results = {c: {} for c in conditions}
    for model_key in models:
        audio_dir = ROOT / "evaluation" / "results" / model_key / "audio"
        # read each clip once, score it under every condition
        cache = []
        for entry in entries:
            wav = audio_dir / f"{entry['id']}.wav"
            if wav.exists():
                audio, sample_rate = read_wav(wav)
                cache.append((entry, audio, sample_rate))

        for condition in conditions:
            rows = []
            for entry, audio, sample_rate in cache:
                shaped = condition_audio(audio, sample_rate, condition)
                metric_audio = resample(shaped, sample_rate, METRIC_SR)
                row = {"id": entry["id"], "group": entry.get("group"),
                       "category": entry.get("category")}
                row["utmos"] = utmos_metric.score(
                    utmos_model, metric_audio, METRIC_SR, device=args.device
                )
                if dnsmos_sessions is not None:
                    row.update(dnsmos_metric.score(dnsmos_sessions, metric_audio, METRIC_SR))
                rows.append(row)
            results[condition][model_key] = rows
            print(f"  {model_key:9s} {condition:9s} "
                  f"utmos median={median([r['utmos'] for r in rows]):.3f}",
                  file=sys.stderr)

    summary = {}
    for condition, per_model in results.items():
        summary[condition] = {}
        for model_key, rows in per_model.items():
            entry = {
                "n": len(rows),
                "utmos_mean": float(np.mean([r["utmos"] for r in rows])),
                "utmos_median": median([r["utmos"] for r in rows]),
            }
            if rows and "ovrl" in rows[0]:
                for k in ("sig", "bak", "ovrl", "p808"):
                    vals = [r[k] for r in rows if r.get(k) is not None]
                    if vals:
                        entry[f"dnsmos_{k}_median"] = median(vals)
            summary[condition][model_key] = entry

    pairs = {}
    if "voxcpm2" in models and "higgs3" in models:
        for condition in conditions:
            pairs[condition] = {
                "voxcpm2_vs_higgs3_utmos": paired_comparison(
                    results[condition]["voxcpm2"], results[condition]["higgs3"], "utmos"
                )
            }
            if not args.no_dnsmos:
                pairs[condition]["voxcpm2_vs_higgs3_dnsmos_ovrl"] = paired_comparison(
                    results[condition]["voxcpm2"], results[condition]["higgs3"], "ovrl"
                )

    payload = {
        "conditions": {
            "raw": "as synthesized",
            "peak": f"peak normalized to {PEAK_TARGET_DBFS} dBFS",
            "loudness": f"BS.1770 loudness normalized to {LOUDNESS_TARGET_LUFS} LUFS",
            "trimmed": "loudness normalized, then leading/trailing silence removed",
        },
        "summary": summary,
        "paired": pairs,
        "utterances": results,
    }
    path = ROOT / "evaluation" / "results" / "naturalness.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"{'condition':10s} {'model':9s} {'UTMOS':>7s} {'DNSMOS OVRL':>12s}")
    print("-" * 42)
    for condition in conditions:
        for model_key in models:
            s = summary[condition][model_key]
            ovrl = s.get("dnsmos_ovrl_median")
            print(f"{condition:10s} {model_key:9s} {s['utmos_median']:7.3f} "
                  f"{ovrl if ovrl is None else f'{ovrl:12.3f}'}")
        print()
    if pairs:
        print("paired voxcpm2 vs higgs3 (UTMOS, per utterance):")
        for condition in conditions:
            p = pairs[condition]["voxcpm2_vs_higgs3_utmos"]
            if p:
                print(f"  {condition:10s} voxcpm2 wins {p['a_wins']:3d}/{p['n']} "
                      f"({p['a_win_rate']:5.1%})  mean delta {p['mean_delta']:+.3f}")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
