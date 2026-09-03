"""
Score synthesized audio: CER (correctness), UTMOS + DNSMOS (naturalness),
and RTF (speed, carried through from synthesize.py).

Each metric model is loaded once and reused across all utterances -- loading
Whisper-large-v3 per file would dominate the runtime.

Usage:
    python evaluation/score.py --model mms
    python evaluation/score.py --model voxcpm2 --metrics cer
    python evaluation/score.py --model fish-s2 --dnsmos-dir D:/models/dnsmos

Reads:  evaluation/results/<model>/synthesis.json + the wavs it lists
Writes: evaluation/results/<model>/scores.json   (full detail, incl. transcripts)
        evaluation/results/<model>/scores.csv    (one flat row per utterance)

Metrics are independently selectable, so you can score CER today and add
DNSMOS once its .onnx files are in place, without re-synthesizing.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.backends import BACKEND_KEYS
from evaluation.common import (
    METRIC_SAMPLE_RATE,
    ROOT,
    load_entries,
    mean,
    median,
    read_json,
    read_wav,
    resample,
    scores_csv_path,
    scores_path,
    synthesis_path,
    write_json,
)
from evaluation.metrics import METRIC_KEYS

CSV_COLUMNS = [
    "id", "group", "category",
    "cer",
    "utmos",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808",
    "rtf", "synth_seconds", "audio_seconds",
    "transcript",
]


def load_metric_models(metrics, device, dnsmos_dir):
    """Load every requested metric up front, so a missing dependency fails in
    the first second rather than 40 minutes into a run."""
    models = {}
    if "cer" in metrics:
        from evaluation.metrics import cer as cer_metric

        print(f"Loading Whisper-{cer_metric.MODEL_SIZE} ({device}) ...")
        models["cer"] = cer_metric.load_asr(device=device)
    if "utmos" in metrics:
        from evaluation.metrics import utmos as utmos_metric

        print("Loading UTMOS ...")
        models["utmos"] = utmos_metric.load_utmos(device=device)
    if "dnsmos" in metrics:
        from evaluation.metrics import dnsmos as dnsmos_metric

        print("Loading DNSMOS ...")
        models["dnsmos"] = dnsmos_metric.load_dnsmos(
            dnsmos_dir=dnsmos_dir, device=device
        )
    return models


def score_utterance(record, sentence, models, device):
    """-> the synthesis record extended with whichever metrics were requested."""
    out = dict(record)
    wav = ROOT / record["wav"]
    if not wav.is_file():
        out["error"] = f"missing wav: {record['wav']}"
        return out

    audio, sample_rate = read_wav(wav)
    # Every metric here wants 16 kHz mono; the wavs are at each model's native
    # rate (MMS 16k, VoxCPM2 48k, Fish 44.1k), so resample in memory only.
    audio16 = resample(audio, sample_rate, METRIC_SAMPLE_RATE)

    if "cer" in models:
        from evaluation.metrics import cer as cer_metric

        out.update(cer_metric.score(models["cer"], audio16, sentence))
    if "utmos" in models:
        from evaluation.metrics import utmos as utmos_metric

        out["utmos"] = utmos_metric.score(
            models["utmos"], audio16, METRIC_SAMPLE_RATE, device=device
        )
    if "dnsmos" in models:
        from evaluation.metrics import dnsmos as dnsmos_metric

        result = dnsmos_metric.score(models["dnsmos"], audio16, METRIC_SAMPLE_RATE)
        out["dnsmos"] = result
    return out


def summarize(records):
    """Overall plus per-group aggregates. Median alongside mean because CER and
    RTF are both heavy-tailed -- one runaway utterance should not define a
    model's score."""

    def block(subset):
        cers = [r.get("cer") for r in subset]
        return {
            "count": len(subset),
            "cer_mean": mean(cers),
            "cer_median": median(cers),
            "utmos_mean": mean([r.get("utmos") for r in subset]),
            "dnsmos_ovrl_mean": mean(
                [(r.get("dnsmos") or {}).get("ovrl") for r in subset]
            ),
            "dnsmos_sig_mean": mean(
                [(r.get("dnsmos") or {}).get("sig") for r in subset]
            ),
            "dnsmos_bak_mean": mean(
                [(r.get("dnsmos") or {}).get("bak") for r in subset]
            ),
            "dnsmos_p808_mean": mean(
                [(r.get("dnsmos") or {}).get("p808") for r in subset]
            ),
            "rtf_mean": mean([r.get("rtf") for r in subset]),
            "rtf_median": median([r.get("rtf") for r in subset]),
        }

    summary = {"overall": block(records)}
    for key in ("group", "category"):
        buckets = {}
        for record in records:
            buckets.setdefault(record.get(key), []).append(record)
        summary[f"by_{key}"] = {k: block(v) for k, v in sorted(buckets.items())
                                if k is not None}
    return summary


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            dnsmos = record.get("dnsmos") or {}
            row = dict(record)
            row.update(
                {
                    "dnsmos_sig": dnsmos.get("sig"),
                    "dnsmos_bak": dnsmos.get("bak"),
                    "dnsmos_ovrl": dnsmos.get("ovrl"),
                    "dnsmos_p808": dnsmos.get("p808"),
                }
            )
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", required=True, choices=BACKEND_KEYS)
    parser.add_argument(
        "--metrics",
        default=",".join(METRIC_KEYS),
        help=f"comma-separated subset of {','.join(METRIC_KEYS)} (default: all)",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--dnsmos-dir", help="directory holding the DNSMOS .onnx files")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRIC_KEYS]
    if unknown:
        parser.error(f"unknown metric(s): {', '.join(unknown)}")

    path = synthesis_path(args.model)
    if not path.is_file():
        parser.error(
            f"no synthesis found at {path} -- run:\n"
            f"  python evaluation/synthesize.py --model {args.model}"
        )
    synthesis = read_json(path)

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    sentences = {e["id"]: e["sentence"] for e in load_entries()}
    utterances = [u for u in synthesis.get("utterances", []) if "error" not in u]
    failed = len(synthesis.get("utterances", [])) - len(utterances)

    models = load_metric_models(metrics, device, args.dnsmos_dir)

    scored = []
    for index, record in enumerate(utterances, start=1):
        result = score_utterance(
            record, sentences.get(record["id"], ""), models, device
        )
        scored.append(result)
        line = f"  [{index}/{len(utterances)}] {result['id']}"
        if result.get("cer") is not None:
            line += f"  CER {result['cer']:.3f}"
        if result.get("utmos") is not None:
            line += f"  UTMOS {result['utmos']:.2f}"
        if (result.get("dnsmos") or {}).get("ovrl") is not None:
            line += f"  DNSMOS {result['dnsmos']['ovrl']:.2f}"
        if "error" in result:
            line += f"  ! {result['error']}"
        print(line)

    payload = {
        "model": args.model,
        "metrics": metrics,
        "env": dict(synthesis.get("env", {}), scoring_device=device),
        "skipped_failed_synthesis": failed,
        "summary": summarize(scored),
        "utterances": scored,
    }
    write_json(scores_path(args.model), payload)
    write_csv(scores_csv_path(args.model), scored)

    overall = payload["summary"]["overall"]
    print(f"\n{args.model}: {overall['count']} scored, {failed} skipped (synthesis failed)")
    for label, key in (
        ("CER mean", "cer_mean"),
        ("CER median", "cer_median"),
        ("UTMOS", "utmos_mean"),
        ("DNSMOS OVRL", "dnsmos_ovrl_mean"),
        ("RTF median", "rtf_median"),
    ):
        value = overall.get(key)
        if value is not None:
            print(f"  {label:<12} {value:.4f}")
    print(f"Wrote {scores_path(args.model)} and {scores_csv_path(args.model)}")


if __name__ == "__main__":
    main()
