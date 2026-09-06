"""
CER for the synthesized clips, scored with a Khmer-capable ASR.

This is the replacement for the Whisper-large-v3 CER in score.py, which is
invalid for Khmer (CLAUDE.md: Whisper collapses into repetition loops, median
CER ~100% for every model). The scorer here is the project author's own
fine-tuned Khmer CTC model -- Meta's Omnilingual ASR 300M encoder with a
grapheme-cluster vocabulary and a self-conditioned CTC head:

    https://huggingface.co/Darayut/Omnilingual-ASR-Khm
    source repo: /run/media/pc/disk1/streaming_asr

Reported CER on held-out *natural* speech: 2.24% (in-domain val),
12.03% (FLEURS), 14.82% (SLR42). Those are the floor to read the numbers here
against -- roughly 12-15% on out-of-domain read speech.

=========================  READ THIS BEFORE USING A NUMBER  ==================
THE SCORER IS NOT NEUTRAL BETWEEN THE MODELS IT IS SCORING.

Its training pool (manifests/train_combined.json) includes ~47 hours of
VoxCPM2-synthesized Khmer:

    data/cs_llm_tts       19,825 rows / 21.7 h   VoxCPM2 (LLM-authored corpus)
    data/cs_synth_packed  14,431 rows / 25.4 h   VoxCPM2 voice cloning

It has heard a great deal of VoxCPM2's exact acoustic signature and zero
Higgs TTS 3. Some of the CER gap between those two models is therefore
acoustic domain familiarity, not synthesis quality, and it points one way:
**the bias favours VoxCPM2.** No VoxCPM2-free checkpoint exists in the ASR
repo -- every saved checkpoint trained on train_combined.json or
train_fleurs_tts.json, and both include the synth shards.

How to read a result honestly:
  * VoxCPM2 ahead  -> confounded. Consistent with the bias; not evidence.
  * Higgs 3 ahead  -> strong. It won against the bias, not with it.
  * mms / fish-s2  -> neither was ever in ASR training, so their numbers are
                      the cleanest calibration points in the table.
Pair this with audio_stats.py, which is ASR-free and cannot be biased at all.
=============================================================================

CER definition is imported unchanged from metrics/khmer_text.py -- the same
normalization the harness has always used (NFC, drop whitespace/ZWSP/
punctuation, Khmer digits to ASCII, lowercase Latin, char edit distance over
reference length). That module is stdlib-only, which is what lets it be
imported from inside the ASR's virtualenv.

RUN IT WITH THE ASR's INTERPRETER, not this repo's .venv -- fairseq2 ships
compiled extensions and pins torch, and this repo's .venv has a different one:

    /run/media/pc/disk1/streaming_asr/venv/bin/python \\
        evaluation/score_cer_khmer.py --model voxcpm2 --model higgs3

Writes evaluation/results/<model>/cer_khmer_asr.json. Never touches
scores.json (the Whisper run is kept for the record) and never touches
eval-set/eval.json.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics.khmer_text import cer as char_error_rate  # noqa: E402
from metrics.khmer_text import normalize  # noqa: E402

DEFAULT_ASR_REPO = Path("/run/media/pc/disk1/streaming_asr")
ASR_REPO_ENV = "KHMER_ASR_REPO"
SAMPLE_RATE = 16000
DEFAULT_MODELS = ("mms", "voxcpm2", "fish-s2", "higgs3")

# Recorded in the output so a stale file is never mistaken for a fresh one.
SCORER_ID = "Darayut/Omnilingual-ASR-Khm (local hub_export)"


def asr_repo(args):
    path = Path(args.asr_repo or os.environ.get(ASR_REPO_ENV) or DEFAULT_ASR_REPO)
    if not (path / "omnilingual_asr_khm").is_dir():
        raise SystemExit(
            f"{path} does not look like the Khmer ASR repo (no omnilingual_asr_khm/).\n"
            f"Pass --asr-repo or set ${ASR_REPO_ENV}."
        )
    return path


def load_asr(repo, checkpoint_dir, device_str=None):
    """Build the CTC model once. Returns (model, tokenizer, device, amp_dtype)."""
    sys.path.insert(0, str(repo))
    import torch
    from safetensors.torch import load_file

    from omnilingual_asr_khm.model import ClusterCTCModel, ModelConfig
    from omnilingual_asr_khm.preprocess import ClusterTokenizer

    ckpt = Path(checkpoint_dir)
    if not ckpt.is_absolute():
        ckpt = repo / ckpt

    state_dict = load_file(ckpt / "model.safetensors")
    model_config = ModelConfig(**json.loads((ckpt / "config.json").read_text()))
    model_config.gradient_checkpointing = False
    tokenizer = ClusterTokenizer.from_file(ckpt / "vocab.json")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise SystemExit(
            f"vocab size {tokenizer.vocab_size} != checkpoint {model_config.vocab_size}"
        )

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Building the architecture pulls the omniASR encoder and prints a rich
    # progress bar over 423 tensors; it is noise in a 400-clip batch run.
    with contextlib.redirect_stdout(io.StringIO()):
        model = ClusterCTCModel(model_config)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), tokenizer, device, amp_dtype


def load_wav_16k(path):
    """-> float32 mono at 16 kHz. Matches the ASR repo's own loader (soxr VHQ),
    so a clip is resampled here exactly as it would be by transcribe.py."""
    import numpy as np
    import soundfile as sf
    import soxr

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="VHQ")
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def transcribe(bundle, audio):
    import torch

    from omnilingual_asr_khm.data import normalize_audio_

    model, tokenizer, device, amp_dtype = bundle
    wave = torch.from_numpy(audio).unsqueeze(0).to(device)
    lens = torch.tensor([wave.shape[1]], device=device)
    normalize_audio_(wave, lens)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not torch.float32
    ):
        logits, frame_lens = model(wave, lens)
    ids = model.greedy_decode(logits, frame_lens)[0]
    return tokenizer.decode(ids)


def score_model(bundle, model_key, entries, quiet=False):
    rows = []
    audio_dir = ROOT / "evaluation" / "results" / model_key / "audio"
    started = time.time()
    for i, entry in enumerate(entries, start=1):
        wav = audio_dir / f"{entry['id']}.wav"
        if not wav.exists():
            rows.append({"id": entry["id"], "error": "missing wav"})
            continue
        audio, native_sr = load_wav_16k(wav)
        hypothesis = transcribe(bundle, audio)
        reference = entry["sentence"]
        rows.append(
            {
                "id": entry["id"],
                "group": entry.get("group"),
                "category": entry.get("category"),
                "cer": char_error_rate(reference, hypothesis),
                "transcript": hypothesis,
                "reference": reference,
                "norm_reference": normalize(reference),
                "norm_hypothesis": normalize(hypothesis),
                "audio_seconds": len(audio) / float(SAMPLE_RATE),
                "native_sample_rate": native_sr,
            }
        )
        if not quiet and i % 20 == 0:
            print(f"  {model_key}: {i}/{len(entries)}", file=sys.stderr, flush=True)
    elapsed = time.time() - started
    if not quiet:
        print(f"  {model_key}: {len(rows)} clips in {elapsed:.1f}s", file=sys.stderr)
    return rows


def summarize(rows):
    values = [r["cer"] for r in rows if r.get("cer") is not None]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(values),
        "cer_mean": sum(values) / len(values),
        "cer_median": median,
        "cer_min": ordered[0],
        "cer_max": ordered[-1],
        "cer_p90": ordered[int(0.9 * (len(ordered) - 1))],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", dest="models",
                    help="model key; repeatable. Default: all four.")
    ap.add_argument("--asr-repo", default=None, help=f"default ${ASR_REPO_ENV} or {DEFAULT_ASR_REPO}")
    ap.add_argument("--checkpoint", default="hub_export",
                    help="export dir with model.safetensors/config.json/vocab.json "
                         "(relative to the ASR repo). Default: hub_export")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N entries only (smoke run)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    models = args.models or list(DEFAULT_MODELS)
    repo = asr_repo(args)
    entries = json.loads((ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    if not args.quiet:
        print(f"scorer: {SCORER_ID} from {repo}/{args.checkpoint}", file=sys.stderr)
    bundle = load_asr(repo, args.checkpoint, args.device)

    for model_key in models:
        rows = score_model(bundle, model_key, entries, quiet=args.quiet)
        out = {
            "model": model_key,
            "scorer": SCORER_ID,
            "scorer_repo": str(repo),
            "scorer_checkpoint": args.checkpoint,
            "cer_definition": "evaluation/metrics/khmer_text.py",
            "bias_warning": (
                "The scoring ASR trained on ~47 h of VoxCPM2-synthesized Khmer and no "
                "Higgs TTS 3. CER comparisons involving voxcpm2 are biased in its favour."
            ),
            "summary": summarize(rows),
            "utterances": rows,
        }
        path = ROOT / "evaluation" / "results" / model_key / "cer_khmer_asr.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        s = out["summary"]
        if s.get("n"):
            print(f"{model_key:9s} n={s['n']:3d}  mean={s['cer_mean']:7.2%}  "
                  f"median={s['cer_median']:7.2%}  p90={s['cer_p90']:7.2%}")
        else:
            print(f"{model_key:9s} no scored utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
