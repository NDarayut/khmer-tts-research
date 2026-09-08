"""
Build the style-labelled Khmer corpus that teaches VoxCPM2 expressive control.

WHAT THIS PRODUCES
------------------
A VoxCPM2 training manifest (JSONL: {"audio", "text", "duration"}) in which the
`text` field is the Khmer transcript prefixed by a control tag:

    <|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។

That prefix is the entire mechanism. VoxCPM2's text path is a single
`text_tokenizer(text)` call -- there is no language channel, no style channel,
no speaker channel to condition on (see docs/09 §9.2). But the training packer
sets `loss_mask` to zero across every text position, so anything put in the
text field is pure conditioning: the model is never asked to *reproduce* it,
only to use it in predicting the audio latents that follow. Prefixing an
attribute string is therefore the cheapest possible control channel, and it
needs no architecture change at all.

WHY THE LABELS ARE MEASURED, NOT ANNOTATED
------------------------------------------
There is no expressive Khmer speech corpus with emotion labels. There is,
however, ~1000 h of real multi-speaker read Khmer in this project's sibling ASR
repo. Every axis below is computed from the waveform itself, which means:

  * the labels are objective and reproducible,
  * we can *verify* the fine-tune afterwards by measuring the same quantity on
    generated audio and checking it moved in the commanded direction.

That last point matters more than it sounds. This project has already
established (CLAUDE.md, "Metrics") that no learned metric ranks Khmer TTS
correctly. A control axis defined as a measurable acoustic quantity dodges that
problem entirely: "did asking for `rate:fast` produce faster speech?" is
arithmetic, not a MOS predictor.

THE FIVE AXES
-------------
  spk     Speaker identity, 20 real voices (f1..f9, m1..m11). Gives voice
          selection with no reference clip -- which the base model cannot do.
  rate    Speaking rate = Khmer characters per second. Ranked GLOBALLY;
          chars/sec means the same thing coming from any voice.
  pitch   Median F0. Ranked WITHIN speaker, because absolute F0 is voice
          identity, not style -- "high" must mean "high for this voice", or
          the axis just re-encodes gender.
  var     F0 standard deviation in SEMITONES: flat / mid / lively. The
          best-known correlate of expressive vs monotone delivery, and the
          axis this whole exercise is really about. Within speaker, same
          reasoning as pitch -- see label_rows() for why the wider global
          ranking was rejected despite measuring better.
  energy  RMS level. Within speaker. Audio is gain-normalised PER SPEAKER,
          not per clip, so within-speaker dynamics survive and this axis
          stays learnable.

Levels are taken from the TAILS, not from tertiles: bottom 15% / middle 70% /
top 15% by default (`--cut`). See percentile_labels() -- balanced tertiles gave
a control range about 1.5x too narrow to be worth shipping.

PER-SLOT DROPOUT
----------------
Each slot is independently replaced by `any` with probability --slot-dropout.
This is what makes partial specification work: at inference you can write
`<|spk:any|rate:any|pitch:any|var:lively|energy:any|>` and control expressive-
ness alone. Without dropout the model only ever sees fully-specified tags and
generalises badly to partial ones.

USAGE
-----
    python finetune/build_corpus.py --clips 6000
    python finetune/build_corpus.py --clips 200 --out-dir finetune/data-pilot
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASR_REPO = Path("/run/media/pc/disk1/streaming_asr")
SHARD_DIR = ASR_REPO / "data" / "dataset" / "train"

SR = 16000
FMIN, FMAX = 65.0, 400.0
FRAME = 1024
HOP = 160                 # 10 ms
SILENCE_DB = -40.0        # relative to clip peak, for silence trimming
KEEP_SILENCE_S = 0.08     # leading/trailing silence to retain after trimming
TARGET_DBFS = -20.0       # per-SPEAKER normalisation target (not per clip)

MIN_DUR, MAX_DUR = 3.0, 12.0

SLOTS = ("spk", "rate", "pitch", "var", "energy")
LEVELS = {
    "rate": ("slow", "mid", "fast"),
    "pitch": ("low", "mid", "high"),
    "var": ("flat", "mid", "lively"),
    "energy": ("soft", "mid", "loud"),
}
ANY = "any"

KHMER_RE = re.compile(r"[ក-៝០-៩]")


# --------------------------------------------------------------------------- #
# tag construction -- the one format the whole project depends on
# --------------------------------------------------------------------------- #
def make_tag(values):
    """values: dict slot -> level. Order is fixed; the model learns positions."""
    return "<|" + "|".join(f"{s}:{values[s]}" for s in SLOTS) + "|>"


TAG_RE = re.compile(r"^<\|(?:[a-z]+:[a-z0-9]+\|)+>")


def strip_tag(text):
    return TAG_RE.sub("", text)


# --------------------------------------------------------------------------- #
# audio measurement
# --------------------------------------------------------------------------- #
def trim_silence(audio):
    """Trim to KEEP_SILENCE_S of padding either side.

    The official VoxCPM FAQ names trailing silence >0.5 s as the single most
    common cause of runaway generation -- the model learns that utterances do
    not end. This is not cosmetic preprocessing.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    thresh = peak * (10.0 ** (SILENCE_DB / 20.0))
    loud = np.flatnonzero(np.abs(audio) > thresh)
    if loud.size == 0:
        return audio
    pad = int(KEEP_SILENCE_S * SR)
    return audio[max(0, loud[0] - pad): min(audio.size, loud[-1] + pad)]


def measure(audio):
    """F0 and level statistics. librosa.yin, not pyin -- pyin's Viterbi decode
    costs ~2.5 s/clip, which is 4 h for this corpus and buys nothing here: we
    only need robust summary statistics, not a frame-accurate contour."""
    import librosa

    if audio.size < FRAME:
        return None
    f0 = librosa.yin(audio, fmin=FMIN, fmax=FMAX, sr=SR,
                     frame_length=FRAME, hop_length=HOP)
    # yin always returns a value; gate on periodicity via frame energy and on
    # the plausible speech band, which is enough to reject silence and noise.
    frames = audio[: (audio.size // HOP) * HOP].reshape(-1, HOP)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    n = min(len(f0), len(energy))
    f0, energy = f0[:n], energy[:n]
    voiced = (energy > np.max(energy) * (10.0 ** (SILENCE_DB / 20.0))) & \
             (f0 > FMIN * 1.05) & (f0 < FMAX * 0.95)
    if voiced.sum() < 10:
        return None

    vf0 = f0[voiced]
    med = float(np.median(vf0))
    st = 12.0 * np.log2(vf0 / med)
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    return {
        "f0_median_hz": med,
        "f0_std_st": float(np.std(st)),
        "rms_dbfs": float(20.0 * np.log10(rms + 1e-12)),
        "voiced_fraction": float(voiced.mean()),
    }


# --------------------------------------------------------------------------- #
# worker: one shard -> a list of candidate rows with audio written to disk
# --------------------------------------------------------------------------- #
def process_shard(job):
    shard_path, wav_dir, row_idxs = job
    import pyarrow.parquet as pq
    import soundfile as sf

    rows = []
    try:
        table = pq.read_table(shard_path)
    except Exception as exc:                      # noqa: BLE001
        print(f"  ! {Path(shard_path).name}: {exc}", file=sys.stderr)
        return rows

    idxs = [i for i in sorted(row_idxs) if i < table.num_rows]
    for rec in table.take(idxs).to_pylist():
        # sentence_id is NOT unique -- the same sentence is read by several
        # speakers and carries the same id in each. Key on speaker+sentence or
        # one speaker's clip silently overwrites another's.
        sid = f"{rec['speaker_id']}_{rec['sentence_id']}"
        try:
            audio, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
        except Exception:                          # noqa: BLE001
            continue
        if sr != SR or audio.ndim != 1:
            continue
        audio = trim_silence(audio)
        dur = audio.size / SR
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue
        stats = measure(audio)
        if stats is None or stats["voiced_fraction"] < 0.25:
            continue

        text = rec["transcript"].strip()
        n_khmer = len(KHMER_RE.findall(text))
        if n_khmer < 10:
            continue

        out = wav_dir / f"{sid}.wav"
        sf.write(out, audio, SR, subtype="PCM_16")
        rows.append({
            "sentence_id": sid,
            "speaker_id": rec["speaker_id"],
            "wav": str(out),
            "text": text,
            "duration": dur,
            "char_rate": n_khmer / dur,
            **stats,
        })
    return rows


# --------------------------------------------------------------------------- #
# bucketing
# --------------------------------------------------------------------------- #
def percentile_labels(values, levels, cut=15.0):
    """Label by percentile, with the outer classes taken from the TAILS.

    `cut=15` means: bottom 15% is the low level, top 15% is the high level, and
    the middle 70% is `mid`. Tertiles (cut=33.3) would balance the classes, and
    that is what the first version of this script did -- but it produced a
    control range too narrow to be worth having. Measured on this corpus:

        axis     tertiles      15/85 tails
        rate     +4.49 ch/s -> +6.64 ch/s
        pitch    +1.86 st   -> +2.88 st
        var      +1.16 st   -> ~+1.6 st
        energy   +3.78 dB   -> ~+5 dB

    Roughly 1.5x on every axis, for 900 exemplars of each extreme instead of
    2000. For teaching a *direction* that is the right trade: the model needs
    unambiguous examples of "this is what fast sounds like" more than it needs
    many borderline ones. The middle class keeps the rest of the data in play
    rather than discarding it.
    """
    lo, hi = np.percentile(values, [cut, 100.0 - cut])

    def label(v):
        return levels[0] if v < lo else (levels[2] if v >= hi else levels[1])

    return label, (float(lo), float(hi))


def label_rows(rows, spk_rows, spk_tag, cut):
    """Attach a control-tag label to every row.

    WHICH AXES ARE RANKED WITHIN SPEAKER, AND WHY
    ---------------------------------------------
    `rate` is ranked GLOBALLY: characters per second means the same thing
    coming from any voice, so a global ranking is the honest one.

    `pitch`, `var` and `energy` are ranked WITHIN SPEAKER. Ranking them
    globally would widen the measured spread -- var especially, from ~1.6 to
    ~3.0 semitones -- but it would do so by letting the axis select a *speaker*
    rather than a delivery. `var:lively` would then mostly mean "use one of the
    animated voices", and the verification sweep, which runs with `spk:any`,
    would report a large effect for the wrong reason. Within-speaker ranking
    keeps each axis about how something is said rather than who says it, at the
    cost of a narrower range. That is the right trade for a control channel.
    """
    rate_label, rate_cuts = percentile_labels(
        [r["char_rate"] for r in rows], LEVELS["rate"], cut)
    within = {}
    for spk, rs in spk_rows.items():
        within[spk] = {
            "pitch": percentile_labels([r["f0_median_hz"] for r in rs], LEVELS["pitch"], cut),
            "var": percentile_labels([r["f0_std_st"] for r in rs], LEVELS["var"], cut),
            "energy": percentile_labels([r["rms_dbfs"] for r in rs], LEVELS["energy"], cut),
        }
    for r in rows:
        spk = r["speaker_id"]
        r["lab"] = {
            "spk": spk_tag[spk],
            "rate": rate_label(r["char_rate"]),
            "pitch": within[spk]["pitch"][0](r["f0_median_hz"]),
            "var": within[spk]["var"][0](r["f0_std_st"]),
            "energy": within[spk]["energy"][0](r["rms_dbfs"]),
        }
    return rows, within, rate_cuts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=6000,
                    help="target number of clips in the finished corpus")
    ap.add_argument("--shards", type=int, default=0,
                    help="how many parquet shards to scan (0 = auto from --clips)")
    ap.add_argument("--per-speaker-cap", type=int, default=0,
                    help="max clips per speaker (0 = clips//n_speakers * 1.6)")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--slot-dropout", type=float, default=0.15,
                    help="probability each slot is replaced by 'any'")
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "data"))
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--cut", type=float, default=15.0,
                    help="percentile for the outer style levels; 15 means "
                         "bottom 15%% low / top 15%% high / middle 70%% mid. "
                         "33.3 gives balanced tertiles and a ~1.5x narrower "
                         "control range -- see percentile_labels().")
    ap.add_argument("--from-meta", action="store_true",
                    help="relabel an existing corpus from its corpus_meta.json "
                         "instead of re-extracting audio. Every measurement is "
                         "already stored there, so changing --cut or "
                         "--slot-dropout costs seconds rather than 20 minutes.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    global MIN_DUR, MAX_DUR
    MIN_DUR, MAX_DUR = args.min_dur, args.max_dur
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    wav_dir = (out_dir / "wavs").resolve()   # absolute: HF datasets
    # resolves manifest audio paths against the CWD, not the manifest directory
    wav_dir.mkdir(parents=True, exist_ok=True)

    if args.from_meta:
        meta_in = out_dir / "corpus_meta.json"
        if not meta_in.exists():
            sys.exit(f"--from-meta needs {meta_in}; run a full build first.")
        prev = json.loads(meta_in.read_text(encoding="utf-8"))
        rows = [dict(c) for c in prev["clips"]]
        if not all("text" in r for r in rows):
            # Metadata written before `text` was recorded there. Recover the
            # transcripts from the existing manifests -- they hold tag+text, and
            # strip_tag() is the exact inverse of how they were written.
            texts = {}
            for split in ("train", "val"):
                mp_ = out_dir / f"{split}.jsonl"
                if not mp_.exists():
                    continue
                for line in mp_.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        texts[d["audio"]] = strip_tag(d["text"])
            missing = [r for r in rows if r["wav"] not in texts]
            if missing:
                # Drop rather than abort: a clip whose transcript cannot be
                # recovered is unusable, but the other several thousand are
                # fine and a rebuild costs 20 minutes for no gain.
                print(f"  ! {len(missing)} clips have no recoverable transcript; "
                      f"dropping them (e.g. {Path(missing[0]['wav']).name})",
                      file=sys.stderr)
                drop = {r["wav"] for r in missing}
                for w in drop:
                    Path(w).unlink(missing_ok=True)
                rows = [r for r in rows if r["wav"] not in drop]
            for r in rows:
                r["text"] = texts[r["wav"]]
        spk_rows = defaultdict(list)
        for r in rows:
            spk_rows[r["speaker_id"]].append(r)
        spk_tag = prev["speaker_tags"]
        print(f"Relabelling {len(rows)} clips from {meta_in} at cut={args.cut}",
              file=sys.stderr)
        rows, within, rate_cuts = label_rows(rows, spk_rows, spk_tag, args.cut)
        write_outputs(rows, out_dir, spk_tag, within, rate_cuts, args, rng)
        return

    # ---------------- pick candidate utterances from the index -------------- #
    tsv = ASR_REPO / "data" / "dataset" / "train.tsv"
    print(f"Reading index {tsv}", file=sys.stderr)
    by_shard = defaultdict(list)
    per_spk = defaultdict(int)
    # Oversample by 2.5x: measurement rejects some clips (unvoiced, too short
    # after trimming), and we want headroom to balance buckets afterwards.
    want = int(args.clips * 2.5)
    cap = args.per_speaker_cap or max(1, int(want / 17))

    header = None
    rows_idx = []
    with tsv.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0:
                header = parts
                continue
            rec = dict(zip(header, parts))
            try:
                dur = float(rec["duration"])
            except (ValueError, KeyError):
                continue
            if not (MIN_DUR <= dur <= MAX_DUR + 2.0):
                continue
            if not rec.get("speaker_id"):
                continue
            rows_idx.append(rec)
    print(f"  {len(rows_idx)} utterances in the duration window", file=sys.stderr)

    rng.shuffle(rows_idx)
    # Derive sentence_id from the shard row; train.tsv has row_index, and the
    # parquet carries sentence_id -- match on shard+row by reading sentence_id
    # out of the shard itself, so select by shard path and row index here.
    chosen = defaultdict(set)
    n_chosen = 0
    for rec in rows_idx:
        spk = rec["speaker_id"]
        if per_spk[spk] >= cap:
            continue
        chosen[rec["shard_path"]].add(int(rec["row_index"]))
        per_spk[spk] += 1
        n_chosen += 1
        if n_chosen >= want:
            break
    print(f"  selected {n_chosen} candidates across {len(chosen)} shards "
          f"({len(per_spk)} speakers)", file=sys.stderr)

    jobs = []
    for shard_rel, row_idxs in sorted(chosen.items()):
        shard_abs = ASR_REPO / shard_rel
        if shard_abs.exists():
            jobs.append((str(shard_abs), wav_dir, row_idxs))

    print(f"Extracting + measuring {len(jobs)} shards on {args.workers} workers",
          file=sys.stderr)
    rows = []
    with mp.Pool(args.workers) as pool:
        for k, shard_rows in enumerate(pool.imap_unordered(process_shard, jobs), 1):
            rows.extend(shard_rows)
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)} shards, {len(rows)} clips", file=sys.stderr, flush=True)
    print(f"  {len(rows)} clips survived measurement", file=sys.stderr)
    if not rows:
        sys.exit("No clips extracted -- check the source dataset path.")

    # ---------------- per-speaker gain normalisation ------------------------ #
    # Per SPEAKER, not per clip: normalising each clip individually would erase
    # exactly the level variation the `energy` axis is supposed to control.
    import soundfile as sf
    spk_rows = defaultdict(list)
    for r in rows:
        spk_rows[r["speaker_id"]].append(r)
    for spk, rs in spk_rows.items():
        med_db = float(np.median([r["rms_dbfs"] for r in rs]))
        gain = 10.0 ** ((TARGET_DBFS - med_db) / 20.0)
        for r in rs:
            audio, _ = sf.read(r["wav"], dtype="float32")
            audio = audio * gain
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0.99:                       # never clip
                audio = audio * (0.99 / peak)
            sf.write(r["wav"], audio, SR, subtype="PCM_16")
            r["rms_dbfs"] = float(20.0 * np.log10(
                np.sqrt(np.mean(audio ** 2) + 1e-12) + 1e-12))

    # ---------------- speaker tags ------------------------------------------ #
    # f-adt2-0002 -> f2 / m3 ... short, stable, and gender is visible in the tag.
    order = defaultdict(list)
    for spk in sorted(spk_rows):
        order[spk[0]].append(spk)
    spk_tag = {}
    for gender, spks in order.items():
        for i, spk in enumerate(spks, 1):
            spk_tag[spk] = f"{gender}{i}"

    # ---------------- trim to --clips --------------------------------------- #
    # Trim BEFORE labelling: the percentile cuts must be computed on the set
    # that actually gets trained on, and tail-based labelling produces its own
    # 15/70/15 class split, so there is no balancing left to do afterwards.
    rng.shuffle(rows)
    if len(rows) > args.clips:
        rows = rows[: args.clips]
        spk_rows = defaultdict(list)
        for r in rows:
            spk_rows[r["speaker_id"]].append(r)
    print(f"  kept {len(rows)} clips", file=sys.stderr)

    rows, within, rate_cuts = label_rows(rows, spk_rows, spk_tag, args.cut)

    # remove wavs we did not keep, so the directory matches the manifest
    keep_paths = {r["wav"] for r in rows}
    removed = 0
    for p in wav_dir.glob("*.wav"):
        if str(p) not in keep_paths:
            p.unlink()
            removed += 1
    print(f"  removed {removed} unused wavs", file=sys.stderr)

    write_outputs(rows, out_dir, spk_tag, within, rate_cuts, args, rng)


def write_outputs(rows, out_dir, spk_tag, within, rate_cuts, args, rng):
    """Shuffle, split, apply per-slot dropout, and write manifests + metadata.

    Shared by the full build and by --from-meta, so a relabel produces byte-
    for-byte the same shape of output as a fresh extraction."""
    # ---------------- write manifests --------------------------------------- #
    # Serialise everything BEFORE truncating any file. The manifests are the
    # only place the transcripts live once the parquet shards are out of scope,
    # so a crash midway through writing would destroy data that took 20 minutes
    # to extract -- as it once did.
    missing = [r for r in rows if not r.get("text")]
    if missing:
        sys.exit(f"{len(missing)} rows have no transcript; refusing to write.")

    rng.shuffle(rows)
    n_val = max(8, int(len(rows) * args.val_frac))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    payload = {}
    for split, rs in splits.items():
        lines = []
        for r in rs:
            vals = dict(r["lab"])
            if split == "train":
                for s in SLOTS:
                    if rng.random() < args.slot_dropout:
                        vals[s] = ANY
            lines.append(json.dumps({
                "audio": r["wav"],
                "text": make_tag(vals) + r["text"],
                "duration": round(r["duration"], 3),
            }, ensure_ascii=False))
        payload[split] = lines

    for split, lines in payload.items():
        path = out_dir / f"{split}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(lines)} rows)", file=sys.stderr)

    # full metadata, for the verification pass and for the write-up
    meta_path = out_dir / "corpus_meta.json"
    meta_path.write_text(json.dumps({
        "n_clips": len(rows),
        "hours": round(sum(r["duration"] for r in rows) / 3600.0, 3),
        "speaker_tags": spk_tag,
        "rate_cuts_chars_per_s": rate_cuts,
        "within_speaker_cuts": {
            spk: {ax: within[spk][ax][1] for ax in ("pitch", "var", "energy")}
            for spk in within
        },
        "slot_dropout": args.slot_dropout,
        "slots": list(SLOTS),
        "levels": LEVELS,
        "clips": [{
            "sentence_id": r["sentence_id"], "wav": r["wav"], "text": r["text"],
            "speaker_id": r["speaker_id"], "duration": r["duration"],
            "char_rate": r["char_rate"], "f0_median_hz": r["f0_median_hz"],
            "f0_std_st": r["f0_std_st"], "rms_dbfs": r["rms_dbfs"],
            "lab": r["lab"],
        } for r in rows],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {meta_path}", file=sys.stderr)

    # ---------------- summary ------------------------------------------------ #
    print("\n== corpus ==", file=sys.stderr)
    print(f"clips {len(rows)}  hours {sum(r['duration'] for r in rows)/3600:.2f}",
          file=sys.stderr)
    for ax in ("rate", "pitch", "var", "energy"):
        counts = defaultdict(int)
        for r in rows:
            counts[r["lab"][ax]] += 1
        print(f"  {ax:7s} " + "  ".join(f"{k}={counts[k]}" for k in LEVELS[ax]),
              file=sys.stderr)
    spk_counts = defaultdict(int)
    for r in rows:
        spk_counts[r["lab"]["spk"]] += 1
    print("  spk     " + "  ".join(f"{k}={v}" for k, v in sorted(spk_counts.items())),
          file=sys.stderr)


if __name__ == "__main__":
    main()
