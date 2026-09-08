"""
Aggregate every model's scores.json into one comparison report.

Usage:
    python evaluation/report.py
Writes: evaluation/results_report.md

Models with no scores.json yet are listed as "not run" rather than omitted --
a partial comparison should look partial.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.backends import BACKEND_KEYS
from evaluation.common import ROOT, read_json, scores_path

OUT_PATH = ROOT / "evaluation" / "results_report.md"

MODEL_LABELS = {
    "mms": "Meta MMS-TTS (`facebook/mms-tts-khm`) -- docs/04",
    "voxcpm2": "VoxCPM2 (`openbmb/VoxCPM2`) -- docs/06",
    "fish-s2": "Fish Audio S2-Pro (`fishaudio/s2-pro`) -- docs/05",
    "higgs3": "Higgs TTS 3 (`bosonai/higgs-tts-3-4b`)",
}

# Models that speak Khmer without documenting it. They belong in the ranking
# -- they do the job -- but there is no vendor figure to check the result
# against, so a surprising score here has nothing to corroborate it.
UNDOCUMENTED_KHMER = {
    "higgs3": (
        "its model card lists 100 languages and Khmer is not among them, yet "
        "it produces Khmer speech (verified by hand before the run)"
    ),
}

# Vendor-published figures, for sanity-checking a local run. Both come from
# OpenBMB's own 30-language benchmark -- see docs/03 section 3.4 on why they
# are upper bounds, not independent results.
PUBLISHED = {
    "voxcpm2": "2.05% CER (OpenBMB internal, Khmer); RTF 0.13-0.30 on an RTX 4090",
    "fish-s2": "75.15% CER (OpenBMB internal, Khmer)",
    "mms": "no published Khmer benchmark",
    "higgs3": "no Khmer figure published; <5 WER/CER on 85 of its 100 languages",
}


def fmt(value, digits=3, scale=1.0, suffix=""):
    if value is None:
        return "--"
    return f"{value * scale:.{digits}f}{suffix}"


def load_all():
    out = {}
    for model in BACKEND_KEYS:
        path = scores_path(model)
        out[model] = read_json(path) if path.is_file() else None
    return out


# A CER at or above this is not a score, it is a failure: the ASR got
# essentially every character wrong. Whisper-large-v3 does this on Khmer by
# collapsing into a repetition loop, which also pushes CER above 100% because
# the hypothesis ends up longer than the reference.
CER_INVALID_THRESHOLD = 0.9


def cer_validity_warning(results, lines):
    """Say so loudly when CER did not measure anything.

    A metric that cannot separate the models is worse than no metric, because
    the table above still renders a number for it."""
    suspect = {
        model: payload["summary"]["overall"]["cer_median"]
        for model, payload in results.items()
        if payload and (payload["summary"]["overall"]["cer_median"] or 0)
        >= CER_INVALID_THRESHOLD
    }
    if not suspect:
        return

    # Identical transcripts for different audio is the giveaway: it means the
    # decoder is landing in the same attractor regardless of its input.
    per_id = {}
    for model, payload in results.items():
        if not payload:
            continue
        for utterance in payload["utterances"]:
            transcript = utterance.get("transcript")
            if transcript:
                per_id.setdefault(utterance["id"], set()).add(transcript)
    shared = sum(
        1 for transcripts in per_id.values() if len(transcripts) == 1
    ) if len(results) > 1 else 0

    lines.append("## :warning: CER is not valid in this run")
    lines.append("")
    lines.append(
        "**Do not rank the models on the CER column above.** The scoring ASR "
        "failed, so those numbers describe Whisper, not the TTS models."
    )
    lines.append("")
    for model, value in sorted(suspect.items()):
        lines.append(
            f"- `{model}` has a median CER of {fmt(value, 1, 100, '%')} -- i.e. "
            "essentially every character is wrong, and above 100% the "
            "hypothesis is longer than the reference it is meant to match."
        )
    if shared:
        lines.append(
            f"- {shared} sentence(s) produced a **byte-identical transcript "
            "across different models' audio** -- impossible unless the decoder "
            "is ignoring the waveform."
        )
    lines.append(
        "- Whisper-large-v3 collapses into a repetition loop on Khmer "
        "(`ប្រាប់ប្រាប់ប្រាប់...`). Raw transcripts in each `scores.json` show it "
        "directly."
    )
    lines.append("")
    lines.append(
        "The other three metrics are unaffected -- UTMOS, DNSMOS and RTF never "
        "touch the ASR. To restore CER, swap `metrics/cer.py` for a "
        "Khmer-capable ASR and re-run `score.py --metrics cer`; the synthesized "
        "audio does not need regenerating."
    )
    lines.append("")


def headline_table(results, lines):
    lines.append("## Headline comparison")
    lines.append("")
    lines.append(
        "| Model | n | CER mean | CER median | UTMOS | DNSMOS OVRL | DNSMOS SIG | "
        "DNSMOS BAK | P.808 | RTF median | RTF mean |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            lines.append(f"| `{model}` | _not run_ | | | | | | | | | |")
            continue
        o = payload["summary"]["overall"]
        lines.append(
            f"| `{model}` | {o['count']} "
            f"| {fmt(o['cer_mean'], 2, 100, '%')} "
            f"| {fmt(o['cer_median'], 2, 100, '%')} "
            f"| {fmt(o['utmos_mean'], 2)} "
            f"| {fmt(o['dnsmos_ovrl_mean'], 2)} "
            f"| {fmt(o['dnsmos_sig_mean'], 2)} "
            f"| {fmt(o['dnsmos_bak_mean'], 2)} "
            f"| {fmt(o['dnsmos_p808_mean'], 2)} "
            f"| {fmt(o['rtf_median'])} "
            f"| {fmt(o['rtf_mean'])} |"
        )
    lines.append("")
    lines.append(
        "CER lower is better (0% = perfect). UTMOS and DNSMOS are 1-5 MOS "
        "scales, higher is better. RTF below 1.0 is faster than real time."
    )
    lines.append("")


def breakdown_table(results, lines, key, title, note):
    lines.append(f"## {title}")
    lines.append("")
    lines.append(note)
    lines.append("")
    buckets = []
    for payload in results.values():
        if payload:
            for name in payload["summary"].get(f"by_{key}", {}):
                if name not in buckets:
                    buckets.append(name)
    if not buckets:
        lines.append("_No scored runs yet._")
        lines.append("")
        return

    header = "| Model | " + " | ".join(f"`{b}`" for b in buckets) + " |"
    lines.append(header)
    lines.append("|---" * (len(buckets) + 1) + "|")
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        by = payload["summary"].get(f"by_{key}", {})
        cells = [
            fmt((by.get(b) or {}).get("cer_mean"), 2, 100, "%") for b in buckets
        ]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.append("")


def worst_table(results, lines, limit=10):
    lines.append("## Worst utterances by CER")
    lines.append("")
    lines.append(
        "Read the transcript before blaming the TTS -- a high CER here can be "
        "Whisper failing on Khmer rather than the model mispronouncing "
        "anything. See the caveat below."
    )
    lines.append("")
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        scored = [u for u in payload["utterances"] if u.get("cer") is not None]
        if not scored:
            continue
        worst = sorted(scored, key=lambda u: -u["cer"])[:limit]
        lines.append(f"### `{model}`")
        lines.append("")
        lines.append("| id | category | CER | ASR transcript |")
        lines.append("|---|---|---|---|")
        for u in worst:
            transcript = (u.get("transcript") or "").replace("|", "\\|")
            if len(transcript) > 60:
                transcript = transcript[:57] + "..."
            lines.append(
                f"| {u['id']} | `{u.get('category')}` | "
                f"{fmt(u['cer'], 1, 100, '%')} | {transcript} |"
            )
        lines.append("")


# fish-s2 runs its codec on the CPU because S2-Pro's 9.65 GB of weights plus a
# 4.58 GB codec do not fit one 12 GB card (backends/fish_s2.py, CODEC_DEVICE).
# Its RTF therefore measures GPU generation + CPU decode and is not the same
# quantity as a fully-GPU model's.
HYBRID_RTF = {"fish-s2": "codec runs on CPU -- RTF not comparable"}

# An utterance far longer than the sentence warrants means the model failed to
# stop, not that it spoke slowly. Those clips inflate RTF and drag the quality
# metrics toward whatever the model babbles.
RUNAWAY_SECONDS = 20.0


def measurement_caveats(results, lines):
    """Flag the things that make a column mean different things per row."""
    from evaluation.common import median as med

    notes = []
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        if model in UNDOCUMENTED_KHMER:
            notes.append(
                f"- `{model}`: **Khmer is undocumented** -- "
                f"{UNDOCUMENTED_KHMER[model]}. Its scores count, but no "
                "published Khmer figure exists to corroborate them."
            )
        if model in HYBRID_RTF:
            notes.append(f"- `{model}`: {HYBRID_RTF[model]}.")
        seconds = [
            u.get("audio_seconds") for u in payload["utterances"]
            if u.get("audio_seconds")
        ]
        runaway = [s for s in seconds if s > RUNAWAY_SECONDS]
        if runaway:
            notes.append(
                f"- `{model}`: {len(runaway)} utterance(s) ran past "
                f"{RUNAWAY_SECONDS:.0f}s (longest {max(runaway):.1f}s, median "
                f"across the set {med(seconds):.1f}s) -- the model did not stop "
                "on its own. Those clips inflate its RTF and its quality scores "
                "reflect the filler, not the sentence."
            )
    if not notes:
        return
    lines.append("## Measurement caveats")
    lines.append("")
    lines.append(
        "These affect what a column *means* for a given row -- read them before "
        "comparing across rows."
    )
    lines.append("")
    lines.extend(notes)
    lines.append("")


def env_section(results, lines):
    lines.append("## Run environment")
    lines.append("")
    lines.append(
        "RTF is hardware-bound and is only comparable between models measured "
        "on the same machine -- check that these rows match before comparing "
        "the speed column."
    )
    lines.append("")
    lines.append("| Model | Device | GPU | Synth load (s) | Timestamp |")
    lines.append("|---|---|---|---|---|")
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        env = payload.get("env", {})
        lines.append(
            f"| `{model}` | {env.get('device', '--')} | {env.get('gpu', '--')} "
            f"| {env.get('load_seconds', '--')} | {env.get('timestamp', '--')} |"
        )
    lines.append("")


def build_report(results):
    lines = []
    lines.append("# TTS Model Comparison -- Khmer")
    lines.append("")
    ran = [m for m in BACKEND_KEYS if results.get(m)]
    lines.append(
        f"{len(ran)} of {len(BACKEND_KEYS)} models scored, over the fixed "
        "100-sentence set in `eval-set/eval.json` (50 `pure_khmer` + 50 "
        "`code_switched`). Metrics: CER (correctness, via Whisper-large-v3), "
        "UTMOS and DNSMOS (naturalness), RTF (speed)."
    )
    lines.append("")

    headline_table(results, lines)
    cer_validity_warning(results, lines)
    breakdown_table(
        results, lines, "group", "CER by group",
        "The set is split 50/50 for exactly this comparison -- code-switching "
        "is where Khmer TTS is expected to degrade.",
    )
    breakdown_table(
        results, lines, "category", "CER by category",
        "Category definitions are in `evaluation/data-promt.md`; counts and "
        "examples in `evaluation/dataset_overview.md`.",
    )
    worst_table(results, lines)
    measurement_caveats(results, lines)
    env_section(results, lines)

    lines.append("## Published figures, for comparison")
    lines.append("")
    lines.append("| Model | Vendor/third-party published |")
    lines.append("|---|---|")
    for model in BACKEND_KEYS:
        lines.append(f"| `{model}` | {PUBLISHED[model]} |")
    lines.append("")

    lines.append("## Reading these numbers")
    lines.append("")
    lines.append(
        "- **The CER floor is not zero.** Per `docs/research/03-evaluation-benchmarking.md` "
        "section 3.4, ASR-based CER is only as good as the scoring ASR, and "
        "Whisper-large-v3's own Khmer accuracy is limited. Differences between "
        "the 3 models are meaningful (same ASR, same sentences); the absolute "
        "value is not."
    )
    lines.append(
        "- **UTMOS and DNSMOS have never heard Khmer.** Both were trained on "
        "MOS studies of mostly English speech. They rate acoustic quality and "
        "prosodic naturalness, which transfers reasonably, but treat them as a "
        "relative ranking rather than a calibrated Khmer MOS."
    )
    lines.append(
        "- **DNSMOS BAK is near its ceiling for clean synthesis** and carries "
        "little signal here; SIG and OVRL are the informative columns."
    )
    lines.append(
        "- **Published figures are self-reported upper bounds** -- both the "
        "VoxCPM2 and Fish S2-Pro Khmer numbers come from OpenBMB's own "
        "benchmark, not an independent evaluation."
    )
    lines.append(
        "- **RTF excludes model load time** (reported separately above) and is "
        "measured after a discarded warm-up utterance."
    )
    lines.append("")

    lines.append(
        "_Generated by `evaluation/report.py` from "
        "`evaluation/results/<model>/scores.json`. See `evaluation/README.md` "
        "for the runbook._"
    )
    return "\n".join(lines) + "\n"


def main():
    results = load_all()
    OUT_PATH.write_text(build_report(results), encoding="utf-8")
    ran = [m for m in BACKEND_KEYS if results.get(m)]
    missing = [m for m in BACKEND_KEYS if not results.get(m)]
    print(f"Wrote {OUT_PATH}")
    print(f"  scored: {', '.join(ran) if ran else 'none'}")
    if missing:
        print(f"  not run: {', '.join(missing)}")


if __name__ == "__main__":
    main()
