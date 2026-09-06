"""
Build a side-by-side listening page from the synthesized audio.

The scores in results_report.md say which model measures better. They cannot
say which one *sounds* right in Khmer -- and with CER unusable (see that
report), listening is currently the only way to judge intelligibility at all.
So this writes a self-contained HTML page: one row per sentence, one player per
model, reference text alongside.

    python evaluation/audio_samples.py
    python evaluation/audio_samples.py --per-category 2 --bitrate 64k

Writes evaluation/audio_samples.html.

Audio is transcoded to mono MP3 and embedded as base64 data: URIs, because the
page has to work as a single file with no sibling assets. That caps how much
audio can go in -- hence one sentence per category by default (11 of 100),
chosen to span every category in the set rather than the first N.
"""

import argparse
import base64
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.backends import BACKEND_KEYS
from evaluation.common import ROOT, load_entries, read_json, scores_path, wav_path

OUT_PATH = ROOT / "evaluation" / "audio_samples.html"

# Total budget for the finished page. The Artifact ceiling is 16 MB and base64
# inflates by 4/3, so stop well short and say so rather than emit something
# that will not open.
MAX_PAGE_BYTES = 14 * 1024 * 1024


def pick_entries(entries, per_category):
    """One (or more) sentence per category, so every category is audible.

    Deterministic: takes the first N of each category in file order, never a
    random sample -- the page should not change between runs.
    """
    seen = {}
    chosen = []
    for entry in entries:
        key = (entry.get("group"), entry.get("category"))
        if seen.get(key, 0) >= per_category:
            continue
        seen[key] = seen.get(key, 0) + 1
        chosen.append(entry)
    return chosen


def encode_mp3(path, bitrate):
    """-> base64 mono MP3, or None if the wav is missing / ffmpeg fails."""
    if not path.is_file():
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-b:a", bitrate, str(tmp)],
            capture_output=True,
        )
        if result.returncode != 0 or not tmp.stat().st_size:
            return None
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


def load_metrics():
    """model -> {id: {cer, utmos, dnsmos_ovrl}} for the per-clip badges."""
    out = {}
    for model in BACKEND_KEYS:
        path = scores_path(model)
        if not path.is_file():
            continue
        out[model] = {
            u["id"]: {
                "cer": u.get("cer"),
                "utmos": u.get("utmos"),
                "dnsmos": (u.get("dnsmos") or {}).get("ovrl"),
                "seconds": u.get("audio_seconds"),
            }
            for u in read_json(path)["utterances"]
        }
    return out


def fmt(value, digits=2, scale=1.0, suffix=""):
    return "--" if value is None else f"{value * scale:.{digits}f}{suffix}"


def render(rows, models, metrics, bitrate):
    parts = [HEAD]
    parts.append(
        f"<p class=note><strong>{len(rows)}</strong> of 100 sentences, one per "
        f"category, across <strong>{len(models)}</strong> models. Audio is the "
        f"unmodified synthesis output, transcoded to {bitrate} mono MP3 for "
        "embedding. The full set is in "
        "<code>evaluation/results/&lt;model&gt;/audio/</code>.</p>"
    )
    parts.append(
        "<p class=warn><strong>CER is not a valid measure in this run.</strong> "
        "Whisper-large-v3 fails on Khmer, so the CER badges below are shown for "
        "completeness only &mdash; see <code>results_report.md</code>. Judge "
        "these clips by ear.</p>"
    )
    for entry, clips in rows:
        parts.append("<section class=row>")
        parts.append(
            f"<div class=meta><span class=id>{html.escape(entry['id'])}</span>"
            f"<span class=cat>{html.escape(entry.get('group',''))}"
            f" &middot; {html.escape(entry.get('category',''))}</span></div>"
        )
        parts.append(f"<p class=sentence lang=km>{html.escape(entry['sentence'])}</p>")
        parts.append("<div class=clips>")
        for model in models:
            audio = clips.get(model)
            stat = metrics.get(model, {}).get(entry["id"], {})
            parts.append("<div class=clip>")
            parts.append(f"<div class=model>{html.escape(model)}</div>")
            if audio:
                parts.append(
                    f'<audio controls preload=none '
                    f'src="data:audio/mpeg;base64,{audio}"></audio>'
                )
            else:
                parts.append('<div class=missing>no audio</div>')
            parts.append(
                "<div class=badges>"
                f"<span title='predicted naturalness MOS, 1-5'>UTMOS "
                f"{fmt(stat.get('utmos'))}</span>"
                f"<span title='DNSMOS OVRL, 1-5'>DNSMOS "
                f"{fmt(stat.get('dnsmos'))}</span>"
                f"<span class=dim title='invalid -- see note above'>CER "
                f"{fmt(stat.get('cer'), 0, 100, '%')}</span>"
                f"<span class=dim>{fmt(stat.get('seconds'), 1, 1, 's')}</span>"
                "</div>"
            )
            parts.append("</div>")
        parts.append("</div></section>")
    parts.append("</main>")
    return "\n".join(parts)


HEAD = """<title>Khmer TTS Listening Room</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2\
?family=IBM+Plex+Mono:wght@450;600\
&family=IBM+Plex+Sans:wght@400;600\
&family=Kantumruy+Pro:wght@400;600&display=swap">
<style>
  /* Cool instrument-panel neutrals, biased toward the teal accent -- the
     vernacular of waveform and spectrogram displays, not document paper. */
  :root {
    --bg: #f7f9fa; --panel: #ffffff; --well: #eef2f4; --ink: #101619;
    --dim: #57676d; --line: #d9e2e5; --accent: #0d7382; --accent-soft: #e2f0f2;
    --warn-bg: #fdf7e9; --warn-line: #e6d3a1; --warn-ink: #6b5417;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0c1214; --panel: #141b1e; --well: #192327; --ink: #e7eef0;
      --dim: #8ba0a6; --line: #232f33; --accent: #52cfdf; --accent-soft: #12303550;
      --warn-bg: #241f13; --warn-line: #4b4126; --warn-ink: #d9c998;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0c1214; --panel: #141b1e; --well: #192327; --ink: #e7eef0;
    --dim: #8ba0a6; --line: #232f33; --accent: #52cfdf; --accent-soft: #12303550;
    --warn-bg: #241f13; --warn-line: #4b4126; --warn-ink: #d9c998;
  }
  body { background: var(--bg); color: var(--ink); margin: 0;
         font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
         font-size: 15px; line-height: 1.55; }
  main { max-width: 1120px; margin: 0 auto; padding: 40px 22px 72px; }
  header { border-bottom: 2px solid var(--ink); padding-bottom: 14px;
           margin-bottom: 22px; }
  h1 { font-size: 28px; font-weight: 600; margin: 0 0 6px;
       letter-spacing: -0.02em; text-wrap: balance; }
  .sub { color: var(--dim); margin: 0; max-width: 62ch; }
  .note { color: var(--dim); font-size: 13.5px; max-width: 74ch; }
  .warn { background: var(--warn-bg); border: 1px solid var(--warn-line);
          color: var(--warn-ink); border-radius: 6px; padding: 11px 14px;
          font-size: 13.5px; max-width: 74ch; }
  /* Rows are separated by rule, not stacked as cards -- the cards are the
     clips, which are the things you actually compare. */
  .row { border-top: 1px solid var(--line); padding: 22px 0 4px; }
  .meta { display: flex; gap: 10px; align-items: baseline; }
  .id { font-family: "IBM Plex Mono", ui-monospace, monospace; font-weight: 600;
        font-size: 12px; color: var(--accent); letter-spacing: 0.04em; }
  .cat { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
         color: var(--dim); text-transform: uppercase; letter-spacing: 0.07em; }
  .sentence { font-family: "Kantumruy Pro", "Noto Sans Khmer", sans-serif;
              font-size: 20px; line-height: 2.0; margin: 10px 0 16px;
              word-break: break-word; max-width: 68ch; }
  .clips { display: grid; gap: 12px;
           grid-template-columns: repeat(auto-fit, minmax(236px, 1fr)); }
  .clip { background: var(--well); border-radius: 7px; padding: 11px 12px 12px; }
  .model { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
           font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
           margin-bottom: 8px; }
  audio { width: 100%; height: 32px; accent-color: var(--accent); }
  audio:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .missing { color: var(--dim); font-size: 13px; padding: 6px 0 10px; }
  .badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px;
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 10.5px; color: var(--dim);
            font-variant-numeric: tabular-nums; }
  .badges span { background: var(--panel); border-radius: 3px;
                 padding: 2px 6px; white-space: nowrap; }
  .badges .dim { opacity: 0.5; text-decoration: line-through; }
</style>
<main>
<header>
<h1>Khmer TTS &mdash; listening room</h1>
<p class=sub>The same Khmer sentence spoken by every model under test, one row
per category. Scores rank what is measurable; only your ear can tell you which
of these is actually speaking Khmer.</p>
</header>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--per-category", type=int, default=1,
                        help="sentences per category (default 1 -> 11 rows)")
    parser.add_argument("--bitrate", default="48k", help="mp3 bitrate, e.g. 64k")
    parser.add_argument("--models", help="comma-separated subset of models")
    args = parser.parse_args()

    models = ([m.strip() for m in args.models.split(",")]
              if args.models else list(BACKEND_KEYS))
    models = [m for m in models if (ROOT / "evaluation" / "results" / m).is_dir()]
    if not models:
        parser.error("no synthesized models found under evaluation/results/")

    entries = pick_entries(load_entries(), args.per_category)
    metrics = load_metrics()

    rows = []
    missing = 0
    for entry in entries:
        clips = {}
        for model in models:
            audio = encode_mp3(wav_path(model, entry["id"]), args.bitrate)
            if audio is None:
                missing += 1
            clips[model] = audio
        rows.append((entry, clips))
        print(f"  {entry['id']}  {entry.get('category')}")

    page = render(rows, models, metrics, args.bitrate)
    size = len(page.encode("utf-8"))
    OUT_PATH.write_text(page, encoding="utf-8")

    print(f"\nWrote {OUT_PATH}  ({size / 1024 / 1024:.2f} MB)")
    print(f"  {len(rows)} sentences x {len(models)} models"
          f"{f', {missing} clips missing' if missing else ''}")
    if size > MAX_PAGE_BYTES:
        print(f"  ! over the {MAX_PAGE_BYTES / 1024 / 1024:.0f} MB budget -- "
              "lower --bitrate or --per-category before publishing.")


if __name__ == "__main__":
    main()
