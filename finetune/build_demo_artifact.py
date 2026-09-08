"""
Turn the demo clips into a single self-contained page you can listen to.

`make_demo.py` writes wavs; this embeds them as data: URIs so the result is one
HTML file with no external requests -- artifacts cannot fetch media from
anywhere, so the audio has to travel inside the page.

Clips are transcoded to mono 64 kbps MP3 first. Raw 48 kHz wav would be ~22 MB
of base64 for a two-sentence set, over the 16 MB page limit; MP3 brings the same
set to roughly 1 MB with no audible cost for judging prosody.

The page is organised as matched rows: one row per axis per sentence, the three
levels side by side, same sentence and same seed, so the only difference within
a row is the commanded level. Base-model rows sit underneath as the control.

    python finetune/build_demo_artifact.py
    python finetune/build_demo_artifact.py --demo-dir finetune/results/demo
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXIS_BLURB = {
    "rate": ("Speaking rate", "slow &rarr; fast, measured in Khmer characters per second"),
    "pitch": ("Pitch register", "low &rarr; high, relative to that voice"),
    "var": ("Pitch variation", "flat &rarr; lively: monotone versus expressive"),
    "energy": ("Level", "soft &rarr; loud"),
}


def mp3_data_uri(wav, kbps=64):
    """Transcode to mono MP3 and return a data: URI. ffmpeg reads/writes pipes."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-b:a", f"{kbps}k",
         "-f", "mp3", "pipe:1"],
        capture_output=True, check=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(out).decode(), len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir", default=str(ROOT / "finetune" / "results" / "demo"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--kbps", type=int, default=64)
    args = ap.parse_args()

    d = Path(args.demo_dir)
    meta = json.loads((d / "demo.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else d / "index.html"

    uri, total = {}, 0
    for c in meta["clips"]:
        u, n = mp3_data_uri(d / c["file"], args.kbps)
        uri[c["file"]] = u
        total += n
    print(f"embedded {len(uri)} clips, {total/1e6:.2f} MB of audio", file=sys.stderr)

    by = {c["file"]: c for c in meta["clips"]}

    def clip(name):
        c = by.get(name)
        if not c:
            return '<div class="cell empty">&mdash;</div>'
        return (f'<div class="cell"><audio controls preload="none" src="{uri[c["file"]]}">'
                f'</audio><div class="meta">{c["duration"]:.2f}s</div></div>')

    rows = []
    for e in meta["sentences"]:
        sid = e["id"]
        rows.append(f'<h3>{sid} <span class="kh">{e["sentence"]}</span></h3>')
        for axis in meta["axes"]:
            title, blurb = AXIS_BLURB[axis]
            levels = meta["levels"][axis]
            cells = "".join(
                f'<div class="col"><div class="lvl">{lv}</div>'
                + clip(f"lora_{axis}_{lv}_{sid}.wav") + "</div>" for lv in levels)
            base = "".join(
                f'<div class="col"><div class="lvl base">{lv}</div>'
                + clip(f"base_{axis}_{lv}_{sid}.wav") + "</div>"
                for lv in (levels[0], levels[2]))
            rows.append(
                f'<section class="axis"><div class="hd"><b>{title}</b>'
                f'<code>{axis}</code><span>{blurb}</span></div>'
                f'<div class="grid">{cells}</div>'
                f'<details><summary>Base model, same sentence &mdash; the control</summary>'
                f'<p class="note">The base model has never seen the tag. If you can hear a '
                f'difference here, it is not the adapter.</p>'
                f'<div class="grid two">{base}</div></details></section>')
        vcells = "".join(
            f'<div class="col"><div class="lvl">{v}</div>'
            + clip(f"lora_spk_{v}_{sid}.wav") + "</div>" for v in meta["voices"])
        rows.append(
            f'<section class="axis"><div class="hd"><b>Voice</b><code>spk</code>'
            f'<span>which corpus speaker, with no reference clip</span></div>'
            f'<div class="grid two">{vcells}</div></section>')

    html = TEMPLATE.replace("__ROWS__", "\n".join(rows)) \
                   .replace("__LORA__", str(meta["lora"])) \
                   .replace("__NCLIPS__", str(len(meta["clips"]))) \
                   .replace("__HOLD__", meta["hold"])
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)", file=sys.stderr)


TEMPLATE = """<title>Khmer Style Control</title>
<style>
:root{--ink:#1a1915;--stone:#6b6a62;--oat:#e5e1d8;--linen:#f4f2ec;--bone:#fbfaf7;
--teal:#0b776f;--teal-soft:#ccfbf1;--paper:#fff;--violet:#6d28d9;}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--ink:#f1efe9;--stone:#a3a199;--oat:#3a3833;--linen:#232220;--bone:#171614;
--teal:#5eead4;--teal-soft:#134e4a;--paper:#1e1d1a;}}
:root[data-theme="dark"]{--ink:#f1efe9;--stone:#a3a199;--oat:#3a3833;--linen:#232220;
--bone:#171614;--teal:#5eead4;--teal-soft:#134e4a;--paper:#1e1d1a;}
*{box-sizing:border-box}
body{background:var(--bone);color:var(--ink);margin:0;
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 72px}
h1{font-family:Georgia,serif;font-size:30px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--stone);margin:0 0 22px;font-size:15px}
.intro{background:var(--linen);border-left:4px solid var(--teal);
padding:14px 16px;border-radius:0 6px 6px 0;margin:0 0 28px;font-size:14px}
.intro p{margin:0 0 8px}.intro p:last-child{margin:0}
h3{font-family:Georgia,serif;font-size:17px;margin:34px 0 12px;
padding-top:16px;border-top:1px solid var(--oat)}
h3 .kh{font-weight:400;color:var(--stone);font-size:14px;margin-left:6px}
.axis{background:var(--paper);border:1px solid var(--oat);border-radius:8px;
padding:14px 16px;margin:0 0 14px}
.hd{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;margin-bottom:11px}
.hd b{font-size:14.5px}
.hd code{background:var(--teal-soft);color:var(--teal);padding:1px 6px;
border-radius:4px;font-size:12px}
.hd span{color:var(--stone);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:11px}
.grid.two{grid-template-columns:repeat(2,1fr)}
@media(max-width:640px){.grid,.grid.two{grid-template-columns:1fr}}
.lvl{font-size:12px;font-weight:600;color:var(--teal);margin-bottom:5px;
text-transform:uppercase;letter-spacing:.04em}
.lvl.base{color:var(--stone)}
audio{width:100%;height:34px}
.meta{color:var(--stone);font-size:11px;margin-top:3px}
.cell.empty{color:var(--stone);font-size:13px;padding:8px 0}
details{margin-top:12px;border-top:1px dashed var(--oat);padding-top:10px}
summary{cursor:pointer;color:var(--stone);font-size:13px}
.note{color:var(--stone);font-size:12.5px;margin:8px 0 10px}
footer{margin-top:38px;padding-top:16px;border-top:1px solid var(--oat);
color:var(--stone);font-size:12.5px}
code.inline{background:var(--linen);padding:1px 5px;border-radius:4px;font-size:12.5px}
</style>
<div class="wrap">
<h1>Khmer style control &mdash; listening set</h1>
<p class="sub">VoxCPM2 with the style-control LoRA adapter &middot; __NCLIPS__ clips</p>

<div class="intro">
<p><b>How to listen.</b> Each row is the same sentence at the same seed, with one
slot of the control tag changed. The only difference between clips in a row is
the level that was commanded, so anything you hear is the tag doing work.</p>
<p>Unswept slots are pinned to <code class="inline">mid</code> rather than left
unspecified, which keeps the tag in the shape the model saw most often during
training.</p>
<p><b>The control condition.</b> Every axis has a collapsed panel with the base
model on the same sentence. It has never seen the tag and demonstrably ignores
it. If a difference is audible there, it is not the adapter &mdash; that panel is
what makes the rest falsifiable rather than suggestive.</p>
</div>

__ROWS__

<footer>
Adapter: <code class="inline">__LORA__</code> &middot; unswept slots held at
<code class="inline">__HOLD__</code> &middot; audio is 64&nbsp;kbps mono MP3,
transcoded from 48&nbsp;kHz for page size. Method, measurements and the
failure that preceded this: <code class="inline">docs/11</code> and
<code class="inline">finetune/results/diagnosis.md</code>.
</footer>
</div>
"""

if __name__ == "__main__":
    main()
