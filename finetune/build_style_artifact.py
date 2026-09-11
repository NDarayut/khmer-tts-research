"""
Build the style-transfer page: description versus example, audible.

WHY A PAGE
----------
The measurement says a reference recording moves pitch modulation by several
semitones where 28 parenthetical wordings moved it by a fraction of one. But
this project's own CLAUDE.md records that global F0 statistics do not track
perceived expressiveness -- `prosody_stats.py` called `mms` the most expressive
model when it had been eliminated by ear for robotic prosody. So a number
moving is not the finding; it only licenses the listening.

Each speaker block plays four things, and the speaker is the SAME in all four:

    reference: flat      a real recording from that speaker's flat band
    reference: lively    a real recording from the same speaker, lively band
    generated from flat  the Khmer test sentence, cloned from the first
    generated from lively  the same sentence and seed, cloned from the second

If the last two sound alike, the channel does not carry style and the numbers
are measuring something else. If they differ the way the first two differ, it
does.

    .venv/bin/python finetune/build_style_artifact.py
"""

import argparse
import base64
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.verify_control import spearman  # noqa: E402

OUT = ROOT / "finetune" / "results" / "style_reference"
CHAN_NAME = {"ref": "reference_wav_path", "cont": "prompt_wav_path + prompt_text"}
CHAN_WHAT = {"ref": "voice cloning, structurally isolated via ref_audio tokens",
             "cont": "continuation mode"}


def mp3_uri(wav, kbps=32):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "24000",
         "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(out).decode(), len(out)


CSS = """
:root{
  --paper:#f5f6f8; --card:#ffffff; --ink:#141b22; --ink-soft:#57646f;
  --rule:#dde2e8; --rule-soft:#eceff3;
  --anchor:#1c3f5e;
  --flat:#54748a; --flat-wash:#e7edf1;
  --lively:#b5461f; --lively-wash:#f8e9e2;
  --shadow:0 1px 2px rgba(20,27,34,.07);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0c1116; --card:#151d24; --ink:#e5eaef; --ink-soft:#93a2af;
    --rule:#26313a; --rule-soft:#1b242b;
    --anchor:#7fb2da;
    --flat:#8fb0c5; --flat-wash:#16242e;
    --lively:#e58254; --lively-wash:#301709;
    --shadow:0 1px 2px rgba(0,0,0,.5);
  }
}
:root[data-theme="dark"]{
  --paper:#0c1116; --card:#151d24; --ink:#e5eaef; --ink-soft:#93a2af;
  --rule:#26313a; --rule-soft:#1b242b;
  --anchor:#7fb2da;
  --flat:#8fb0c5; --flat-wash:#16242e;
  --lively:#e58254; --lively-wash:#301709;
  --shadow:0 1px 2px rgba(0,0,0,.5);
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:56px 26px 90px}
.kh{font-family:"Noto Sans Khmer","Khmer OS",sans-serif;line-height:2}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}

header{border-bottom:2px solid var(--ink);padding-bottom:26px;margin-bottom:34px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-soft);margin:0 0 14px}
h1{font-family:Newsreader,Georgia,serif;font-weight:600;
  font-size:clamp(32px,5.2vw,46px);line-height:1.1;margin:0 0 14px;
  text-wrap:balance;letter-spacing:-.01em}
.lede{font-size:18.5px;color:var(--ink-soft);margin:0;max-width:64ch}
.lede b{color:var(--ink);font-weight:600}

h2{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:26px;
  margin:52px 0 6px;letter-spacing:-.005em}
p{max-width:70ch}
.sub{color:var(--ink-soft);margin:0 0 20px;max-width:70ch}

/* headline contrast */
.verdict{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:26px 0 8px}
.vcard{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:18px 20px;box-shadow:var(--shadow)}
.vcard .k{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-soft);margin-bottom:9px}
.vcard .v{font-family:Newsreader,Georgia,serif;font-size:38px;line-height:1;
  font-variant-numeric:tabular-nums}
.vcard .w{font-size:13.5px;color:var(--ink-soft);margin-top:8px}
.vcard.desc .v{color:var(--flat)}
.vcard.exam .v{color:var(--lively)}
@media (max-width:640px){.verdict{grid-template-columns:1fr}}

/* speaker blocks */
.spk{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  margin:14px 0;box-shadow:var(--shadow);overflow:hidden}
.spkhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  padding:11px 16px;border-bottom:1px solid var(--rule-soft);
  font-family:"IBM Plex Mono",monospace;font-size:12.5px}
.spkhead .id{font-weight:600}
.spkhead .meta{color:var(--ink-soft);font-size:11.5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.cell{padding:13px 16px}
.cell+.cell{border-left:1px solid var(--rule-soft)}
.cell.f{background:var(--flat-wash)}
.cell.l{background:var(--lively-wash)}
.band{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;font-weight:600;margin-bottom:9px}
.cell.f .band{color:var(--flat)}
.cell.l .band{color:var(--lively)}
.btns{display:flex;flex-direction:column;gap:6px;align-items:flex-start}
@media (max-width:640px){.grid{grid-template-columns:1fr}.cell+.cell{border-left:0;
  border-top:1px solid var(--rule-soft)}}

.play{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.03em;
  border:1px solid var(--rule);background:var(--card);color:var(--ink-soft);
  border-radius:3px;padding:6px 10px;cursor:pointer;white-space:nowrap;
  display:inline-flex;align-items:center;gap:7px;width:100%;justify-content:flex-start}
.play:hover{border-color:var(--anchor);color:var(--anchor)}
.play.on{border-color:var(--anchor);color:var(--anchor)}
.play .tri{width:0;height:0;border-left:6px solid currentColor;
  border-top:4px solid transparent;border-bottom:4px solid transparent;flex:none}
.play.on .tri{border:0;width:7px;height:7px;background:currentColor;border-radius:1px}
.play .st{margin-left:auto;font-variant-numeric:tabular-nums;opacity:.75}
.play.real{border-style:dashed}

.note{background:var(--card);border:1px solid var(--rule);
  border-left:3px solid var(--anchor);border-radius:0 3px 3px 0;
  padding:15px 18px;margin:22px 0;box-shadow:var(--shadow);font-size:15px}
.note.warn{border-left-color:var(--lively)}

table{border-collapse:collapse;width:100%;font-size:14.5px;margin:16px 0 8px}
th,td{text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid var(--rule-soft)}
th{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-soft);font-weight:600;
  border-bottom:1px solid var(--rule)}
td.num{text-align:right;font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
.demo{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:14px 18px;margin:18px 0;box-shadow:var(--shadow)}
.demo .lbl{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-soft);margin-bottom:6px}
figure{margin:22px 0}
figcaption{font-size:13.5px;color:var(--ink-soft);margin-top:8px;max-width:66ch}
svg{max-width:100%;height:auto;display:block}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--ink-soft);max-width:72ch}
"""

JS = """
let cur=null,curBtn=null;
document.addEventListener('click',e=>{
  const p=e.target.closest('.play'); if(!p) return;
  if(cur) cur.pause();
  if(curBtn) curBtn.classList.remove('on');
  if(curBtn===p){cur=null;curBtn=null;return;}
  cur=new Audio(p.dataset.src); curBtn=p; p.classList.add('on');
  cur.onended=()=>p.classList.remove('on');
  cur.play();
});
"""


def dose_svg(rows):
    """Reference f0_std against generated f0_std, with the identity line.
    Identity is the right comparison: it asks not just whether the output moved
    but whether it landed WHERE THE REFERENCE WAS."""
    W, H, PL, PR, PT, PB = 620, 400, 56, 16, 16, 44
    lo, hi = 1.0, 10.0
    def X(v): return PL + (v - lo) / (hi - lo) * (W - PL - PR)
    def Y(v): return PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reference pitch '
         f'variation against generated pitch variation">']
    o.append(f'<rect x="{PL}" y="{PT}" width="{W-PL-PR}" height="{H-PT-PB}" '
             f'fill="none" stroke="var(--rule)"/>')
    for g in (2, 4, 6, 8, 10):
        o.append(f'<line x1="{X(g):.1f}" y1="{PT}" x2="{X(g):.1f}" y2="{H-PB}" '
                 f'stroke="var(--rule-soft)"/>')
        o.append(f'<line x1="{PL}" y1="{Y(g):.1f}" x2="{W-PR}" y2="{Y(g):.1f}" '
                 f'stroke="var(--rule-soft)"/>')
        o.append(f'<text x="{X(g):.1f}" y="{H-PB+16}" font-size="11" '
                 f'fill="var(--ink-soft)" text-anchor="middle" '
                 f'font-family="IBM Plex Mono, monospace">{g}</text>')
        o.append(f'<text x="{PL-8}" y="{Y(g)+4:.1f}" font-size="11" '
                 f'fill="var(--ink-soft)" text-anchor="end" '
                 f'font-family="IBM Plex Mono, monospace">{g}</text>')
    o.append(f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" '
             f'y2="{Y(hi):.1f}" stroke="var(--ink-soft)" stroke-width="1.4" '
             f'stroke-dasharray="5 4"/>')
    o.append(f'<text x="{X(hi)-6:.1f}" y="{Y(hi)+16:.1f}" font-size="11" '
             f'fill="var(--ink-soft)" text-anchor="end" '
             f'font-family="IBM Plex Mono, monospace">output = reference</text>')
    for r in rows:
        x, y = r["ref_f0_std_st"], r["f0_std_st"]
        if lo <= x <= hi and lo <= y <= hi:
            c = "var(--lively)" if r["band"] == "lively" else "var(--flat)"
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.6" '
                     f'fill="{c}" fill-opacity=".42"/>')
    o.append(f'<text x="{(PL+W-PR)/2:.0f}" y="{H-6}" font-size="12" '
             f'fill="var(--ink-soft)" text-anchor="middle">'
             f'pitch variation of the reference recording (st)</text>')
    o.append(f'<text x="15" y="{(PT+H-PB)/2:.0f}" font-size="12" '
             f'fill="var(--ink-soft)" text-anchor="middle" '
             f'transform="rotate(-90 15 {(PT+H-PB)/2:.0f})">'
             f'pitch variation of the generated Khmer (st)</text>')
    o.append('</svg>')
    return "".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "style_transfer.html"))
    ap.add_argument("--kbps", type=int, default=32)
    ap.add_argument("--channel", default="ref", help="channel used for the audio blocks")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (OUT / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    doc = json.loads((OUT / "style_reference.json").read_text(encoding="utf-8"))
    refs = json.loads((OUT / "references.json").read_text(encoding="utf-8"))
    sents = {e["id"]: e["sentence"] for e in json.loads(
        (ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}

    ref_of = {(r["speaker_id"], r["band"], r["slot"]): r for r in refs["refs"]}
    ch = args.channel
    chr_ = [r for r in rows if r["channel"] == ch]

    # the sentence with the most saved audio, so blocks are complete
    have = defaultdict(set)
    for w in (OUT / "audio" / ch).glob("*.wav"):
        sid, spk, band = w.stem.split("_", 2)
        have[sid].add((spk, band))
    if not have:
        sys.exit(f"no audio under {OUT/'audio'/ch}")
    demo_id = max(have, key=lambda k: len(have[k]))
    demo_txt = sents[demo_id]

    total = [0]
    def uri(p):
        u, n = mp3_uri(p, args.kbps)
        total[0] += n
        return u

    gen_of = {}
    for r in chr_:
        if r["seed"] == 0 and r["slot"] == 0 and r["id"] == demo_id:
            gen_of[(r["speaker_id"], r["band"])] = r

    P = []
    a = P.append
    a("<title>Style by Example</title>")
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:wght@500;600&family=Source+Sans+3:wght@400;600&'
      'family=IBM+Plex+Mono:wght@400;500;600&'
      'family=Noto+Sans+Khmer:wght@400;500&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div class="wrap"><header>')
    a('<p class="eyebrow">VoxCPM2 &middot; base model, no adapter &middot; '
      'Khmer &middot; 576 generations</p>')
    a('<h1>Style by Example</h1>')
    a('<p class="lede">Twenty-eight ways of <i>asking</i> for expressive delivery '
      'moved Khmer pitch modulation by a fraction of a semitone. Handing the model '
      'a <b>recording</b> of it moves several &mdash; with the speaker held fixed, '
      'so what transfers is style and not voice.</p>')
    a('</header>')

    d = doc.get(ch, {}).get("f0_std_st")
    dose = doc.get(ch, {}).get("_dose", {})
    a('<div class="verdict">')
    a('<div class="vcard desc"><div class="k">Asking for it &mdash; 28 wordings</div>'
      '<div class="v">&plusmn;0.7 st</div>'
      '<div class="w">and once the pitch shift those wordings cause is removed, '
      'nothing survives at p &lt; 0.05.</div></div>')
    if d:
        a(f'<div class="vcard exam"><div class="k">Showing it &mdash; one recording</div>'
          f'<div class="v">{d["median_delta"]:+.2f} st</div>'
          f'<div class="w">lively reference over flat reference, same speaker and '
          f'sentence: {d["hits"]}/{d["n"]} pairs '
          f'({d["hits"]/d["n"]*100:.0f}%).</div></div>')
    a('</div>')

    a(f'<div class="demo"><div class="lbl">every generated clip below says</div>'
      f'<div class="kh">{html.escape(demo_txt)}</div></div>')

    a('<h2>Same speaker, two reference recordings</h2>')
    a('<p class="sub">The dashed buttons are real recordings of that speaker &mdash; '
      'one from the flat band, one from the lively band. The solid buttons are the '
      'Khmer sentence above, generated at the same seed, cloned from the recording '
      'beside it. Voice is constant within a block by construction, so anything you '
      'hear change is style.</p>')

    spks = sorted({s for s, _ in gen_of})
    for spk in spks:
        rf = ref_of.get((spk, "flat", 0)); rl = ref_of.get((spk, "lively", 0))
        gf = gen_of.get((spk, "flat")); gl = gen_of.get((spk, "lively"))
        if not (rf and rl and gf and gl):
            continue
        a('<div class="spk">')
        a(f'<div class="spkhead"><span class="id">{html.escape(spk)}</span>'
          f'<span class="meta">reference recordings differ by '
          f'{rl["ref_f0_std_st"]-rf["ref_f0_std_st"]:+.2f} st &middot; '
          f'the generated pair differs by '
          f'{gl["f0_std_st"]-gf["f0_std_st"]:+.2f} st</span></div>')
        a('<div class="grid">')
        for band, rr, gg, cls in (("flat", rf, gf, "f"), ("lively", rl, gl, "l")):
            a(f'<div class="cell {cls}"><div class="band">{band} reference</div>'
              f'<div class="btns">')
            a(f'<button class="play real" data-src="{uri(rr["wav"])}">'
              f'<span class="tri"></span>real recording'
              f'<span class="st">{rr["ref_f0_std_st"]:.2f} st</span></button>')
            w = OUT / "audio" / ch / f"{demo_id}_{spk}_{band}.wav"
            if w.exists():
                a(f'<button class="play" data-src="{uri(w)}">'
                  f'<span class="tri"></span>generated Khmer'
                  f'<span class="st">{gg["f0_std_st"]:.2f} st</span></button>')
            a('</div></div>')
        a('</div></div>')

    a('<h2>It does not just move &mdash; it lands where the reference was</h2>')
    a('<p class="sub">A control that reliably increases variation would be useful. '
      'One that reproduces the <i>amount</i> in the reference is a different and '
      'stronger thing, so the comparison here is against the identity line rather '
      'than against zero.</p>')
    a('<figure>')
    a(dose_svg(chr_))
    cap = (f'All {len(chr_)} generations on the <code>{CHAN_NAME[ch]}</code> '
           f'channel. Spearman rho = <b>{dose.get("rho_pooled", 0):+.3f}</b> '
           f'between the reference clip\'s own pitch variation and the generated '
           f'clip\'s.')
    if dose.get("rho_within_speaker_median") is not None:
        cap += (f' Within a single speaker, median rho = '
                f'<b>{dose["rho_within_speaker_median"]:+.3f}</b> across '
                f'{dose["n_speakers"]} speakers &mdash; so the relationship is not '
                f'carried by differences between voices.')
    a(f'<figcaption>{cap}</figcaption>')
    a('</figure>')

    a('<h2>Both audio channels work; the stronger one is not the cleaner one</h2>')
    a('<p class="sub">A control is only an axis if it moves the thing it names '
      'and leaves the rest alone. The last three columns are what LEAKED &mdash; '
      'a hit rate near 50% means the measure did not move at all, which is the '
      'good outcome here.</p>')
    a('<div class="scroll"><table><thead><tr><th>channel</th>'
      '<th class="num">&Delta; f0_std</th><th class="num">lively &gt; flat</th>'
      '<th class="num">dose rho</th><th class="num">pitch</th>'
      '<th class="num">rate</th><th class="num">level</th>'
      '</tr></thead><tbody>')
    for c in ("ref", "cont"):
        cc = doc.get(c)
        if not cc or "f0_std_st" not in cc:
            continue
        m = cc["f0_std_st"]; dz = cc.get("_dose", {})
        def leak(k, unit):
            q = cc.get(k)
            if not q:
                return "&mdash;"
            return (f'{q["median_delta"]:+.2f} {unit}<br>'
                    f'<span style="opacity:.6">{q["hits"]/q["n"]*100:.0f}%</span>')
        a(f'<tr><td><code>{CHAN_NAME[c]}</code><br>'
          f'<span style="opacity:.65;font-size:12.5px">{CHAN_WHAT[c]}</span></td>'
          f'<td class="num">{m["median_delta"]:+.2f} st</td>'
          f'<td class="num">{m["hits"]}/{m["n"]}</td>'
          f'<td class="num">{dz.get("rho_pooled", float("nan")):+.3f}</td>'
          f'<td class="num">{leak("f0_median_hz", "Hz")}</td>'
          f'<td class="num">{leak("char_rate", "ch/s")}</td>'
          f'<td class="num">{leak("rms_dbfs", "dB")}</td></tr>')
    a('</tbody></table></div>')
    a('<p><code>reference_wav_path</code> leaks nothing: pitch, rate and level all '
      'land within a few points of 50%, which is chance. '
      '<code>prompt_wav_path</code> moves variation harder &mdash; +2.27 st against '
      '+1.72 &mdash; but it also drags speaking rate down by about one Khmer '
      'character per second (42/144, p &lt; 1e-4), which is what continuation mode '
      'would be expected to do: it inherits the reference\'s timing along with its '
      'melody. Reach for cloning when you want the style alone, and continuation '
      'when you want the whole delivery.</p>')

    a('<h2>What this does not show</h2>')
    a('<div class="note warn">These are global pitch statistics, and this project '
      'has already established they do not measure how expressive speech '
      '<i>sounds</i>: <code>prosody_stats.py</code> ranked <code>mms</code> as the '
      'most expressive of four models (F0 std 5.66 st) when it had been eliminated '
      'by ear for flat, robotic prosody. Wide but wrongly-placed pitch movement '
      'reads as robotic and the statistic cannot tell the difference. So the '
      'numbers here show the channel <b>carries the reference\'s pitch behaviour</b>. '
      'Whether the result sounds good is what the blocks above are for, and your '
      'ears outrank the table.</div>')
    a('<p>The practical consequence is a different shape of control from the '
      'prompt list. Prosody by description gives you named levels you can type. '
      'Prosody by example gives you whatever is in the recording, which means the '
      'reference clip becomes the interface &mdash; and this corpus already has '
      '881 lively and 880 flat Khmer clips to draw one from.</p>')

    a('<footer>')
    a(f'Built by <code>finetune/build_style_artifact.py</code> from '
      f'<code>finetune/results/style_reference/</code>. Base VoxCPM2, no adapter. '
      f'Six speakers &times; two variation bands &times; two reference clips '
      f'&times; six Khmer sentences &times; two seeds &times; two channels = 576 '
      f'generations. Every comparison pins the speaker, the sentence and the seed, '
      f'so only the reference recording differs. Reference clips and their band '
      f'labels come from this project\'s hand-labelled Khmer corpus '
      f'(<code>finetune/data/corpus_meta.json</code>). '
      f'Audio {args.kbps}&nbsp;kbps MP3, {total[0]/1e6:.1f}&nbsp;MB embedded.')
    a('</footer></div>')
    a(f"<script>{JS}</script>")

    outp = Path(args.out)
    outp.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {outp} ({outp.stat().st_size/1e6:.2f} MB, "
          f"{total[0]/1e6:.2f} MB audio)")


if __name__ == "__main__":
    main()
