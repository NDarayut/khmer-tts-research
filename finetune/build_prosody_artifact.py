"""
Build the prosody-prompt reference page: the vetted list, audible.

WHY A PAGE AND NOT JUST THE MARKDOWN
------------------------------------
PROSODY_PROMPTS.md carries the same numbers, but two of the findings are
claims about how something SOUNDS and a table cannot settle them:

  * "(speaking quickly)" is genuinely faster rather than truncated -- the CER
    guard says so, but one listen says so faster.
  * the Khmer-language prompts are SPOKEN ALOUD rather than obeyed. That is
    the kind of thing nobody should have to take on a CER number.

So every prompt row plays its own clip against the same sentence with no
parenthetical, which is exactly the pairing the measurement used.

    .venv/bin/python finetune/build_prosody_artifact.py
"""

import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SWEEP = ROOT / "finetune" / "results" / "prompt_sweep"
PRIMARY = {"var": "f0_std_st", "rate": "char_rate",
           "pitch": "f0_median_hz", "energy": "rms_dbfs"}
UNIT = {"char_rate": "char/s", "f0_median_hz": "Hz",
        "rms_dbfs": "dB", "f0_std_st": "st"}

# The demo sentence. Short enough that a row of clips stays small, long enough
# that a rate change is audible.
DEMO = "A03"

# tier, axis, prompt_name, what you would reach for it FOR
ROWS = [
    ("use",  "pitch",  "low",           "a lower voice"),
    ("use",  "pitch",  "deep",          "a lower voice, male"),
    ("use",  "pitch",  "high",          "a higher voice"),
    ("use",  "pitch",  "very_high",     "a much higher voice"),
    ("use",  "rate",   "very_slow",     "slower speech"),
    ("use",  "rate",   "quick",         "faster speech"),
    ("care", "rate",   "slow",          "slower speech, gentler"),
    ("care", "energy", "soft",          "quieter"),
    ("care", "energy", "loud",          "louder"),
    ("no",   "rate",   "slight_quick",  "inert -- 'slightly' is not a hedge the model reads"),
    ("no",   "energy", "whisper",       "worse than the plain (speaking softly, quietly)"),
    ("no",   "energy", "shout",         "worse than the plain (speaking loudly)"),
    ("no",   "var",    "lively_expressive", "no expressiveness control exists -- it moves pitch"),
    ("no",   "var",    "kh_lively",     "written in Khmer: the model SPEAKS it"),
]

TIER = {
    "use":  ("Recommended", "Reliable in direction and in magnitude. "
                            "Hit rate at least 75%, effect at least as large as "
                            "the seed-to-seed noise."),
    "care": ("Usable with care", "Moves the right way about 85% of the time, but "
                                 "the effect is smaller than the run-to-run spread. "
                                 "Bias a batch with these; do not expect to hit a "
                                 "target in one generation."),
    "no":   ("Do not use", "Measured, and not worth reaching for."),
}


def mp3_uri(wav, kbps=32):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "24000",
         "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(out).decode(), len(out)


CSS = """
:root{
  --paper:#f6f8f6; --card:#ffffff; --ink:#131d1b; --ink-soft:#556763;
  --rule:#dbe3df; --rule-soft:#eaf0ec;
  --teal:#0b776f; --teal-wash:#e3f0ed;
  --amber:#9a5b12; --amber-wash:#f6ecdd;
  --stone:#79837f; --stone-wash:#edf0ee;
  --shadow:0 1px 2px rgba(19,29,27,.07);
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0d1413; --card:#16201e; --ink:#e4ebe8; --ink-soft:#93a6a1;
    --rule:#26332f; --rule-soft:#1c2725;
    --teal:#5cc6b6; --teal-wash:#10302c;
    --amber:#dda05c; --amber-wash:#33240f;
    --stone:#8d9793; --stone-wash:#1d2624;
    --shadow:0 1px 2px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --paper:#0d1413; --card:#16201e; --ink:#e4ebe8; --ink-soft:#93a6a1;
  --rule:#26332f; --rule-soft:#1c2725;
  --teal:#5cc6b6; --teal-wash:#10302c;
  --amber:#dda05c; --amber-wash:#33240f;
  --stone:#8d9793; --stone-wash:#1d2624;
  --shadow:0 1px 2px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:56px 26px 90px}
.kh{font-family:"Noto Sans Khmer","Khmer OS",sans-serif;line-height:2}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}

header{border-bottom:2px solid var(--ink);padding-bottom:26px;margin-bottom:34px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-soft);margin:0 0 14px}
h1{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:clamp(32px,5.2vw,46px);
  line-height:1.1;margin:0 0 14px;text-wrap:balance;letter-spacing:-.01em}
.lede{font-size:18.5px;color:var(--ink-soft);margin:0;max-width:64ch}
.lede b{color:var(--ink);font-weight:600}

h2{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:26px;
  margin:52px 0 6px;letter-spacing:-.005em}
h3{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:19px;margin:30px 0 8px}
p{max-width:70ch}
.sub{color:var(--ink-soft);margin:0 0 20px;max-width:70ch}

/* ---- prompt rows ---------------------------------------------------- */
.tier{margin:34px 0 0}
.tierhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  border-bottom:1px solid var(--rule);padding-bottom:7px;margin-bottom:4px}
.tiername{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:600}
.tier.use .tiername{color:var(--teal)}
.tier.care .tiername{color:var(--amber)}
.tier.no .tiername{color:var(--stone)}
.tierwhy{font-size:14px;color:var(--ink-soft);flex:1;min-width:260px}

.row{display:grid;grid-template-columns:minmax(0,1fr) 150px 132px 92px;
  gap:14px;align-items:center;padding:13px 0 13px 13px;
  border-bottom:1px solid var(--rule-soft);border-left:3px solid var(--stone)}
.tier.use .row{border-left-color:var(--teal)}
.tier.care .row{border-left-color:var(--amber)}
.tier.no  .row{border-left-color:var(--rule)}
.for{font-size:12.5px;color:var(--ink-soft);margin:0 0 3px}
.pstr{font-family:"IBM Plex Mono",monospace;font-size:13.5px;font-weight:500;
  cursor:pointer;border:0;background:none;color:var(--ink);padding:0;text-align:left;
  border-bottom:1px dashed var(--rule);line-height:1.45}
.pstr:hover{color:var(--teal);border-bottom-color:var(--teal)}
.pstr.copied{color:var(--teal)}
.tier.no .pstr{color:var(--ink-soft);text-decoration:line-through;
  text-decoration-color:var(--rule)}

/* effect bar, centred on zero */
.bar{position:relative;height:26px}
.bar .axis{position:absolute;left:50%;top:2px;bottom:2px;width:1px;background:var(--rule)}
.bar .fill{position:absolute;top:7px;height:12px;border-radius:2px;background:var(--stone)}
.tier.use .bar .fill{background:var(--teal)}
.tier.care .bar .fill{background:var(--amber)}
.bar .val{position:absolute;top:3px;font-family:"IBM Plex Mono",monospace;
  font-size:11.5px;color:var(--ink-soft);white-space:nowrap}

/* hit-rate strip: literally one cell per (sentence, seed) pair */
.dots{display:flex;gap:2px;align-items:center}
.dots i{width:4px;height:15px;border-radius:1px;background:var(--rule);display:block}
.dots i.on{background:var(--stone)}
.tier.use .dots i.on{background:var(--teal)}
.tier.care .dots i.on{background:var(--amber)}
.dots .n{font-family:"IBM Plex Mono",monospace;font-size:11.5px;
  color:var(--ink-soft);margin-left:7px;white-space:nowrap}

.play{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.04em;
  border:1px solid var(--rule);background:var(--card);color:var(--ink-soft);
  border-radius:3px;padding:5px 9px;cursor:pointer;white-space:nowrap;
  display:inline-flex;align-items:center;gap:6px}
.play:hover{border-color:var(--teal);color:var(--teal)}
.play.on{border-color:var(--teal);color:var(--teal);background:var(--teal-wash)}
.play .tri{width:0;height:0;border-left:6px solid currentColor;
  border-top:4px solid transparent;border-bottom:4px solid transparent}
.play.on .tri{border:0;width:7px;height:7px;background:currentColor;border-radius:1px}

@media (max-width:760px){
  .row{grid-template-columns:1fr 1fr;gap:10px}
  .row .who{grid-column:1/-1}
}

/* ---- reference / note blocks ---------------------------------------- */
.note{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--teal);
  border-radius:0 3px 3px 0;padding:15px 18px;margin:22px 0;box-shadow:var(--shadow);
  font-size:15px}
.note.warn{border-left-color:var(--amber)}
.note b{font-weight:600}

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
footer code{font-size:12.5px}
"""

JS = """
let cur=null,curBtn=null;
document.addEventListener('click',e=>{
  const p=e.target.closest('.play');
  if(p){
    if(cur){cur.pause();}
    if(curBtn){curBtn.classList.remove('on');}
    if(curBtn===p){cur=null;curBtn=null;return;}
    cur=new Audio(p.dataset.src);curBtn=p;p.classList.add('on');
    cur.onended=()=>p.classList.remove('on');
    cur.play();
    return;
  }
  const s=e.target.closest('.pstr');
  if(s){
    navigator.clipboard?.writeText(s.dataset.copy||s.textContent).then(()=>{
      const o=s.textContent;s.textContent='copied';s.classList.add('copied');
      setTimeout(()=>{s.textContent=o;s.classList.remove('copied');},900);
    }).catch(()=>{});
  }
});
"""


def scatter_svg(pairs, slope, icept):
    """dPitch vs dVariation, with the fitted line. This one picture is the
    whole argument for why the variation axis reads as inverted."""
    W, H, PL, PR, PT, PB = 620, 300, 54, 14, 14, 42
    xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
    x0, x1 = -140.0, 200.0
    y0, y1 = -3.0, 3.0
    def X(v): return PL + (v - x0) / (x1 - x0) * (W - PL - PR)
    def Y(v): return PT + (1 - (v - y0) / (y1 - y0)) * (H - PT - PB)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pitch change versus '
         f'variation change, {len(pairs)} paired generations">']
    o.append(f'<rect x="{PL}" y="{PT}" width="{W-PL-PR}" height="{H-PT-PB}" '
             f'fill="none" stroke="var(--rule)"/>')
    for gx in (-100, 0, 100, 200):
        o.append(f'<line x1="{X(gx):.1f}" y1="{PT}" x2="{X(gx):.1f}" y2="{H-PB}" '
                 f'stroke="var(--rule-soft)"/>')
        o.append(f'<text x="{X(gx):.1f}" y="{H-PB+16}" font-size="11" '
                 f'fill="var(--ink-soft)" text-anchor="middle" '
                 f'font-family="IBM Plex Mono, monospace">{gx:+d}</text>')
    for gy in (-2, 0, 2):
        o.append(f'<line x1="{PL}" y1="{Y(gy):.1f}" x2="{W-PR}" y2="{Y(gy):.1f}" '
                 f'stroke="var(--rule-soft)"/>')
        o.append(f'<text x="{PL-8}" y="{Y(gy)+4:.1f}" font-size="11" '
                 f'fill="var(--ink-soft)" text-anchor="end" '
                 f'font-family="IBM Plex Mono, monospace">{gy:+d}</text>')
    for x, y in zip(xs, ys):
        if x0 <= x <= x1 and y0 <= y <= y1:
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.4" '
                     f'fill="var(--teal)" fill-opacity=".33"/>')
    o.append(f'<line x1="{X(x0):.1f}" y1="{Y(slope*x0+icept):.1f}" '
             f'x2="{X(x1):.1f}" y2="{Y(slope*x1+icept):.1f}" '
             f'stroke="var(--amber)" stroke-width="2"/>')
    o.append(f'<text x="{(PL+W-PR)/2:.0f}" y="{H-6}" font-size="12" '
             f'fill="var(--ink-soft)" text-anchor="middle">'
             f'change in median pitch (Hz)</text>')
    o.append(f'<text x="14" y="{(PT+H-PB)/2:.0f}" font-size="12" '
             f'fill="var(--ink-soft)" text-anchor="middle" '
             f'transform="rotate(-90 14 {(PT+H-PB)/2:.0f})">'
             f'change in variation (st)</text>')
    o.append('</svg>')
    return "".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SWEEP / "prosody_prompts.html"))
    ap.add_argument("--kbps", type=int, default=32)
    args = ap.parse_args()

    doc = json.loads((SWEEP / "prompt_sweep.json").read_text(encoding="utf-8"))["axes"]
    idx = {(ax, e["prompt_name"]): e for ax, es in doc.items() for e in es}
    rows = [json.loads(l) for l in
            (SWEEP / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cers = json.loads((SWEEP / "prompt_cer.json").read_text(encoding="utf-8"))
    cer_of = {(c["axis"], c["prompt_name"]): c["cer"] for c in cers if c["id"] == DEMO}
    sent = {e["id"]: e["sentence"] for e in json.loads(
        (ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}[DEMO]

    # the scatter behind the variation finding
    ref = {(r["id"], r["seed"]): r for r in rows
           if r["prompt_name"] == "__bare__" and r["axis"] == "var"}
    pairs = []
    for r in rows:
        if r["axis"] != "var" or r["prompt_name"] == "__bare__":
            continue
        b = ref.get((r["id"], r["seed"]))
        if b:
            pairs.append((r["f0_median_hz"] - b["f0_median_hz"],
                          r["f0_std_st"] - b["f0_std_st"]))
    slope, icept = np.polyfit([p[0] for p in pairs], [p[1] for p in pairs], 1)

    total = [0]
    def clip(axis, name):
        w = SWEEP / "audio" / axis / f"{DEMO}_{name}.wav"
        if not w.exists():
            return None
        uri, n = mp3_uri(w, args.kbps)
        total[0] += n
        return uri

    P = []
    a = P.append
    a("<title>Khmer Prosody Prompts</title>")
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:wght@500;600&family=Source+Sans+3:wght@400;600&'
      'family=IBM+Plex+Mono:wght@400;500;600&'
      'family=Noto+Sans+Khmer:wght@400;500&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div class="wrap"><header>')
    a('<p class="eyebrow">VoxCPM2 &middot; base model, no adapter &middot; '
      '1,176 generations</p>')
    a('<h1>Khmer Prosody Prompts</h1>')
    a('<p class="lede">Which parenthetical prompts steer Khmer speech, how often '
      'they work, and the one axis that has <b>no control at all</b> &mdash; '
      'measured across 28 wordings and seven strategies, each paired against '
      'the same sentence at the same seed with no prompt.</p>')
    a('</header>')

    a('<div class="demo"><div class="lbl">every clip on this page says</div>'
      f'<div class="kh">{html.escape(sent)}</div></div>')

    a('<p class="sub">Prefix the prompt straight onto the text, no separator: '
      '<code>model.generate(text=prompt + khmer_sentence)</code>. '
      'Click any wording to copy it.</p>')

    bare = clip("rate", "__bare__")
    if bare:
        a('<div class="note"><b>Start here.</b> This is the sentence with no '
          'parenthetical at all &mdash; the reference every number below is '
          'measured against. '
          f'<button class="play" data-src="{bare}"><span class="tri"></span>'
          'no prompt</button></div>')

    for tier in ("use", "care", "no"):
        name, why = TIER[tier]
        a(f'<div class="tier {tier}"><div class="tierhead">'
          f'<span class="tiername">{name}</span>'
          f'<span class="tierwhy">{why}</span></div>')
        for t, axis, pname, who in ROWS:
            if t != tier:
                continue
            e = idx[(axis, pname)]
            mk = PRIMARY[axis]
            m = e["measures"][mk]
            d, hits, n = m["median_delta"], m["hits"], m["n"]
            span = {"char_rate": 3.0, "f0_median_hz": 90.0,
                    "rms_dbfs": 4.0, "f0_std_st": 1.2}[mk]
            frac = max(-1.0, min(1.0, d / span))
            w = abs(frac) * 50
            left = 50 if frac >= 0 else 50 - w
            lab = (f'left:calc(50% + {w+4:.1f}%)' if frac >= 0
                   else f'right:calc(50% + {w+4:.1f}%)')
            a('<div class="row">')
            a(f'<div class="who"><p class="for">{html.escape(who)}</p>'
              f'<button class="pstr" data-copy="{html.escape(e["prompt"])}">'
              f'{html.escape(e["prompt"])}</button></div>')
            a(f'<div class="bar"><span class="axis"></span>'
              f'<span class="fill" style="left:{left:.1f}%;width:{w:.1f}%"></span>'
              f'<span class="val" style="{lab}">{d:+.2f} {UNIT[mk]}</span></div>')
            dots = "".join(f'<i class="{"on" if i < hits else ""}"></i>'
                           for i in range(n))
            a(f'<div class="dots">{dots}<span class="n">{hits}/{n}</span></div>')
            u = clip(axis, pname)
            a(f'<button class="play" data-src="{u}"><span class="tri"></span>play</button>'
              if u else '<span></span>')
            a('</div>')
        a('</div>')

    a('<h2>The two numbers, and why both</h2>')
    a('<p class="sub">A prompt you can plug into any text has to clear two '
      'different bars, and they disagree often enough to be worth separating.</p>')
    a('<p><b>Hit rate</b> &mdash; the strip above &mdash; is one cell per '
      '(sentence, seed) pair: did this wording move the measure the way it asked? '
      'That answers <i>will it work on the next sentence</i>. '
      '<b>snr</b> is the median effect divided by the seed-to-seed spread within '
      'one sentence, and answers <i>can I predict how much</i>. Below 1, the noise '
      'is bigger than the effect.</p>')
    a('<div class="note warn"><b>Energy is the case where they disagree.</b> '
      '<code>(speaking softly, quietly)</code> moves the right way on 21 of 24 '
      'pairs &mdash; p&nbsp;=&nbsp;0.0003, thoroughly reliable &mdash; but its '
      '2.9&nbsp;dB effect sits under a 6.5&nbsp;dB seed spread, so any single '
      'generation lands somewhere unpredictable. Use it to bias a batch, not to '
      'hit a level.</div>')

    a('<h2>Plain wordings beat vivid ones</h2>')
    a('<p class="sub">The most useful rule the sweep produced, and the one most '
      'likely to be guessed backwards.</p>')
    a('<div class="scroll"><table><thead><tr><th>instead of</th><th>use</th>'
      '<th class="num">effect</th><th class="num">effect</th></tr></thead><tbody>')
    for bad, good, ax in ((("energy", "whisper"), ("energy", "soft"), "dB"),
                          (("energy", "shout"), ("energy", "loud"), "dB"),
                          (("pitch", "very_high"), ("pitch", "high"), "Hz")):
        eb, eg = idx[bad], idx[good]
        mb = eb["measures"][PRIMARY[bad[0]]]; mg = eg["measures"][PRIMARY[good[0]]]
        a(f'<tr><td><code>{html.escape(eb["prompt"])}</code></td>'
          f'<td><code>{html.escape(eg["prompt"])}</code></td>'
          f'<td class="num">{mb["median_delta"]:+.2f} {ax}</td>'
          f'<td class="num">{mg["median_delta"]:+.2f} {ax}</td></tr>')
    a('</tbody></table></div>')
    a('<p>The one exception is rate, where <code>(speaking very slowly and '
      'deliberately)</code> beats <code>(speaking slowly)</code> &mdash; '
      '&minus;2.32 against &minus;1.50&nbsp;char/s. Intensifiers help on the slow '
      'side of rate and nowhere else. <code>slightly</code> is inert everywhere.</p>')

    a('<h2>Faster, not truncated</h2>')
    a('<p class="sub">Speaking rate is characters divided by duration, so a model '
      'that drops the end of the sentence scores as faster. This project has been '
      'burned by exactly that before, so no rate prompt is recommended without the '
      'transcription check.</p>')
    a('<div class="scroll"><table><thead><tr><th>prompt</th>'
      '<th class="num">rate</th><th class="num">median CER</th></tr></thead><tbody>')
    for ax, nm in (("rate", "__bare__"), ("rate", "quick"), ("rate", "very_slow"),
                   ("var", "kh_lively"), ("var", "kh_flat")):
        c = cer_of.get((ax, nm))
        e = idx.get((ax, nm))
        lbl = "no prompt" if nm == "__bare__" else html.escape(e["prompt"])
        rt = (f'{e["measures"][PRIMARY[ax]]["median_delta"]:+.2f} char/s'
              if e and ax == "rate" else "&mdash;")
        a(f'<tr><td><code>{lbl}</code></td><td class="num">{rt}</td>'
          f'<td class="num">{c*100:.1f}%</td></tr>' if c is not None else "")
    a('</tbody></table></div>')
    a('<p>Against a 12&ndash;15% floor for this ASR on natural speech, '
      '<code>(speaking quickly)</code> is clean. The two Khmer-language prompts '
      'are not &mdash; and the reason is audible in their rows above: '
      '<b>the model reads the prompt out loud</b> instead of obeying it. '
      'The control channel is English-bound.</p>')

    a('<h2>Expressiveness: there is no control to find</h2>')
    a('<p class="sub">28 wordings across seven strategies &mdash; graded intensity, '
      'naming the acoustic quantity, role framing, emotion, pitch-pinning, prosodic '
      'correlates, and Khmer phrasing. None raised measured pitch variation. '
      'The reason is that expressive vocabulary is routed into the '
      '<i>pitch</i> control.</p>')
    a('<div class="scroll"><table><thead><tr><th>wording</th>'
      '<th class="num">change in pitch</th></tr></thead><tbody>')
    for nm in ("pauses", "storyteller", "anim_extreme", "lively_expressive",
               "level_tone", "flat_monotone"):
        e = idx[("var", nm)]
        a(f'<tr><td><code>{html.escape(e["prompt"])}</code></td>'
          f'<td class="num">{e["measures"]["f0_median_hz"]["median_delta"]:+.1f} Hz</td>'
          f'</tr>')
    a('</tbody></table></div>')
    a('<p>Ask for animation, get a higher voice &mdash; 40 to 84&nbsp;Hz higher, '
      'which is a voice-sized change, not a delivery-sized one. Ask for flatness '
      'and pitch does not move at all.</p>')

    a('<figure>')
    a(scatter_svg(pairs, slope, icept))
    a(f'<figcaption>Every one of the {len(pairs)} paired generations on the '
      f'variation axis. Raising pitch mechanically narrows relative semitone '
      f'spread: the fit is {slope:+.5f} st per Hz, so a 50&nbsp;Hz rise costs '
      f'{slope*50:.2f}&nbsp;st on its own. That is what made the axis look '
      f'<i>inverted</i> &mdash; <code>(a lively, expressive delivery)</code> '
      f'measured &minus;0.66&nbsp;st, of which &minus;0.39 was simply the pitch '
      f'rise it caused.</figcaption>')
    a('</figure>')

    a('<div class="note warn"><b>With the pitch component removed, nothing is '
      'left.</b> No wording raises variation at p&nbsp;&lt;&nbsp;0.05. Intent '
      'stops predicting the outcome: the top of the residual ranking is '
      '<code>(reading with dramatic pauses between phrases)</code>, but third and '
      'fourth are <code>(a robotic, emotionless machine voice)</code> and '
      '<code>(a bored, deadpan tone)</code> &mdash; which asked for the opposite. '
      'Pinning the pitch does not rescue it either: <code>(a normal-pitched voice '
      'with wide pitch variation)</code> still rose +39.9&nbsp;Hz, because the '
      'expressive half wins.</div>')
    a('<p>So expressiveness is not a prompt problem, and more wordings will not '
      'fix it. It needs training.</p>')

    a('<footer>')
    a(f'Built by <code>finetune/build_prosody_artifact.py</code> from '
      f'<code>finetune/results/prompt_sweep/</code>. Base VoxCPM2, no adapter. '
      f'Eight sentences from the frozen Khmer eval set &times; 3 seeds &times; '
      f'{len(idx)} conditions = 1,176 generations. Each prompt is paired against '
      f'the same sentence at the same seed with no parenthetical; the bars are '
      f'median within-pair changes and the strips are per-pair sign counts. '
      f'CER from the project\'s Khmer CTC ASR '
      f'(<code>finetune/score_prompt_sweep_cer.py</code>). '
      f'Audio {args.kbps}&nbsp;kbps MP3, {total[0]/1e6:.1f}&nbsp;MB embedded. '
      f'Full tables in <code>finetune/PROSODY_PROMPTS.md</code>.')
    a('</footer></div>')
    a(f"<script>{JS}</script>")

    out = Path(args.out)
    out.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, "
          f"{total[0]/1e6:.2f} MB audio)")


if __name__ == "__main__":
    main()
