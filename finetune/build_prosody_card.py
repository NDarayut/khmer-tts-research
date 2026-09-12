"""
The prosody prompt card: the eight strings to use, playable, plus a file bundle.

WHY THIS EXISTS SEPARATELY
--------------------------
`build_prosody_artifact.py` is the study -- how the prompts were found, which
families failed, how the resolution was computed. This is the answer stripped of
the reasoning: the exact strings, their levels, and a button per clip.

It also packages the clips as files, because the artifact viewer's sandbox
blocks any download the page itself starts -- `<a download>` and script-driven
saves are both inert there. So a download button on the page would look like it
worked and do nothing. Files have to be handed over out of band, and this writes
the bundle for that.

    .venv/bin/python finetune/build_prosody_card.py
"""

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SWEEP = ROOT / "finetune" / "results" / "prompt_sweep"
DEMOS = ("A03", "A19")

# axis, rung name, level label, file slug. Order is the ladder's own order.
CARD = [
    ("rate", "very_slow",     "slow",   "rate-1-slow"),
    ("rate", "normal_pace",  "normal", "rate-2-normal"),
    ("rate", "quick",        "fast",   "rate-3-fast"),
    ("pitch", "low",         "low",    "pitch-1-low"),
    ("pitch", "normal_pitch", "normal", "pitch-2-normal"),
    ("pitch", "high",        "high",   "pitch-3-high"),
    ("energy", "lv_e_adv_m3", "quiet", "energy-1-quiet"),
    ("energy", "lv_e_adv_p4", "loud",  "energy-2-loud"),
]
REF = ("rate", "__bare__", "no prompt", "00-no-prompt")

AXIS_TITLE = {"rate": "Rate", "pitch": "Pitch", "energy": "Energy"}
AXIS_UNIT = {"rate": "char/s", "pitch": "Hz", "energy": "dB"}
AXIS_NOTE = {
    "rate": "Three levels.",
    "pitch": "Three levels.",
    "energy": "Two levels. The full span is 6.28&nbsp;dB and the run-to-run "
              "noise is 5.63&nbsp;dB, so a middle level would sit inside the "
              "noise of both ends. There is no third level to give.",
}

DONT = [
    ("(speaking at 0.8x speed)", "number ignored"),
    ("(a voice three semitones lower than normal)",
     "number ignored &mdash; and it moves pitch <i>up</i>"),
    ("(speaking at 50% volume)", "number ignored"),
    ("(speaking at a pace of 3 out of 5)", "number ignored"),
    ("(speaking slightly quickly)", "+0.18 char/s &mdash; inert"),
    ("(whispering)", "weaker than <code>(speaking very quietly)</code>"),
    ("(shouting, projecting the voice)", "weaker than <code>(speaking loudly)</code>"),
    ("any prompt written in Khmer", "the model reads it aloud, 30&ndash;44% CER"),
]

CSS = """
:root{
  --paper:#f6f8f6; --card:#ffffff; --ink:#131d1b; --ink-soft:#556763;
  --rule:#dbe3df; --rule-soft:#eaf0ec;
  --teal:#0b776f; --teal-wash:#e3f0ed; --amber:#9a5b12; --stone:#79837f;
  --shadow:0 1px 2px rgba(19,29,27,.07);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0d1413; --card:#16201e; --ink:#e4ebe8; --ink-soft:#93a6a1;
    --rule:#26332f; --rule-soft:#1c2725;
    --teal:#5cc6b6; --teal-wash:#10302c; --amber:#dda05c; --stone:#8d9793;
    --shadow:0 1px 2px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --paper:#0d1413; --card:#16201e; --ink:#e4ebe8; --ink-soft:#93a6a1;
  --rule:#26332f; --rule-soft:#1c2725;
  --teal:#5cc6b6; --teal-wash:#10302c; --amber:#dda05c; --stone:#8d9793;
  --shadow:0 1px 2px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:52px 24px 88px}
.kh{font-family:"Noto Sans Khmer","Khmer OS",sans-serif;line-height:1.9}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}

header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-soft);margin:0 0 12px}
h1{font-family:Newsreader,Georgia,serif;font-weight:600;
  font-size:clamp(30px,5vw,42px);line-height:1.1;margin:0 0 12px;
  text-wrap:balance;letter-spacing:-.01em}
.lede{font-size:18px;color:var(--ink-soft);margin:0;max-width:60ch}

h2{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:25px;
  margin:44px 0 4px}
.axnote{color:var(--ink-soft);font-size:14.5px;margin:0 0 16px;max-width:62ch}
p{max-width:68ch}

.lvl{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  margin:9px 0;box-shadow:var(--shadow);padding:14px 16px;
  display:grid;grid-template-columns:74px minmax(0,1fr) 92px;
  gap:14px;align-items:center}
.tag{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;font-weight:600;color:var(--teal);
  background:var(--teal-wash);border-radius:2px;padding:4px 0;text-align:center}
.lvl.mid .tag{color:var(--stone);background:var(--rule-soft)}
.pstr{font-family:"IBM Plex Mono",monospace;font-size:14.5px;font-weight:500;
  cursor:pointer;border:0;background:none;color:var(--ink);padding:0;
  text-align:left;border-bottom:1px dashed var(--rule);line-height:1.5;
  word-break:break-word}
.pstr:hover,.pstr.copied{color:var(--teal);border-bottom-color:var(--teal)}
.eff{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-soft);
  margin-top:4px;font-variant-numeric:tabular-nums}
.plays{display:flex;flex-direction:column;gap:5px}
.play{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.03em;
  border:1px solid var(--rule);background:var(--card);color:var(--ink-soft);
  border-radius:3px;padding:5px 8px;cursor:pointer;white-space:nowrap;
  display:inline-flex;align-items:center;gap:6px;justify-content:flex-start}
.play:hover,.play.on{border-color:var(--teal);color:var(--teal)}
.play.on{background:var(--teal-wash)}
.play .tri{width:0;height:0;border-left:6px solid currentColor;
  border-top:4px solid transparent;border-bottom:4px solid transparent;flex:none}
.play.on .tri{border:0;width:7px;height:7px;background:currentColor;border-radius:1px}
@media (max-width:620px){
  .lvl{grid-template-columns:74px 1fr}
  .plays{grid-column:1/-1;flex-direction:row}
}

.note{background:var(--card);border:1px solid var(--rule);
  border-left:3px solid var(--teal);border-radius:0 3px 3px 0;
  padding:14px 17px;margin:20px 0;box-shadow:var(--shadow);font-size:15px}
.note.warn{border-left-color:var(--amber)}
pre{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:14px 16px;overflow-x:auto;font-family:"IBM Plex Mono",monospace;
  font-size:13.5px;line-height:1.65;margin:14px 0}
.demo{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:13px 17px;margin:16px 0;box-shadow:var(--shadow)}
.demo .lbl{font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-soft);
  margin-bottom:5px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:14px 0}
th,td{text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid var(--rule-soft);
  vertical-align:top}
th{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-soft);font-weight:600;
  border-bottom:1px solid var(--rule)}
.scroll{overflow-x:auto}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--ink-soft);max-width:70ch}
"""

JS = """
let cur=null,curBtn=null;
document.addEventListener('click',e=>{
  const p=e.target.closest('.play');
  if(p){
    if(cur) cur.pause();
    if(curBtn) curBtn.classList.remove('on');
    if(curBtn===p){cur=null;curBtn=null;return;}
    cur=new Audio(p.dataset.src);curBtn=p;p.classList.add('on');
    cur.onended=()=>p.classList.remove('on');
    cur.play();
    return;
  }
  const s=e.target.closest('.pstr');
  if(s){
    const t=s.dataset.copy||s.textContent;
    navigator.clipboard?.writeText(t).then(()=>{
      const o=s.textContent;s.textContent='copied';s.classList.add('copied');
      setTimeout(()=>{s.textContent=o;s.classList.remove('copied');},900);
    }).catch(()=>{});
  }
});
"""


def wav_for(axis, name, sid):
    return SWEEP / "audio" / axis / f"{sid}_{name}.wav"


def mp3_uri(wav, kbps):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "24000",
         "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(out).decode(), len(out)


def bundle(out_dir, kbps):
    """Write the clips as real files plus a zip. The page cannot hand these over
    -- the artifact sandbox makes downloads inert -- so they travel separately."""
    pack = out_dir / "prosody_clips"
    if pack.exists():
        shutil.rmtree(pack)
    (pack / "mp3").mkdir(parents=True)
    (pack / "wav").mkdir(parents=True)
    sents = {e["id"]: e["sentence"] for e in json.loads(
        (ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}
    lv = json.loads((SWEEP / "prompt_levels.json").read_text(encoding="utf-8"))

    manifest = []
    for si, sid in enumerate(DEMOS, 1):
        for axis, name, label, slug in [REF] + CARD:
            w = wav_for(axis, name, sid)
            if not w.exists():
                continue
            stem = f"sentence{si}-{slug}"
            shutil.copyfile(w, pack / "wav" / f"{stem}.wav")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(w), "-ac", "1",
                 "-ar", "44100", "-b:a", "192k", str(pack / "mp3" / f"{stem}.mp3")],
                check=True)
            prompt = ""
            for fam in lv.get(axis, {}).values():
                if name in fam.get("rungs", {}):
                    prompt = fam["rungs"][name]["prompt"]
                    break
            manifest.append({"file": stem, "sentence_id": sid,
                             "sentence": sents[sid], "axis": axis,
                             "level": label, "prompt": prompt})

    (pack / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["Khmer prosody prompt clips -- VoxCPM2 base model, no fine-tuning.",
             "",
             "mp3/  44.1 kHz 192 kbps mono -- for slides",
             "wav/  original model output, lossless",
             "",
             "file                              axis    level   prompt",
             "-" * 100]
    for m in manifest:
        lines.append(f"{m['file']:34}{m['axis']:8}{m['level']:8}{m['prompt']}")
    lines += ["", "Sentences:"]
    for si, sid in enumerate(DEMOS, 1):
        lines.append(f"  sentence{si} ({sid}): {sents[sid]}")
    (pack / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zp = out_dir / "prosody_clips.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pack.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pack.parent))
    return pack, zp, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SWEEP / "prosody_card.html"))
    ap.add_argument("--kbps", type=int, default=40)
    args = ap.parse_args()

    lv = json.loads((SWEEP / "prompt_levels.json").read_text(encoding="utf-8"))
    sw = json.loads((SWEEP / "prompt_sweep.json").read_text(encoding="utf-8"))["axes"]
    hits = {}
    for ax, es in sw.items():
        for e in es:
            for mk, m in e["measures"].items():
                hits[(ax, e["prompt_name"])] = (m.get("hits"), m.get("n"))
    sents = {e["id"]: e["sentence"] for e in json.loads(
        (ROOT / "eval-set" / "eval.json").read_text(encoding="utf-8"))}

    # Per-SENTENCE reliability, not just per-generation. 24 pairs is 8 sentences
    # at 3 seeds, so "18/24" can hide a prompt that fails on two whole texts.
    # The count that matters for reuse is how many distinct sentences moved the
    # way the wording asked, taking each sentence's median over its own seeds.
    import numpy as np
    rows_ = [json.loads(l) for l in
             (SWEEP / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    PRIM = {"rate": "char_rate", "pitch": "f0_median_hz", "energy": "rms_dbfs"}
    texts = {}
    for ax, name, label, slug in CARD:
        if label == "normal":
            continue
        mk = PRIM[ax]
        d = -1 if label in ("slow", "low", "quiet") else 1
        ref_ = {(r["id"], r["seed"]): r for r in rows_
                if r["prompt_name"] == "__bare__" and r["axis"] == ax}
        per = {}
        for r in rows_:
            if r["axis"] == ax and r["prompt_name"] == name:
                b = ref_.get((r["id"], r["seed"]))
                if b:
                    per.setdefault(r["id"], []).append(r[mk] - b[mk])
        if per:
            ok = sum(1 for v in per.values() if np.median(v) * d > 0)
            texts[(ax, name)] = (ok, len(per))

    pack, zp, manifest = bundle(SWEEP, args.kbps)

    def rung(axis, name):
        for fam in lv.get(axis, {}).values():
            if name in fam.get("rungs", {}):
                return fam["rungs"][name], fam["sd"]
        return None, None

    total = [0]
    P, a = [], None
    P = []
    a = P.append
    a("<title>Khmer Prosody Prompt Card</title>")
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:wght@500;600&family=Source+Sans+3:wght@400;600&'
      'family=IBM+Plex+Mono:wght@400;500;600&'
      'family=Noto+Sans+Khmer:wght@400;500&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div class="wrap"><header>')
    a('<p class="eyebrow">VoxCPM2 &middot; base model, no fine-tuning &middot; Khmer</p>')
    a('<h1>Khmer Prosody Prompt Card</h1>')
    a('<p class="lede">The eight strings to use, and nothing else. Click a prompt '
      'to copy it; click a sentence to hear it.</p>')
    a('</header>')

    a('<pre>audio = model.generate(text=PROMPT + khmer_sentence)</pre>')
    a('<div class="note"><b>How these were chosen.</b> Each was generated over '
      '8 Khmer sentences at 3 seeds &mdash; 24 clips &mdash; each paired against '
      'the same sentence at the same seed with no prompt. The two counts under '
      'each string are how many of those 24 clips moved the way the wording '
      'asked, and how many of the 8 <i>distinct sentences</i> did, taking each '
      'sentence\'s median over its own seeds. The second is the one that says '
      'whether a prompt travels to new text.</div>')
    a('<div class="note"><b>One prompt at a time.</b> Do not put two of these in '
      'the same parenthetical. Combining is what broke the expressiveness axis '
      '&mdash; asked for a pitch and a delivery together, the model obeyed only '
      'one of them.</div>')

    # reference clip
    rw = wav_for(*REF[:2], DEMOS[0])
    if rw.exists():
        u, n = mp3_uri(rw, args.kbps)
        total[0] += n
        a('<div class="note"><b>No prompt at all</b> &mdash; the baseline every '
          f'number below is measured against. '
          f'<button class="play" data-src="{u}"><span class="tri"></span>'
          'play</button></div>')

    for si, sid in enumerate(DEMOS, 1):
        a(f'<div class="demo"><div class="lbl">sentence {si}</div>'
          f'<div class="kh">{html.escape(sents[sid])}</div></div>')

    for axis in ("rate", "pitch", "energy"):
        a(f'<h2>{AXIS_TITLE[axis]}</h2>')
        a(f'<p class="axnote">{AXIS_NOTE[axis]}</p>')
        for ax, name, label, slug in CARD:
            if ax != axis:
                continue
            r, sd = rung(ax, name)
            if not r:
                continue
            h, nn = hits.get((ax, name), (None, None))
            tx = texts.get((ax, name))
            mid = " mid" if label == "normal" else ""
            a(f'<div class="lvl{mid}"><span class="tag">{label}</span>')
            a(f'<div><button class="pstr" data-copy="{html.escape(r["prompt"])}">'
              f'{html.escape(r["prompt"])}</button>'
              f'<div class="eff">{r["median_delta"]:+.2f} {AXIS_UNIT[axis]}'
              + (f' &middot; {h}/{nn} clips' if h else '')
              + (f' &middot; {tx[0]}/{tx[1]} sentences' if tx else '')
              + '</div></div>')
            a('<div class="plays">')
            for si, sid in enumerate(DEMOS, 1):
                w = wav_for(ax, name, sid)
                if not w.exists():
                    continue
                u, n = mp3_uri(w, args.kbps)
                total[0] += n
                a(f'<button class="play" data-src="{u}">'
                  f'<span class="tri"></span>s{si}</button>')
            a('</div></div>')

    a('<h2>Do not use these</h2>')
    a('<div class="scroll"><table><thead><tr><th>prompt</th><th>why</th></tr>'
      '</thead><tbody>')
    for p_, why in DONT:
        a(f'<tr><td><code>{html.escape(p_)}</code></td><td>{why}</td></tr>')
    a('</tbody></table></div>')

    a('<h2>The audio files</h2>')
    a('<div class="note warn"><b>They are not downloadable from this page.</b> '
      'The viewer this artifact runs in blocks any download a page starts itself, '
      'so a download button here would look like it worked and do nothing. '
      f'The {len(manifest)} clips were sent to you as '
      '<code>prosody_clips.zip</code> instead &mdash; '
      '<code>mp3/</code> at 44.1&nbsp;kHz 192&nbsp;kbps for slides, '
      '<code>wav/</code> lossless, plus a <code>README.txt</code> mapping every '
      'file to its prompt.</div>')
    a('<p>Filenames are <code>sentence1-pitch-3-high.mp3</code> and so on, so '
      'they sort into ladder order in a file picker.</p>')

    a('<footer>')
    a(f'Built by <code>finetune/build_prosody_card.py</code>. Effects are median '
      f'within-sentence changes against the same sentence at the same seed with no '
      f'parenthetical, over 8 sentences &times; 3 seeds. Hit counts are how many '
      f'of those 24 moved in the direction asked. Energy carries two levels '
      f'rather than three because its 6.28&nbsp;dB span sits inside a 5.63&nbsp;dB '
      f'noise floor. Full study: <code>finetune/PROSODY_PROMPTS.md</code>. '
      f'Page audio {args.kbps}&nbsp;kbps, {total[0]/1e6:.1f}&nbsp;MB embedded.')
    a('</footer></div>')
    a(f"<script>{JS}</script>")

    outp = Path(args.out)
    outp.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {outp} ({outp.stat().st_size/1e6:.2f} MB)")
    print(f"bundle {zp} ({zp.stat().st_size/1e6:.2f} MB, {len(manifest)} clips)")


if __name__ == "__main__":
    main()
