"""
Build a self-contained listening page from one or two nvv_infer.py result dirs.

WHY THIS EXISTS
---------------
The question this experiment asks -- did the model learn to cough when told to
cough -- has no trustworthy automatic answer. This project has already
established that the learned quality predictors are inverted for its purposes,
and no NVV detector is installed. The instrument is a person listening.

So the page is built to be listened to, not read: every clip is paired with its
own untagged control, generated from the same sentence at the same seed, so the
only difference between the two is the tag. If the "cough" is also there
without the tag, the tag did nothing.

Clips are embedded as MP3 data URIs, so the page is one file with no external
requests and keeps working after the audio directory is deleted.

    python finetune/build_nvv_artifact.py --base finetune/results/nvv/base \
        [--lora finetune/results/nvv/lora] --out finetune/results/nvv/listen.html
"""

import argparse
import base64
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def mp3_data_uri(wav, kbps):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "24000",
         "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(out).decode(), len(out)


def load(dirpath):
    d = Path(dirpath)
    return json.loads((d / "manifest.json").read_text(encoding="utf-8")), d


CSS = """
:root{
  --paper:#f4f6f4; --card:#ffffff; --ink:#131e1d; --ink-soft:#4c605d;
  --line:#d7ded9; --line-soft:#e8ede9;
  --teal:#0d6b62; --teal-soft:#e2efec; --amber:#a35a08; --amber-soft:#f7ecdd;
  --shadow:0 1px 2px rgba(19,30,29,.06);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0e1615; --card:#16211f; --ink:#e6ece9; --ink-soft:#9aada9;
    --line:#273633; --line-soft:#1e2b29;
    --teal:#5dc8b8; --teal-soft:#12312d; --amber:#e2a45c; --amber-soft:#33240f;
    --shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --paper:#0e1615; --card:#16211f; --ink:#e6ece9; --ink-soft:#9aada9;
  --line:#273633; --line-soft:#1e2b29;
  --teal:#5dc8b8; --teal-soft:#12312d; --amber:#e2a45c; --amber-soft:#33240f;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  font-size:15px; line-height:1.55;
}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--teal);
  margin:0 0 10px;
}
h1{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600;
  font-size:clamp(28px,4.2vw,40px); line-height:1.15; margin:0 0 12px;
  text-wrap:balance; letter-spacing:-.01em;
}
.lede{max-width:64ch;color:var(--ink-soft);margin:0}
.lede strong{color:var(--ink)}
h2{
  font-family:"IBM Plex Serif",Georgia,serif; font-size:21px; font-weight:600;
  margin:44px 0 6px; letter-spacing:-.005em;
}
h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--ink-soft);font-weight:400}
p{max-width:64ch}
.note{
  background:var(--card); border:1px solid var(--line); border-left:3px solid var(--teal);
  padding:14px 18px; margin:22px 0 0; font-size:14px; color:var(--ink-soft);
  box-shadow:var(--shadow);
}
.note b{color:var(--ink)}
.taghead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin:40px 0 4px}
.tagname{
  font-family:"IBM Plex Mono",monospace; font-size:20px; font-weight:600;
  color:var(--ink);
}
.badge{
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; padding:3px 8px; border-radius:2px; font-weight:500;
}
.badge.novel{background:var(--amber-soft);color:var(--amber);border:1px solid var(--amber)}
.badge.doc{background:var(--teal-soft);color:var(--teal);border:1px solid var(--teal)}
.tagsub{font-size:13.5px;color:var(--ink-soft);margin:0 0 16px;max-width:64ch}
.carrier{
  background:var(--card); border:1px solid var(--line); border-radius:3px;
  margin-bottom:12px; box-shadow:var(--shadow); overflow:hidden;
}
.carrier-text{
  padding:12px 16px; border-bottom:1px solid var(--line-soft);
  font-size:14px; font-family:"IBM Plex Mono",monospace; line-height:1.5;
}
.carrier-text .slot{
  background:var(--teal-soft); color:var(--teal); padding:1px 5px;
  border-radius:2px; font-weight:600;
}
.seeds{display:flex;flex-wrap:wrap;gap:0}
.seed{
  flex:1 1 200px; min-width:190px; padding:12px 16px;
  border-right:1px solid var(--line-soft);
}
.seed:last-child{border-right:0}
.seedlabel{
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-soft); margin-bottom:8px;
}
.pair{display:flex;flex-direction:column;gap:6px}
.btn{
  display:flex; align-items:center; gap:9px; width:100%; text-align:left;
  border:1px solid var(--line); background:transparent; color:var(--ink);
  padding:7px 10px; border-radius:2px; cursor:pointer; font:inherit;
  font-size:13px; transition:background .12s,border-color .12s;
}
.btn:hover{background:var(--teal-soft);border-color:var(--teal)}
.btn:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
.btn.playing{background:var(--teal-soft);border-color:var(--teal)}
.btn .ico{
  width:16px;height:16px;flex:none;border-radius:50%;
  border:1.5px solid currentColor;position:relative;color:var(--teal);
}
.btn .ico::after{
  content:"";position:absolute;left:5.5px;top:3.5px;
  border-left:5px solid currentColor;border-top:3.5px solid transparent;
  border-bottom:3.5px solid transparent;
}
.btn.playing .ico::after{
  left:4.5px;top:4px;border:0;width:6px;height:7px;
  background:currentColor;
}
.btn.ctl{color:var(--ink-soft);font-size:12.5px}
.btn.ctl .ico{color:var(--ink-soft)}
.dur{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:11.5px;
  color:var(--ink-soft);font-variant-numeric:tabular-nums}
.verdict{display:flex;gap:5px;margin-top:8px}
.vbtn{
  flex:1; border:1px solid var(--line); background:transparent; color:var(--ink-soft);
  font:inherit; font-size:11.5px; padding:4px 6px; border-radius:2px; cursor:pointer;
  font-family:"IBM Plex Mono",monospace;
}
.vbtn:hover{border-color:var(--ink-soft)}
.vbtn[aria-pressed="true"]{background:var(--teal);border-color:var(--teal);color:var(--paper)}
.vbtn.no[aria-pressed="true"]{background:var(--ink-soft);border-color:var(--ink-soft)}
.tally{
  font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--ink-soft);
  margin-top:10px; font-variant-numeric:tabular-nums;
}
.tally b{color:var(--teal);font-size:15px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--ink-soft)}
footer code{font-family:"IBM Plex Mono",monospace;font-size:12px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
let cur=null,curBtn=null;
document.addEventListener('click',e=>{
  const b=e.target.closest('.btn'); if(!b) return;
  if(cur){cur.pause();}
  if(curBtn){curBtn.classList.remove('playing');}
  if(curBtn===b){cur=null;curBtn=null;return;}
  const a=new Audio(b.dataset.src);
  a.play().catch(()=>{});
  a.addEventListener('ended',()=>{b.classList.remove('playing');});
  b.classList.add('playing');cur=a;curBtn=b;
});
function key(t){return 'nvv:'+t;}
let store={};
try{store=JSON.parse(localStorage.getItem('nvv-verdicts')||'{}');}catch(e){store={};}
function save(){try{localStorage.setItem('nvv-verdicts',JSON.stringify(store));}catch(e){}}
function tally(){
  document.querySelectorAll('[data-tally]').forEach(el=>{
    const tag=el.dataset.tally;
    let yes=0,tot=0;
    document.querySelectorAll(`[data-vtag="${tag}"]`).forEach(g=>{
      tot++; if(store[g.dataset.vid]==='yes') yes++;
    });
    el.innerHTML=`heard the event in <b>${yes}</b> of ${tot} takes`;
  });
}
document.addEventListener('click',e=>{
  const v=e.target.closest('.vbtn'); if(!v) return;
  const g=v.closest('[data-vid]'); const id=g.dataset.vid;
  const val=v.dataset.val;
  store[id]= store[id]===val ? null : val;
  g.querySelectorAll('.vbtn').forEach(b=>b.setAttribute('aria-pressed',String(store[id]===b.dataset.val)));
  save();tally();
});
document.querySelectorAll('[data-vid]').forEach(g=>{
  g.querySelectorAll('.vbtn').forEach(b=>b.setAttribute('aria-pressed',String(store[g.dataset.vid]===b.dataset.val)));
});
tally();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kbps", type=int, default=40)
    args = ap.parse_args()

    base_man, base_dir = load(args.base)
    lora_man, lora_dir = (load(args.lora) if args.lora else (None, None))

    doc_tags = set(base_man["documented_tags"])
    carriers = base_man["carriers"]
    seeds = base_man["seeds"]

    by = defaultdict(dict)
    total_bytes = 0
    for man, d, which in [(base_man, base_dir, "base"),
                          (lora_man, lora_dir, "lora")]:
        if man is None:
            continue
        for c in man["clips"]:
            uri, n = mp3_data_uri(d / c["wav"], args.kbps)
            total_bytes += n
            by[(c["tag"], c["carrier_index"], c["seed"])][f"{which}_{c['condition']}"] = {
                "uri": uri, "dur": c["duration"]}

    has_lora = lora_man is not None
    parts = []
    ap_ = parts.append
    ap_('<title>Can VoxCPM2 Learn to Cough</title>')
    ap_('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&'
        'family=IBM+Plex+Serif:wght@600&display=swap">')
    ap_(f"<style>{CSS}</style>")
    ap_('<div class="wrap"><header>')
    ap_('<p class="eyebrow">VoxCPM2 &middot; tag expansion &middot; English</p>')
    ap_('<h1>Can VoxCPM2 learn to cough?</h1>')
    n_novel = len([t for t in base_man["novel_tags"]])
    ap_(f'<p class="lede">VoxCPM2 documents eight non-verbal tags, but nothing in '
        f'the model treats them as a feature &mdash; they are ordinary text the model '
        f'picked up by accident from its training data. If that is right, then a tag '
        f'it has <em>never</em> seen should be no harder to teach than one it half '
        f'knows. This page is the listening test for that claim: '
        f'<strong>{n_novel} invented tags</strong> against '
        f'<strong>{len(doc_tags)} documented ones</strong>, '
        f'{"before and after fine-tuning" if has_lora else "on the untrained base model"}.</p>')
    ap_('</header>')

    ap_('<div class="note"><b>How to listen.</b> Every take has a control: the same '
        'sentence, the same random seed, with the tag deleted. Play both. If you hear '
        'the cough in the control too, the tag did nothing and the model was going to '
        'make that sound anyway. Mark what you hear &mdash; the tally is kept in this '
        'browser only.</div>')

    if not has_lora:
        ap_('<div class="note"><b>This is the &ldquo;before&rdquo; column.</b> No training '
            'has happened yet. The expectation is that the documented tags do something '
            'at least sometimes, and the invented ones do nothing at all. If an invented '
            'tag already works, that is worth knowing before spending a GPU on it.</div>')

    order = list(doc_tags) + [t for t in base_man["novel_tags"]]
    for tag in order:
        novel = tag not in doc_tags
        slug = tag.strip("[]")
        ap_('<div class="taghead">')
        ap_(f'<span class="tagname">{html.escape(tag)}</span>')
        ap_(f'<span class="badge {"novel" if novel else "doc"}">'
            f'{"invented" if novel else "documented"}</span>')
        ap_('</div>')
        ap_(f'<p class="tagsub">{"Never documented by OpenBMB. The model has no reason to know this string means anything." if novel else "Listed in the VoxCPM2 model card."}</p>')
        ap_(f'<p class="tally" data-tally="{html.escape(slug)}"></p>')

        for ci, carrier in enumerate(carriers):
            shown = html.escape(carrier).replace(
                "{tag}", f'<span class="slot">{html.escape(tag)}</span>')
            ap_('<div class="carrier">')
            ap_(f'<div class="carrier-text">{shown}</div><div class="seeds">')
            for seed in seeds:
                cell = by.get((tag, ci, seed), {})
                ap_('<div class="seed">')
                ap_(f'<div class="seedlabel">seed {seed}</div>')
                ap_(f'<div class="pair" data-vid="{html.escape(slug)}-{ci}-{seed}" '
                    f'data-vtag="{html.escape(slug)}">')
                rows = [("base_tagged", "with tag", ""),
                        ("base_untagged", "control, no tag", " ctl")]
                if has_lora:
                    rows = [("lora_tagged", "after training", ""),
                            ("base_tagged", "before training", " ctl"),
                            ("base_untagged", "control, no tag", " ctl")]
                for k, label, cls in rows:
                    c = cell.get(k)
                    if not c:
                        continue
                    ap_(f'<button class="btn{cls}" data-src="{c["uri"]}">'
                        f'<span class="ico"></span>{label}'
                        f'<span class="dur">{c["dur"]:.1f}s</span></button>')
                ap_('<div class="verdict">'
                    '<button class="vbtn" data-val="yes">heard it</button>'
                    '<button class="vbtn no" data-val="no">nothing</button>'
                    '</div>')
                ap_('</div></div>')
            ap_('</div></div>')

    ap_('<footer>')
    ap_(f'Generated by <code>finetune/build_nvv_artifact.py</code> from '
        f'<code>{html.escape(str(Path(args.base).name))}</code>'
        f'{" and <code>" + html.escape(Path(args.lora).name) + "</code>" if has_lora else ""}. '
        f'{len(by)} takes, MP3 at {args.kbps} kbps, '
        f'{total_bytes/1e6:.1f} MB of audio embedded.')
    ap_('</footer></div>')
    ap_(f"<script>{JS}</script>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    size = out.stat().st_size
    print(f"wrote {out} ({size/1e6:.1f} MB, {len(by)} takes)")
    if size > 15e6:
        print("WARNING: over 15 MB; lower --kbps", file=sys.stderr)


if __name__ == "__main__":
    main()
