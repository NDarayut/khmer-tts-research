"""
Listening page contrasting trained-on sentences with held-out ones.

WHY THIS EXISTS
---------------
The tag-sensitivity probe returned a null on held-out text (185/359 clips
worse when the tag is deleted, p=0.30, against the base model's 180/359). A
null is exactly the kind of result that deserves to be checked by ear, and the
page is the check.

Every row plays the same four things, so the comparison is always like for
like: the real recording, the sentence generated with the tag, the same
sentence with the tag deleted, and the tagged sentence on the untrained model.

The rows are labelled `trained on` or `held out`, and both are present, because
the contrast between them is the diagnostic. A model that produces the event on
sentences it memorized but not on new ones has a generalization problem. A
model that produces it in neither never learned the tag at all. Those call for
different work, and the loss numbers alone do not separate them as plainly as
listening does.

    .venv/bin/python finetune/build_heldout_artifact.py \
        --lora finetune/results/nvv/heldout_lora \
        --base finetune/results/nvv/heldout_base \
        --out finetune/results/nvv/heldout.html
"""

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.build_nvv_artifact import CSS, JS, mp3_data_uri  # noqa: E402

EXTRA_CSS = """
.clip{background:var(--card);border:1px solid var(--line);border-radius:3px;
  margin-bottom:12px;box-shadow:var(--shadow);overflow:hidden}
.clip-text{padding:12px 16px;border-bottom:1px solid var(--line-soft);
  font-size:14px;font-family:"IBM Plex Mono",monospace;line-height:1.5}
.btn.orig{border-color:var(--amber);color:var(--amber)}
.btn.orig .ico{color:var(--amber)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;
  color:var(--ink-soft);margin:14px 0 0;
  font-family:"IBM Plex Mono",monospace}
.legend span{display:flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;border:1px solid currentColor}
.split{display:inline-block;font-family:"IBM Plex Mono",monospace;
  font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  padding:2px 7px;border-radius:2px;margin-left:10px;vertical-align:2px}
.split.tr{background:#8a5a2b1a;color:#8a5a2b;border:1px solid #8a5a2b55}
.split.ho{background:#0b776f1a;color:var(--teal);border:1px solid #0b776f55}
@media (prefers-color-scheme:dark){
  .split.tr{color:#d9a05b;border-color:#d9a05b55;background:#d9a05b14}
}
.statline{font-family:"IBM Plex Mono",monospace;font-size:12.5px;
  color:var(--ink-soft);margin:6px 0 0}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kbps", type=int, default=40)
    ap.add_argument("--per-tag", type=int, default=4,
                    help="clips shown per tag per split")
    args = ap.parse_args()

    ld = Path(args.lora)
    lman = json.loads((ld / "manifest.json").read_text(encoding="utf-8"))
    bd = Path(args.base) if args.base else None
    bman = (json.loads((bd / "manifest.json").read_text(encoding="utf-8"))
            if bd else None)

    base_by = {}
    if bman:
        for c in bman["clips"]:
            base_by[(c["overfit_index"], c["seed"], c["condition"])] = c

    meta = json.loads((ROOT / "finetune" / "data-nvv" / "heldout" /
                       "heldout_meta.json").read_text(encoding="utf-8"))
    split_of = {c["overfit_index"]: c["split"] for c in meta["clips"]}

    by_tag = defaultdict(lambda: defaultdict(dict))
    originals = {}
    for c in lman["clips"]:
        by_tag[c["tag"]][c["overfit_index"]][(c["seed"], c["condition"])] = c
        originals[c["overfit_index"]] = c["original"]

    total = 0
    parts = []
    ap_ = parts.append
    ap_("<title>Does the Tag Do Anything?</title>")
    ap_('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&'
        'family=IBM+Plex+Serif:wght@600&display=swap">')
    ap_(f"<style>{CSS}{EXTRA_CSS}</style>")
    ap_('<div class="wrap"><header>')
    ap_('<p class="eyebrow">VoxCPM2 &middot; tag expansion &middot; English &middot; '
        '5.94 h, 6,000 steps</p>')
    ap_('<h1>Does the tag do anything?</h1>')
    ap_('<p class="lede">A LoRA fine-tune on 5.94 hours of tagged English, '
        '6,000 steps, about 21 passes over the corpus. Measured on 359 held-out '
        'clips from speakers it never heard, deleting the tag from the '
        'transcript changed the loss no more than chance &mdash; '
        '<strong>185 of 359</strong> clips got worse, against the untrained '
        'model\'s 180. This page is that result, audible.</p>')
    ap_('</header>')
    ap_('<div class="note"><b>Two kinds of row, deliberately.</b> '
        '<span class="split tr">trained on</span> sentences the model saw about '
        '21 times. <span class="split ho">held out</span> sentences it never saw, '
        'from speakers absent from training. If the event appears in the first '
        'and not the second, the tag was learned but did not generalize. If it '
        'appears in neither, the fine-tune installed nothing. '
        'Compare <b>with tag</b> against <b>no tag</b> within a row &mdash; if '
        'they sound the same, the tag is doing no work.</div>')
    ap_('<div class="legend">'
        '<span><i class="sw" style="border-color:var(--amber)"></i>original recording &mdash; what the event sounds like</span>'
        '<span><i class="sw" style="border-color:var(--teal)"></i>generated with the tag</span>'
        '<span><i class="sw"></i>controls</span></div>')

    for tag, clips in by_tag.items():
        ap_('<div class="taghead">')
        ap_(f'<span class="tagname">{html.escape(tag)}</span>')
        doc = tag in ("[laughing]", "[sigh]")
        ap_(f'<span class="badge {"doc" if doc else ""}">'
            f'{"documented by OpenBMB" if doc else "novel tag"}</span>')
        ap_(f'<span class="tally" data-tally="{html.escape(tag.strip("[]"))}"></span>')
        ap_('</div>')
        # train rows first, then held out, so the contrast reads top to bottom
        order = sorted(clips, key=lambda i: (split_of.get(i) != "train", i))
        for idx in order:
            cells = clips[idx]
            any_c = next(iter(cells.values()))
            sp = split_of.get(idx, "?")
            badge = ('<span class="split tr">trained on</span>' if sp == "train"
                     else '<span class="split ho">held out</span>')
            shown = html.escape(any_c["text"]).replace(
                html.escape(tag), f'<span class="slot">{html.escape(tag)}</span>')
            ap_('<div class="clip">')
            ap_(f'<div class="clip-text">{shown}{badge}</div><div class="seeds">')

            uri, n = mp3_data_uri(originals[idx], args.kbps)
            total += n
            ap_('<div class="seed"><div class="seedlabel">ground truth</div>'
                '<div class="pair">'
                f'<button class="btn orig" data-src="{uri}">'
                '<span class="ico"></span>original recording</button></div></div>')

            for seed in lman["seeds"]:
                ap_('<div class="seed">')
                ap_(f'<div class="seedlabel">seed {seed}</div>')
                ap_(f'<div class="pair" data-vid="{idx}-{seed}" '
                    f'data-vtag="{html.escape(tag.strip("[]"))}">')
                rows = [(cells.get((seed, "tagged")), ld, "with tag", ""),
                        (cells.get((seed, "untagged")), ld, "no tag", " ctl"),
                        (base_by.get((idx, seed, "tagged")), bd,
                         "before training", " ctl")]
                for c, d, label, cls in rows:
                    if not c or d is None:
                        continue
                    uri, n = mp3_data_uri(d / c["wav"], args.kbps)
                    total += n
                    ap_(f'<button class="btn{cls}" data-src="{uri}">'
                        f'<span class="ico"></span>{label}'
                        f'<span class="dur">{c["duration"]:.1f}s</span></button>')
                ap_('<div class="verdict">'
                    '<button class="vbtn" data-val="yes">event present</button>'
                    '<button class="vbtn no" data-val="no">no event</button>'
                    '</div>')
                ap_('</div></div>')
            ap_('</div></div>')

    ap_('<footer>')
    ap_(f'Generated by <code>finetune/build_heldout_artifact.py</code>. '
        f'{args.per_tag} clips per tag per split, {meta["n_train"]} trained-on and '
        f'{meta["n_val"]} held-out sentences in total. '
        f'MP3 at {args.kbps} kbps, {total/1e6:.1f} MB embedded. '
        f'Corpus: NonverbalTTS, 5.94 h. Adapter: r=64, LoRA on lm + dit + proj, '
        f'6,000 steps. Probe: removing the tag raised the loss on 185 of 359 '
        f'held-out clips (p=0.30); scrambling the transcript raised it on 339 '
        f'of 359 (p=3e-76), so the measurement works.')
    ap_('</footer></div>')
    ap_(f"<script>{JS}</script>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
