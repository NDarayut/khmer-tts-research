"""
Listening page for the overfitting run, with the ground-truth recording included.

WHY THIS EXISTS
---------------
Every earlier page compared generated audio against generated audio. For the
overfitting run there is a better reference available: the model was trained to
reproduce specific recordings, so the recording itself is the answer key. Each
row here plays four things over the same sentence --

    original      the real NonverbalTTS recording, what the event sounds like
    with tag      the memorized sentence, tag inline
    no tag        the same sentence, tag deleted
    before        the same tagged sentence on the untrained model

-- so the question stops being "is there a sound?" and becomes "is it the sound
in the original?", which is the one that matters and the one a duration or an
ASR pass cannot answer.

The CSS and the player are shared with build_nvv_artifact.py so the two pages
stay visually one family.

    python finetune/build_overfit_artifact.py \
        --lora finetune/results/nvv/overfit_lora \
        --base finetune/results/nvv/overfit_base \
        --out finetune/results/nvv/overfit.html
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
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kbps", type=int, default=40)
    ap.add_argument("--per-tag", type=int, default=6,
                    help="clips shown per tag; the run trains on more")
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

    by_tag = defaultdict(lambda: defaultdict(dict))
    originals = {}
    for c in lman["clips"]:
        by_tag[c["tag"]][c["overfit_index"]][(c["seed"], c["condition"])] = c
        originals[c["overfit_index"]] = c["original"]

    total = 0
    parts = []
    ap_ = parts.append
    ap_("<title>Overfitting the Tag</title>")
    ap_('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&'
        'family=IBM+Plex+Serif:wght@600&display=swap">')
    ap_(f"<style>{CSS}{EXTRA_CSS}</style>")
    ap_('<div class="wrap"><header>')
    ap_('<p class="eyebrow">VoxCPM2 &middot; capability probe &middot; English</p>')
    ap_('<h1>Can it learn the tag if we let it cheat?</h1>')
    ap_('<p class="lede">A normal fine-tune left the tag nearly inert: deleting '
        '<code>[cough]</code> cost about 2&ndash;6% of what scrambling the whole '
        'sentence cost, and moving the tag to the wrong place cost nothing at all. '
        'That could mean the model <em>cannot</em> tie a written tag to a sound, or '
        'just that 78 coughs was too little to learn from. So this run memorizes '
        '<strong>60 clips, 5.4 minutes</strong>, for 2,000 steps &mdash; about 265 '
        'passes over the same handful of recordings.</p>')
    ap_('</header>')
    ap_('<div class="note"><b>These are the training sentences.</b> The model has '
        'seen each one hundreds of times, so this shows what it is <em>capable</em> '
        'of, not what it can do on new text. If <span style="color:var(--amber)">'
        'original</span> and <b>with tag</b> now carry the same event while '
        '<b>no tag</b> does not, the mechanism works and the earlier failure was '
        'about data. If they still sound alike, the mechanism is the problem and '
        'no corpus fixes it.</div>')
    ap_('<div class="legend">'
        '<span><i class="sw" style="border-color:var(--amber)"></i>original recording &mdash; the answer key</span>'
        '<span><i class="sw" style="border-color:var(--teal)"></i>generated with the tag</span>'
        '<span><i class="sw"></i>controls</span></div>')

    for tag, clips in by_tag.items():
        ap_('<div class="taghead">')
        ap_(f'<span class="tagname">{html.escape(tag)}</span>')
        ap_('<span class="badge doc">memorized</span>')
        ap_(f'<span class="tally" data-tally="{html.escape(tag.strip("[]"))}"></span>')
        ap_('</div>')
        for idx in sorted(clips)[: args.per_tag]:
            cells = clips[idx]
            any_c = next(iter(cells.values()))
            shown = html.escape(any_c["text"]).replace(
                html.escape(tag), f'<span class="slot">{html.escape(tag)}</span>')
            ap_('<div class="clip">')
            ap_(f'<div class="clip-text">{shown}</div><div class="seeds">')

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
                    '<button class="vbtn" data-val="yes">matches original</button>'
                    '<button class="vbtn no" data-val="no">no event</button>'
                    '</div>')
                ap_('</div></div>')
            ap_('</div></div>')

    ap_('<footer>')
    ap_(f'Generated by <code>finetune/build_overfit_artifact.py</code>. '
        f'{args.per_tag} of 12 clips shown per tag; the run trained on all 60. '
        f'MP3 at {args.kbps} kbps, {total/1e6:.1f} MB embedded. '
        f'Evaluated on the training sentences &mdash; memorization, not '
        f'generalization.')
    ap_('</footer></div>')
    ap_(f"<script>{JS}</script>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
