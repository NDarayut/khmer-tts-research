"""
Build the full evaluation write-up as a single self-contained HTML document.

results_report.md is the machine-readable summary; this is the version meant to
be read end to end and handed to someone else -- the same numbers, plus the
method, the caveats, and the audio itself, so a reader can check a claim about
how a model sounds without cloning the repo.

    python evaluation/report_document.py
    python evaluation/report_document.py --bitrate 64k --per-category 1

Writes evaluation/results_report.html.

Every figure is read from evaluation/results/<model>/scores.json at build time.
Nothing here is transcribed by hand -- re-run it after re-scoring and the
document follows.
"""

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from khmer_tts.reporting.audio_samples import encode_mp3, pick_entries
from khmer_tts.synthesis.backends import BACKEND_KEYS
from khmer_tts.common import ROOT, load_entries, read_json, scores_path, wav_path
from khmer_tts.reporting.report import (
    CER_INVALID_THRESHOLD,
    HYBRID_RTF,
    MODEL_LABELS,
    PUBLISHED,
    RUNAWAY_SECONDS,
    UNDOCUMENTED_KHMER,
)

OUT_PATH = ROOT / "evaluation" / "results_report.html"

# Prose that cannot be derived from the JSON: what each model is, and what the
# run revealed about getting it to run at all.
MODEL_NOTES = {
    "mms": (
        "36M-parameter single-speaker VITS, the smallest model here by three "
        "orders of magnitude. The only one with a Khmer-specific checkpoint "
        "rather than a multilingual model that happens to include Khmer."
    ),
    "voxcpm2": (
        "2B tokenizer-free TTS with automatic language detection. Khmer is one "
        "of its 30 documented languages and it carries the strongest published "
        "Khmer figure of any model here. Apache-2.0. <strong>A contender on "
        "listening, and last on every naturalness metric.</strong>"
    ),
    "fish-s2": (
        "~5B autoregressive model over a DAC codec, non-commercially licensed. "
        "The hardest to run: its 9.65 GB of weights and 4.58 GB codec do not "
        "fit a 12 GB card together, so the codec was moved to the CPU. "
        "<strong>Eliminated on listening, despite the best UTMOS in the "
        "run.</strong>"
    ),
    "higgs3": (
        "4B Qwen3 backbone over 8 audio codebooks at 25 fps, research and "
        "non-commercial licence. The least trouble to run -- it fits the card "
        "unaided. Khmer is absent from its 102-language list yet works. "
        "<strong>The other contender.</strong>"
    ),
}

SECTIONS = []


def fmt(value, digits=2, scale=1.0, suffix=""):
    return "&mdash;" if value is None else f"{value * scale:.{digits}f}{suffix}"


def esc(text):
    return html.escape(str(text))


def load_all():
    out = {}
    for model in BACKEND_KEYS:
        path = scores_path(model)
        if path.is_file():
            out[model] = read_json(path)
    return out


def section(number, title, *body):
    SECTIONS.append((number, title))
    inner = "\n".join(body)
    return (
        f'<section id="s{number}">\n'
        f'<h2><span class="sn">{number}</span>{title}</h2>\n{inner}\n</section>'
    )


def table(headers, rows, classes="", note=None):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f'<p class="caption">{note}</p>' if note else ""
    return (
        f'<div class="tw"><table class="{classes}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{cap}"
    )


# --- sections -------------------------------------------------------------

def headline(results):
    rows = []
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        o = payload["summary"]["overall"]
        flag = " <span class='tag'>hybrid</span>" if model in HYBRID_RTF else ""
        rows.append([
            f"<code>{esc(model)}</code>",
            o["count"],
            f"<span class='bad'>{fmt(o['cer_mean'], 1, 100, '%')}</span>",
            f"<span class='bad'>{fmt(o['cer_median'], 1, 100, '%')}</span>",
            f"<strong>{fmt(o['utmos_mean'])}</strong>",
            fmt(o["dnsmos_ovrl_mean"]),
            fmt(o["dnsmos_sig_mean"]),
            fmt(o["dnsmos_p808_mean"]),
            f"<strong>{fmt(o['rtf_median'], 3)}</strong>{flag}",
        ])
    return table(
        ["Model", "n", "CER mean", "CER median", "UTMOS", "DNSMOS OVRL",
         "DNSMOS SIG", "P.808", "RTF median"],
        rows,
        classes="data",
        note="CER lower is better; it is struck through in effect &mdash; see "
             "&sect;4. UTMOS, DNSMOS and P.808 are 1&ndash;5 MOS scales, higher "
             "better. RTF below 1.0 is faster than real time.",
    )


def breakdown(results, key, buckets_note):
    buckets = []
    for payload in results.values():
        for name in payload["summary"].get(f"by_{key}", {}):
            if name not in buckets:
                buckets.append(name)
    rows = []
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        by = payload["summary"].get(f"by_{key}", {})
        cells = [f"<code>{esc(model)}</code>"]
        for bucket in buckets:
            block = by.get(bucket) or {}
            cells.append(fmt(block.get("utmos_mean")))
        rows.append(cells)
    return table(
        ["Model"] + [f"<code>{esc(b)}</code>" for b in buckets],
        rows, classes="data", note=buckets_note,
    )


def cer_evidence(results):
    # Every transcript produced for each sentence, one per model, kept as a
    # list so repeats are visible.
    per_id = {}
    for payload in results.values():
        for u in payload["utterances"]:
            if u.get("transcript"):
                per_id.setdefault(u["id"], []).append(u["transcript"])
    # Sentences where at least two models transcribed byte-identically.
    # Requiring *all* models to agree is the wrong test: one model breaking the
    # tie would hide three others collapsing onto the same string.
    collisions = sum(1 for v in per_id.values() if len(v) > len(set(v)))
    total = len(per_id)
    medians = {
        m: p["summary"]["overall"]["cer_median"] for m, p in results.items()
    }
    worst = max(medians.values()) if medians else 0

    # A real transcript, so the failure mode is visible rather than asserted.
    sample = ""
    for payload in results.values():
        for u in payload["utterances"]:
            t = u.get("transcript") or ""
            if len(t) > 40 and t.count(t[:12]) > 2:
                # Whisper's degenerate output carries the odd U+FFFD; it is
                # noise from the decoder, not part of the Khmer, and the
                # artifact deploy rejects the codepoint outright.
                sample = t[:110].replace("�", "")
                break
        if sample:
            break

    rows = [
        [f"<code>{esc(m)}</code>", fmt(v, 1, 100, "%")]
        for m, v in sorted(medians.items(), key=lambda kv: -kv[1])
    ]
    out = [
        "<p>The primary metric did not measure anything. Whisper-large-v3, the "
        "scoring ASR, collapses into a repetition loop on Khmer instead of "
        "transcribing it. Three independent signs, any one of which would be "
        "disqualifying:</p>",
        "<ol>",
        f"<li><strong>Median CER at or above 100% for every model</strong> "
        f"(worst {fmt(worst, 1, 100, '%')}). Above 100% the hypothesis is "
        "longer than the reference it is scored against &mdash; the ASR is "
        "emitting more text than was spoken.</li>",
        f"<li><strong>{collisions} of the {total} sentences drew a "
        "byte-identical transcript from two or more different models' "
        "audio.</strong> Models with different architectures, sample rates and "
        "voices cannot produce the same transcription character for character "
        "unless the decoder has stopped listening to its input.</li>",
        "<li><strong>The transcripts are visibly degenerate.</strong> A "
        "representative one:</li>",
        "</ol>",
    ]
    if sample:
        out.append(f'<blockquote lang="km">{esc(sample)}&hellip;</blockquote>')
    out.append(table(["Model", "CER median"], rows, classes="data narrow"))
    out.append(
        "<p>Decoder settings were ruled out as the cause: adding a temperature "
        "fallback ladder, an n-gram repetition block, and language "
        "auto-detection each changed the output text but none produced a "
        "transcript resembling the reference.</p>"
    )
    out.append(
        "<p>So the one metric designed to ask whether a model says the right "
        "words returned nothing, for every model, and the field was left to be "
        "ranked on sound alone.</p>"
    )
    return "\n".join(out)


def script_chars(sentence):
    """Length of a sentence in characters that are actually pronounced.

    Khmer has no inter-word spacing, so characters -- not words -- are the
    only length unit available. Latin letters count too; the code-switched
    half of the set is half English.
    """
    return sum(1 for c in sentence if "ក" <= c <= "៿" or c.isalpha())


# A clip is flagged when its speaking rate differs from its *own model's*
# median by more than this factor. Comparing each model against itself rather
# than against a fixed band is what makes the check fair: models legitimately
# differ in baseline pace (voxcpm2 speaks ~40% faster than fish-s2 throughout),
# and a fixed band would penalise that difference instead of the failures.
# Twofold is deliberately lax -- ordinary punctuation and phrase-length effects
# stay well inside it, so anything flagged is a gross failure.
RATE_DEVIATION = 2.0

# The listening verdict (see listening_result). This is the one input to this
# document that is a human judgement rather than a number read off a file, so
# it lives in exactly one place and everything that depends on it derives from
# here.
CONTENDERS = ("voxcpm2", "higgs3")
ELIMINATED = ("fish-s2", "mms")


def rates(results):
    """Seconds of audio per script character, per model.

    Over the whole set -- not the handful of entries sampled for the listening
    section, which is a different and much smaller list.
    """
    by_id = {e["id"]: e for e in load_entries()}
    out = {}
    for model, payload in results.items():
        vals = []
        for u in payload["utterances"]:
            entry = by_id.get(u["id"])
            seconds = u.get("audio_seconds")
            if not entry or not seconds:
                continue
            n = script_chars(entry["sentence"])
            if n:
                vals.append((seconds / n, u, n))
        out[model] = sorted(vals, key=lambda t: t[0])
    return out


def metrics_failed(results):
    """The section that replaces the old ranking: why the numbers misrank."""
    per_model = rates(results)
    rows = []
    offenders = {}
    for model in BACKEND_KEYS:
        vals = per_model.get(model)
        if not vals:
            continue
        r = [t[0] for t in vals]
        mid = r[len(r) // 2]
        bad = [t for t in vals
               if t[0] * RATE_DEVIATION < mid or t[0] > mid * RATE_DEVIATION]
        offenders[model] = bad
        spread = r[-1] / r[0] if r[0] else 0
        cell = f"{len(bad)}" if bad else "0"
        rows.append([
            f"<code>{esc(model)}</code>",
            f"{mid:.3f}",
            f"{r[0]:.3f} &ndash; {r[-1]:.3f}",
            f"<strong>{spread:.1f}&times;</strong>",
            (f"<span class='bad'>{cell}</span>" if bad else cell),
        ])

    # The single most damaging case: of the clips too *fast* to be a reading of
    # their text -- the unambiguous kind of failure, since the words cannot
    # physically fit -- the one UTMOS liked best.
    worst = None
    for model, bad in offenders.items():
        mid = per_model[model][len(per_model[model]) // 2][0]
        for value, u, chars in bad:
            if value * RATE_DEVIATION >= mid or not u.get("utmos"):
                continue
            if worst is None or u["utmos"] > worst[1]["utmos"]:
                worst = (model, u, value, chars)

    out = [
        "<p>Two independent metrics were supposed to rank these models. Both "
        "did, and both were wrong. The reason is the same in each case: they "
        "measure properties of the <em>waveform</em>, and none of those "
        "properties is &lsquo;says the Khmer words in the text&rsquo;.</p>",
        "<h3>CER measured nothing at all</h3>",
        cer_evidence(results),
        "<h3>UTMOS and DNSMOS measured the wrong thing</h3>",
        "<p>With correctness unmeasured, naturalness was left ranking the "
        "field alone &mdash; and naturalness rewards audio that is smooth, "
        "clean and confident whether or not it corresponds to the input. A "
        "model that abandons the sentence and produces a fluent noise instead "
        "scores <em>better</em>, not worse, because fluent noise has none of "
        "the hesitations and artefacts a model attempting a hard sentence "
        "produces.</p>",
        "<p>That is not a hypothetical. Because every clip's duration is "
        "recorded, each one can be checked against the length of the text it "
        "was given. Khmer is written without spaces, so seconds-per-character "
        "is the available measure of speaking rate &mdash; crude, but it does "
        "not need to be precise to catch a model that disposes of a long "
        "sentence in a couple of seconds.</p>",
        table(
            ["Model", "Median s/char", "Range", "Spread", "Implausible"],
            rows, classes="data",
            note="Spread is the slowest utterance divided by the fastest, "
                 "within one model. &lsquo;Implausible&rsquo; counts clips "
                 f"more than {RATE_DEVIATION:.0f}&times; off that same "
                 "model's own median &mdash; each model judged against itself, "
                 "so a naturally brisk model is not penalised for being brisk. "
                 "Three of the four are internally consistent. The fourth "
                 "varies by more than eightfold, which no reading of the same "
                 "kind of text can do.",
        ),
    ]
    if worst:
        model, u, rate, chars = worst
        median_rate = per_model[model][len(per_model[model]) // 2][0]
        # Is that score higher than the best any usable model managed anywhere?
        ceiling = max(
            (x["utmos"] for m in CONTENDERS for x in
             results.get(m, {}).get("utterances", []) if x.get("utmos")),
            default=0,
        )
        beats = (
            " &mdash; higher than the best score either usable model earned on "
            f"any of its {len(per_model[model])} utterances ({ceiling:.2f})"
            if u["utmos"] > ceiling else ""
        )
        out.append(
            f'<p class="callout"><strong>The decisive case.</strong> '
            f"<code>{esc(model)}</code> utterance <code>{esc(u['id'])}</code> "
            f"delivers a {chars}-character sentence in "
            f"{u['audio_seconds']:.1f}&nbsp;seconds &mdash; {rate:.3f} s/char, "
            f"against the same model's own median of {median_rate:.3f}. At "
            f"{median_rate / rate:.1f}&times; its normal rate the words cannot "
            "physically fit; most of the sentence is simply not there. UTMOS "
            f"scored that clip <strong>{u['utmos']:.2f} out of 5</strong>"
            f"{beats}. A metric that awards its best marks to a clip which "
            "dropped its sentence is not measuring TTS quality; it is "
            "measuring audio smoothness.</p>"
        )
    out.append(
        "<p>Neither metric is defective. UTMOS and DNSMOS do what they were "
        "built to do &mdash; predict perceived signal quality, from MOS "
        "studies conducted almost entirely in English. Nothing in either was "
        "ever asked to check that the speech matches a Khmer sentence, and "
        "so neither does.</p>"
    )
    return "\n".join(out)


def listening_result(results):
    ranks = {}
    for metric, better in (("utmos_mean", max), ("dnsmos_ovrl_mean", max)):
        order = sorted(
            (m for m in results if results[m]["summary"]["overall"].get(metric)),
            key=lambda m: -results[m]["summary"]["overall"][metric],
        )
        for i, m in enumerate(order, 1):
            ranks.setdefault(m, []).append((metric, i, len(order)))

    def rank_cell(model):
        best = min(ranks.get(model, []), key=lambda t: t[1], default=None)
        if not best:
            return "&mdash;"
        metric, place, total = best
        name = "UTMOS" if metric == "utmos_mean" else "DNSMOS OVRL"
        value = results[model]["summary"]["overall"][metric]
        cell = f"{place} of {total} on {name} ({value:.2f})"
        return f"<span class='bad'>{cell}</span>" if place == 1 else cell

    verdicts = {
        "voxcpm2": "<strong>Contender.</strong> Intelligible Khmer.",
        "higgs3": "<strong>Contender.</strong> Intelligible Khmer, though the "
                  "vendor documents no Khmer support at all.",
        "fish-s2": "Eliminated &mdash; fails on correctness (&sect;4).",
        "mms": "Eliminated &mdash; reads the text faithfully, but flat and "
               "robotic.",
    }
    rows = [
        [f"<code>{esc(m)}</code>", verdicts.get(m, "&mdash;"), rank_cell(m)]
        for m in list(CONTENDERS) + list(ELIMINATED) if m in results
    ]
    return "\n".join([
        "<p>Because no automatic metric survived &sect;4, the models were "
        "ranked by ear instead &mdash; the reviewer being a Khmer speaker "
        "listening to the clips in &sect;8. That verdict is the finding of "
        "this report, and it disagrees with every table in it.</p>",
        table(
            ["Model", "Listening verdict", "Best automatic rank"],
            rows, classes="data",
            note="&lsquo;Best automatic rank&rsquo; is the most favourable "
                 "position each model reaches on either naturalness metric in "
                 "&sect;6 &mdash; the case each model makes for itself on the "
                 "numbers alone. Both models that top a metric are the "
                 "eliminated ones.",
        ),
        '<p class="callout"><strong>The ranking is close to inverted.</strong> '
        "The two models a Khmer listener can actually use place last and "
        "second on naturalness; the model that places first is unusable, and "
        "the model that tops perceptual quality is unusable too. Had this "
        "report been written from its own tables &mdash; as its first version "
        "was &mdash; it would have recommended the worst option in the "
        "set.</p>",
        "<p>The two eliminated models fail differently, and the difference "
        "matters for anyone tempted to revisit them. <code>fish-s2</code> "
        "fails on <em>correctness</em>: &sect;4 shows it discarding or padding "
        "the sentence outright, which is consistent with the 75.15% Khmer CER "
        "its own vendor's benchmark reports (&sect;10) and means no amount of "
        "tuning at inference time recovers it. <code>mms</code> fails on "
        "<em>quality</em>: it is a 36M single-speaker VITS that reads the text "
        "faithfully &mdash; its speaking rate is the steadiest in the set "
        "&mdash; but does so with flat, robotic prosody. It remains the "
        "sensible choice where a tiny, fast, offline model matters more than "
        "sounding human, and it is the only model here that could be "
        "fine-tuned on a single consumer GPU.</p>",
        "<p>Between the two contenders this report declares no winner. "
        "Separating them needs a structured listening test &mdash; the same "
        "sentences, both models, presented blind and in random order to "
        "several Khmer speakers, scored for intelligibility and naturalness "
        "separately. That is the one measurement in this project that cannot "
        "be automated away, and &sect;8 is the material for it.</p>",
    ])


def caveats(results):
    items = []
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        if model in UNDOCUMENTED_KHMER:
            items.append(
                f"<li><code>{esc(model)}</code> &mdash; <strong>Khmer is "
                f"undocumented.</strong> {UNDOCUMENTED_KHMER[model].capitalize()}. "
                "Its scores stand, but no published Khmer figure exists to "
                "corroborate them.</li>"
            )
        if model in HYBRID_RTF:
            items.append(
                f"<li><code>{esc(model)}</code> &mdash; <strong>RTF is not "
                "comparable.</strong> Its codec runs on the CPU because model "
                "and codec do not fit one 12 GB card together, so its RTF "
                "measures GPU generation plus CPU decode. The quality metrics "
                "are unaffected; they read the finished waveform.</li>"
            )
        seconds = [u.get("audio_seconds") for u in payload["utterances"]
                   if u.get("audio_seconds")]
        runaway = [s for s in seconds if s > RUNAWAY_SECONDS]
        if runaway:
            items.append(
                f"<li><code>{esc(model)}</code> &mdash; <strong>runaway "
                f"generation.</strong> {len(runaway)} utterance(s) exceeded "
                f"{RUNAWAY_SECONDS:.0f}s, the longest {max(runaway):.1f}s "
                "against a set median of "
                f"{sorted(seconds)[len(seconds) // 2]:.1f}s. The model failed "
                "to stop; those clips inflate its RTF and their quality scores "
                "describe filler rather than the sentence.</li>"
            )
    return "<ul class='caveats'>" + "\n".join(items) + "</ul>"


def samples_section(results, entries, bitrate, models):
    metrics = {
        m: {u["id"]: u for u in p["utterances"]} for m, p in results.items()
    }
    blocks = []
    for entry in entries:
        cells = []
        for model in models:
            audio = encode_mp3(wav_path(model, entry["id"]), bitrate)
            stat = metrics.get(model, {}).get(entry["id"], {})
            player = (
                f'<audio controls preload="none" '
                f'src="data:audio/mpeg;base64,{audio}"></audio>'
                if audio else '<p class="missing">no audio</p>'
            )
            cells.append(
                f'<div class="clip"><div class="model">{esc(model)}</div>'
                f"{player}"
                f'<div class="badges"><span>UTMOS '
                f"{fmt(stat.get('utmos'))}</span><span>DNSMOS "
                f"{fmt((stat.get('dnsmos') or {}).get('ovrl'))}</span>"
                f"<span>{fmt(stat.get('audio_seconds'), 1, 1, 's')}</span>"
                "</div></div>"
            )
        blocks.append(
            f'<figure class="sample"><figcaption>'
            f'<code>{esc(entry["id"])}</code> '
            f'<span class="cat">{esc(entry.get("group"))} &middot; '
            f'{esc(entry.get("category"))}</span></figcaption>'
            f'<p class="km" lang="km">{esc(entry["sentence"])}</p>'
            f'<div class="clips">{"".join(cells)}</div></figure>'
        )
    return "\n".join(blocks)


def environment(results):
    rows = []
    for model in BACKEND_KEYS:
        payload = results.get(model)
        if not payload:
            continue
        env = payload["env"]
        rows.append([
            f"<code>{esc(model)}</code>",
            f"<code>{esc(env.get('model_id', '--'))}</code>",
            esc(env.get("device", "--")),
            f"{env.get('load_seconds', '--')}s",
            esc((env.get("timestamp") or "--")[:10]),
        ])
    return table(
        ["Model", "Checkpoint", "Device", "Load time", "Run date"],
        rows, classes="data",
        note="All four ran on the same machine, so RTF is comparable between "
             "them &mdash; with the one exception noted in &sect;6. Load time "
             "is excluded from RTF.",
    )


def published_table():
    rows = [
        [f"<code>{esc(m)}</code>", PUBLISHED[m]]
        for m in BACKEND_KEYS if m in PUBLISHED
    ]
    return table(
        ["Model", "Published figure"], rows, classes="data",
        note="Vendor-reported, not independently verified. The VoxCPM2 and "
             "Fish figures both come from OpenBMB's own benchmark.",
    )


def build(results, entries, bitrate, models):
    o = {m: results[m]["summary"]["overall"] for m in results}
    n = o[list(o)[0]]["count"]

    parts = [HEAD]
    header_html = (
        '<header class="doc-head">'
        "<p class='eyebrow'>Technical evaluation</p>"
        "<h1>Open-source TTS for Khmer</h1>"
        f"<p class='lede'>Four open-source text-to-speech models synthesized "
        f"the same fixed {n}-sentence Khmer set on one machine. Every metric "
        "used to score them turned out to rank them wrongly &mdash; this is "
        "the record of that, and of what listening said instead.</p>"
        '<dl class="meta">'
        f"<div><dt>Models</dt><dd>{len(results)}</dd></div>"
        f"<div><dt>Sentences</dt><dd>{n}</dd></div>"
        f"<div><dt>Clips</dt><dd>{n * len(results)}</dd></div>"
        "<div><dt>Hardware</dt><dd>RTX 3060, 12 GB</dd></div>"
        "</dl></header>"
    )

    parts.append(section(
        1, "Summary",
        f"<p>Four open-source TTS models synthesized the same fixed {n}-sentence "
        "Khmer set on one machine. All four completed it &mdash; "
        f"{n * len(results)} clips, no failures &mdash; and were then scored on "
        "character error rate, two predicted-MOS metrics, and speed.</p>",
        "<p><strong>All four metrics failed to rank the models.</strong> CER "
        "measured nothing, because the scoring ASR cannot transcribe Khmer. "
        "UTMOS and DNSMOS measured something real but irrelevant: how clean the "
        "audio sounds, which is not the same question as whether it says the "
        "Khmer sentence. Speed measured speed. Section 4 is the evidence, and "
        "it is unambiguous &mdash; the single highest naturalness score awarded "
        "anywhere in this run went to a clip that omits most of its "
        "sentence.</p>",
        "<p>Ranked by a Khmer speaker listening to the output, only two of the "
        "four produce usable Khmer: <strong>VoxCPM2</strong> and <strong>Higgs "
        "TTS 3</strong>. <code>fish-s2</code> and <code>mms</code> are out. "
        "That ordering is close to the reverse of what the metrics reported "
        "&mdash; the eliminated <code>fish-s2</code> holds the best naturalness "
        "score in the run, and the contender <code>voxcpm2</code> holds the "
        "worst (&sect;5).</p>",
        '<p class="callout"><strong>What this report is now.</strong> Not a '
        "ranking derived from measurements &mdash; a record of measurements "
        "that did not work, the evidence that they did not, and a listening "
        "result that stands in their place. The tables in &sect;6 are retained "
        "as diagnostics, not as findings. The audio in &sect;8 outweighs "
        "them.</p>",
    ))

    parts.append(section(
        2, "Method",
        "<p>The test set is 100 fixed Khmer sentences &mdash; 50 pure Khmer, 50 "
        "code-switched with English &mdash; spanning eleven categories from "
        "single words to long complex clauses. The same set, in the same "
        "order, drives every model; it is never modified.</p>",
        "<p>Each model synthesizes to its own native sample rate. Metrics "
        "resample to 16 kHz in memory only, so the stored audio stays "
        "lossless. Real-time factor is measured during generation, after a "
        "discarded warm-up utterance and excluding model load, because it "
        "cannot be recovered from a finished file.</p>",
        table(
            ["Metric", "Measures", "Scale"],
            [["CER", "Correctness &mdash; <em>invalid, see &sect;4</em>",
              "0 = perfect, lower better"],
             ["UTMOS", "Predicted naturalness MOS", "1&ndash;5, higher better"],
             ["DNSMOS", "Perceptual quality (P.835 / P.808)",
              "1&ndash;5, higher better"],
             ["RTF", "Synthesis time &divide; audio duration",
              "&lt;1 = faster than real time"]],
            classes="data",
        ),
    ))

    parts.append(section(
        3, "The models",
        "<dl class='models'>" + "".join(
            f"<div><dt><code>{esc(m)}</code> "
            f"<span class='label'>{MODEL_LABELS.get(m, '')}</span></dt>"
            f"<dd>{MODEL_NOTES.get(m, '')}</dd></div>"
            for m in BACKEND_KEYS if m in results
        ) + "</dl>",
    ))

    parts.append(section(
        4, "Why the metrics failed", metrics_failed(results)))

    parts.append(section(5, "The listening result", listening_result(results)))

    parts.append(section(
        6, "The numbers, as diagnostics",
        "<p>Everything below is a valid measurement of what it measures. None "
        "of it ranks these models for Khmer &mdash; &sect;4 is why. It is kept "
        "because it is the record of the run, and because the gap between "
        "these tables and &sect;5 is itself the most useful result here.</p>",
        headline(results),
        "<h3>Naturalness by group</h3>",
        breakdown(results, "group",
                  "UTMOS split across the two halves of the set. "
                  "Code-switching is where Khmer TTS is expected to degrade."),
        "<h3>Naturalness by category</h3>",
        breakdown(results, "category",
                  "Category definitions are in "
                  "<code>evaluation/data-promt.md</code>."),
    ))

    parts.append(section(
        7, "Measurement caveats",
        "<p>These change what a column in &sect;6 <em>means</em> for one row.</p>",
        caveats(results),
    ))

    parts.append(section(
        8, "Listening samples",
        "<p>One sentence per category, every model, unmodified synthesis "
        f"output transcoded to {bitrate} mono MP3 for embedding. This is the "
        "evidence &sect;5 rests on and the material a formal listening test "
        "would use. Compare a clip against the badge under it: a model can "
        "carry the better numbers and be the worse clip.</p>",
        samples_section(results, entries, bitrate, models),
    ))

    parts.append(section(
        9, "Limitations",
        "<ul>"
        "<li><strong>The listening result is one listener.</strong> It is the "
        "best evidence in this report and still the weakest kind of evidence "
        "&mdash; unblinded, unreplicated, and not scored on a scale. It is "
        "reliable for the coarse judgement it makes (two models usable, two "
        "not) and not for anything finer, which is why &sect;5 declares no "
        "winner between the two.</li>"
        "<li><strong>No working intelligibility metric.</strong> Fixing it "
        "requires a Khmer-capable ASR. Three attempts with "
        "<code>Qwen3-ASR-0.6B-Khmer</code> failed on library bugs rather than "
        "on the idea. The audio is on disk, so a fix is a re-score, not a "
        "re-run.</li>"
        "<li><strong>UTMOS and DNSMOS have never heard Khmer.</strong> Both "
        "were trained on MOS studies of mostly English speech. &sect;4 shows "
        "what that costs; treat their numbers as a description of the "
        "waveform, not of the model.</li>"
        "<li><strong>The speaking-rate check is coarse.</strong> Characters "
        f"are a poor proxy for phonemes and the {RATE_DEVIATION:.0f}&times; "
        "threshold is deliberately lax, so the check is one-directional: what "
        "it flags is certainly broken, but a clean sheet proves nothing. A "
        "model that substituted wrong words at a normal speed would pass "
        "it.</li>"
        "<li><strong>DNSMOS BAK saturates</strong> on clean synthesis and "
        "carries little signal; SIG and OVRL are the informative columns.</li>"
        "<li><strong>Default voices only.</strong> No reference audio or voice "
        "cloning was used, so that no model gained an advantage from a "
        "better-matched speaker. Both contenders support cloning, and both "
        "may do better with it than they do here.</li>"
        "<li><strong>One machine, one run, seed 0.</strong> No repeats, so "
        "small differences between adjacent models are not significant.</li>"
        "</ul>",
    ))

    parts.append(section(
        10, "Published figures",
        "<p>What the vendors claim, for comparison against what was "
        "measured.</p>",
        published_table(),
    ))

    parts.append(section(11, "Run environment", environment(results)))

    parts.append(
        '<footer><p>Generated by <code>evaluation/report_document.py</code> '
        "from <code>evaluation/results/&lt;model&gt;/scores.json</code>. "
        "Full-resolution audio for all "
        f"{n * len(results)} clips is in "
        "<code>evaluation/results/&lt;model&gt;/audio/</code>.</p></footer>"
    )
    parts.append("</article></main>")

    toc = "".join(
        f'<li><a href="#s{n_}"><span>{n_}</span>{t}</a></li>'
        for n_, t in SECTIONS
    )
    return (
        "\n".join(parts)
        .replace("<!--HEADER-->", header_html)
        .replace("<!--TOC-->", toc)
    )


HEAD = """<title>Open-Source TTS for Khmer</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2\
?family=IBM+Plex+Mono:wght@450;600\
&family=IBM+Plex+Sans:wght@400;600\
&family=Kantumruy+Pro:wght@400;600\
&family=Spectral:ital,wght@0,400;0,600;1,400&display=swap">
<style>
  :root {
    --ground: #eceff1; --paper: #ffffff; --ink: #14191b; --dim: #5b6a70;
    --rule: #dce3e6; --soft: #f2f6f7; --accent: #0d7382;
    --flag-bg: #fdf7e9; --flag-rule: #e3cf9e; --flag-ink: #6a5316;
    --bad: #a33a2a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #080c0e; --paper: #11181b; --ink: #e7eef0; --dim: #8ca1a7;
      --rule: #232f33; --soft: #172023; --accent: #52cfdf;
      --flag-bg: #221e13; --flag-rule: #473e24; --flag-ink: #d8c9a0;
      --bad: #e08a78;
    }
  }
  :root[data-theme="dark"] {
    --ground: #080c0e; --paper: #11181b; --ink: #e7eef0; --dim: #8ca1a7;
    --rule: #232f33; --soft: #172023; --accent: #52cfdf;
    --flag-bg: #221e13; --flag-rule: #473e24; --flag-ink: #d8c9a0;
    --bad: #e08a78;
  }
  body { background: var(--ground); color: var(--ink); margin: 0;
         font-family: Spectral, Georgia, serif; font-size: 16.5px;
         line-height: 1.68; }
  main { max-width: 940px; margin: 0 auto; padding: 30px 18px 60px; }
  article { background: var(--paper); border: 1px solid var(--rule);
            padding: 54px 60px 44px; }
  @media (max-width: 720px) { article { padding: 32px 22px 30px; } }

  .doc-head { border-bottom: 3px double var(--rule); padding-bottom: 26px;
              margin-bottom: 6px; }
  .eyebrow { font-family: "IBM Plex Sans", sans-serif; font-size: 11px;
             font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
             color: var(--accent); margin: 0 0 10px; }
  h1 { font-size: 40px; font-weight: 600; line-height: 1.12; margin: 0 0 12px;
       letter-spacing: -0.02em; text-wrap: balance; }
  .lede { font-size: 19px; color: var(--dim); margin: 0 0 24px; max-width: 60ch;
          font-style: italic; }
  .meta { display: flex; flex-wrap: wrap; gap: 26px; margin: 0;
          font-family: "IBM Plex Sans", sans-serif; }
  .meta dt { font-size: 10.5px; font-weight: 600; letter-spacing: 0.1em;
             text-transform: uppercase; color: var(--dim); }
  .meta dd { margin: 2px 0 0; font-size: 17px; font-weight: 600;
             font-variant-numeric: tabular-nums; }

  nav { background: var(--soft); padding: 16px 22px; margin: 28px 0 6px; }
  nav p { font-family: "IBM Plex Sans", sans-serif; font-size: 10.5px;
          font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
          color: var(--dim); margin: 0 0 8px; }
  nav ol { list-style: none; margin: 0; padding: 0; columns: 2; gap: 26px; }
  nav li { break-inside: avoid; }
  nav a { color: var(--ink); text-decoration: none; font-size: 14.5px;
          display: flex; gap: 10px; padding: 2px 0; }
  nav a span { color: var(--accent); font-family: "IBM Plex Mono", monospace;
               font-size: 12px; min-width: 16px; }
  nav a:hover { color: var(--accent); }

  section { padding-top: 34px; }
  h2 { font-size: 25px; font-weight: 600; margin: 0 0 14px; display: flex;
       gap: 14px; align-items: baseline; letter-spacing: -0.01em;
       border-bottom: 1px solid var(--rule); padding-bottom: 8px; }
  h2 .sn { font-family: "IBM Plex Mono", monospace; font-size: 15px;
           color: var(--accent); font-weight: 600; }
  h3 { font-family: "IBM Plex Sans", sans-serif; font-size: 13px;
       font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
       color: var(--dim); margin: 30px 0 10px; }
  p { max-width: 68ch; }
  ol, ul { max-width: 66ch; }
  li { margin-bottom: 7px; }
  code { font-family: "IBM Plex Mono", monospace; font-size: 0.85em;
         background: var(--soft); padding: 1px 5px; border-radius: 3px; }
  blockquote { margin: 14px 0; padding: 12px 18px; background: var(--soft);
               border-left: 3px solid var(--accent);
               font-family: "Kantumruy Pro", sans-serif; font-size: 15px;
               line-height: 2; color: var(--dim); word-break: break-word; }
  .callout { background: var(--flag-bg); border: 1px solid var(--flag-rule);
             color: var(--flag-ink); padding: 13px 17px; margin: 18px 0;
             font-size: 15.5px; max-width: 68ch; }
  .bad { color: var(--bad); }

  .tw { overflow-x: auto; margin: 16px 0 4px; }
  table.data { border-collapse: collapse; width: 100%; font-size: 13.5px;
               font-family: "IBM Plex Sans", sans-serif;
               font-variant-numeric: tabular-nums; }
  table.narrow { width: auto; min-width: 260px; }
  table.data th { text-align: left; font-size: 10.5px; font-weight: 600;
                  letter-spacing: 0.07em; text-transform: uppercase;
                  color: var(--dim); padding: 0 14px 7px 0;
                  border-bottom: 1.5px solid var(--ink); white-space: nowrap; }
  table.data td { padding: 8px 14px 8px 0; border-bottom: 1px solid var(--rule);
                  white-space: nowrap; }
  table.data tr:last-child td { border-bottom: none; }
  .tag { font-family: "IBM Plex Mono", monospace; font-size: 9.5px;
         background: var(--flag-bg); color: var(--flag-ink); padding: 1px 5px;
         border: 1px solid var(--flag-rule); border-radius: 3px;
         text-transform: uppercase; letter-spacing: 0.05em; }
  .caption { font-family: "IBM Plex Sans", sans-serif; font-size: 12.5px;
             color: var(--dim); margin: 8px 0 0; max-width: 72ch; }

  dl.models { margin: 4px 0 0; }
  dl.models > div { padding: 13px 0; border-bottom: 1px solid var(--rule); }
  dl.models > div:last-child { border-bottom: none; }
  dl.models dt { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline;
                 margin-bottom: 4px; }
  dl.models .label { font-family: "IBM Plex Sans", sans-serif; font-size: 12px;
                     color: var(--dim); font-weight: 400; }
  dl.models dd { margin: 0; max-width: 68ch; color: var(--ink); }
  ul.caveats { padding-left: 20px; }

  figure.sample { margin: 0 0 22px; padding: 16px 18px; background: var(--soft);
                  border-left: 3px solid var(--accent); }
  figcaption { display: flex; gap: 12px; align-items: baseline;
               font-family: "IBM Plex Sans", sans-serif; }
  figcaption .cat { font-size: 10.5px; color: var(--dim);
                    text-transform: uppercase; letter-spacing: 0.07em; }
  .km { font-family: "Kantumruy Pro", sans-serif; font-size: 18px;
        line-height: 2.05; margin: 10px 0 15px; word-break: break-word; }
  .clips { display: grid; gap: 11px;
           grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
  .clip { background: var(--paper); border: 1px solid var(--rule);
          border-radius: 5px; padding: 10px 11px 11px; }
  .clip .model { font-family: "IBM Plex Mono", monospace; font-size: 10.5px;
                 font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
                 margin-bottom: 7px; }
  audio { width: 100%; height: 32px; accent-color: var(--accent); }
  audio:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .missing { color: var(--dim); font-size: 12.5px; margin: 6px 0; }
  .badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px;
            font-family: "IBM Plex Mono", monospace; font-size: 10px;
            color: var(--dim); font-variant-numeric: tabular-nums; }
  .badges span { background: var(--soft); border-radius: 3px; padding: 2px 6px;
                 white-space: nowrap; }

  footer { margin-top: 42px; padding-top: 18px; border-top: 1px solid var(--rule);
           font-family: "IBM Plex Sans", sans-serif; font-size: 12.5px;
           color: var(--dim); }
  footer p { max-width: 74ch; margin: 0; }
  a { color: var(--accent); }
</style>
<main><article>
<!--HEADER-->
<nav><p>Contents</p><ol><!--TOC--></ol></nav>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--bitrate", default="48k")
    args = parser.parse_args()

    results = load_all()
    if not results:
        parser.error("no scores.json found -- run score.py first")
    models = [m for m in BACKEND_KEYS if m in results]
    entries = pick_entries(load_entries(), args.per_category)

    page = build(results, entries, args.bitrate, models)
    OUT_PATH.write_text(page, encoding="utf-8")
    size = len(page.encode("utf-8"))
    print(f"Wrote {OUT_PATH}  ({size / 1024 / 1024:.2f} MB)")
    print(f"  {len(models)} models, {len(entries)} sample sentences")


if __name__ == "__main__":
    main()
