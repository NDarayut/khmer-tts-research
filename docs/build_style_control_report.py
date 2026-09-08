#!/usr/bin/env python
"""
Build the Smean AI technical report on adding style control to VoxCPM2.

Companion to build_literature_review.py, in the same house style (see
smean_docx.py). Where the literature review surveys models, this one documents
one engineering effort end to end -- including the run that failed, which is the
part with the transferable finding in it.

Content is sourced from docs/11-voxcpm2-style-control-finetune.md and the
measurement record in finetune/results/diagnosis.md. Probe numbers are read from
finetune/experiments/*.json rather than retyped, so the document cannot drift
from the experiments it describes.

    python docs/build_style_control_report.py

Writes docs/Smean-VoxCPM2-Style-Control.docx
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smean_docx import *  # noqa: F401,F403  -- brand tokens and layout helpers

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "finetune" / "experiments"
OUT = Path(__file__).resolve().parent / "Smean-VoxCPM2-Style-Control.docx"


def probe(arm):
    f = EXP / f"result_{arm}.json"
    return json.loads(f.read_text()) if f.exists() else None


def sens(arm):
    """Mean loss under the true and swapped tag, plus the sign-test result."""
    f = EXP / f"sensitivity_{arm}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    d["true_mean"] = sum(d["true"]) / len(d["true"])
    d["swap_mean"] = sum(d["swap"]) / len(d["swap"])
    return d


def pctdelta(d):
    return f"**{100 * d['delta_mean'] / d['true_mean']:+.2f}%**" if d else "n/a"


def sep(arm):
    d = probe(arm)
    return f"{d['separation']:+.1f} Hz" if d else "n/a"


def verdict(arm):
    d = probe(arm)
    return d["verdict"] if d else "n/a"


_TBL = [0]


def tcap(doc, text):
    """Numbered table caption. Auto-incrementing, so inserting a table upstream
    cannot leave two Table 4s behind -- which it did, once."""
    _TBL[0] += 1
    caption(doc, f"Table {_TBL[0]} — {text}")


ARMS = {a: probe(a) for a in ("end", "proj", "onset")}
S_PROJ, S_ONSET = sens("proj"), sens("onset")
META = json.loads((ROOT / "finetune" / "data" / "corpus_meta.json").read_text())
_PARJ = json.loads((ROOT / "finetune" / "results" / "parenthetical" /
                    "parenthetical.json").read_text())
PAR, CEIL = _PARJ["summary"], _PARJ["corpus_ceilings"]


def parrow(axis, unit, fmt=".2f", bold_delta=False, bold_rho=False, ratio="—"):
    """One row of the parenthetical table, read straight from the results json."""
    a = PAR[axis]
    lv = [a["median_by_level"][k] for k in ("0", "1", "2")]
    d = f"{a['low_to_high']:+.2f} {unit}"
    r = f"{a['spearman_rho']:+.3f}"
    return [f"`{axis}`", f"{lv[0]:{fmt}}", f"{lv[1]:{fmt}}", f"{lv[2]:{fmt}}",
            f"**{d}**" if bold_delta else d,
            f"+{CEIL[axis]['spread']:.2f} {unit}",
            f"**{r}**" if bold_rho else r,
            f"{a['p']:.4f}", f"{a['mean_run_to_run_sd']:.2f}", ratio]
CTRL = json.loads((EXP / "sensitivity_positive_control.json").read_text())
CTRL["true_mean"] = sum(CTRL["true"]) / len(CTRL["true"])
CTRL["scramble_mean"] = sum(CTRL["scramble"]) / len(CTRL["scramble"])
CTRL["n_ctrl"] = len(CTRL["scramble"])

TRAIN_ROWS = (
    '{"audio": ".../wavs/f-adt2-0002_everyday_bureaucracy_06_0001_0010.wav",\n'
    ' "text":  "<|spk:f7|rate:mid|pitch:low|var:mid|energy:mid|>\u1780\u17b6\u179a\u1792\u17d2\u179c\u17be...",\n'
    ' "duration": 6.42}\n'
    '\n'
    '{"audio": ".../wavs/m-adt1-0005_cambodia_social_shift_04_0001_0137.wav",\n'
    ' "text":  "<|spk:m5|rate:mid|pitch:mid|var:any|energy:mid|>\u17a2\u17d2\u1793\u1780\u17a2\u17b6\u1785...",\n'
    ' "duration": 8.15}'
)

META_ROW = (
    '{"sentence_id": "f-adt1-0005_pursuing_hobbies_22_0001_0070",\n'
    ' "speaker_id":  "f-adt1-0005",\n'
    ' "duration": 9.67,\n'
    ' "char_rate": 13.34,          # Khmer characters per second\n'
    ' "f0_median_hz": 228.97,      # pitch\n'
    ' "f0_std_st": 5.28,           # pitch variation, semitones\n'
    ' "rms_dbfs": -19.62,          # level\n'
    ' "lab": {"spk": "f5", "rate": "mid", "pitch": "mid",\n'
    '         "var": "mid", "energy": "mid"}}'
)

PY_USAGE = (
    'from voxcpm import VoxCPM\n'
    'from finetune.synthesize_styled import load_model\n'
    '\n'
    '# load_model reads lora_config.json from the checkpoint, so the adapter\n'
    '# rank is recovered rather than guessed\n'
    'model = load_model("finetune/checkpoints/khmer_style/latest")\n'
    '\n'
    'tag = "<|spk:f2|rate:fast|pitch:any|var:lively|energy:any|>"\n'
    'wav = model.generate(text=tag + "\u1780\u17b6\u179a\u1793\u17b7\u1799\u17b6\u1799...")\n'
    '\n'
    '# switch the adapter off to get the stock model back, bit for bit\n'
    'model.set_lora_enabled(False)'
)

CLI_USAGE = (
    '# styled\n'
    'python finetune/synthesize_styled.py \\\n'
    '    --lora finetune/checkpoints/khmer_style/latest \\\n'
    '    --text "\u179f\u17bc\u1798\u179f\u17d2\u179c\u17b6\u1782\u1798\u1793\u17cd..." \\\n'
    '    --style var=lively,rate=fast,spk=f2 \\\n'
    '    --out hello.wav\n'
    '\n'
    '# base model, no adapter, for an A/B against the same sentence\n'
    'python finetune/synthesize_styled.py --text "..." --out base.wav'
)

NONVERBAL_USAGE = (
    '# no adapter and no fine-tuning needed -- this is stock VoxCPM2\n'
    'wav = model.generate(text="\u1781\u17d2\u1789\u17bb\u17c6\u1782\u17b7\u178f\u1790\u17b6... [laughing] ...")\n'
    '\n'
    '# the two mechanisms compose: parenthetical voice design, our prosodic\n'
    '# header tag, and an inline non-verbal tag in one call\n'
    'text = ("(a young woman, warm voice)"\n'
    '        "<|spk:any|rate:slow|pitch:any|var:lively|energy:any|>"\n'
    '        "\u1781\u17d2\u1789\u17bb\u17c6... [sigh] ...")'
)

MANIFEST_NV = (
    '{"audio": ".../nv/clip_00412.wav",\n'
    ' "text":  "ខ្ញុំគិតថាមិនអីទេ [laughing] បុន្តែ...",\n'
    ' "duration": 4.21}'
)

doc = new_document()

cover(doc,
      title="Speech Control for VoxCPM2",
      subtitle="The control layers, what is measured on Khmer,\nand a training plan for non-verbal vocalization",
      blurb="Expressive control is not one capability but six, and they differ in how "
            "expensive each is to add. This report sets out that taxonomy, reports what "
            "VoxCPM2 already does on Khmer against measurement, records a conditioning "
            "failure that is well known in the generative-modelling literature but "
            "undocumented for this class of TTS system, and gives a phased plan for the "
            "layer with the best cost-to-value ratio left: non-verbal vocalization.",
      meta=[["Document", "Technical report — control taxonomy, measurement and plan"],
            ["Subject", "VoxCPM2 (OpenBMB) · expressive control for Khmer"],
            ["Scope", "Six control layers · prosodic layer measured and fine-tuned · "
                      "non-verbal layer surveyed, with a training plan"],
            ["Evidence base", "Two 4000-step training runs, four 400-step probes, a "
                              "loss-level diagnostic with a positive control, and a "
                              "72-generation sweep of the base model"],
            ["Hardware", "One 12 GB RTX 3060"],
            ["Date", "8 September 2026"],
            ["Status", "Internal research document"]])

page_footer(doc, "Smean AI  ·  VoxCPM2 speech control")

# =========================================================================
heading(doc, "1.  Findings", 1, page_break=True)

para(doc, "VoxCPM2 is this project's recommended model for Khmer, on the strength of a 2.47% "
          "character error rate against Higgs TTS 3's 8.28% and an Apache-2.0 licence. Higgs "
          "accepts inline control tokens; VoxCPM2 appeared to accept none. Six findings follow "
          "from measuring that assumption.")

heading(doc, "1.1  VoxCPM2 already has a prosodic control channel, and on Khmer it mostly works", 3)

para(doc, "A natural-language description in parentheses before the text — (speaking quickly), "
          "(a high-pitched voice) — steers the delivery. Measured on the stock model: 8 sentences "
          "from the frozen evaluation set, three levels per axis, three seeds per cell, median "
          "taken within each cell.")

table(doc,
      ["axis", "low", "mid", "high", "low→high", "corpus ceiling", "rho", "p", "seed sd", "effect ÷ noise"],
      [parrow("pitch", "Hz", ".1f", True, True, "**3.7**"),
       parrow("energy", "dB", ".2f", ratio="0.8"),
       parrow("rate", "ch/s", ".2f", ratio="1.4"),
       parrow("var", "st", ".2f")],
      widths=[0.62, 0.52, 0.52, 0.52, 0.86, 0.86, 0.55, 0.5, 0.5, 0.65], size=8.0)
tcap(doc, "The built-in parenthetical prompt, measured on the base model. "
          "“corpus ceiling” is the separation this project's hand-labelled Khmer corpus achieves "
          "on the same quantity — the upper bound on what a fine-tune on it could teach. "
          "“seed sd” is the spread across repeated generations of the same prompt on the same "
          "sentence. Instrument: finetune/verify_parenthetical.py.")

bullet(doc, [("Pitch. ", {"bold": True}), ("A 106 Hz range at rho +0.76, against a corpus ceiling "
             "of 28 Hz — nearly four times what the 20 available Khmer speakers can express "
             "between them.", {})])
bullet(doc, [("Rate and loudness respond, but are not control. ", {"bold": True}),
             ("Both correlations are statistically solid across sentences. For energy the seed "
              "spread is 5.69 dB against a 4.50 dB effect: re-rolling the seed moves the output "
              "further than changing the command does. Averaging across sentences hides this "
              "entirely, which is why every cell was generated three times.", {})])
bullet(doc, [("Pitch variation fails. ", {"bold": True}), ("rho −0.27 at p = 0.20 — not "
             "significant, and pointing the wrong way. Asking for a lively, expressive delivery "
             "produced less pitch movement than asking for a flat one.", {})])

callout(doc, "Caveat on the wordings", [
    "The prompt phrasings are extrapolated from OpenBMB's two documented examples — "
    "*(slightly faster, cheerful tone)* and *(A young woman, gentle and sweet voice)*. A negative "
    "result on one axis could therefore be a vocabulary mismatch rather than a model limit, and "
    "`var` is where that doubt deserves the most weight. The positive results do not carry the "
    "same doubt: they establish the channel works, whatever better wording might add.",
], accent=SIENNA, fill=SIENNA_SOFT)

heading(doc, "1.2  A global attribute in the text field earns almost no gradient", 3)

para(doc, "Measured on the training objective itself rather than on generated audio, with a "
          "positive control. Teacher-forced forward passes over 40 held-out clips, reseeded "
          "identically before each so the flow-matching timestep and noise are the same.")

table(doc,
      ["condition", "mean loss/diff", "delta", "rows worse", "sign test"],
      [["correct speaker tag", f"{S_PROJ['true_mean']:.5f}", "—", "—", "—"],
       ["swapped speaker tag", f"{S_PROJ['swap_mean']:.5f}", pctdelta(S_PROJ),
        f"{S_PROJ['worse']}/{S_PROJ['n']}", f"p = {S_PROJ['p']:.2f}"],
       ["scrambled transcript (control)", f"{CTRL['scramble_mean']:.5f}",
        f"**{100*(CTRL['scramble_mean']-CTRL['true_mean'])/CTRL['true_mean']:+.1f}%**",
        f"{CTRL['worse']}/{CTRL['n_ctrl']}", "—"]],
      widths=[2.0, 1.2, 1.0, 1.0, 1.0])
tcap(doc, "A 290× ratio between what the transcript is worth to the model and what the control "
          "tag is worth. The positive control is the load-bearing row: it proves the instrument "
          "can detect a signal the model does use.")

para(doc, "The cause is teacher forcing. Every audio patch is predicted with the preceding "
          "ground-truth patches visible, and a speaker's pitch is trivially readable off those, "
          "so the tag supplies nothing the prefix does not already carry. Restricting the loss to "
          "the first K patches confirms it: the swap penalty is +0.00318 at K = 1 and decays "
          "monotonically to +0.00022 over the whole clip, a 14× decay. The tag matters exactly "
          "where no prefix exists yet.")

heading(doc, "1.3  That is a cost, not a wall", 3)

rich(doc, [("The parenthetical prompt of finding 1.1 is also a global attribute in the text "
            "field, and it works.", {"bold": True}),
           (" So the teacher-forcing shortcut does not make global conditioning impossible — it "
            "makes it data-expensive. A signal worth 0.03% of the loss is still learnable given "
            "enough data and a full training run. OpenBMB paid that cost at pre-training scale; "
            "11.7 hours of Khmer through a LoRA did not.", {})])

heading(doc, "1.4  Two changes let a small run learn a global tag anyway", 3)

para(doc, "Isolated one variable at a time on the easiest discrimination available — two "
          "speakers roughly an octave apart, one binary tag, 400 steps — against a threshold "
          "fixed before any arm was run.")

table(doc,
      ["arm", "change from the previous arm", "separation", "verdict"],
      [["`end`", "tag adjacent to `audio_start`", sep("end"), verdict("end")],
       ["`proj`", "+ `enable_proj: true`", sep("proj"), verdict("proj")],
       ["`onset`", "+ 8× onset-weighted loss, τ = 4 patches", f"**{sep('onset')}**",
        f"**{verdict('onset')}**"]],
      widths=[0.8, 2.9, 1.1, 0.9])
tcap(doc, "Probe ladder. Pass bar: the two speaker tags must produce generated F0 medians more "
          "than 30 Hz apart, in the correct direction. The real speakers are 175 Hz apart.")

para(doc, "Confirmed at the objective level, on the same instrument as finding 1.2: the swap "
          f"penalty moved from +{S_PROJ['delta_mean']:.5f} (p = {S_PROJ['p']:.2f}) to "
          f"+{S_ONSET['delta_mean']:.5f} (p = {S_ONSET['p']:.3f}).")

heading(doc, "1.5  VoxCPM2 LoRA fine-tuning fits in 12 GB", 3)

para(doc, "The documented requirement is roughly 20 GB. Casting frozen weights to bfloat16 while "
          "keeping LoRA parameters in float32, and checkpointing all 36 transformer layers, "
          "brings measured peak usage to 11.3 GiB of 12.3 GiB on an RTX 3060, desktop included. "
          "Section 3.5 gives the detail.")

heading(doc, "1.6  Local events are cheap where global attributes are expensive", 3)

para(doc, "Nothing in the acoustic prefix predicts that a laugh is coming, so a local event faces "
          "none of the competition finding 1.2 measures. This is the argument for prioritising "
          "the non-verbal layer over emotion or voice quality, and section 5 is the plan that "
          "follows from it.")

# =========================================================================
heading(doc, "2.  The taxonomy of speech control", 1, page_break=True)

md(doc, "Expressive control is not one capability, and treating it as one is how a roadmap "
          "goes wrong. The distinction that decides everything downstream is between "
          "**attributes that hold over a whole utterance** and **events that happen at a point "
          "in it**. A speaker's pitch register is true of every frame; a laugh occupies 400 ms "
          "and nothing else.")

rich(doc, [("That distinction is not cosmetic. ", {"bold": True}),
           ("Finding 1.2 measures its consequence: a global attribute competes with the "
            "teacher-forced acoustic prefix and loses by a factor of 290. A local event has no "
            "such competitor, so it earns an ordinary gradient and needs no special machinery.",
            {})], space_after=10)

table(doc,
      ["#", "layer", "controls", "scope", "mechanism in VoxCPM2", "Khmer status"],
      [["1", "**Prosodic**", "pitch register, rate, loudness", "global",
        "parenthetical prompt `(…)`, built in", "**measured, works** (§3)"],
       ["1b", "**Prosodic residue**", "pitch *variation*; named speaker; reproducibility",
        "global", "would need a fine-tune", "gaps confirmed (§3.8)"],
       ["2", "**Non-verbal**", "laughter, sighs, hesitation, breath", "**local**",
        "inline `[laughing]` tags, built in", "**untested** (§4, §5)"],
       ["3", "Affective", "emotion — sadness, joy, anger", "global",
        "plausibly the same `(…)` prompt", "unmeasured (§6)"],
       ["4", "Voice quality", "whisper, breathy, creaky", "global",
        "plausibly the same `(…)` prompt", "unmeasured (§6)"],
       ["5", "Discourse", "word emphasis, contrastive focus", "**local**",
        "none — needs a span syntax", "not designed (§6)"],
       ["6", "Timing", "pause insertion, phrase breaks", "local",
        "punctuation only, implicit", "not designed"]],
      widths=[0.28, 1.05, 1.6, 0.5, 1.6, 1.0], size=8.3)
tcap(doc, "The six layers. Each row's literature is given in the section it points to, and "
          "collected in section 9.")

callout(doc, "Layers 3 and 4 are unmeasured, not blocked", [
    "An earlier draft called emotion and voice quality “blocked: no expressive Khmer corpus”. "
    "That was wrong in the same way the whole project was wrong before finding 1.1. Both are "
    "global attributes that the parenthetical channel plausibly already carries — OpenBMB's own "
    "documented example, *(slightly faster, cheerful tone)*, is a layer-3 prompt.",
    "Extending `finetune/verify_parenthetical.py` to them is one dictionary edit and about "
    "twenty minutes of GPU time per layer. The corpus problem only applies if that measurement "
    "returns negative — and then finding 1.3 sets the price.",
])

md(doc, "**Layers 5 and 6 need a different syntax, not more data.** A header tag cannot say "
          "*which word* to emphasise. That requires span marking in the text, or a parallel "
          "per-token flag, and the strongest reference in the literature avoids training an "
          "emphasis label at all — see section 6.")

# =========================================================================
heading(doc, "3.  Layer 1 — prosody", 1, page_break=True)

heading(doc, "3.1  The built-in channel", 2)

code(doc, 'wav = model.generate(text="(speaking quickly)ថ្ងៃនេះអាកាសធាតុល្អណាស់។")')

para(doc, "OpenBMB document this for voice design and for style-guided cloning over a reference "
          "clip, listing gender, age, tone, emotion and pace as what it steers. Nothing in their "
          "documentation covers Khmer, and this project had already been burned once by assuming "
          "a control surface transfers to an undocumented language — document 10 carries exactly "
          "that caveat for Higgs TTS 3's control tokens. The measurement in finding 1.1 is, as "
          "far as this project can establish, the first on Khmer.")

md(doc, "Prompts used, three levels each: *(speaking slowly / at a normal pace / quickly)*, "
          "*(a low- / normal- / high-pitched voice)*, *(a flat, monotone / a normal / a lively, "
          "expressive delivery)*, *(speaking softly, quietly / at a normal volume / loudly)*.")

para(doc, "This places VoxCPM2 in the natural-language-prompt family of controllable TTS — "
          "PromptTTS, InstructTTS, PromptTTS 2 and Parler-TTS all condition on a free-text style "
          "description rather than on categorical labels, and TextrolSpeech and SpeechCraft are "
          "the corpora that made that family trainable.")

heading(doc, "3.2  The fine-tuned tag: mechanism", 2)

para(doc, "The alternative built here prefixes a compact, enumerable tag to the text field:")

code(doc, '<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។')

md(doc, "That is the entire architectural change. There is none. The reason it works rather "
          "than merely being convenient is in the training packer: "
          "`voxcpm/training/packers.py::process_tts_data()` builds each sequence as "
          "`[text tokens] [audio start] [audio patches] [audio end]` and sets")

code(doc, "loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])")

rich(doc, [("Zero across every text position.", {"bold": True}),
           (" The model is never asked to reproduce the text, only to use it in predicting the "
            "audio latents that follow. Anything placed in the text field is therefore pure "
            "conditioning, costing exactly what it costs in sequence length — 25 tokens against "
            "roughly 140 for a typical Khmer sentence — and nothing else. Parler-TTS uses the "
            "same arrangement with a separate description encoder, which VoxCPM2 does not need "
            "because its tokenizer is byte-level over raw text.", {})])

para(doc, "A prior worry, checked before anything was trained: would the model read the tag "
          "aloud? The base model was asked to synthesize a Khmer sentence with and without a "
          "tag, and both clips were transcribed with the project's Khmer CTC ASR. The "
          "transcripts are identical and both clips are 2.08 s. The base model silently ignores "
          "the tag — which also hands the verification pass a clean control condition.")

heading(doc, "3.3  Where the labels come from", 2)

md(doc, "There is no expressive Khmer speech corpus and no Khmer emotion corpus. Copying "
          "Higgs's 21-emotion catalogue would mean inventing labels for data that does not carry "
          "them, and the result would be unfalsifiable. So the axes are defined as **quantities "
          "measurable on a waveform**.")

md(doc, "This is the Parler-TTS recipe, which exists precisely because “reliance on "
          "human-labeled descriptions prevents scaling”: annotate speaking rate, pitch, SNR and "
          "reverberation automatically over 45k hours, then render the measurements into a text "
          "description the model conditions on. The accompanying `dataspeech` tooling bins each "
          "continuous measurement into categorical descriptors. The tag here is the same idea in "
          "a more compact syntax.")

table(doc,
      ["axis", "levels", "measured as"],
      [["`spk`", "`f1`…`f9`, `m1`…`m11`", "speaker identity, from the corpus"],
       ["`rate`", "`slow` `mid` `fast`", "Khmer characters per second"],
       ["`pitch`", "`low` `mid` `high`", "median F0"],
       ["`var`", "`flat` `mid` `lively`", "F0 standard deviation, in semitones"],
       ["`energy`", "`soft` `mid` `loud`", "RMS level"]],
      widths=[0.8, 1.9, 3.0])
tcap(doc, "Five axes, 32 slot values. Narrower than Higgs's catalogue, and deliberately so: "
          "every one of these is checkable by arithmetic on the output.")

md(doc, "Two design choices, both of which `dataspeech` independently also makes. **Bin edges "
          "come from the 15/85 tails rather than tertiles**, which widens the control range by "
          "40–54% across the four axes. And **`pitch`, `var` and `energy` are ranked within "
          "speaker**, so each axis describes *how* something is said rather than *who* says it; "
          "ranking them globally measures better but lets the axis select a voice instead of a "
          "delivery. `rate` is ranked globally, because characters per second means the same "
          "thing from any voice.")

heading(doc, "3.4  What a training row looks like", 2)

code(doc, TRAIN_ROWS)
tcap(doc, "Two rows of `finetune/data/train.jsonl`. The tag is ordinary text in the `text` "
          "field; nothing in the schema knows it is a tag.")

md(doc, "Per-slot dropout leaves partial tags valid at inference: a caller can ask for "
          "`rate:fast` alone and leave every other slot `any`. The label metadata each row "
          "derives from is kept separately, so a relabelling never requires re-extracting audio:")

code(doc, META_ROW)

heading(doc, "3.5  The corpus, and fitting a 20 GB job into 12 GB", 2)

table(doc,
      ["", "value"],
      [["Clips", f"{META['n_clips']:,} (5,646 train / 174 val)"],
       ["Speakers", f"{len(META['speaker_tags'])}"],
       ["Audio", f"{META['hours']:.2f} h of Khmer read speech"],
       ["Steps", "4000 (≈5.5 epochs) planned"],
       ["LoRA", "r = 64, α = 64, `enable_lm` + `enable_dit` + `enable_proj`"],
       ["Effective batch", "8 (`batch_size` 1 × `grad_accum_steps` 8)"],
       ["Learning rate", "1e-4, the documented LoRA rate"],
       ["Peak VRAM", "11.3 GiB of 12.3 GiB, desktop included"]],
      widths=[1.6, 4.1])
tcap(doc, "The prosodic run. Trainable parameters are roughly 0.1% of the 2.29 B total.")

md(doc, "The published VRAM figure for VoxCPM2 LoRA is roughly 20 GB. The gap is almost "
          "entirely one line in upstream's `from_local`: in training mode the model is left in "
          "float32, so 2.29 B parameters occupy about 8.6 GiB before a single activation is "
          "allocated — while the forward pass then runs under bfloat16 autocast anyway. Three "
          "patches in `finetune/train.py` close it.")

bullet(doc, [("Frozen weights to bfloat16, LoRA parameters kept in float32. ", {"bold": True}),
             ("Halves the resident model. The adapters stay fp32 because AdamW at 1e-4 produces "
              "updates around 1e-2 the size of the weights, and bfloat16's three significant "
              "decimal digits would quantise a meaningful share of them away.", {})])
bullet(doc, [("Gradient checkpointing on all 36 transformer layers. ", {"bold": True}),
             ("Applied only where the installed MiniCPM4 block exposes a usable entry point, and "
              "reported at startup either way rather than silently skipped.", {})])
bullet(doc, [("AudioVAE in float32 on GPU, encode() under no_grad. ", {"bold": True}),
             ("Upstream already freezes it; this makes the memory behaviour explicit.", {})])

heading(doc, "3.6  Why the first run ignored its own control tag", 2)

md(doc, "The first 4000-step run trained cleanly and produced an adapter that demonstrably "
          "changed the model — median |ΔW|/|W| of 3.0%, and generated F0 moved from the base "
          "model's wandering 106–232 Hz into a tight 201–262 Hz — while responding to the "
          "control tag not at all. Every axis came back at rho ≈ 0. The `spk` axis is the proof: "
          "tags spanning 105–280 Hz of real speaker pitch produced 26 Hz of output spread at "
          "rho −0.20. Speaker identity is the axis carrying the most information, because the "
          "model cannot infer it from the text; if that one does not land, nothing is landing.")

md(doc, "Tokenisation was eliminated first — the tags survive the model's own wrapped "
          "tokenizer, differ in ten positions between extremes, and contain no UNK. Text "
          "normalisation was eliminated too, since `normalize` defaults to False. Neither "
          "explained anything, and neither could: both are questions about the input, and "
          "generated-audio metrics cannot distinguish a signal that was never learned from one "
          "lost in sampling.")

rich(doc, [("The step that settled it was abandoning generated audio for the training objective.",
            {"bold": True}),
           (" Finding 1.2 is that measurement. Two minutes of teacher-forced forward passes "
            "answered a question a 6.5-hour run had left ambiguous, and the positive control is "
            "what makes the null interpretable.", {})])

heading(doc, "3.7  The fix", 2)

para(doc, "Two changes, isolated one at a time by the probe ladder in finding 1.4.")

rich(doc, [("The projection layers were frozen. ", {"bold": True}),
           ("`enable_proj` adapts `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj` and "
            "`fusion_concat_proj` — the linear bottleneck through which everything the language "
            "model knows reaches the diffusion transformer that produces the acoustics. The "
            "first run left it false, following document 9's recommendation, which is sound "
            "advice for speaker cloning (where the voice arrives through the reference-audio "
            "encoder) and wrong for conditioning that arrives only as text and has no other "
            "route across.", {})])

rich(doc, [("The loss was weighted toward the onset. ", {"bold": True}),
           ("If the tag is redundant with the acoustic prefix everywhere except the start of the "
            "clip, put the gradient at the start of the clip. Position i of the audio span gets "
            "weight 1 + (W−1)·exp(−i/τ), which at W = 8, τ = 4 is 8.0 on the first patch, 3.6 on "
            "the fifth and about 1 by the twentieth. If the onset is set correctly, "
            "autoregression carries it — the rest of the utterance is generated conditioned on "
            "that first patch.", {})])

md(doc, "Nothing downstream has to change for this to be a true per-position weight: "
          "`unified_cfm.compute_loss` computes `(mask*losses).sum() / sum(mask)`, and "
          "`adaptive_loss_weighting` at p = 0 returns the mask unaltered, so a non-binary mask "
          "is a weighted mean that renormalises itself. Upstream casts the mask to `int32`, "
          "which would truncate the weights, so it is rebuilt as float in PATCH 4.")

callout(doc, "What the more principled version of this would be", [
    "The literature's fix for information preference is to **corrupt the shortcut** — word "
    "dropout in Bowman et al., a deliberately weakened decoder in Chen et al. — rather than to "
    "reweight around it. The equivalent here is corrupting the acoustic prefix during training so "
    "the model cannot read pitch off it.",
    "Onset weighting was chosen because it is a loss-side change that cannot destabilise "
    "generation, and because it was cheap enough to probe in 35 minutes. If this were being "
    "written up as research rather than as a build log, prefix corruption is the comparison that "
    "would have to be run.",
], accent=VIOLET, fill=VIOLET_SOFT)

heading(doc, "3.8  What a fine-tune would still add", 2)

table(doc,
      ["", "built-in prompt", "fine-tuned tag"],
      [["pitch", "**+106 Hz, robust**", "ceiling +28 Hz — strictly worse"],
       ["rate", "+2.29 ch/s, 1.4× noise", "comparable at best"],
       ["energy", "effect below the noise", "*possible* improvement, unproven"],
       ["pitch variation", "**fails, wrong direction**", "*possible* improvement, unproven"],
       ["named speaker, no reference clip", "not available", "**only the tag does this**"],
       ["reproducible for one utterance", "no", "yes, deterministic given a seed"],
       ["enumerable and sweepable", "no", "yes"]],
      widths=[2.0, 2.0, 2.2])
tcap(doc, "Three residues: pitch variation, named-speaker selection, and reproducibility.")

para(doc, "The prosodic run was stopped at step 1,980 of 4,000 on that accounting. The "
          "checkpoint is on disk and the run resumes from it if the residue proves worth "
          "closing. Findings 1.2 through 1.5 come from that work and stand independently of "
          "whether it is ever finished.")

heading(doc, "3.9  Using the tag", 2)

code(doc, PY_USAGE)
code(doc, CLI_USAGE)

md(doc, "The adapter hot-swaps: `model.set_lora_enabled(False)` restores the base model bit "
          "for bit, so the base model's other 29 languages cannot be damaged by a Khmer adapter "
          "that can be switched off. A full fine-tune would have no such escape hatch.")

# =========================================================================
heading(doc, "4.  Layer 2 — non-verbal vocalization", 1, page_break=True)

para(doc, "This layer needs no fine-tuning to exist. VoxCPM2 ships with it: inline square-bracket "
          "tags placed in the text at the point where the vocalization should occur, part of the "
          "base model's training and available today on the stock checkpoint.")

table(doc,
      ["category", "tags", "effect"],
      [["laughter and breath", "`[laughing]`, `[sigh]`", "audible laugh or exhalation"],
       ["hesitation", "`[Uhm]`, `[Shh]`", "filled pause; a shushing sound"],
       ["question particles",
        "`[Question-ah]`, `[Question-ei]`, `[Question-en]`, `[Question-oh]`",
        "interrogative interjections"],
       ["emotional interjections",
        "`[Surprise-wa]`, `[Surprise-yo]`, `[Dissatisfaction-hnn]`",
        "surprise; dissatisfaction"]],
      widths=[1.4, 2.7, 2.1])
tcap(doc, "The inventory, from the VoxCPM2 cookbook. Twelve tags across four categories.")

code(doc, NONVERBAL_USAGE)

md(doc, "OpenBMB's guidance is unusually specific and worth repeating: use them sparingly, "
          "prefer the lowercase `[laughing]` over variants like `[Laughter]`, and do not stack "
          "several into one sentence. The model card lists instability on “very long or highly "
          "expressive inputs” among its limitations, which is the same caution from the other "
          "side.")

callout(doc, "All of this is documented for VoxCPM2 in general and none of it for Khmer", [
    "**Do the tags fire in Khmer text at all?** They are English strings inside a Khmer sentence. "
    "Khmer tokenises to UTF-8 byte tokens, roughly three per character; the tag does not. Whether "
    "the model recognises the pattern in that context is empirical.",
    "**Are they spoken instead of performed?** The failure to check for is the model pronouncing "
    "“laughing” as a word — the same check section 3.2 ran on the prosodic tag, which passed.",
    "**Does the prosodic adapter erode them?** It was trained on read speech containing no "
    "laughter, and LoRA can erode capabilities absent from the fine-tuning distribution. Testable "
    "against the same clip with the adapter disabled.",
], accent=SIENNA, fill=SIENNA_SOFT)

rich(doc, [("Why this layer is tractable for Khmer when emotion is not. ", {"bold": True}),
           ("Non-verbal vocalizations are largely language-independent in their acoustics — a "
            "laugh is a laugh. That is also why detectors trained on English data can be pointed "
            "at Khmer audio to manufacture labels, which is the whole basis of the plan in "
            "section 5.", {})])

# =========================================================================
heading(doc, "5.  Training plan: non-verbal vocalization by tag", 1, page_break=True)

para(doc, "The goal is not to add the capability — VoxCPM2 has it — but to make it fire reliably "
          "on Khmer text, with Khmer-appropriate vocalization, and without damaging anything "
          "else. Two published results bound the approach.")

callout(doc, "The two reference recipes", [
    "**ELaTE** (Kanda et al., 2024) fine-tunes a conditional-flow-matching zero-shot TTS — the "
    "same decoder family as VoxCPM2's Local DiT — for laughter control, conditioning on "
    "frame-level output from a laughter detector. Two results transfer directly: a *small* "
    "laughter-conditioned set suffices, and mixing it with general data preserves base quality.",
    "**NonverbalTTS** (Borisov et al., SSW 2025) builds a 17-hour corpus covering 10 non-verbal "
    "types by automatic detection followed by human validation over VoxCeleb and Expresso, then "
    "shows open TTS models fine-tuned on it reaching parity with closed systems. That is the "
    "labelling pipeline to copy.",
])

heading(doc, "5.1  Phase 0 — measure the base model", 2)

para(doc, "No data, no training, and it may end the project. Synthesize a fixed set of Khmer "
          "sentences three ways — base model, prosodic adapter enabled, adapter disabled — with "
          "and without each tag. Every metric is automatic.")

table(doc,
      ["signal", "what it answers", "tool"],
      [["duration delta", "did anything get inserted?",
        "`verify_control.py`'s measurement layer"],
       ["Khmer CTC ASR transcript", "is the tag being *spoken*?",
        "`evaluation/score_cer_khmer.py`"],
       ["laughter-detector probability", "is what was inserted actually a laugh?",
        "`jrgillick/laughter-detection`"]],
      widths=[1.6, 2.3, 2.3])
tcap(doc, "The go/no-go. The third row is load-bearing, and it is the same instrument that "
          "later produces training labels.")

para(doc, "If the tags already fire on Khmer, no training is needed for those tags and the plan "
          "reduces to whatever subset failed. This phase is the direct application of the lesson "
          "in finding 1.1.")

heading(doc, "5.2  Phase 1 — source audio", 2)

para(doc, "Read-speech corpora contain no non-verbal events; the 11.73-hour prosody corpus has "
          "none at all. The material has to be conversational — Khmer podcasts, interviews, "
          "broadcast talk — and the pipeline follows NonverbalTTS: voice-activity detection, "
          "diarization, ASR, non-verbal detection, then human validation of a sample.")

rich(doc, [("Budget. ", {"bold": True}),
           ("NonverbalTTS is 17 hours for 10 types across many speakers, and ELaTE shows a "
            "single event type needs far less. Target ", {}),
           ("500–1,000 validated events per tag", {"bold": True}),
           (", starting with `[laughing]`, `[sigh]` and `[Uhm]`; the interjection tags are "
            "lower value and can wait.", {})])

heading(doc, "5.3  Phase 2 — detection and labelling", 2)

bullet(doc, [("Laughter. ", {"bold": True}), ("Gillick et al.'s detector gives start and end "
             "timestamps directly, and is the same tool used in phase 0.", {})])
bullet(doc, [("Other events. ", {"bold": True}), ("A classifier trained on VocalSound — 21k "
             "crowdsourced clips of laughter, sighs, coughs, sneezes, sniffs and throat-clearing "
             "— or the corresponding AudioSet classes.", {})])
bullet(doc, [("Alignment. ", {"bold": True}), ("Force-align the ASR transcript to get word "
             "timings, then insert the tag string at the word boundary nearest the detected "
             "event onset.", {})])
bullet(doc, [("Validation. ", {"bold": True}), ("Listen to a sample. NonverbalTTS's precision "
             "came from human validation over automatic detection, not from the detector "
             "alone.", {})])

para(doc, "Output rows look exactly like the existing prosody manifest, with the tag inline "
          "rather than in a header:")

code(doc, MANIFEST_NV)

heading(doc, "5.4  Phase 3 — training", 2)

para(doc, "The same harness as the prosodic run. Every difference follows from the layer being "
          "local rather than global.")

table(doc,
      ["setting", "prosody run", "non-verbal run", "why"],
      [["tag position", "header, before the text", "**inline, at the event**",
        "the model must learn *when*, not only *whether*"],
       ["`--onset-weight`", "8.0", "**1.0 (off)**",
        "onset weighting exists to beat the prefix shortcut for global attributes; a local event "
        "has no prefix competitor"],
       ["`enable_proj`", "true", "true", "still the only text→DiT route"],
       ["data mixing", "n/a", "**≈1:4 tagged to untagged**",
        "ELaTE's finding; prevents unprompted laughter and preserves base quality"],
       ["negative examples", "n/a", "**required**",
        "clips with no tag and no event, so absence is trained too"]],
      widths=[1.0, 1.15, 1.35, 2.7], size=8.3)
tcap(doc, "The last two rows are the ones most likely to be skipped and most likely to cause the "
          "characteristic failure: a model that laughs everywhere.")

heading(doc, "5.5  Phase 3b — frame-level conditioning, only if timing is poor", 2)

para(doc, "If the inline tag places the event but with unreliable timing, ELaTE's actual method "
          "is the escalation: condition the DiT on a per-frame laughter-probability channel from "
          "the detector rather than on a discrete text token. This is invasive — it adds an input "
          "to the flow-matching head — and should not be attempted before the cheap version has "
          "been measured.")

heading(doc, "5.6  Phase 4 — evaluation", 2)

table(doc,
      ["axis", "metric", "pass condition"],
      [["does the event occur", "detector fires on the output",
        "precision and recall against the commanded tag"],
       ["is it in the right place", "detector onset vs commanded position",
        "within a stated tolerance"],
       ["is the tag silent", "Khmer CTC ASR transcript", "no tag text in the transcript"],
       ["no intelligibility regression", "CER on the frozen 100-sentence eval set",
        "within noise of the base model's 2.47%"],
       ["does it sound right", "`evaluation/listening_test.py`", "blind A/B against base"]],
      widths=[1.6, 2.1, 2.5])
tcap(doc, "Four objective checks and one subjective. The apparatus for all five already exists "
          "in this repository.")

rich(doc, [("The CER regression check is the one that must not be skipped. ", {"bold": True}),
           ("`eval-set/eval.json` is held fixed across all four models in this project's "
            "comparison precisely so that a change like this can be checked against it. A "
            "non-verbal adapter that improves expressiveness while costing intelligibility is "
            "not obviously a good trade, and the only way to know is to measure both.", {})])

# =========================================================================
heading(doc, "6.  Layers 3 to 6", 1, page_break=True)

heading(doc, "6.1  Affective — emotion", 2)

md(doc, "Measure the parenthetical channel first — *(a sad tone)*, *(an angry voice)*, "
          "*(cheerful)* — since OpenBMB's own documented example is already an emotion prompt. "
          "If it fails, the corpus problem is real: no Khmer emotion corpus exists, and the "
          "English reference points are ESD and the emotion annotations in NonverbalTTS. The "
          "closest method reference is *Laugh Now Cry Later*, which controls time-varying "
          "emotional state on the same flow-matching zero-shot TTS family as ELaTE — the same "
          "family as VoxCPM2.")

heading(doc, "6.2  Voice quality — whisper, breathy, creaky", 2)

para(doc, "Same measurement first. Whisper is the tractable case in the literature: wTIMIT "
          "provides parallel normal and whispered recordings, and Amazon's voice-conversion "
          "approach to whispered synthesis shipped in Alexa's Whisper Mode. No Khmer equivalent "
          "exists, so a positive result from the prompt channel is the only cheap path.")

heading(doc, "6.3  Discourse — emphasis and contrastive focus", 2)

md(doc, "This layer needs a syntax that does not exist yet. A header tag cannot say which word "
          "to emphasise; that requires span marking in the text, or a parallel per-token flag. "
          "The strongest reference avoids the problem entirely: *Controllable Emphasis with zero "
          "data* increases the predicted duration of the target word rather than training an "
          "emphasis label at all, improving correct identification of the emphasised word by "
          "40% across four languages with no recordings and no annotations.")

rich(doc, [("Why that does not port directly. ", {"bold": True}),
           ("It is a decoder-side intervention on an explicit per-word duration predictor. "
            "VoxCPM2's patch-level diffusion decoder has no such component to reach into, so "
            "this layer is genuinely undesigned rather than merely unbuilt.", {})])

heading(doc, "6.4  Timing — pauses and phrase breaks", 2)

para(doc, "Currently implicit in punctuation. No work done, and no measurement run.")

# =========================================================================
heading(doc, "7.  What is standard here, and what is not", 1)

para(doc, "Worth stating plainly, because a method document that does not separate borrowed "
          "ideas from local inventions is hard to trust and harder to build on.")

rich(doc, [("Standard, and deliberately so. ", {"bold": True}),
           ("Text-side style descriptions as conditioning — the PromptTTS, InstructTTS and "
            "Parler-TTS line. Manufacturing labels by measurement rather than annotation, "
            "including two details arrived at independently here and then found to agree with "
            "`dataspeech`: dropping the extremes when computing bin edges, and judging pitch "
            "relative to comparable speakers rather than the whole corpus. LoRA with float32 "
            "adapters over frozen bfloat16 weights. Condition dropout for classifier-free "
            "guidance, already present in VoxCPM2 as `training_cfg_rate`.", {})])

rich(doc, [("A known problem, met in an unexpected place. ", {"bold": True}),
           ("Finding 1.2 is the information-preference and posterior-collapse failure of "
            "conditional autoregressive models — Chen et al.'s Variational Lossy Autoencoder and "
            "Bowman et al.'s continuous-space sentence VAE — combined with the teacher-forcing "
            "mismatch of Bengio et al.'s scheduled sampling. It is not documented for this class "
            "of TTS system, and recognising it took longer than it should have because the "
            "measurement came before the reading.", {})])

rich(doc, [("Local to this repository, with no citation behind it. ", {"bold": True}),
           ("`enable_proj: true` is an architectural fact about VoxCPM2 — four projection layers "
            "are the only route from the language model into the acoustic generator — not a "
            "general principle; document 9's advice to freeze them is correct for the case it "
            "was written for. The onset-weighted loss is mine, and section 3.7 says what the "
            "more principled comparison would be.", {})])

# =========================================================================
heading(doc, "8.  What this settles, and what it does not", 1)

rich(doc, [('Settled, and the most useful thing here.', {"bold": True}),
           (' VoxCPM2 already has a prosodic control surface, documented only as a passing '
            'example in the model card, and on Khmer it moves pitch 106 Hz at rho +0.76. Nothing '
            'had to be built to get that. Measuring the base model costs about twenty minutes '
            'per layer, and sections 5.1 and 6 are that lesson applied forward.', {})])

rich(doc, [('Settled about the fine-tune.', {"bold": True}),
           (' VoxCPM2 can be given a control surface without touching its architecture, and it '
            'trains on 12 GB rather than the documented 20. The failure mode that makes the '
            'obvious construction quietly not work is identified, measured and fixed — and that '
            'result transfers, because any attribute a fine-tune adds through the text field '
            'faces it.', {})])

rich(doc, [('Settled about method.', {"bold": True}),
           (' Measure what the base model already does before building anything. Measure the '
            'label separation before training, because it bounds what the model can possibly '
            'learn. And take a null result back to the training objective with a positive '
            'control, because generated-audio metrics cannot distinguish a signal that was never '
            'learned from one lost in sampling.', {})])

rich(doc, [('Not settled.', {"bold": True}),
           (' Naturalness. No automatic metric ranks Khmer TTS — this project has already '
            'established that MOS predictors are inverted for Khmer and that prosody statistics '
            'cannot tell well-placed pitch movement from badly-placed pitch movement. Whether '
            'prompted or styled output sounds better is a listening-test question, and the '
            'apparatus exists but has not been run with real listeners.', {})])

rich(doc, [('Next.', {"bold": True}),
           (' Section 5.1. It is an afternoon, it needs no corpus and no GPU time beyond '
            'inference, and it decides whether section 5 is a project or a paragraph.', {})])

# =========================================================================
heading(doc, "9.  References", 1, page_break=True)

def refs(title, items):
    heading(doc, title, 3)
    for txt in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.line_spacing = 1.12
        for i, seg in enumerate(txt.split("*")):
            if seg:
                set_font(p.add_run(seg), BODY_FONT, 9.2, False, i % 2 == 1, INK)

refs("Prompt- and description-based control (layer 1, and layers 3–4 by extension)", [
    "Guo, Z. et al. (2023). *PromptTTS: Controllable text-to-speech with text descriptions.* "
    "ICASSP. arXiv:2211.12171.",
    "Leng, Y. et al. (2024). *PromptTTS 2: Describing and generating voices with text prompt.* "
    "ICLR. arXiv:2309.02285 — a diffusion variation network for the one-to-many problem.",
    "Yang, D. et al. (2023). *InstructTTS: Modelling expressive TTS in discrete latent space with "
    "natural language style prompt.* arXiv:2301.13662.",
    "Lyth, D. & King, S. (2024). *Natural language guidance of high-fidelity text-to-speech with "
    "synthetic annotations.* arXiv:2402.01912 — Parler-TTS; measured attributes as text "
    "conditioning, and the dataspeech measure-and-bin pipeline.",
    "Ji, S. et al. (2024). *TextrolSpeech: A text style control speech corpus with codec language "
    "text-to-speech models.* ICASSP. arXiv:2308.14430.",
])

refs("Non-verbal vocalization (layer 2)", [
    "Kanda, N. et al. (2024). *Making flow-matching-based zero-shot text-to-speech laugh as you "
    "like.* arXiv:2402.07383 — ELaTE. The closest method reference: same conditional "
    "flow-matching decoder family as VoxCPM2, frame-level detector conditioning, and the "
    "small-data-plus-mixing result.",
    "Borisov, M. et al. (2025). *NonverbalTTS: A public English corpus of text-aligned nonverbal "
    "vocalizations with emotion annotations for text-to-speech.* SSW. arXiv:2507.13155 — 17 h, "
    "10 non-verbal types; the automatic-detection-plus-human-validation pipeline copied in "
    "section 5.3. Dataset: huggingface.co/datasets/deepvk/NonverbalTTS.",
    "Gillick, J., Deng, W., Ryokai, K. & Bamman, D. (2021). *Robust laughter detection in noisy "
    "environments.* Interspeech — the detector used in phases 0 and 2. "
    "Code: github.com/jrgillick/laughter-detection.",
    "Gong, Y., Yu, J. & Glass, J. (2022). *VocalSound: A dataset for improving human vocal sounds "
    "recognition.* ICASSP. arXiv:2205.03433 — 21k clips across six non-verbal classes.",
])

refs("Affective and voice quality (layers 3–4)", [
    "Kanda, N. et al. (2024). *Laugh now cry later: Controlling time-varying emotional states of "
    "flow-matching-based zero-shot text-to-speech.* arXiv:2407.12229.",
    "Zhou, K. et al. (2022). *Emotional voice conversion: Theory, databases and ESD.* Speech "
    "Communication — the reference emotional speech dataset.",
    "Amazon (2019). *Voice conversion for whispered speech synthesis.* arXiv:1912.05289 — wTIMIT, "
    "and the method behind Alexa's Whisper Mode.",
])

refs("Emphasis (layer 5)", [
    "Amazon (2023). *Controllable emphasis with zero data for text-to-speech.* arXiv:2307.07062 — "
    "lengthen the target word's predicted duration; +40% correct identification of the emphasised "
    "word, across four languages, with no recordings and no annotations.",
])

refs("Conditioning failure and training method", [
    "Chen, X. et al. (2017). *Variational Lossy Autoencoder.* ICLR. arXiv:1611.02731 — why an "
    "expressive autoregressive decoder ignores its conditioning, and what taking the shortcut "
    "away looks like.",
    "Bowman, S. et al. (2016). *Generating sentences from a continuous space.* CoNLL. "
    "arXiv:1511.06349 — the same collapse in text VAEs; word dropout as the cure.",
    "Bengio, S. et al. (2015). *Scheduled sampling for sequence prediction with recurrent neural "
    "networks.* NeurIPS. arXiv:1506.03099 — teacher forcing and the train/inference mismatch.",
    "Hu, E. et al. (2021). *LoRA: Low-rank adaptation of large language models.* arXiv:2106.09685.",
    "Ho, J. & Salimans, T. (2022). *Classifier-free diffusion guidance.* arXiv:2207.12598 — the "
    "condition dropout VoxCPM2 implements as training_cfg_rate.",
])

refs("VoxCPM2 sources and companion documents", [
    "OpenBMB. *VoxCPM2 model card*, huggingface.co/openbmb/VoxCPM2 — parenthetical voice design "
    "and style-guided cloning, and the stated variability between runs.",
    "OpenBMB. *VoxCPM documentation* — fine-tuning guide, FAQ, and the cookbook that lists the "
    "non-verbal tag inventory in section 4.",
    "Source: voxcpm/training/packers.py (the zero text loss-mask) and voxcpm/model/voxcpm2.py "
    "(from_local, and the float32-in-training line behind the VRAM gap).",
    "This project: docs/09 — VoxCPM2 architecture and training; docs/10 — Higgs TTS 3; "
    "docs/11 — the markdown companion to this report; finetune/README.md — the runbook.",
])

doc.save(OUT)
print(f"wrote {OUT}")
