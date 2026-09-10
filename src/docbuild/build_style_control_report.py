#!/usr/bin/env python
"""
Build the internal engineering report on speech control in VoxCPM2.

An internal ML engineering memo: title page, contents with resolved page
numbers, numbered sections (background, framework, experiments, findings)
and a references list. It is a living document, updated as further
experiments are run, so it deliberately stops at findings and does not
carry a recommendation or next-steps section. Layout primitives are in
src/docbuild/academic_docx.py.

Measurement figures are read from finetune/results/parenthetical/*.json and
finetune/results/nvv/*.json rather than retyped, so the document cannot drift
from the experiments it reports.

Page numbers in the contents are resolved by building the document, converting
it with LibreOffice, reading back which page each heading landed on, and
rebuilding. The pass repeats until the mapping is stable.

    python src/docbuild/build_style_control_report.py

Writes reports/Speech-Control-VoxCPM2.docx
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from academic_docx import *  # noqa: F401,F403
import academic_docx as A

def _yaml(path):
    """Flat scalars and one nested block from the training config, so the
    experiment section reads its hyperparameters out of the file the run
    actually used."""
    out, sect = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body = line.split("#")[0].rstrip()
        key, _, val = body.partition(":")
        val = val.strip()
        if body.startswith(" "):
            if sect is not None:
                out[sect][key.strip()] = _scalar(val)
            continue
        sect = None
        if not val:
            sect = key.strip()
            out[sect] = {}
        else:
            out[key.strip()] = _scalar(val)
    return out


def _scalar(v):
    if v in ("true", "false"):
        return v == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


HERE = Path(__file__).resolve().parent      # src/docbuild
ROOT = HERE.parents[1]                      # repository root
OUT = ROOT / "reports" / "Speech-Control-VoxCPM2.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)

_PARJ = json.loads((ROOT / "finetune" / "results" / "parenthetical" /
                    "parenthetical.json").read_text(encoding="utf-8"))
PAR, CEIL, PROMPTS = _PARJ["summary"], _PARJ["corpus_ceilings"], _PARJ["prompts"]
N_GEN = len(_PARJ["rows"])
N_SENT, N_SEED = _PARJ["n_sentences"], _PARJ["repeats"]

# Model-selection evidence (Section 2.2). Read from the same CER scoring
# results the project's 4-model comparison uses, so the two documents cannot
# disagree about the numbers.
_CER = {m: json.loads((ROOT / "evaluation" / "results" / m / "cer_khmer_asr.json")
                       .read_text(encoding="utf-8"))["summary"]
         for m in ("voxcpm2", "higgs3", "mms", "fish-s2")}

# Tag-conditioning experiments (Section 4.2, 4.3). Every figure quoted there is
# read from the experiment's own output, for the same reason Section 2.2 reads
# the CER results and Section 4.1 reads parenthetical.json: the document
# cannot then drift from the run it reports.
_NVV = ROOT / "finetune" / "results" / "nvv"
_DATA = ROOT / "finetune" / "data-nvv"
OVF = json.loads((_NVV / "tag_sensitivity_overfit.json").read_text(encoding="utf-8"))
NORM = json.loads((_NVV / "tag_sensitivity_n150.json").read_text(encoding="utf-8"))
BASE = json.loads((_NVV / "tag_sensitivity_base_n150.json").read_text(encoding="utf-8"))
HELD = json.loads((_NVV / "tag_sensitivity_heldout.json").read_text(encoding="utf-8"))
BHELD = json.loads((_NVV / "tag_sensitivity_base_heldout.json").read_text(encoding="utf-8"))
REPR = json.loads((_NVV / "tag_representation.json").read_text(encoding="utf-8"))
OMETA = json.loads((_DATA / "overfit" / "overfit_meta.json").read_text(encoding="utf-8"))
CMETA = json.loads((_DATA / "manifests" / "corpus_meta.json").read_text(encoding="utf-8"))
OCFG = _yaml(ROOT / "finetune" / "conf" / "nvv_overfit.yaml")

OVF_MIN = sum(c["duration"] for c in OMETA["clips"]) / 60
OVF_EPOCHS = (OCFG["max_steps"] * OCFG["batch_size"] * OCFG["grad_accum_steps"]
              / OMETA["n"])
CORPUS_H = CMETA["hours"]
DOCUMENTED = set(CMETA["documented_by_voxcpm2"])

SECTIONS = [
    (1, "1", "Executive Summary"),
    (1, "2", "Background"),
    (2, "2.1", "Problem"),
    (2, "2.2", "Model Selection: Why VoxCPM2"),
    (1, "3", "Speech Control Framework"),
    (2, "3.1", "Scope and Definitions"),
    (2, "3.2", "Prosody"),
    (2, "3.3", "Emotion"),
    (2, "3.4", "Non-Verbal Vocalization"),
    (1, "4", "Experiments"),
    (2, "4.1", "Baseline Prosodic Control on Khmer"),
    (2, "4.2", "Tag Conditioning: Capability Test"),
    (2, "4.3", "Tag Conditioning: Generalization Test"),
    (1, "5", "Findings"),
    (1, "", "References"),
]

PAREN_CODE = '''model.generate(
    text="(speaking quickly, a high-pitched voice)"
         "ថ្ងៃនេះអាកាសធាតុល្អណាស់។")'''

TAG_CODE = '''model.generate(text="ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…")'''

TAG_INVENTORY = """[laughing]   [laughter]   [sigh]   [Uhm]   [Shh]
[Question-ah] [Question-ei] [Question-en] [Question-oh]
[Surprise-wa] [Surprise-yo] [Dissatisfaction-hnn]"""


TABLES = ["cer", "prosody", "nvv", "prompts", "paren",
          # Section 4.2 / 4.3
          "ovf_conf", "ovf_cond", "ovf_res", "ovf_cmp", "ovf_tags", "ovf_held"]


def T(key):
    """`Table N` for a table declared in TABLES, so prose references cannot drift
    out of step with the order the tables are actually emitted in."""
    return f"Table {TABLES.index(key) + 1}"


def tbl(doc, key, *args, **kw):
    assert A._TABLE_N[0] == TABLES.index(key), f"{key} emitted out of order"
    return table(doc, *args, **kw)


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}


def word(n):
    return WORDS.get(n, f"{n:,}")


def fp(p):
    """A p-value for a table cell. Sign-test values here span 10^-38 to 0.98,
    and the exponents carry no information a reader needs: what matters is
    whether the condition moved the loss."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def sens(rep, cond):
    """One condition of a tag-sensitivity report as a table row body."""
    c = rep["conditions"][cond]
    return [f"{c['mean']:.5f}", f"{c['delta_mean']:+.5f}",
            f"{c['worse']}/{rep['n']}", fp(c["p"])]


def _ovf_tag_rows():
    """Mean deletion-condition loss per tag. The probe writes one value per clip
    in manifest order, and the overfit manifest records the tag of each clip, so
    the two are zipped rather than the split being restated here."""
    vals = OVF["conditions"]["removed"]["values"]
    per = {}
    for clip, v in zip(OMETA["clips"], vals):
        per.setdefault(clip["tag"], []).append(v)
    rows = sorted(((t, sum(v) / len(v)) for t, v in per.items()),
                  key=lambda r: -r[1])
    return [[f"`{t}`", "yes" if t in DOCUMENTED else "no", f"{m:.4f}"]
            for t, m in rows]


def parrow(axis, label, unit):
    a = PAR[axis]
    lv = [a["median_by_level"][k] for k in ("0", "1", "2")]
    return [label, unit,
            f"{lv[0]:.2f}", f"{lv[1]:.2f}", f"{lv[2]:.2f}",
            f"{a['low_to_high']:+.2f}",
            f"{a['mean_run_to_run_sd']:.2f}",
            f"{CEIL[axis]['spread']:.2f}",
            f"{a['spearman_rho']:+.3f}",
            f"{a['p']:.4f}"]


def cerrow(model, label):
    s = _CER[model]
    return [label, f"{100 * s['cer_median']:.2f}%", f"{100 * s['cer_mean']:.2f}%"]


# =========================================================================
def build(pages):
    """Render the document. `pages` maps '1.2' -> page number, or is empty on
    the first pass, in which case the contents shows a placeholder."""
    A._TABLE_N[0] = 0
    A._FIGURE_N[0] = 0
    doc = new_document()

    title_page(
        doc,
        title="Speech Control for Smean’s TTS on VoxCPM2:\n"
              "Experiments, Findings, and Results",
        prepared_by="R&D Department",
        date="8 September 2026")

    contents(doc, [(lvl, num, txt, pages.get(num or txt, "—"))
                   for lvl, num, txt in SECTIONS])

    # -- 1 Executive Summary -----------------------------------------------
    heading(doc, "1   Executive Summary", 1, page_break=True)

    para(doc, "This report covers speech control for Khmer text-to-speech on VoxCPM2, the "
              "model this project has adopted for Khmer (Section 2.2): the ability to "
              "specify how an utterance is spoken, not just what is said. It is a living "
              "document — it records what has been measured so far and will be updated as "
              "further experiments are run, rather than fixing a conclusion up front.")

    para(doc, "Three results stand out so far.")

    numbered(doc, [
        f"**Prosody (pitch, energy, speaking rate) already works on Khmer, out of the box.** "
        f"A prompt describing a high pitch moves Khmer pitch by "
        f"{PAR['pitch']['low_to_high']:.0f} Hz, about four times what this project's own "
        f"labelled corpus could teach (Section 4.1).",

        "**Emotion is blocked on data, not on the model.** Reliable emotional TTS work "
        "depends on acted, labelled emotional speech corpora, and no such corpus exists for "
        "Khmer (Section 3.3).",

        "**Non-verbal vocalization's tag mechanism can be taught, but the first training "
        "recipe tried does not generalize.** A cheap overfitting test shows the model *can* "
        "learn to place a tag like `[cough]` at a specific point in a sentence (Section "
        "4.2). A held-out test on the same recipe shows that capability does not transfer "
        "to sentences the model was not trained on (Section 4.3).",
    ])

    # -- 2 Background -------------------------------------------------------
    heading(doc, "2   Background", 1, page_break=True)

    heading(doc, "2.1   Problem", 2)

    para(doc, "Getting Khmer text read aloud intelligibly is not enough for most production "
              "use cases — audiobooks, assistants, dubbing, and similar applications need "
              "control over delivery: pacing, pitch, emotional coloring, and non-verbal "
              "sounds like laughs or sighs inserted at the right place. This report answers "
              "two questions so far: what does VoxCPM2 already give us on Khmer for free "
              "(Section 4.1), and can it be taught more (Sections 4.2–4.3).")

    heading(doc, "2.2   Model Selection: Why VoxCPM2", 2)

    para(doc, "VoxCPM2 is an open-weight text-to-speech model released by OpenBMB under the "
              "Apache 2.0 licence, and it is the model this project has adopted for Khmer. "
              "Two practical properties motivated that choice.")

    para(doc, "First, it holds up on Khmer where alternatives fail outright. In the project's "
              "four-model comparison, each model was scored for character error rate (CER) "
              f"on the same fixed set of {word(N_SENT)} Khmer sentences ({T('cer')}). Fish "
              "Audio S2-Pro and Meta MMS both degrade badly on Khmer; VoxCPM2 does not.")

    tbl(doc, "cer",
          "Khmer character error rate across the four models this project evaluated, on "
          "the same fixed 100-sentence set.",
          ["Model", "Median CER", "Mean CER"],
          [cerrow("voxcpm2", "VoxCPM2"),
           cerrow("higgs3", "Higgs TTS 3"),
           cerrow("mms", "Meta MMS"),
           cerrow("fish-s2", "Fish Audio S2-Pro")],
          widths=[1.7, 1.3, 1.3])
    note(doc, "Lower is better. Full methodology and caveats (including a known scorer bias "
              "toward VoxCPM2) are in the project's evaluation report, not repeated here.")

    para(doc, "Second, it needs no Khmer-specific preprocessing. VoxCPM2 reads raw text "
              "directly and infers the language from the script, so Khmer requires no "
              "pronunciation lexicon and no word segmenter — components that do not exist "
              "for Khmer in a production-ready form, and that ordinarily consume most of the "
              "effort in a low-resource TTS project.")

    para(doc, "One more property matters for everything that follows: VoxCPM2 never trains "
              "on the input text as a prediction target — text is purely a conditioning "
              "signal. Practically, this means new markup can be added to the input (a "
              "parenthetical style description, an inline event tag) without disturbing how "
              "the model was originally trained. Both control mechanisms described next rely "
              "on this.")

    para(doc, "VoxCPM2 exposes two ways to influence delivery. A parenthetical description "
              "placed before the sentence characterises the whole utterance:")
    code(doc, PAREN_CODE)
    para(doc, "and a bracketed tag placed inline marks a single event at one position in the "
              "text:")
    code(doc, TAG_CODE)
    para(doc, "The documented tag inventory is:")
    code(doc, TAG_INVENTORY)
    para(doc, "OpenBMB's documentation covers both mechanisms only for Chinese and English; "
              "it says nothing about Khmer. Their behaviour on Khmer is therefore something "
              "this project has to measure rather than assume, which Sections 4.1–4.3 do.")

    # -- 3 Speech Control Framework -----------------------------------------
    heading(doc, "3   Speech Control Framework", 1, page_break=True)

    heading(doc, "3.1   Scope and Definitions", 2)

    para(doc, "\"Speech control\" covers several distinct capabilities. This report separates "
              "three that matter for this project — prosody, emotion, and non-verbal "
              "vocalization — using one distinction that recurs throughout: whether the "
              "thing being controlled is a **global attribute** or a **local event**.")

    para(doc, "A global attribute is a property of the whole utterance (or a long span of "
              "one) — pitch register, speaking rate, emotional tone. A local event is "
              "bounded: it starts, occupies a short interval, and ends, tied to one specific "
              "position in the text — a laugh, a sigh, a filled pause. The distinction is "
              "not just about duration: a global attribute has to compete for influence over "
              "frames the surrounding speech already mostly determines, whereas a local "
              "event is the only thing that determines the frames it occupies. That "
              "difference is why the two are harder or easier to train, and it comes back in "
              "Section 5.")

    para(doc, "Three further categories — voice quality, emphasis, and pause timing — are "
              "adjacent to the three above but out of scope for this report: VoxCPM2 "
              "provides no syntax for them today, and adding one is a larger undertaking than "
              "extending an existing tag.")

    heading(doc, "3.2   Prosody", 2)

    para(doc, "Prosody covers rate, pitch, loudness, pitch variation and pause placement — "
              "everything about delivery once the words themselves are set aside.")

    tbl(doc, "prosody",
          "Prosodic dimensions and how this project measures them.",
          ["Dimension", "What it means", "Measurement used here"],
          [["Speaking rate", "how fast the sentence is spoken", "Khmer characters per second"],
           ["Pitch register", "how high or low the voice is set", "median F0, hertz"],
           ["Pitch variation", "how much the pitch moves", "F0 standard deviation, semitones"],
           ["Loudness", "signal level", "root-mean-square level, dBFS"],
           ["Phrasing", "pause placement and duration", "inter-pausal unit statistics"]],
          widths=[1.4, 2.4, 2.6])

    para(doc, "Prosody is a global attribute, and — as Section 4.1 shows — it is the "
              "capability VoxCPM2 already handles on Khmer without any further work.")

    heading(doc, "3.3   Emotion", 2)

    para(doc, "Emotion is the affective state conveyed by delivery — typically a small closed "
              "set such as neutral, happy, angry, sad, surprised. It overlaps with prosody "
              "(anger and excitement share elevated pitch and energy) but does not reduce to "
              "it: the same rate/pitch/energy setting can read as sarcastic, pleased, or "
              "resigned depending on voice quality and timing that a prosody knob does not "
              "capture.")

    para(doc, "In practice, work on emotional TTS depends almost entirely on acted, parallel, "
              "utterance-labelled emotional speech corpora, and building one for Khmer is not "
              "proportionate to the value it would return right now. VoxCPM2's parenthetical "
              "channel would plausibly accept a description like *(sounding angry)*, since "
              "it is the same free-text field that already carries *(speaking quickly)*, but "
              "this has not been tested. Emotion is therefore out of scope for this phase of "
              "work — see Section 5.")

    heading(doc, "3.4   Non-Verbal Vocalization", 2)

    para(doc, "A non-verbal vocalization is a sound a speaker makes that is not a word: "
              "laughter, a sigh, an audible breath, a filled pause, a cough, a gasp, a sob, a "
              "hesitation particle. These carry stance, regulate turn-taking, and convey "
              "affect, and their presence is a big part of what makes conversational speech "
              "sound conversational rather than read aloud.")

    tbl(doc, "nvv",
          "Non-verbal vocalization types. Tags in the first five rows are documented in "
          "VoxCPM2; the sixth row is not.",
          ["Type", "Tag", "What it communicates"],
          [["Laughter", "`[laughing]`", "amusement, affiliation, mitigation"],
           ["Sigh", "`[sigh]`", "resignation, fatigue, relief"],
           ["Filled pause", "`[Uhm]`", "planning, hesitation, floor-holding"],
           ["Attention marker", "`[Shh]`", "silencing, conspiratorial framing"],
           ["Discourse particle", "`[Question-ah]`, `[Surprise-wa]`, `[Dissatisfaction-hnn]`",
            "stance, back-channelling, question marking"],
           ["Breath, gasp, cough, sob", "—", "phrasing, surprise, distress, physical state"]],
          widths=[1.3, 2.5, 2.6])

    para(doc, "Unlike prosody, non-verbal vocalization is a **local event**: it happens at "
              "one place and leaves the speech before and after it unaffected. That makes it "
              "an easier training target than a global attribute, because the tagged event "
              "has no competing explanation for the frames it occupies — the tag is the only "
              "signal available for them. Whether the tags actually fire on Khmer text is "
              "untested, and Sections 4.2–4.3 measure whether the underlying mechanism can be "
              "taught to work at all, in English first.")

    # -- 4 Experiments --------------------------------------------------------
    heading(doc, "4   Experiments", 1, page_break=True)

    para(doc, "Three experiments were run. The first measures a capability VoxCPM2 already "
              "has, on Khmer. The other two test whether a new capability — inline "
              "non-verbal vocalization tags — can be taught at all, and whether a first "
              "attempt at teaching it generalizes.")

    heading(doc, "4.1   Baseline Prosodic Control on Khmer", 2)

    para(doc, f"The parenthetical mechanism was evaluated on the unmodified model, using "
              f"{word(N_SENT)} sentences drawn from the project's fixed Khmer evaluation "
              f"set. For each of four prosodic axes, three prompts were written to span the "
              f"axis ({T('prompts')}). Every sentence was synthesised under every prompt at "
              f"{word(N_SEED)} random seeds, giving {N_GEN} generations, and the median taken "
              f"within each cell of the design. Pitch, its variation, loudness, and speaking "
              f"rate were measured directly from the audio.")

    tbl(doc, "prompts",
          "Prompts used to span each prosodic axis.",
          ["Axis", "Level 0", "Level 1", "Level 2"],
          [["Pitch"] + [f"*{s}*" for s in PROMPTS["pitch"]],
           ["Energy"] + [f"*{s}*" for s in PROMPTS["energy"]],
           ["Rate"] + [f"*{s}*" for s in PROMPTS["rate"]],
           ["Variation"] + [f"*{s}*" for s in PROMPTS["var"]]],
          widths=[0.85, 1.85, 1.85, 1.85])

    tbl(doc, "paren",
          "Response of the unmodified model to a parenthetical prompt on Khmer. "
          "The corpus bound is the separation between the low and high bands of this "
          "project's hand-labelled Khmer corpus, and represents an upper limit on what "
          "fine-tuning against that corpus could teach.",
          ["Axis", "Unit", "Level 0", "Level 1", "Level 2", "Change",
           "Seed s.d.", "Corpus bound", "ρ", "p"],
          [parrow("pitch", "Pitch", "Hz"),
           parrow("energy", "Energy", "dBFS"),
           parrow("rate", "Rate", "char/s"),
           parrow("var", "Variation", "st")],
          widths=[0.72, 0.5, 0.58, 0.58, 0.58, 0.58, 0.55, 0.72, 0.55, 0.5],
          size=8.2, align_right=(2, 3, 4, 5, 6, 7, 8, 9))
    note(doc, "ρ is the Spearman rank correlation between prompt level and measured value; "
              "p is a permutation test over 10,000 relabellings. Seed s.d. is the mean "
              "run-to-run standard deviation within a cell.")

    para(doc, "Three findings follow.")

    para(doc, f"**Pitch** is controlled reliably. The separation between the extreme prompts is "
              f"{PAR['pitch']['low_to_high']:.2f} Hz with a rank correlation of "
              f"{PAR['pitch']['spearman_rho']:+.3f} (p = {PAR['pitch']['p']:.4f}). This "
              f"exceeds the corpus bound of {CEIL['pitch']['spread']:.2f} Hz by a factor of "
              f"approximately four — a fine-tune on this project's own labelled data could "
              f"not improve on this axis.")

    para(doc, f"**Energy** and **Speaking Rate** are controlled in aggregate but not per "
              f"generation. Both show significant rank correlations "
              f"({PAR['energy']['spearman_rho']:+.3f} and "
              f"{PAR['rate']['spearman_rho']:+.3f}), but the energy effect of "
              f"{PAR['energy']['low_to_high']:.2f} dB is smaller than the "
              f"{PAR['energy']['mean_run_to_run_sd']:.2f} dB standard deviation observed "
              f"across seeds within a single cell. The ordering of the levels is dependable; "
              f"the value of any individual synthesis is not.")

    para(doc, f"**Pitch Variation** is not controlled. The correlation is negative "
              f"({PAR['var']['spearman_rho']:+.3f}) and not significant "
              f"(p = {PAR['var']['p']:.2f}). Prompts requesting a lively delivery produced "
              f"marginally less pitch movement than prompts requesting a monotone one, "
              f"consistent with the axis being unaddressed rather than inverted.")

    para(doc, "Inline tag behaviour on Khmer was not measured here; Sections 4.2–4.3 test the "
              "underlying tagging mechanism in English first, as a prerequisite.")

    heading(doc, "4.2   Tag Conditioning: Capability Test", 2)

    para(doc, f"A conventional fine-tune had already been performed on "
              f"{CORPUS_H:.2f} hours of tagged English speech derived from "
              f"{CMETA['source']}, evaluated on {NORM['n']} clips drawn from its own "
              f"training set, and its effect on the inline tag was close to nothing. Deleting "
              f"the tag from the transcript raised the loss on only "
              f"{NORM['conditions']['removed']['worse']} of {NORM['n']} clips — no better "
              f"than chance (p = {NORM['conditions']['removed']['p']:.3f}) — and moving the "
              f"tag to the end of the sentence made no difference either "
              f"(p = {NORM['conditions']['moved']['p']:.3f}).")

    para(doc, "That null result has two possible explanations, and generated audio alone "
              "cannot tell them apart: a dependency that was never learned sounds the same as "
              "one that was learned and then washed out at generation time.")

    numbered(doc, [
        "**The mechanism is absent.** The model has no way to bind an event to a specific "
        "position in the text. If true, inline tags are not achievable at any corpus size, "
        "and the project should fall back to global descriptors of the kind Section 4.1 "
        "measures.",
        f"**The signal is too weak.** The mechanism exists, but "
        f"{CMETA['tag_counts']['[cough]']} examples of the most common non-laughter tag, "
        f"spread over {CORPUS_H:.2f} hours, is too sparse for the model to pick up on. If "
        f"true, the approach is sound and the fix is more/better data.",
    ])

    para(doc, "Deliberate overfitting is the cheapest way to tell these apart: memorize a "
              "small set of recordings, then check whether the tag's influence shows up on "
              "those same recordings. A model that still ignores the tag after seeing the "
              "answer hundreds of times has problem (1). A model that picks it up has "
              "problem (2), which is fixable.")

    para(doc, f"{T('ovf_conf')} summarises the run. It deliberately does everything a "
              f"production fine-tune would not: no mix of untagged data (its only job is "
              f"preventing regression, which this run isn't worried about), no weight decay, "
              f"a 5x learning rate, and validation on the training clips themselves, so "
              f"validation loss reads memorization rather than generalization.")

    tbl(doc, "ovf_conf",
        "Configuration of the overfitting run. The full file is "
        "`finetune/conf/nvv_overfit.yaml`.",
        ["Quantity", "Value", "Rationale"],
        [["Clips", f"{OMETA['n']} ({OVF_MIN:.1f} min)",
          f"{OMETA['per_tag']} per tag; small enough to memorize in about an hour"],
         ["Tags", f"{len(OMETA['tags'])}", "four undocumented, one documented as control"],
         ["Untagged mixture", "none", "regularization is not wanted here"],
         ["Learning rate", f"{OCFG['learning_rate']:g}",
          "five times the documented rate for this adapter"],
         ["Weight decay", f"{OCFG['weight_decay']:g}", "no regularization, deliberately"],
         ["Warmup", f"{OCFG['warmup_steps']} steps", "short, for the same reason"],
         ["Steps", f"{OCFG['max_steps']:,}",
          f"about {OVF_EPOCHS:.0f} passes over the same clips"],
         ["Fine-tune scope", f"rank {OCFG['lora']['r']} (alpha {OCFG['lora']['alpha']})",
          "a lightweight adapter covering the relevant model components"],
         ["Validation set", "identical to training",
          "validation loss reads memorization by design"]],
        widths=[1.05, 0.95, 2.4], size=8.4)

    para(doc, f"Clips were admitted only if they carried exactly one tag, occurring exactly "
              f"once, in a 2–8 second recording — the cleanest possible mapping from one "
              f"string to one event. Four of the five tags used — "
              f"{', '.join('`' + t + '`' for t in OMETA['tags'] if t not in DOCUMENTED)} "
              f"— appear nowhere in the inventory OpenBMB publishes ({T('nvv')}). The fifth, "
              f"`[laughing]`, is documented and kept as a control: the unmodified model "
              f"already produces laughter, so if it trained appreciably better than the "
              f"undocumented tags, that would suggest the model was retrieving prior "
              f"knowledge of the word rather than learning a new binding.")

    para(doc, "This test was run in English rather than Khmer, on purpose: it asks only "
              "whether the tagging mechanism itself can be taught to work at all, not "
              "whether it works specifically in Khmer. That second question is deferred "
              "until this one is settled.")

    para(doc, f"The instrument is a forward pass, one per clip per condition, comparing the "
              f"loss across four rewritings of the transcript ({T('ovf_cond')}). Generated "
              f"audio is not used to decide the question, for the reason given above.")

    tbl(doc, "ovf_cond",
        "The four conditions of the sensitivity probe.",
        ["Condition", "Transcript", "What it tests"],
        [["True", "tag inline, at the event", "reference"],
         ["Removed", "tag deleted", "whether the tag carries information at all"],
         ["Moved", "tag pushed to the end of the sentence",
          "whether the tag's **position** carries information"],
         ["Scrambled", "word order destroyed, tags left in place",
          "positive control"]],
        widths=[0.75, 1.6, 2.05], size=8.4)

    para(doc, "The **moved** condition is what distinguishes a local event from an "
              "utterance-level flag: a model that only learned \"this clip contains a "
              "cough\" scores the same when the tag is relocated, whereas a model that "
              "learned *where* the cough belongs does not. The **scrambled** condition is "
              "the sanity check: it must raise the loss, or the probe isn't measuring "
              "anything and no other row can be trusted.")

    heading(doc, "Results", 3)

    tbl(doc, "ovf_res",
        f"Effect of each rewriting on the loss after memorization "
        f"(n = {OVF['n']}; reference loss {OVF['true_mean']:.5f}). *Worse* counts the "
        f"clips on which the rewriting raised the loss; *p* is a two-sided sign test.",
        ["Condition", "Mean loss", "Change", "Worse", "p"],
        [["Removed"] + sens(OVF, "removed"),
         ["Moved"] + sens(OVF, "moved"),
         ["Scrambled"] + sens(OVF, "scrambled")],
        widths=[1.0, 0.85, 0.8, 0.7, 0.75], size=8.6,
        align_right=(1, 2, 3, 4))
    note(doc, f"The positive control raised the loss by "
              f"{OVF['conditions']['scrambled']['delta_mean']:+.5f} on "
              f"{OVF['conditions']['scrambled']['worse']} of {OVF['n']} clips, so the probe "
              f"is measuring the objective and the other two rows can be read.")

    para(doc, f"**The tag now carries information.** Deleting it raises the loss on "
              f"{OVF['conditions']['removed']['worse']} of {OVF['n']} clips. **Its position "
              f"matters too** — the more important finding: moving the tag without deleting "
              f"it raises the loss on {OVF['conditions']['moved']['worse']} of {OVF['n']} "
              f"clips, at "
              f"{100 * OVF['conditions']['moved']['delta_mean'] / OVF['conditions']['removed']['delta_mean']:.0f}"
              f"% of the cost of deleting it — whereas the same test on the unmodified model "
              f"gives p = {BASE['conditions']['moved']['p']:.2f}. The model isn't just "
              f"flagging \"this utterance contains a cough\"; it's reading where in the "
              f"sentence the tag sits and placing the event there.")

    para(doc, f"Absolute losses aren't comparable across model states, evaluated on "
              f"different clips at different points in training. {T('ovf_cmp')} instead "
              f"compares the cost of deleting the tag as a fraction of the cost of "
              f"destroying the whole transcript — a ratio internal to each model.")

    tbl(doc, "ovf_cmp",
        "Salience of the inline tag relative to the transcript, across the three "
        "states of the model. The ratio is the cost of deleting the tag divided by the "
        "cost of scrambling the transcript, measured within each model. All three rows "
        "are measured on clips the model in question was trained on; the held-out "
        "comparison is reported separately in Section 4.3.",
        ["Model state", "n", "Deletion p", "Relocation p", "Tag / transcript"],
        [["Unmodified", f"{BASE['n']}", fp(BASE["conditions"]["removed"]["p"]),
          fp(BASE["conditions"]["moved"]["p"]),
          f"{100 * BASE['removed_over_scrambled']:.1f}%"],
         [f"Fine-tuned, {CORPUS_H:.2f} h", f"{NORM['n']}",
          fp(NORM["conditions"]["removed"]["p"]),
          fp(NORM["conditions"]["moved"]["p"]),
          f"{100 * NORM['removed_over_scrambled']:.1f}%"],
         [f"Overfitted, {OVF_MIN:.1f} min", f"{OVF['n']}",
          fp(OVF["conditions"]["removed"]["p"]),
          fp(OVF["conditions"]["moved"]["p"]),
          f"{100 * OVF['removed_over_scrambled']:.1f}%"]],
        widths=[1.25, 0.4, 0.9, 0.9, 1.0], size=8.6,
        align_right=(1, 2, 3, 4))
    note(doc, f"The two p-value columns are sign tests on the deletion and relocation "
              f"conditions respectively. The overfitted figure is "
              f"{OVF['removed_over_scrambled'] / NORM['removed_over_scrambled']:.1f} times "
              f"the fine-tuned one and "
              f"{OVF['removed_over_scrambled'] / BASE['removed_over_scrambled']:.1f} times "
              f"the unmodified one.")

    tbl(doc, "ovf_tags",
        "Loss under the deletion condition, by tag. The control tag is the only one "
        "OpenBMB documents.",
        ["Tag", "Documented", "Mean loss when deleted"],
        _ovf_tag_rows(),
        widths=[1.1, 0.9, 1.4], size=8.6, align_right=(2,))
    note(doc, "The four undocumented tags are not weaker than the documented one, which "
              f"argues that the effect in {T('ovf_res')} is a newly learned binding rather "
              "than the model retrieving prior knowledge of the English words in brackets.")

    heading(doc, "Interpretation and limits", 3)

    para(doc, "This experiment shows VoxCPM2 *can* bind an arbitrary bracketed tag to a "
              "specific non-verbal event at a specific position in a sentence, and that a "
              "lightweight fine-tune is enough to install that binding. The \"mechanism is "
              "absent\" explanation from above is ruled out: the earlier null result was a "
              "problem with the training data, not the model. That's the finding the project "
              "needed, obtained for the cost of an hour of training.")

    para(doc, f"It proves nothing about generalization, and should not be quoted as though it "
              f"did. The evaluation runs on the same {OMETA['n']} sentences the model was "
              f"trained on, roughly {OVF_EPOCHS:.0f} times each — that's what overfitting "
              f"means. Whether the same tags work on text the model hasn't seen is a "
              f"separate question, addressed next.")

    heading(doc, "4.3   Tag Conditioning: Generalization Test", 2)

    para(doc, f"The generalization test Section 4.2 deferred to has since been run, and the "
              f"result is negative. The validation split of the corpus was never shown to "
              f"the model during training; {HELD['n']} of its clips carry an inline tag, and "
              f"those {HELD['n']} are the entire available sample. The same four-condition "
              f"probe was applied to the fine-tuned model and to the unmodified model on "
              f"those identical clips, so the two are compared within one clip set rather "
              f"than across sets. {T('ovf_held')} reports the outcome.")

    tbl(doc, "ovf_held",
        f"Tag sensitivity on {HELD['n']} clips held out of training, for the fine-tuned "
        f"model and the unmodified model over the same clips. The two are "
        f"indistinguishable.",
        ["Model state", "Reference loss", "Deletion Δ", "Clips worse", "p",
         "Tag / transcript"],
        [["Unmodified", f"{BHELD['true_mean']:.5f}",
          f"{BHELD['conditions']['removed']['delta_mean']:+.5f}",
          f"{BHELD['conditions']['removed']['worse']}/{BHELD['n']}",
          fp(BHELD["conditions"]["removed"]["p"]),
          f"{100 * BHELD['removed_over_scrambled']:.1f}%"],
         ["Fine-tuned", f"{HELD['true_mean']:.5f}",
          f"{HELD['conditions']['removed']['delta_mean']:+.5f}",
          f"{HELD['conditions']['removed']['worse']}/{HELD['n']}",
          fp(HELD["conditions"]["removed"]["p"]),
          f"{100 * HELD['removed_over_scrambled']:.1f}%"]],
        align_right=(1, 2, 3, 4, 5))

    para(doc, f"The two rows describe the same behaviour. The cost of deleting the tag "
              f"differs between them by only "
              f"{abs(HELD['conditions']['removed']['delta_mean'] - BHELD['conditions']['removed']['delta_mean']):.4f}, "
              f"and the sign-test counts differ by a single clip in the unmodified model's "
              f"favour. Whatever sensitivity to the tag these sentences show, the model had "
              f"it before training — {OCFG['max_steps']:,} steps on {CORPUS_H:.2f} hours "
              f"added nothing that reaches text the model hasn't seen.")

    para(doc, f"Two caveats. First, statistical power: at n = {HELD['n']}, a sign test needs "
              f"roughly twenty clips of twenty-eight to reach significance, so a small true "
              f"effect could be hiding here — what this rules out is a large one, which is "
              f"what a working method should produce given the capability shown in Section "
              f"4.2. Second, the final column isn't comparable to the same column in "
              f"{T('ovf_cmp')}: these are different sentences, and the unmodified model "
              f"scores {100 * BHELD['removed_over_scrambled']:.0f}% here against "
              f"{100 * BASE['removed_over_scrambled']:.0f}% on the clips used there — which "
              f"is why the unmodified model was re-measured on this specific held-out set "
              f"rather than reusing the earlier number.")

    para(doc, "Together, Sections 4.2 and 4.3 tell a consistent story: the mechanism exists "
              "— VoxCPM2 can bind a bracketed tag to a local event and to its position — but "
              "the fine-tune as configured does not install that binding in a form that "
              "survives to unseen text. The fixes below are prerequisites, not refinements.")

    note(doc, "Provenance. The figures in Sections 4.2–4.3 are read at build time from "
              "`finetune/results/nvv/tag_sensitivity_overfit.json` and its counterparts "
              "for the other model states and the held-out set. The trained adapter itself "
              "was deleted in error after the run and has not been regenerated; the recorded "
              "measurements and the synthesized audio survive, but reproducing the audio "
              "would require repeating the training. The procedure is documented in "
              "`finetune/OVERFIT_EXPERIMENT.md`.")

    # -- 5 Findings ----------------------------------------------------------
    heading(doc, "5   Findings", 1, page_break=True)

    para(doc, "Of the three speech-control categories in Section 3, one is already handled by "
              "the base model, one is blocked on data, and one has a working mechanism whose "
              "training recipe does not yet generalize. This section restates what each "
              "experiment established; it will be extended as further experiments are run.")

    para(doc, f"**Prosody is already handled.** The parenthetical mechanism moves Khmer pitch "
              f"by {PAR['pitch']['low_to_high']:.2f} Hz at a rank correlation of "
              f"{PAR['pitch']['spearman_rho']:+.3f}, roughly four times the "
              f"{CEIL['pitch']['spread']:.2f} Hz this project's own labelled corpus could "
              f"express (Section 4.1). A prosody fine-tune trained on that corpus would "
              f"reproduce a capability the base model already has, in a conditioning "
              f"channel that's already in use. What remains — pitch variation and "
              f"seed-to-seed consistency — is real but small.")

    para(doc, "**Emotion is blocked on data, not method.** Every reliable result in this "
              "space depends on acted, parallel, labelled emotional speech, and no Khmer "
              "corpus of that kind exists (Section 3.3).")

    para(doc, "**Non-verbal vocalization has a mechanism that works but a recipe that "
              "doesn't yet generalize.** It is a local event, not a global attribute "
              "(Section 3.1) — the tagged event has no competing explanation for the frames "
              "it occupies, unlike prosody or emotion. The interface already exists: "
              "`[laughing]`, `[sigh]` and `[Uhm]` are documented in the unmodified model. "
              "Section 4.2 shows the underlying binding can be taught; Section 4.3 shows "
              "the specific recipe tried does not carry that binding to text the model "
              "hasn't seen. Whether a different recipe closes that gap is open, and is the "
              "next thing this document will report on.")

    # -- refs -------------------------------------------------------------
    heading(doc, "References", 1, page_break=True)

    for r in [
        "Gillick, J., Deng, W., Ryokai, K. and Bamman, D. (2021). Robust laughter detection "
        "in noisy environments. *Interspeech 2021*, 2481–2485.",

        "Gong, Y., Yu, J. and Glass, J. (2022). VocalSound: a dataset for improving human "
        "vocal sounds recognition. *ICASSP 2022*. arXiv:2205.03433.",

        "Guo, Z., Leng, Y., Wu, Y., Zhao, S. and Tan, X. (2023). PromptTTS: controllable "
        "text-to-speech with text descriptions. *ICASSP 2023*. arXiv:2211.12171.",

        "Hsu, C.-C., Kanda, N., Zhu, Y. et al. (2024). Laugh Now Cry Later: controlling "
        "time-varying emotional states of flow-matching-based zero-shot text-to-speech. "
        "*IEEE SLT 2024*. arXiv:2407.12229.",

        "Ji, S., Zuo, J., Fang, M. et al. (2024). TextrolSpeech: a text style control speech "
        "corpus with codec language text-to-speech models. *ICASSP 2024*. arXiv:2308.14430.",

        "Kanda, N., Wang, X., Eskimez, S. E. et al. (2024). Making flow-matching-based "
        "zero-shot text-to-speech laugh as you like. arXiv:2402.07383.",

        "Leng, Y., Guo, Z., Shen, K. et al. (2024). PromptTTS 2: describing and generating "
        "voices with text prompt. *ICLR 2024*. arXiv:2309.02285.",

        "Lyth, D. and King, S. (2024). Natural language guidance of high-fidelity "
        "text-to-speech with synthetic annotations. arXiv:2402.01912. Released as Parler-TTS.",

        "MNV-17: a high-quality performative Mandarin dataset for nonverbal vocalization "
        "recognition in speech (2025). arXiv:2509.18196.",

        "NonverbalTTS: a public English corpus of text-aligned nonverbal vocalizations with "
        "emotion annotations for text-to-speech (2025). *Speech Synthesis Workshop 2025*. "
        "arXiv:2507.13155.",

        "NVMOS: non-verbal vocalization quality assessment in speech (2026). arXiv:2606.15888.",

        "NVSpeech: an integrated and scalable pipeline for human-like speech modeling with "
        "paralinguistic vocalizations (2025). arXiv:2508.04195.",

        "NVV-SuperBench: beyond words, beyond quality — benchmarking nonverbal vocalizations "
        "in speech generation (2026). arXiv:2604.16211.",

        "OpenBMB (2025). VoxCPM2 model card. huggingface.co/openbmb/VoxCPM2.",

        "Yang, D., Liu, S., Huang, R. et al. (2023). InstructTTS: modelling expressive TTS in "
        "discrete latent space with natural language style prompt. arXiv:2301.13662.",

        "Zhou, K., Sisman, B., Liu, R. and Li, H. (2022). Emotional voice conversion: theory, "
        "databases and ESD. *Speech Communication*, 137, 1–18. arXiv:2105.14762.",
    ]:
        reference(doc, r)

    doc.save(OUT)
    return doc


# =========================================================================
def measure_pages():
    """Convert with LibreOffice and read back the page each heading landed on."""
    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                                "--outdir", td, str(OUT)],
                               capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            print("  soffice not found on PATH; leaving contents page numbers "
                  "as placeholders", file=sys.stderr)
            return {}
        pdf = Path(td) / (OUT.stem + ".pdf")
        if not pdf.exists():
            print("  pdf conversion failed:", r.stdout.strip(), r.stderr.strip(),
                  file=sys.stderr)
            return {}
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                             capture_output=True, text=True, check=True).stdout
    pages = [re.sub(r"\s+", " ", pg) for pg in txt.split("\f")]
    # locate the contents page rather than assuming its index: the title page
    # is allowed to change length without silently breaking the mapping
    toc = next((i for i, pg in enumerate(pages) if pg.strip().startswith("Contents")), 1)
    out = {}
    for lvl, num, title in SECTIONS:
        needle = re.sub(r"\s+", " ", f"{num} {title}" if num else title).strip()
        for i, pg in enumerate(pages[toc + 1:], start=toc + 2):
            if needle in pg:
                out[num or title] = i
                break
    return out


if __name__ == "__main__":
    pages, prev = {}, None
    for attempt in range(4):
        build(pages)
        found = measure_pages()
        missing = [n or t for _, n, t in SECTIONS if (n or t) not in found]
        if found == pages:
            break
        prev, pages = pages, found
    else:
        print("warning: page numbering did not settle", file=sys.stderr)
    if missing:
        print(f"warning: no page found for {missing}", file=sys.stderr)
    print(f"wrote {OUT}")
    print("  contents:", ", ".join(f"{k}={v}" for k, v in pages.items()))
