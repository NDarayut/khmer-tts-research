#!/usr/bin/env python
"""
Build the Smean AI technical report on speech control in VoxCPM2.

Companion to build_literature_review.py, in the same house style (see
smean_docx.py). Where the literature review surveys models, this one sets out
what can be controlled in synthetic speech, what VoxCPM2 already controls on
Khmer, and what the published work has built for the rest.

Content is sourced from docs/11-voxcpm2-style-control-finetune.md. The
measurement numbers are read from finetune/results/parenthetical/*.json rather
than retyped, so the document cannot drift from the experiment it describes.

    python docs/build_style_control_report.py

Writes docs/Smean-VoxCPM2-Style-Control.docx
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smean_docx import *  # noqa: F401,F403  -- brand tokens and layout helpers

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "Smean-VoxCPM2-Style-Control.docx"

_PARJ = json.loads((ROOT / "finetune" / "results" / "parenthetical" /
                    "parenthetical.json").read_text())
PAR, CEIL, PROMPTS = _PARJ["summary"], _PARJ["corpus_ceilings"], _PARJ["prompts"]
N_GEN = len(_PARJ["rows"])

_TBL = [0]


def tcap(doc, text):
    """Numbered table caption. Auto-incrementing, so inserting a table upstream
    cannot leave two Table 4s behind -- which it did, once."""
    _TBL[0] += 1
    caption(doc, f"Table {_TBL[0]} — {text}")


def parrow(axis, label, unit, fmt=".2f", bold=False):
    """One row of the parenthetical table, read straight from the results json."""
    a = PAR[axis]
    lv = [a["median_by_level"][k] for k in ("0", "1", "2")]
    d = f"{a['low_to_high']:+.2f}"
    r = f"{a['spearman_rho']:+.3f}"
    return [label,
            f"{lv[0]:{fmt}}", f"{lv[1]:{fmt}}", f"{lv[2]:{fmt}}",
            f"**{d}**" if bold else d,
            f"+{CEIL[axis]['spread']:.2f}",
            f"**{r}**" if bold else r,
            f"{a['p']:.4f}",
            f"{a['mean_run_to_run_sd']:.2f}",
            unit]


STACK = """Khmer text
  |   byte-level tokenizer (vocab 73,448 -- no G2P, no language tag)
  v
+--------------------------------------------------+
| MiniCPM4 backbone LM        2048 dim x 28 layers |  one latent per audio patch
| GQA 16 query / 2 KV heads, LongRoPE to 32k       |
+--------------------------------------------------+
  |   hidden state
  v
+--------------------------------------------------+
| Residual LM                 8 layers, no RoPE    |  refines the latent
+--------------------------------------------------+
  |   conditioning vector
  v
+--------------------------------------------------+
| Local DiT       1024 dim x 12 layers, CFM head   |  diffusion: latent -> features
| euler solver, log-norm schedule, CFG 2.0         |  10 steps at inference
+--------------------------------------------------+
  |   64-dim features, 4 frames per patch
  v
+--------------------------------------------------+
| AudioVAE V2     enc 16 kHz  ->  dec 48 kHz       |  waveform
+--------------------------------------------------+"""

LOSS_MASK = (
    "# voxcpm/training/packers.py :: process_tts_data\n"
    "loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])\n"
    "#            ^^^^^^^^^^^^^^^^^^^^ zero across every text position"
)

PAREN_USAGE = (
    "# a natural-language description of the whole utterance\n"
    'model.generate(text="(speaking quickly, a high-pitched voice)"\n'
    '                    "ថ្ងៃនេះអាកាសធាតុល្អណាស់។")'
)

NV_USAGE = (
    "# an inline tag marking one event at one position\n"
    'model.generate(text="ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…")\n'
    "\n"
    "# documented inventory\n"
    "[laughing] [laughter] [sigh] [Uhm] [Shh]\n"
    "[Question-ah|ei|en|oh] [Surprise-wa|yo] [Dissatisfaction-hnn]"
)

doc = new_document()

cover(doc,
      title="Speech Control for VoxCPM2",
      subtitle="What can be controlled in synthetic speech, what VoxCPM2\n"
               "already does on Khmer, and what the literature has built",
      blurb="Expressive control is not one capability. Prosody, emotion and non-verbal "
            "vocalization are separable problems that differ in what they describe and in "
            "whether they are properties of a whole utterance or events at a single "
            "position. This report introduces VoxCPM2 and the control it ships with, "
            "reports what that control actually does on Khmer against measurement, "
            "reviews the published work on each layer, and states which layer this "
            "project will build.",
      meta=[["Document", "Technical report — speech control, measurement and literature"],
            ["Subject", "VoxCPM2 (OpenBMB) · expressive control for Khmer"],
            ["Scope", "Prosody · emotion · non-verbal vocalization"],
            ["Evidence base",
             f"A {N_GEN}-generation sweep of the stock model over the frozen "
             "evaluation set, plus 20 published papers"],
            ["Date", "8 September 2026"],
            ["Status", "Internal research document"]])

page_footer(doc, "Smean AI  ·  VoxCPM2 speech control")

# =========================================================================
heading(doc, "Contents", 1, page_break=True)

table(doc,
      ["§", "Section", "What is in it"],
      [["1", "**Overview**",
        "VoxCPM2, its architecture, and the two control mechanisms it ships with — "
        "including the measurement showing the parenthetical prompt controls Khmer pitch"],
       ["2", "**Speech control**",
        "What prosody, emotion and non-verbal vocalization are, with examples, and the "
        "global/local distinction that decides engineering cost"],
       ["3", "**Literature review**",
        "Natural-language style control, emotional speech synthesis, non-verbal "
        "vocalization, and the evaluation problem — closing with the layer we will build"],
       ["4", "**Methodology**", "To be written"],
       ["", "**References**", "Grouped by layer"]],
      widths=[0.5, 1.7, 5.0], size=9.2, zebra=True)

# =========================================================================
heading(doc, "1.  Overview", 1, page_break=True)

heading(doc, "1.1  What VoxCPM2 is", 3)

md(doc, "VoxCPM2 is OpenBMB's **tokenizer-free, diffusion-autoregressive** text-to-speech "
        "model — 2.29 B parameters, Apache-2.0. It is this project's recommended model for "
        "Khmer: on the frozen 100-sentence evaluation set it scores a **2.47% median "
        "character error rate**, against 8.28% for Higgs TTS 3, 25.12% for Meta MMS and "
        "78.01% for Fish Audio S2-Pro.")

md(doc, "*Tokenizer-free* is the property that matters for a low-resource language. The "
        "model does not quantize speech into a discrete codebook; it predicts *continuous* "
        "latent vectors, one per four-frame patch of audio. There is therefore no audio "
        "vocabulary that can be badly fitted for Khmer — the failure that eliminated Fish "
        "Audio S2 has no analogue here. On the text side the model reads raw UTF-8 bytes "
        "through a 73,448-entry tokenizer, so Khmer needs **no G2P, no phonemizer, no "
        "lexicon and no word segmenter**, none of which exist in usable form for the "
        "language.")

heading(doc, "1.2  Architecture", 3)

code(doc, STACK, size=6.6)
caption(doc, "The VoxCPM2 stack. A Local Encoder (1024 × 12) sits on the input side and "
             "encodes a reference clip into the same latent space for zero-shot cloning.")

table(doc,
      ["Property", "Value", "Consequence"],
      [["`patch_size`", "4", "One LM step covers four VAE frames."],
       ["`feat_dim`", "64", "Latent width the DiT predicts."],
       ["AudioVAE frame rate", "25 fps", "One second of audio ≈ 6.25 LM positions."],
       ["Encoder rate", "16 kHz", "Training audio **must** be 16 kHz; the validator rejects mismatches."],
       ["Decoder rate", "48 kHz", "Super-resolution is built in; you do not supply 48 kHz data."],
       ["`inference_cfg_rate`", "2.0", "Classifier-free guidance default, exposed as `--cfg-value`."]],
      widths=[1.5, 1.0, 4.0], size=8.8)
tcap(doc, "Architecture properties that govern training and control.")

md(doc, "Two structural facts govern everything that follows.")

md(doc, "**The Local DiT is a conditional flow-matching decoder.** That is the same model "
        "family the non-verbal vocalization literature fine-tunes, which is why those "
        "recipes are applicable to this model rather than merely adjacent to it.")

md(doc, "**The training loss mask is zero across every text position.**")

code(doc, LOSS_MASK)

md(doc, "Nothing in the text field is ever a prediction target. The text field is a pure "
        "conditioning channel, and anything can be placed in it — a tag, a description, a "
        "marker — without disturbing the training objective.")

heading(doc, "1.3  The control VoxCPM2 ships with", 3)

md(doc, "VoxCPM2 has two built-in control mechanisms, and they operate at different scopes. "
        "A **parenthetical natural-language prompt** describes the whole utterance:")

code(doc, PAREN_USAGE)

md(doc, "**Inline square-bracket tags** mark a single event at one position:")

code(doc, NV_USAGE)

heading(doc, "1.4  The parenthetical prompt works on Khmer", 3)

md(doc, f"Measured on the stock model, no fine-tuning: 8 sentences drawn from the frozen "
        f"evaluation set, three prompt levels per axis, three seeds per cell, median taken "
        f"within each cell — {N_GEN} generations in total. The *ceiling* column is the "
        f"separation between the low and high bands of this project's own hand-labelled "
        f"Khmer corpus: what a fine-tune on that corpus could teach, at best.")

table(doc,
      ["Axis", "low", "mid", "high", "Δ", "ceiling", "ρ", "p", "seed sd", "unit"],
      [parrow("pitch", "**pitch**", "Hz", ".2f", bold=True),
       parrow("energy", "energy", "dBFS"),
       parrow("rate", "rate", "char/s"),
       parrow("var", "variation", "st")],
      widths=[1.0, 0.75, 0.75, 0.75, 0.85, 0.8, 0.8, 0.7, 0.7, 0.7],
      size=8.2, align_right=(1, 2, 3, 4, 5, 6, 7, 8))
tcap(doc, "Stock VoxCPM2 under a parenthetical prompt. ρ is Spearman rank correlation "
          "between prompt level and measured value; p is a permutation test; *seed sd* is "
          "the mean run-to-run standard deviation within a cell.")

md(doc, "Read it as three results, not one.")

bullet(doc, [("Pitch is controlled, strongly. ", {"bold": True}),
             (f"The +{PAR['pitch']['low_to_high']:.2f} Hz span is nearly four times the "
              f"+{CEIL['pitch']['spread']:.2f} Hz separating the low and high pitch bands of "
              "the hand-labelled corpus. Whatever a fine-tune could teach about pitch, the "
              "base model already exceeds.", {})])
bullet(doc, [("Energy and rate move in the right direction but are noise-limited. ",
              {"bold": True}),
             (f"The {PAR['energy']['low_to_high']:.2f} dB energy effect sits under a "
              f"{PAR['energy']['mean_run_to_run_sd']:.2f} dB seed-to-seed standard deviation. "
              "The rank correlation is real; the per-generation effect is not reliable.", {})])
bullet(doc, [("Pitch variation fails, and fails in the wrong direction. ", {"bold": True}),
             (f"Asking for a lively delivery produces *less* pitch movement than asking for a "
              f"monotone one (ρ {PAR['var']['spearman_rho']:+.3f}, p "
              f"{PAR['var']['p']:.2f} — indistinguishable from noise).", {})])

callout(doc, "What is and is not established",
        ["One prosodic axis is solved by the base model, two are usable in aggregate, and "
         "one is not addressed at all.",
         "Whether the **inline non-verbal tags fire on Khmer** has not been measured. The "
         "inventory is documented for Chinese and English, and nothing establishes that a "
         "Khmer text context triggers them."])

# =========================================================================
heading(doc, "2.  Speech control", 1, page_break=True)

md(doc, "“Expressive control” is not one capability. It is a set of separable ones that "
        "differ in what they describe, how long they last, and — the part that decides "
        "engineering cost — whether they are properties of a whole utterance or events at a "
        "single position. The three that matter for this project are **prosody**, "
        "**emotion** and **non-verbal vocalization**.")

heading(doc, "2.1  Prosody", 3)

md(doc, "**What it is.** The suprasegmental properties of speech — everything carried "
        "*above* the individual sounds. Prosody is what remains when the words are stripped "
        "out: how fast, how high, how loud, how varied, and where the pauses fall.")

table(doc,
      ["Dimension", "Acoustic correlate", "Measured as"],
      [["Speaking rate", "phones or syllables per second", "characters per second"],
       ["Pitch register", "fundamental frequency, F0", "median F0 in Hz"],
       ["Pitch variation", "F0 range and contour movement", "F0 standard deviation, semitones"],
       ["Loudness / projection", "signal energy", "RMS in dBFS"],
       ["Phrasing", "pause placement and length", "inter-pausal unit statistics"]],
      widths=[1.5, 2.5, 2.5], size=8.8)
tcap(doc, "The prosodic dimensions, and the measurement each reduces to.")

md(doc, "**Examples.** *“I never said she stole my money”* read at three syllables per "
        "second versus six is a rate change. The same sentence at 120 Hz versus 240 Hz is a "
        "register change. Read flat it sounds robotic; read with a six-semitone F0 range it "
        "sounds engaged — that is variation. All three leave the words untouched.")

md(doc, "**Scope: global.** A prosodic setting is true of the whole utterance, or of a long "
        "span of it. It has no onset and no offset.")

md(doc, "**Status here.** Largely solved by the parenthetical prompt — §1.4.")

heading(doc, "2.2  Emotion", 3)

md(doc, "**What it is.** The affective state the delivery conveys: neutral, happy, angry, "
        "sad, surprised — the five categories the standard corpora use. Emotion is *realised "
        "through* prosody plus voice quality, but it does not reduce to a prosody setting: "
        "anger and excitement share high energy and high pitch and are not the same thing, "
        "and the difference lives in voice quality, articulatory precision and timing detail "
        "that a rate/pitch/energy vector does not capture.")

md(doc, "**Examples.** *“Oh, that's great.”* — flat and slow reads as sarcasm; fast, high "
        "and bright reads as delight; slow with creaky voice reads as resignation. Identical "
        "text, three affects.")

md(doc, "**Scope: global**, in the usual formulation. One label per utterance is the "
        "convention in essentially every emotional-speech corpus.")

md(doc, "**Status here.** Unmeasured. The parenthetical channel plausibly accepts *(sounding "
        "angry)* — it is the same free-text field that already carries *(speaking quickly)* — "
        "but nothing has been tested on Khmer.")

heading(doc, "2.3  Non-verbal vocalization", 3)

md(doc, "**What it is.** Sounds a speaker makes that are not words: laughter, sighs, "
        "breaths, filled pauses, throat-clearing, coughs, gasps, sobs, hesitation particles. "
        "They carry stance, turn-taking cues and affect, and they are a large part of what "
        "separates conversational speech from read speech.")

table(doc,
      ["Type", "Example tag", "What it signals"],
      [["Laughter", "`[laughing]`", "amusement, affiliation, softening"],
       ["Sigh", "`[sigh]`", "resignation, fatigue, relief"],
       ["Filled pause", "`[Uhm]`", "planning, hesitation, floor-holding"],
       ["Breath", "`[breath]`", "phrasing boundary, effort"],
       ["Discourse particle", "`[Question-ah]`, `[Dissatisfaction-hnn]`", "stance, back-channel"],
       ["Gasp, sob, cough, throat-clear", "—", "surprise, distress, physical state"]],
      widths=[1.7, 2.0, 2.8], size=8.8)
tcap(doc, "Non-verbal vocalization types. Tags in the first five rows are documented in "
          "stock VoxCPM2.")

md(doc, "**Examples.** *“I thought it was fine* `[laughing]` *but apparently not.”* The "
        "laugh is at one place, lasts a few hundred milliseconds, and everything before and "
        "after it is ordinary speech. Compare *“So we should* `[Uhm]` *probably wait”* — a "
        "filled pause inserted mid-clause.")

callout(doc, "Scope: local — and this is the structural difference",
        ["A non-verbal vocalization is a bounded event with an onset, a duration and an "
         "offset, and it lives at a specific token position.",
         "That makes it a *different engineering problem*. A global attribute has to compete "
         "for influence over every frame against an acoustic context that already predicts "
         "those frames. A local event owns the frames it occupies, and nothing else predicts "
         "them."])

md(doc, "**Status here.** VoxCPM2 ships the tag inventory. Whether the tags fire on Khmer "
        "text is unmeasured.")

heading(doc, "2.4  Adjacent categories", 3)

md(doc, "Three more exist and are worth naming so they are not confused with the above.")

table(doc,
      ["Category", "What it is", "Scope", "Example"],
      [["Voice quality", "phonation mode", "global", "whispered, breathy, creaky, tense"],
       ["Emphasis / focus", "which word carries contrastive stress", "local (a span)",
        "“**I** never said that” vs “I never said **that**”"],
       ["Timing", "pause insertion and length", "local", "a deliberate beat before a punchline"]],
      widths=[1.3, 2.2, 1.0, 2.0], size=8.8)
tcap(doc, "Categories adjacent to the three above.")

md(doc, "Voice quality behaves like emotion — global, prompt-addressable in principle. "
        "Emphasis and timing behave like non-verbal vocalization, in that they are local, but "
        "they need a *span* syntax rather than a point tag, which VoxCPM2 does not have.")

# =========================================================================
heading(doc, "3.  Literature review", 1, page_break=True)

md(doc, "Four bodies of work bear on this: natural-language prompt control of prosody and "
        "style, emotional speech synthesis, non-verbal vocalization, and the evaluation "
        "problem that runs under all of them.")

heading(doc, "3.1  Prosody and natural-language style control", 3)

md(doc, "The dominant idea of the last three years is to replace a reference clip with a "
        "*description*, and to obtain the descriptions by machine rather than by hand.")

table(doc,
      ["Work", "Contribution", "Why it matters here"],
      [["**PromptTTS** (Guo et al., ICASSP 2023) `arXiv:2211.12171`",
        "Style prompt plus content prompt, encoded separately and fed to an acoustic model.",
        "Established the format. Its dataset had to be built by hand, which capped it."],
       ["**InstructTTS** (Yang et al., 2023) `arXiv:2301.13662`",
        "Free-form instructions rather than attribute lists; discrete diffusion decoder, "
        "cross-modal representation aligning instruction text with speech style.",
        "Shows the channel tolerates arbitrary description, not a fixed vocabulary."],
       ["**PromptTTS 2** (Leng et al., ICLR 2024) `arXiv:2309.02285`",
        "A variation network predicting the reference-speech representation from the prompt, "
        "and an LLM pipeline that writes prompts from recognised attributes. 44k hours.",
        "Addresses the one-to-many problem — a description underdetermines the voice — and "
        "removes the labelling cost."],
       ["**Natural language guidance of high-fidelity TTS with synthetic annotations** "
        "(Lyth & King, 2024) `arXiv:2402.01912`",
        "Gender, accent, pitch, speaking rate and recording conditions labelled "
        "*computationally* across 45k hours of found data, then turned into descriptive "
        "sentences. Released openly as Parler-TTS.",
        "**The labels for prosodic control can be measured, not annotated.** That is exactly "
        "the property this project used when it built its Khmer corpus by measuring F0, "
        "character rate and RMS per clip."],
       ["**TextrolSpeech** (Ji et al., ICASSP 2024) `arXiv:2308.14430`",
        "236 hours, 33k utterances, style descriptions generated by an LLM pipeline over "
        "five attribute dimensions.",
        "The corpus side of the same idea."]],
      widths=[1.9, 2.6, 2.5], size=8.2)
tcap(doc, "Natural-language style control.")

md(doc, "This line explains why VoxCPM2's parenthetical prompt exists and why it works: it "
        "is the same conditioning format, trained the same way. It also sets the bar — these "
        "systems control pitch, rate and energy from description, which is precisely the set "
        "§1.4 finds already working on Khmer.")

heading(doc, "3.2  Emotion", 3)

table(doc,
      ["Work", "Contribution", "Why it matters here"],
      [["**Emotional voice conversion: Theory, databases and ESD** (Zhou, Sisman, Liu & Li, "
        "*Speech Communication* 2022) `arXiv:2105.14762`",
        "29+ hours, 350 parallel utterances from 10 English and 10 Mandarin speakers across "
        "five emotions, for multi-speaker and cross-lingual work; plus a survey of the field.",
        "The reference point for data — and the demonstration of how much acted, parallel, "
        "labelled recording an emotion result rests on."],
       ["**Laugh Now Cry Later** (Hsu et al., SLT 2024) `arXiv:2407.12229`",
        "Controls speaker emotion *and* laughter in one flow-matching zero-shot system, "
        "treating a laugh as a controllable event rather than an emotional label.",
        "The bridge to the next section: emotion and non-verbal vocalization are not "
        "separate problems in practice."]],
      widths=[2.1, 2.4, 2.5], size=8.2)
tcap(doc, "Emotional speech synthesis and conversion.")

md(doc, "The corpus shape is the obstacle. Every result here rests on parallel, acted, "
        "per-utterance-labelled emotional speech, and no such corpus exists for Khmer.")

heading(doc, "3.3  Non-verbal vocalization", 3)

md(doc, "This is the active area, and the one whose recipes transfer directly.")

table(doc,
      ["Work", "Contribution", "Why it matters here"],
      [["**NVSpeech** (2025) `arXiv:2508.04195`",
        "An integrated pipeline across recognition *and* synthesis: 48,430 human-annotated "
        "utterances over **18 word-level paralinguistic categories**; a paralinguistic-aware "
        "ASR emitting cues as **inline decodable tokens** (“You're so funny [Laughter]”), "
        "used to auto-label 573 hours / 174,179 utterances of Chinese with word-level "
        "alignment; then a zero-shot TTS fine-tuned on human- plus auto-labelled data for "
        "explicit control at arbitrary token positions.",
        "The closest match to what this project wants. Three ideas transfer: the tag is "
        "**inline, at the position of the event**; an ASR-shaped detector can bootstrap the "
        "corpus from found audio; and a small human-validated seed set is enough to bootstrap "
        "the automatic labeller."],
       ["**ELaTE** (Kanda et al., 2024) `arXiv:2402.07383`",
        "Fine-tunes a **conditional flow-matching** zero-shot TTS with frame-level "
        "conditioning from a laughter detector, controlling both when the laugh happens and "
        "how it sounds.",
        "The closest *method* reference — the same decoder family as VoxCPM2's Local DiT. Two "
        "results govern any plan: a small conditioned set suffices, and **mixing with general "
        "data preserves base-model quality**."],
       ["**NonverbalTTS** (2025, SSW) `arXiv:2507.13155`",
        "17 hours over 10 non-verbal types, built from open sources by automatic detection "
        "followed by human validation; reported at parity with proprietary systems.",
        "The template for corpus-building at a scale one person can reach."],
       ["**Robust laughter detection in noisy environments** (Gillick et al., Interspeech "
        "2021)",
        "Frame-level laughter detection and segmentation trained on found audio.",
        "The detector those pipelines depend on, already released."],
       ["**VocalSound** (Gong, Yu & Glass, ICASSP 2022) `arXiv:2205.03433`",
        "21k crowdsourced recordings of laughter, sighs, coughs, throat-clearing, sneezes and "
        "sniffs from 3,365 speakers, with a classifier baseline.",
        "Covers the rest of the inventory, so the automatic detection stage needs no model "
        "training on our side."],
       ["**MNV-17** (2025) `arXiv:2509.18196`",
        "A high-quality performative Mandarin non-verbal vocalization corpus for recognition.",
        "Evidence that deliberate recording is a viable route when found audio is thin."]],
      widths=[1.8, 2.8, 2.4], size=8.2)
tcap(doc, "Non-verbal vocalization: datasets, detectors and synthesis methods.")

md(doc, "Taken together this is a complete published recipe — detector → alignment → human "
        "validation → inline-tagged manifest → flow-matching fine-tune with mixed general "
        "data. Every stage has a reference implementation or a released model, and the target "
        "model class matches VoxCPM2's Local DiT.")

heading(doc, "3.4  Evaluating naturalness and non-verbal quality", 3)

md(doc, "Any control claim needs a measurement, and this project has already found that the "
        "standard naturalness predictors are unusable for Khmer: across 400 clips the rank "
        "correlation between UTMOS and Khmer character error rate is **ρ = +0.55** — the "
        "clips UTMOS likes best are the ones that get the Khmer most wrong. The evaluation "
        "question therefore has to be answered by instruments specific to the thing being "
        "controlled.")

table(doc,
      ["Work", "Contribution", "Why it matters here"],
      [["**NVV-SuperBench / NVBench** (2026) `arXiv:2604.16211`",
        "A unified **45-type taxonomy** of non-verbal vocalizations, a bilingual "
        "English/Chinese dataset, and a multi-axis protocol that **separates general speech "
        "naturalness from NVV-specific controllability, placement and salience**. 15 TTS "
        "systems evaluated.",
        "Those four axes are the right ones to report against: did the event appear, was it "
        "the right type, was it in the right place, did it sound like the thing."],
       ["**NVMOS** (2026) `arXiv:2606.15888`",
        "Predicts a MOS-like 0–5 perceptual quality score for a *specific marked non-verbal "
        "event*, taking audio plus text containing an explicit tag such as `[laugh]`. Reports "
        "that general-purpose audio LLMs disagree measurably with expert raters on this task.",
        "Supplies event quality as a model rather than a listening panel — and its input "
        "format, tagged text plus audio, is the format a training manifest is already in."]],
      widths=[1.9, 2.8, 2.3], size=8.2)
tcap(doc, "Evaluation instruments for non-verbal vocalization.")

md(doc, "Together these give an evaluation plan that does not depend on UTMOS: "
        "controllability and placement measured automatically with a detector, event quality "
        "with NVMOS, and intelligibility regression with the project's existing Khmer CTC "
        "scorer against the frozen evaluation set.")

heading(doc, "3.5  What we will focus on, and why", 3)

callout(doc, "Non-verbal vocalization",
        ["The prosodic layer is already available on this model, the emotional layer is "
         "blocked on data, and the non-verbal layer is the one where the cost-to-value ratio "
         "is best."],
        accent=TEAL)

md(doc, "**Prosody is already available.** The parenthetical prompt moves Khmer pitch across "
        f"+{PAR['pitch']['low_to_high']:.2f} Hz at ρ {PAR['pitch']['spearman_rho']:+.3f} — "
        f"roughly four times the +{CEIL['pitch']['spread']:.2f} Hz separation our own "
        "labelled corpus can express (§1.4). Building a prosodic controller would reproduce a "
        "capability the base model has, in a channel that is already occupied. The residue — "
        "pitch variation, reproducibility across seeds — is real but small.")

md(doc, "**Emotion is blocked on data, not method.** Every result in §3.2 depends on acted, "
        "parallel, per-utterance-labelled emotional speech. No Khmer corpus of that "
        "description exists, and recording one is out of proportion to the value.")

md(doc, "Non-verbal vocalization is the remaining layer, and four things recommend it.")

bullet(doc, [("It is local. ", {"bold": True}),
             ("A tagged event owns the frames it occupies. Global attributes have to compete "
              "for influence over frames the acoustic context already predicts; a laugh at "
              "position *k* has no competitor. This is a structural advantage, and it is why "
              "the conditioning difficulty that dominates global-attribute work is not "
              "expected to recur.", {})])
bullet(doc, [("The model already has the interface. ", {"bold": True}),
             ("`[laughing]`, `[sigh]` and `[Uhm]` are documented tags in stock VoxCPM2. The "
              "work is making them fire on Khmer, not inventing a syntax.", {})])
bullet(doc, [("The acoustics are substantially language-independent. ", {"bold": True}),
             ("A laugh is a laugh; a sigh is a sigh. What is language-specific is *where* "
              "they go and what they mean in context — which is what the tag position "
              "supplies. This is why a small Khmer corpus can plausibly be enough, and it is "
              "the assumption the methodology must test first.", {})])
bullet(doc, [("Every stage has a published reference. ", {"bold": True}),
             ("NVSpeech for the pipeline and the inline-token format, ELaTE for the "
              "flow-matching fine-tune and the data-mixing ratio, NonverbalTTS for the "
              "human-validation loop, Gillick and VocalSound for detection, NVV-SuperBench "
              "and NVMOS for evaluation.", {})])

# =========================================================================
heading(doc, "4.  Methodology", 1, page_break=True)

md(doc, "*To be written.*", size=10.5)

# =========================================================================
heading(doc, "References", 1, page_break=True)


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


refs("Prosody and natural-language style control", [
    "Guo, Z. et al. (2023). *PromptTTS: Controllable text-to-speech with text descriptions.* "
    "ICASSP. arXiv:2211.12171",
    "Yang, D. et al. (2023). *InstructTTS: Modelling expressive TTS in discrete latent space "
    "with natural language style prompt.* arXiv:2301.13662",
    "Leng, Y. et al. (2024). *PromptTTS 2: Describing and generating voices with text "
    "prompt.* ICLR. arXiv:2309.02285",
    "Lyth, D. & King, S. (2024). *Natural language guidance of high-fidelity text-to-speech "
    "with synthetic annotations.* arXiv:2402.01912 — released as Parler-TTS.",
    "Ji, S. et al. (2024). *TextrolSpeech: A text style control speech corpus with codec "
    "language text-to-speech models.* ICASSP. arXiv:2308.14430",
])

refs("Emotion", [
    "Zhou, K., Sisman, B., Liu, R. & Li, H. (2022). *Emotional voice conversion: Theory, "
    "databases and ESD.* Speech Communication. arXiv:2105.14762",
    "Hsu, C.-C. et al. (2024). *Laugh Now Cry Later: Controlling time-varying emotional "
    "states of flow-matching-based zero-shot text-to-speech.* SLT. arXiv:2407.12229",
])

refs("Non-verbal vocalization", [
    "*NVSpeech: An integrated and scalable pipeline for human-like speech modeling with "
    "paralinguistic vocalizations.* (2025). arXiv:2508.04195",
    "Kanda, N. et al. (2024). *ELaTE: Making flow-matching-based zero-shot text-to-speech "
    "laugh as you like.* arXiv:2402.07383",
    "*NonverbalTTS: A public English corpus of text-aligned nonverbal vocalizations with "
    "emotion annotations for text-to-speech.* (2025). SSW. arXiv:2507.13155",
    "Gillick, J. et al. (2021). *Robust laughter detection in noisy environments.* "
    "Interspeech. ISCA archive.",
    "Gong, Y., Yu, J. & Glass, J. (2022). *VocalSound: A dataset for improving human vocal "
    "sounds recognition.* ICASSP. arXiv:2205.03433",
    "*MNV-17: A high-quality performative Mandarin dataset for nonverbal vocalization "
    "recognition in speech.* (2025). arXiv:2509.18196",
])

refs("Evaluation", [
    "*NVV-SuperBench: Beyond words, beyond quality — benchmarking nonverbal vocalizations in "
    "speech generation.* (2026). arXiv:2604.16211",
    "*NVMOS: Non-verbal vocalization quality assessment in speech.* (2026). arXiv:2606.15888",
])

refs("VoxCPM2 sources and companion documents", [
    "OpenBMB. *VoxCPM2 model card*, huggingface.co/openbmb/VoxCPM2 — parenthetical voice "
    "design, style-guided cloning, and the non-verbal tag inventory.",
    "Source: voxcpm/training/packers.py (the zero text loss mask) and "
    "voxcpm/model/voxcpm2.py.",
    "This project: docs/09 — VoxCPM2 architecture and training; docs/03 — evaluation and "
    "benchmarking; docs/11 — the markdown companion to this report; finetune/README.md — the "
    "runbook; finetune/results/parenthetical/ — the measurement in section 1.4.",
])

doc.save(OUT)
print(f"wrote {OUT}")
