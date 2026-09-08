#!/usr/bin/env python
"""
Build the technical report on speech control in VoxCPM2.

An unbranded academic report: title page, contents with resolved page numbers,
four numbered sections and a bibliography. Layout primitives are in
src/docbuild/academic_docx.py.

Measurement figures are read from finetune/results/parenthetical/*.json rather
than retyped, so the document cannot drift from the experiment it reports.

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

HERE = Path(__file__).resolve().parent      # src/docbuild
ROOT = HERE.parents[1]                      # repository root
OUT = ROOT / "reports" / "Speech-Control-VoxCPM2.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)

_PARJ = json.loads((ROOT / "finetune" / "results" / "parenthetical" /
                    "parenthetical.json").read_text())
PAR, CEIL, PROMPTS = _PARJ["summary"], _PARJ["corpus_ceilings"], _PARJ["prompts"]
N_GEN = len(_PARJ["rows"])
N_SENT, N_SEED = _PARJ["n_sentences"], _PARJ["repeats"]

SECTIONS = [
    (1, "1", "Overview"),
    (2, "1.1", "Model Description"),
    (2, "1.2", "System Architecture"),
    (2, "1.3", "Built-in Control Mechanisms"),
    (2, "1.4", "Measurement of Parenthetical Prosodic Control on Khmer"),
    (1, "2", "Speech Control"),
    (2, "2.1", "Definition and Scope"),
    (2, "2.2", "Prosody"),
    (2, "2.3", "Emotion"),
    (2, "2.4", "Non-Verbal Vocalization"),
    (2, "2.5", "Related Categories"),
    (1, "3", "Literature Review"),
    (2, "3.1", "Natural Language Style Control"),
    (2, "3.2", "Emotional Speech Synthesis"),
    (2, "3.3", "Non-Verbal Vocalization"),
    (2, "3.4", "Evaluation of Controlled Speech"),
    (2, "3.5", "Research Focus"),
    (1, "4", "Methodology"),
    (1, "", "References"),
]

LOSS_MASK = """# voxcpm/training/packers.py, process_tts_data
loss_mask = cat([zeros(text_length), ones(audio_length), zeros(1)])
#                ^^^^^^^^^^^^^^^^^^ zero at every text position"""

PAREN_CODE = '''model.generate(
    text="(speaking quickly, a high-pitched voice)"
         "ថ្ងៃនេះអាកាសធាតុល្អណាស់។")'''

TAG_CODE = '''model.generate(text="ខ្ញុំគិតថាមិនអីទេ [laughing] ប៉ុន្តែ…")'''

TAG_INVENTORY = """[laughing]   [laughter]   [sigh]   [Uhm]   [Shh]
[Question-ah] [Question-ei] [Question-en] [Question-oh]
[Surprise-wa] [Surprise-yo] [Dissatisfaction-hnn]"""


TABLES = ["arch", "params", "prompts", "paren", "prosody", "nvv", "adjacent", "nvv_refs"]


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


# =========================================================================
def build(pages):
    """Render the document. `pages` maps '1.2' -> page number, or is empty on
    the first pass, in which case the contents shows a placeholder."""
    A._TABLE_N[0] = 0
    A._FIGURE_N[0] = 0
    doc = new_document()

    title_page(
        doc,
        title="Finetuning VoxCPM\n"
              "for Speech Control: Integrating Non-Verbal Vocalization",
        prepared_by="R&D Department",
        date="8 September 2026")

    contents(doc, [(lvl, num, txt, pages.get(num or txt, "—"))
                   for lvl, num, txt in SECTIONS])

    # -- 1 ----------------------------------------------------------------
    heading(doc, "1   Overview", 1, page_break=True)

    para(doc, "This report concerns the control of delivery in synthetic speech: the ability "
              "to specify how an utterance is spoken rather than only what is said. It has "
              "three purposes. Section 1 describes VoxCPM2, the model this project has "
              "adopted for Khmer, and reports a measurement of the control it already "
              "provides. Section 2 defines the categories of speech control and distinguishes "
              "them from one another. Section 3 reviews the published work addressing each "
              "category and identifies the one this project will pursue. Section 4, the "
              "methodology, is reserved.")

    heading(doc, "1.1   Model Description", 2)

    para(doc, "VoxCPM2 is an open-weight text-to-speech model released by OpenBMB under the "
              "Apache 2.0 licence. It has 2.29 billion parameters and is tokenizer-free: "
              "rather than mapping speech onto a discrete codebook, it predicts continuous "
              "latent vectors, one for every four-frame patch of audio.")

    para(doc, "This property has a direct bearing on low-resource languages. Systems built on "
              "discrete audio codebooks depend on the codebook having been fitted to the "
              "target language during pre-training, and where it has not, synthesis degrades "
              "in a way that fine-tuning does not readily repair. In the four-model "
              "comparison conducted for this project, Fish Audio S2-Pro failed in exactly "
              "that manner, returning a median character error rate of 78.01 per cent on "
              "Khmer. Over the same fixed set of one hundred sentences VoxCPM2 returned 2.47 "
              "per cent, against 8.28 per cent for Higgs TTS 3 and 25.12 per cent for Meta "
              "MMS. VoxCPM2 was selected on that evidence.")

    para(doc, "A second property follows from the input side. The model reads raw UTF-8 bytes "
              "through a 73,448-entry tokenizer and is given no language identifier, "
              "inferring the language from the script. Khmer consequently requires no "
              "grapheme-to-phoneme conversion, no pronunciation lexicon and no word "
              "segmenter. None of these components exists for Khmer in a form suitable for "
              "production use, and assembling them ordinarily accounts for the largest share "
              "of effort in a low-resource text-to-speech project.")

    heading(doc, "1.2   System Architecture", 2)

    figure(doc, ROOT / "reports" / "assets" / "voxcpm-architecture.png",
           "Overview of VoxCPM's Architecture")

    para(doc, f"The generation path comprises the stages set out in {T('arch')}. A backbone language "
              "model emits one latent vector per audio patch; a residual language model "
              "refines it; a local diffusion transformer converts the refined latent into "
              "acoustic features under a conditional flow-matching objective; and a "
              "variational autoencoder decodes those features to a waveform, accepting "
              "features derived at 16 kHz and emitting audio at 48 kHz. A separate local "
              "encoder, not shown, maps a reference recording into the same latent space and "
              "is the mechanism by which zero-shot voice cloning is performed.")

    tbl(doc, "arch",
          "The VoxCPM2 generation path. Dimensions and layer counts are taken from "
          "`config.json` in the `openbmb/VoxCPM2` release.",
          ["Stage", "Configuration", "Output"],
          [["Byte-level tokenizer", "vocabulary 73,448; no grapheme-to-phoneme stage, no "
            "language identifier", "token sequence"],
           ["MiniCPM4 backbone LM", "2048 dimensions, 28 layers; grouped-query attention, "
            "16 query and 2 key-value heads; LongRoPE to 32k", "one latent vector per audio patch"],
           ["Residual LM", "8 layers, no positional encoding", "refined latent"],
           ["Local DiT", "1024 dimensions, 12 layers; conditional flow matching; Euler "
            "solver, guidance scale 2.0, 10 steps at inference",
            "64-dimensional acoustic features, 4 frames per patch"],
           ["AudioVAE V2", "encoder at 16 kHz, decoder at 48 kHz", "waveform"]],
          widths=[1.25, 3.15, 1.7])

    tbl(doc, "params",
          "Architecture parameters bearing on training and control.",
          ["Parameter", "Value", "Consequence"],
          [["`patch_size`", "4", "One language-model step spans four autoencoder frames."],
           ["`feat_dim`", "64", "Width of the latent the diffusion transformer predicts."],
           ["Frame rate", "25 fps", "One second of audio occupies 6.25 language-model positions."],
           ["Encoder rate", "16 kHz", "Training audio must be supplied at 16 kHz; the validator rejects other rates."],
           ["Decoder rate", "48 kHz", "Bandwidth extension is internal; 48 kHz training data is not required."],
           ["`inference_cfg_rate`", "2.0", "Default classifier-free guidance scale, exposed as `--cfg-value`."]],
          widths=[1.35, 0.75, 4.3])

    para(doc, "Two of these properties bear on the remainder of the report.")

    para(doc, "The first is the identity of the acoustic decoder. The local diffusion "
              "transformer is a conditional flow-matching model. The methods reviewed in "
              "Section 3.3 for controlling non-verbal vocalization were developed for, and "
              "evaluated on, models of this class. They are therefore applicable to VoxCPM2 "
              "without alteration of the underlying training objective, which is not true of "
              "methods developed for discrete-codec systems.")

    para(doc, "The second concerns the treatment of the text field during training. The loss "
              "mask constructed in the data packer is zero at every text position:")

    code(doc, LOSS_MASK)

    para(doc, "No token in the text field is ever a prediction target. The field operates "
              "purely as a conditioning channel, and its contents may therefore be extended "
              "with tags, markers or descriptive text without perturbing the objective the "
              "model is trained under. Both control mechanisms described below exploit this, "
              "and any mechanism added later would do the same.")

    heading(doc, "1.3   Built-in Control Mechanisms", 2)

    para(doc, "VoxCPM2 provides two mechanisms for influencing delivery. They differ in the "
              "interval over which they apply, and that difference is developed in Section 2.")

    para(doc, "The first is a parenthetical description placed before the text, which "
              "characterises the utterance as a whole:")
    code(doc, PAREN_CODE)

    para(doc, "The second is a bracketed tag placed inline, which marks a single event at one "
              "position in the text:")
    code(doc, TAG_CODE)

    para(doc, "The documented tag inventory is:")
    code(doc, TAG_INVENTORY)

    para(doc, "The model documentation describes both mechanisms with reference to Chinese "
              "and English. It makes no statement about their behaviour in other languages, "
              "and the training corpus composition is not published in sufficient detail to "
              "infer one. Their efficacy on Khmer is therefore an empirical question, which "
              "the next section addresses for the first mechanism.")

    heading(doc, "1.4   Measurement of Parenthetical Prosodic Control on Khmer", 2)

    para(doc, f"The parenthetical mechanism was evaluated on the unmodified model, using "
              f"{word(N_SENT)} sentences drawn from the project's fixed Khmer evaluation "
              f"set. For each of four prosodic axes, three prompts were written to span the "
              f"axis ({T('prompts')}). Every sentence was synthesised under every prompt at "
              f"{word(N_SEED)} random seeds, giving {N_GEN} generations, and the median taken "
              f"within each cell of the design. Acoustic measurement follows the definitions "
              f"used elsewhere in the project: median fundamental frequency for pitch, its "
              f"standard deviation in semitones for variation, root-mean-square level for "
              f"energy, and Khmer characters per second for rate.")

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
              f"approximately four. The capability a fine-tune on the project's own labelled "
              f"data could add to this axis is therefore negative.")

    para(doc, f"**Energy** and **Speaking Rate** are controlled in aggregate but not per generation. "
              f"Both show significant rank correlations "
              f"({PAR['energy']['spearman_rho']:+.3f} and "
              f"{PAR['rate']['spearman_rho']:+.3f}), but the energy effect of "
              f"{PAR['energy']['low_to_high']:.2f} dB is smaller than the "
              f"{PAR['energy']['mean_run_to_run_sd']:.2f} dB standard deviation observed "
              f"across seeds within a single cell. The ordering of the levels is dependable; "
              f"the value of any individual synthesis is not.")

    para(doc, f"**Pitch Variation** is not controlled. The correlation is negative "
              f"({PAR['var']['spearman_rho']:+.3f}) and not significant "
              f"(p = {PAR['var']['p']:.2f}). Prompts requesting a lively delivery produced "
              f"marginally less pitch movement than prompts requesting a monotone one, which "
              f"is consistent with the axis being unaddressed rather than inverted.")

    para(doc, "The behaviour of the inline tags on Khmer was not measured, and no claim is "
              "made about it here. Establishing whether they fire at all on Khmer text is a "
              "prerequisite for the work Section 3.5 selects.")

    # -- 2 ----------------------------------------------------------------
    heading(doc, "2   Speech Control", 1, page_break=True)

    heading(doc, "2.1   Definition and Scope", 2)

    para(doc, "The term expressive control is used loosely in the literature to denote any "
              "influence over delivery not exercised through the choice of words. It "
              "subsumes several capabilities that differ in what they describe, over what "
              "interval they apply, and in the effort each requires to implement. This "
              "section separates the three that concern this project, and names three "
              "further categories that are adjacent to them.")

    para(doc, "One distinction cuts across all of them and is used throughout what follows. "
              "A global attribute is a property of a whole utterance, or of a long span of "
              "one: it has no onset and no offset, and it is realised in every frame. A "
              "local event is bounded: it begins, occupies a short interval, and ends, and "
              "it is associated with a specific position in the text. The two are not merely "
              "different in duration. A global attribute must influence frames whose "
              "acoustic content is already largely determined by the surrounding speech, "
              "whereas a local event is the only thing that determines the frames it "
              "occupies. The consequences for training are taken up in Section 3.5.")

    heading(doc, "2.2   Prosody", 2)

    para(doc, "Prosody comprises the suprasegmental properties of speech, that is, those "
              "carried above the level of the individual sound. It is what remains when the "
              "identity of the words is set aside: the rate at which they are spoken, the "
              "pitch at which they are set, the loudness with which they are projected, the "
              "degree of pitch movement, and the placement of pauses.")

    tbl(doc, "prosody",
          "Prosodic dimensions and their acoustic correlates.",
          ["Dimension", "Acoustic correlate", "Measurement used here"],
          [["Speaking rate", "phones or syllables per unit time", "Khmer characters per second"],
           ["Pitch register", "fundamental frequency", "median F0, hertz"],
           ["Pitch variation", "F0 range and contour movement", "F0 standard deviation, semitones"],
           ["Loudness", "signal energy", "root-mean-square level, dBFS"],
           ["Phrasing", "pause placement and duration", "inter-pausal unit statistics"]],
          widths=[1.4, 2.4, 2.6])

    para(doc, "The sentence *I never said she stole my money* spoken at three syllables per "
              "second and at six differs in rate alone. Spoken with a median fundamental "
              "frequency of 120 hertz and of 240 hertz, it differs in register. Spoken with "
              "an F0 standard deviation near zero it is heard as mechanical, and with a "
              "range of six semitones as engaged; that is variation. In each case the words, "
              "and the meaning they carry, are unchanged.")

    para(doc, "Prosody is a global attribute. It is the category most extensively treated in "
              "the literature, and, as Section 1.4 established, the one VoxCPM2 already "
              "addresses on Khmer.")

    heading(doc, "2.3   Emotion", 2)

    para(doc, "Emotion denotes the affective state conveyed by the delivery. The standard "
              "corpora encode it as a small closed set of categories, most commonly neutral, "
              "happy, angry, sad and surprised.")

    para(doc, "Emotion is realised through prosody together with voice quality, but it does "
              "not reduce to a prosodic setting. Anger and excitement share elevated pitch "
              "and elevated energy and are not perceptually similar; the distinction resides "
              "in phonation, articulatory precision and fine timing, none of which a "
              "rate-pitch-energy specification captures. The sentence *Oh, that's great* "
              "spoken flatly and slowly is heard as sarcastic, spoken quickly and brightly "
              "as pleased, and spoken slowly with creaky phonation as resigned. The three "
              "readings differ in affect, not in prosodic setting alone.")

    para(doc, "Emotion is treated as a global attribute in essentially all of the literature: "
              "one label per utterance is the near-universal convention. Its status in "
              "VoxCPM2 on Khmer is unmeasured. The parenthetical channel would plausibly "
              "accept a description such as *(sounding angry)*, since it is the same "
              "free-text field that already carries *(speaking quickly)*, but this has not "
              "been tested.")

    heading(doc, "2.4   Non-Verbal Vocalization", 2)

    para(doc, "A non-verbal vocalization is a sound produced by a speaker that is not a word: "
              "laughter, a sigh, an audible breath, a filled pause, a cough, a gasp, a sob, "
              "a hesitation particle. Such sounds carry stance, regulate turn-taking and "
              "convey affect, and their presence is a substantial part of what distinguishes "
              "conversational speech from read speech.")

    tbl(doc, "nvv",
          "Non-verbal vocalization types. Tags in the first five rows are documented in "
          "VoxCPM2; the sixth row is not.",
          ["Type", "Tag", "Communicative function"],
          [["Laughter", "`[laughing]`", "amusement, affiliation, mitigation"],
           ["Sigh", "`[sigh]`", "resignation, fatigue, relief"],
           ["Filled pause", "`[Uhm]`", "planning, hesitation, floor-holding"],
           ["Attention marker", "`[Shh]`", "silencing, conspiratorial framing"],
           ["Discourse particle", "`[Question-ah]`, `[Surprise-wa]`, `[Dissatisfaction-hnn]`",
            "stance, back-channelling, question marking"],
           ["Breath, gasp, cough, sob", "—", "phrasing, surprise, distress, physical state"]],
          widths=[1.3, 2.5, 2.6])

    para(doc, "In the utterance *I thought it was fine* `[laughing]` *but apparently not*, "
              "the laugh occurs at one place, occupies a few hundred milliseconds, and "
              "leaves the speech before and after it unaffected. In *So we should* `[Uhm]` "
              "*probably wait*, a filled pause is inserted mid-clause and has the same "
              "bounded character.")

    para(doc, "Non-verbal vocalization is therefore a local event, and this is its "
              "significant property for present purposes. A global attribute competes for "
              "influence over frames that the surrounding acoustic context already predicts. "
              "A tagged event has no such competitor: the frames it occupies are predicted "
              "by nothing else, and the conditioning signal is the only available "
              "explanation for them. The engineering problem is correspondingly different, "
              "and, as Section 3.5 argues, easier.")

    para(doc, "VoxCPM2 supplies the tag inventory. Whether the tags are realised on Khmer "
              "text is unmeasured.")

    heading(doc, "2.5   Related Categories", 2)

    para(doc, "Three further categories are named here so that they are not conflated with "
              "the preceding three.")

    tbl(doc, "adjacent",
          "Categories adjacent to prosody, emotion and non-verbal vocalization.",
          ["Category", "Definition", "Scope", "Example"],
          [["Voice quality", "mode of phonation", "global",
            "whispered, breathy, creaky, tense"],
           ["Emphasis", "placement of contrastive stress", "local span",
            "*I* never said that; I never said *that*"],
           ["Timing", "insertion and duration of pauses", "local",
            "a deliberate pause before a resolution"]],
          widths=[1.15, 2.15, 0.85, 2.25])

    para(doc, "Voice quality behaves as emotion does: it is global, and in principle "
              "addressable through the same descriptive channel. Emphasis and timing are "
              "local, as non-verbal vocalization is, but they require a syntax that "
              "delimits a span rather than marking a point. VoxCPM2 provides no such syntax, "
              "and adding one is a larger undertaking than adding a tag.")

    # -- 3 ----------------------------------------------------------------
    heading(doc, "3   Literature Review", 1, page_break=True)

    para(doc, "Four bodies of work bear on the categories set out above: natural language "
              "control of prosody and style, emotional speech synthesis, non-verbal "
              "vocalization, and the evaluation of controlled speech. Each is reviewed in "
              "turn, and Section 3.5 states which category this project will pursue and on "
              "what grounds.")

    heading(doc, "3.1   Natural Language Style Control", 2)

    para(doc, "The organising idea of this literature is to replace the reference recording "
              "with a description, and to obtain the descriptions by automatic means rather "
              "than by annotation.")

    para(doc, "Guo et al. (2023) established the format with PromptTTS, which takes a style "
              "prompt and a content prompt, encodes them separately, and conditions an "
              "acoustic model on both. The approach was constrained by its data: the "
              "style-annotated corpus had to be constructed manually, which limited its "
              "scale. Yang et al. (2023) relaxed the input format in InstructTTS, accepting "
              "free-form instructions rather than attribute lists and learning a cross-modal "
              "representation that aligns instruction text with speech style, decoded in a "
              "discrete latent space.")

    para(doc, "Leng et al. (2024) addressed the two limitations that constrain the family as "
              "a whole. The first is that a description underdetermines a voice: many "
              "distinct voices satisfy *a young woman speaking quickly*, and a model trained "
              "to map descriptions to speech must resolve that ambiguity somehow. PromptTTS "
              "2 introduces a variation network that predicts, from the prompt "
              "representation, the reference-speech representation that would otherwise have "
              "been supplied. The second is annotation cost, addressed by a pipeline in "
              "which a speech understanding model recognises attributes and a large language "
              "model writes the corresponding prompt sentence. The system was trained on 44,000 "
              "hours.")

    para(doc, "Lyth and King (2024) give the clearest statement of the annotation argument "
              "and the one released without restriction, as Parler-TTS. Gender, accent, "
              "pitch, speaking rate and recording conditions are labelled computationally "
              "across a 45,000-hour corpus of found data using classifiers and signal "
              "measurements, and the resulting attributes are rendered as descriptive "
              "sentences on which the model is conditioned. The system outperforms prior work "
              "on fidelity while using no manually annotated data. The transferable result "
              "is that the labels required for prosodic control can be measured rather than "
              "annotated, which is the property this project relied upon when it labelled a "
              "Khmer corpus by measuring fundamental frequency, character rate and "
              "root-mean-square level per clip. Ji et al. (2024) supply the corpus "
              "counterpart in TextrolSpeech: 236 hours and 33,000 utterances with style "
              "descriptions generated by a language-model pipeline over five attribute "
              "dimensions.")

    para(doc, "This literature accounts for the parenthetical mechanism in VoxCPM2, which "
              "uses the same conditioning format and was presumably trained in the same "
              "manner. It also sets the expectation against which Section 1.4 should be "
              "read: these systems control pitch, rate and energy from description, and "
              "those are precisely the axes the measurement finds already functioning on "
              "Khmer.")

    heading(doc, "3.2   Emotional Speech Synthesis", 2)

    para(doc, "Work on emotion is organised around acted, parallel corpora. Zhou et al. "
              "(2022) survey the field and introduce the Emotional Speech Dataset, which "
              "remains the standard reference: 350 parallel utterances from ten English and "
              "ten Mandarin speakers across five emotion categories, exceeding 29 hours "
              "recorded under controlled acoustic conditions, and designed to support "
              "multi-speaker and cross-lingual conversion.")

    para(doc, "Hsu et al. (2024) are the bridge between this section and the next. Their "
              "system controls speaker emotion and laughter within a single flow-matching "
              "zero-shot model, and in doing so treats a laugh as a controllable event "
              "rather than as an emotional label. The separation between emotion and "
              "non-verbal vocalization, clear enough as a definition, is not clean in "
              "practice.")

    para(doc, "The obstacle in this category is the shape of the data rather than the "
              "adequacy of the method. Every result rests on parallel, acted, "
              "utterance-labelled emotional speech. No Khmer corpus of that description "
              "exists, and commissioning one is not proportionate to the value it would "
              "return.")

    heading(doc, "3.3   Non-Verbal Vocalization", 2)

    para(doc, "This is the most active of the four areas and the one whose methods transfer "
              "most directly to VoxCPM2.")

    para(doc, "NVSpeech (2025) is the closest match to the problem stated in Section 2.4. It "
              "treats recognition and synthesis as one pipeline. A manually annotated set of "
              "48,430 utterances covering eighteen word-level paralinguistic categories is "
              "used to train a paralinguistic-aware speech recogniser that emits the cues as "
              "inline decodable tokens, so that a transcript reads *You're so funny "
              "[Laughter]*. That recogniser then labels a corpus of 174,179 Chinese "
              "utterances, 573 hours, with word-level alignment. A zero-shot synthesiser is "
              "finally fine-tuned on the combined human- and machine-labelled data, yielding "
              "explicit control over vocalizations inserted at arbitrary token positions. "
              "Three elements transfer: the tag is placed inline at the position of the "
              "event rather than in a header; a recogniser-shaped detector can bootstrap a "
              "corpus from found audio; and a modest human-validated seed set suffices to "
              "bootstrap the automatic labeller.")

    para(doc, "Kanda et al. (2024) provide the closest methodological reference. ELaTE "
              "fine-tunes a conditional flow-matching zero-shot synthesiser using "
              "frame-level conditioning derived from a laughter detector, obtaining control "
              "over both the timing of a laugh and its acoustic character. Two of its "
              "results constrain any plan built on it. A comparatively small conditioned "
              "dataset is sufficient; and mixing the conditioned data with general training "
              "data preserves the quality of the base model, so the fine-tune need not be "
              "paid for in intelligibility. The relevance is direct: the decoder ELaTE "
              "modifies belongs to the same class as the VoxCPM2 local diffusion "
              "transformer.")

    para(doc, "NonverbalTTS (2025) is the reference for corpus construction at a tractable "
              "scale. Seventeen hours covering ten non-verbal types were assembled from open "
              "sources by automatic detection followed by human validation, and the "
              "resulting system is reported at parity with proprietary alternatives. The "
              "detectors on which such pipelines depend are themselves available: Gillick et "
              "al. (2021) release robust frame-level laughter detection and segmentation "
              "trained on found audio, and Gong et al. (2022) release VocalSound, 21,000 "
              "crowdsourced recordings of laughter, sighs, coughs, throat-clearing, sneezes "
              "and sniffs from 3,365 speakers, together with a classifier baseline. Between "
              "them the automatic detection stage requires no model training. Where found "
              "audio is too thin to support detection, deliberate recording remains viable; "
              "MNV-17 (2025) demonstrates this for Mandarin.")

    tbl(doc, "nvv_refs",
          "Principal references for non-verbal vocalization, by pipeline stage.",
          ["Stage", "Reference", "Contribution"],
          [["Detection", "Gillick et al. (2021); Gong et al. (2022)",
            "Released frame-level laughter detector; released classifier over six vocalization types."],
           ["Corpus construction", "NVSpeech (2025); NonverbalTTS (2025)",
            "Recogniser-driven auto-labelling at 573 hours; detection plus human validation at 17 hours."],
           ["Synthesis", "Kanda et al. (2024); Hsu et al. (2024)",
            "Flow-matching fine-tuning with detector conditioning; joint emotion and laughter control."],
           ["Deliberate recording", "MNV-17 (2025)",
            "Performative corpus where found audio is insufficient."]],
          widths=[1.15, 1.95, 3.3])

    para(doc, "Taken together these constitute a complete and published procedure: detection, "
              "alignment, human validation, an inline-tagged manifest, and a flow-matching "
              "fine-tune with general data mixed in. Every stage has either a reference "
              "implementation or a released model.")

    heading(doc, "3.4   Evaluation of Controlled Speech", 2)

    para(doc, "A control claim requires an instrument, and this project has already "
              "established that the usual instruments are unsuitable for Khmer. Across 400 "
              "clips the rank correlation between UTMOS, a learned mean-opinion-score "
              "predictor, and Khmer character error rate is +0.55: the recordings the "
              "predictor scores highest are those that render the Khmer least correctly. The "
              "inversion is between models rather than within any one model's output, which "
              "is precisely the comparison the predictor was being used to make. Naturalness "
              "predictors trained without Khmer in view cannot be relied upon here.")

    para(doc, "Two recent contributions address the evaluation of non-verbal vocalization "
              "specifically. NVV-SuperBench (2026) pairs a unified taxonomy of 45 "
              "vocalization types with a bilingual English and Chinese dataset and, more "
              "usefully, defines a protocol that separates general speech naturalness from "
              "vocalization-specific controllability, placement and salience; fifteen "
              "systems are evaluated under it. Those four axes are the appropriate ones to "
              "report against, since they decompose the question into whether the event "
              "occurred, whether it was of the requested type, whether it occurred in the "
              "requested place, and whether it was acoustically convincing.")

    para(doc, "NVMOS (2026) supplies the last of these as a model rather than as a listening "
              "panel. It predicts a mean-opinion-score-like value between zero and five for "
              "a specific marked non-verbal event, taking as input the audio together with "
              "text containing an explicit tag such as `[laugh]`. The authors additionally "
              "report that general-purpose audio-capable multimodal models disagree "
              "measurably with expert raters on this task, so a multimodal model is not an "
              "acceptable substitute. Its input format, tagged text paired with audio, is the "
              "format in which a training manifest for this work would already exist.")

    para(doc, "An evaluation plan that avoids the inverted predictors therefore exists: "
              "controllability and placement measured automatically with a detector, "
              "acoustic quality of the event measured with NVMOS, and intelligibility "
              "regression measured with the project's existing Khmer connectionist temporal "
              "classification scorer against the frozen evaluation set.")

    heading(doc, "3.5   Research Focus", 2)

    para(doc, "This project will pursue non-verbal vocalization. The reasoning proceeds by "
              "elimination and is then stated positively.")

    para(doc, f"Prosody is already provided. The parenthetical mechanism moves Khmer pitch "
              f"across {PAR['pitch']['low_to_high']:.2f} hertz at a rank correlation of "
              f"{PAR['pitch']['spearman_rho']:+.3f}, roughly four times the "
              f"{CEIL['pitch']['spread']:.2f} hertz separation the project's own labelled "
              f"corpus is able to express (Section 1.4). A prosodic controller trained on "
              f"that corpus would reproduce a capability the base model possesses, in a "
              f"conditioning channel that is already occupied. The residue, principally "
              f"pitch variation and reproducibility across random seeds, is genuine but "
              f"small.")

    para(doc, "Emotion is blocked on data rather than on method. The results reviewed in "
              "Section 3.2 depend without exception on acted, parallel, utterance-labelled "
              "emotional speech, and no Khmer corpus of that description exists.")

    para(doc, "Non-verbal vocalization is the remaining category, and four considerations "
              "recommend it.")

    numbered(doc, [
        "It is a local event. A tagged vocalization is the only predictor of the frames it "
        "occupies, whereas a global attribute must compete for influence over frames the "
        "surrounding context already determines. The conditioning difficulty that attends "
        "global-attribute training is therefore not expected to arise.",
        "The interface exists. The tags `[laughing]`, `[sigh]` and `[Uhm]` are documented in "
        "the unmodified model. The work is to make them operate on Khmer, not to design a "
        "syntax and persuade the model to read it.",
        "The acoustics are substantially language-independent. Laughter and sighing are not "
        "language-specific gestures; what is language-specific is where they are placed and "
        "what they signal in context, and placement is supplied by the tag position. This is "
        "why a small Khmer corpus may be sufficient, and it is the assumption that any "
        "methodology must test before committing resources.",
        "Every stage has a published reference: NVSpeech for the pipeline and the "
        "inline-token format, ELaTE for the flow-matching fine-tune and the data-mixing "
        "ratio, NonverbalTTS for the human-validation loop, Gillick et al. and VocalSound "
        "for detection, and NVV-SuperBench and NVMOS for evaluation.",
    ])

    # -- 4 ----------------------------------------------------------------
    heading(doc, "4   Methodology", 1, page_break=True)
    para(doc, "Reserved.", align=None)

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

        "OpenBMB (2025). VoxCPM2 model card. huggingface.co/openbmb/VoxCPM2. Source "
        "referenced in Section 1.2: `voxcpm/training/packers.py` and "
        "`voxcpm/model/voxcpm2.py`.",

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
        r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                            "--outdir", td, str(OUT)],
                           capture_output=True, text=True, timeout=300)
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
