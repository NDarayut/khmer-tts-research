#!/usr/bin/env python
"""
Build the Smean AI literature review on VoxCPM2 and Higgs TTS 3 as a .docx.

Content is sourced from docs/02, docs/03, docs/06, docs/09 and docs/10 in this
repo, plus the in-house evaluation run in evaluation/results_report.md.

Brand tokens (colors, logo, typography) are taken from https://www.smean.ai/ --
the palette below is the site's own :root custom-property block.

    python docs/build_literature_review.py

Writes docs/Smean-TTS-Literature-Review.docx
"""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# --------------------------------------------------------------------------
# Smean AI brand tokens (from the site's :root CSS custom properties)
# --------------------------------------------------------------------------
TEAL = "0B776F"        # --teal        : primary
TEAL_LIGHT = "2DD4BF"  # --teal-light
TEAL_INK = "134E4A"    # --teal-ink
TEAL_SOFT = "CCFBF1"   # --teal-soft
VIOLET = "6D28D9"      # --violet      : accent
VIOLET_SOFT = "EDE9FE"
SIENNA = "C2410C"      # --sienna      : warning / caution
SIENNA_SOFT = "FFEDD5"
INK = "1A1915"         # --ink         : body text
STONE = "6B6A62"       # --stone       : muted text
OAT = "E5E1D8"         # --oat         : borders
LINEN = "F4F2EC"       # --linen       : surfaces
BONE = "FBFAF7"        # --bone        : page background
PAPER = "FFFFFF"

HEADING_FONT = "Georgia"     # brand --font-display is Fraunces, Georgia fallback
BODY_FONT = "Calibri"        # stand-in for Inter Tight (brand --font-sans)
MONO_FONT = "Courier New"   # metric-compatible everywhere; Consolas is Windows-only
KHMER_FONT = "Khmer OS System"

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "assets" / "smean-logo.png"
OUT = ROOT / "Smean-TTS-Literature-Review.docx"


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), v)
    return e


def shade(element, color):
    """Apply a solid fill to a <w:tcPr> or <w:pPr> owner."""
    pr = element.get_or_add_tcPr() if element.tag.endswith("}tc") else element.get_or_add_pPr()
    pr.append(_el("w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": color}))


def cell_borders(cell, color=OAT, size=4, edges=("top", "bottom", "left", "right"), **overrides):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = _el("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        spec = overrides.get(edge)
        if edge in edges or spec:
            c, s = (spec if spec else (color, size))
            borders.append(_el(f"w:{edge}", **{"w:val": "single", "w:sz": str(s),
                                               "w:space": "0", "w:color": c}))
        else:
            borders.append(_el(f"w:{edge}", **{"w:val": "nil"}))
    tcPr.append(borders)


def set_font(run, name=BODY_FONT, size=None, bold=None, italic=None, color=None):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = _el("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), KHMER_FONT if name == MONO_FONT else name)
    if size is not None:
        run.font.size = Pt(size)
        # match the complex-script size too, or Khmer runs inflate the line box
        rPr.append(_el("w:szCs", **{"w:val": str(int(round(size * 2)))}))
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def para(doc, text="", size=10.5, style=None, space_after=6, space_before=0,
         color=INK, bold=False, italic=False, align=None, font=BODY_FONT, indent=None):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = 1.18
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.left_indent = Inches(indent)
    if text:
        set_font(p.add_run(text), font, size, bold, italic, color)
    return p


def rich(doc, chunks, size=10.5, space_after=6, indent=None, align=None):
    """chunks: list of (text, {bold/italic/color/font/size}) tuples."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.18
    if indent is not None:
        pf.left_indent = Inches(indent)
    if align is not None:
        p.alignment = align
    for text, opts in chunks:
        set_font(p.add_run(text),
                 opts.get("font", BODY_FONT), opts.get("size", size),
                 opts.get("bold"), opts.get("italic"), opts.get("color", INK))
    return p


def heading(doc, text, level=1, page_break=False):
    if page_break:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    sizes = {1: 19, 2: 14, 3: 11.5}
    colors = {1: TEAL, 2: TEAL_INK, 3: INK}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt({1: 4, 2: 16, 3: 12}[level])
    pf.space_after = Pt({1: 10, 2: 6, 3: 4}[level])
    pf.keep_with_next = True
    set_font(p.add_run(text), HEADING_FONT, sizes[level], True, False, colors[level])
    if level == 1:
        pPr = p._p.get_or_add_pPr()
        bdr = _el("w:pBdr")
        bdr.append(_el("w:bottom", **{"w:val": "single", "w:sz": "12",
                                      "w:space": "6", "w:color": TEAL}))
        pPr.append(bdr)
    return p


def bullet(doc, chunks, indent=0.25, size=10.5):
    if isinstance(chunks, str):
        chunks = [(chunks, {})]
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.left_indent = Inches(indent + 0.2)
    pf.first_line_indent = Inches(-0.2)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.15
    for text, opts in chunks:
        set_font(p.add_run(text), opts.get("font", BODY_FONT), opts.get("size", size),
                 opts.get("bold"), opts.get("italic"), opts.get("color", INK))
    return p


def table(doc, headers, rows, widths=None, size=8.8, header_fill=TEAL,
          header_color=PAPER, zebra=True, align_right=(), header=True):
    t = doc.add_table(rows=1 if header else 0, cols=len(headers))
    t.autofit = False
    t.style = "Table Grid"

    def write(cell, text, *, bold=False, color=INK, fill=None, sz=size, right=False):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.paragraphs[0].text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2.5)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = 1.06
        if right:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # inline **bold** markers
        for i, seg in enumerate(str(text).split("**")):
            if seg:
                set_font(p.add_run(seg), BODY_FONT, sz, bold or i % 2 == 1, False, color)
        if fill:
            shade(cell._tc, fill)
        cell_borders(cell, OAT, 4)

    if header:
        for i, h in enumerate(headers):
            write(t.rows[0].cells[i], h, bold=True, color=header_color, fill=header_fill, sz=size)
    for r, row in enumerate(rows):
        cells = t.add_row().cells
        fill = LINEN if (zebra and r % 2 == 1) else PAPER
        for i, val in enumerate(row):
            write(cells[i], val, fill=fill, right=(i in align_right))

    if widths:
        total = sum(widths)
        avail = 6.5
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(avail * w / total)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(doc, text):
    para(doc, text, size=8.5, color=STONE, italic=True, space_after=12, space_before=0)


def callout(doc, title, body, accent=TEAL, fill=TEAL_SOFT):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.5)
    shade(cell._tc, fill)
    cell_borders(cell, edges=(), left=(accent, 24), top=(fill, 4),
                 bottom=(fill, 4), right=(fill, 4))
    cell.paragraphs[0].text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.1)
    set_font(p.add_run(title), BODY_FONT, 9.5, True, False, accent)
    for line in body:
        q = cell.add_paragraph()
        q.paragraph_format.space_after = Pt(5)
        q.paragraph_format.left_indent = Inches(0.1)
        q.paragraph_format.line_spacing = 1.15
        for i, seg in enumerate(line.split("**")):
            if seg:
                set_font(q.add_run(seg), BODY_FONT, 9.5, i % 2 == 1, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def code(doc, text, size=7.2):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.5)
    shade(cell._tc, LINEN)
    cell_borders(cell, OAT, 4)
    cell.paragraphs[0].text = ""
    first = True
    for line in text.strip("\n").split("\n"):
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Inches(0.06)
        set_font(p.add_run(line if line else " "), MONO_FONT, size, False, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def field(paragraph, instr):
    r = paragraph.add_run()
    r._r.append(_el("w:fldChar", **{"w:fldCharType": "begin"}))
    it = _el("w:instrText", **{"xml:space": "preserve"})
    it.text = instr
    r._r.append(it)
    r._r.append(_el("w:fldChar", **{"w:fldCharType": "end"}))
    return r


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = BODY_FONT
normal.font.size = Pt(10.5)
normal.font.color.rgb = RGBColor.from_string(INK)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
normal.element.rPr.rFonts.set(qn("w:cs"), KHMER_FONT)

sec = doc.sections[0]
sec.top_margin = Inches(0.85)
sec.bottom_margin = Inches(0.85)
sec.left_margin = Inches(1.0)
sec.right_margin = Inches(1.0)

# ---- cover -------------------------------------------------------------
cover = doc.add_paragraph()
cover.alignment = WD_ALIGN_PARAGRAPH.LEFT
cover.paragraph_format.space_before = Pt(90)
cover.paragraph_format.space_after = Pt(4)
if LOGO.exists():
    cover.add_run().add_picture(str(LOGO), width=Inches(0.82))

rich(doc, [("SMEAN AI", {"bold": True, "size": 10, "color": TEAL}),
           ("   ·   ", {"color": OAT, "size": 10}),
           ("Speech Technology Research", {"size": 10, "color": STONE})],
     space_after=26)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing = 1.05
set_font(p.add_run("Neural Text-to-Speech for Khmer"), HEADING_FONT, 30, True, False, INK)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(20)
p.paragraph_format.line_spacing = 1.1
set_font(p.add_run("VoxCPM2 and Higgs TTS 3 — architecture, training,\ndata requirements, and evaluation"),
         HEADING_FONT, 15, False, True, TEAL)

hr = doc.add_paragraph()
hr.paragraph_format.space_after = Pt(16)
pPr = hr._p.get_or_add_pPr()
b = _el("w:pBdr")
b.append(_el("w:bottom", **{"w:val": "single", "w:sz": "18", "w:space": "1", "w:color": TEAL}))
pPr.append(b)

para(doc, "A literature review of the two open-weight multilingual TTS systems that "
          "produce intelligible Khmer, prepared as the technical basis for a Khmer "
          "speech-synthesis capability at Smean AI.",
     size=11, color=STONE, space_after=26)

table(doc,
      ["", ""],
      [["Document", "Literature review — TTS model survey"],
       ["Subjects", "VoxCPM2 (OpenBMB) · Higgs TTS 3 (Boson AI)"],
       ["Scope", "Architecture · training procedure · data requirements · evaluation metrics"],
       ["Evidence base", "Model checkpoints, shipped source code, vendor documentation, "
                         "and a 400-clip in-house synthesis run"],
       ["Date", "6 September 2026"],
       ["Status", "Internal research document"]],
      widths=[1.1, 3.6], size=9.5, zebra=True, header=False)

# ---- footer ------------------------------------------------------------
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(footer.add_run("Smean AI  ·  Neural TTS for Khmer  ·  page "), BODY_FONT, 8, False, False, STONE)
field(footer, " PAGE ")
for r in footer.runs:
    set_font(r, BODY_FONT, 8, False, False, STONE)

# =========================================================================
heading(doc, "1.  Executive summary", 1, page_break=True)

para(doc, "Khmer is a low-resource language for speech synthesis. It has no eSpeak-NG voice, "
          "hence no off-the-shelf grapheme-to-phoneme frontend; its script is an abugida with no "
          "inter-word spacing, so even tokenising a sentence into words requires a dedicated tool; "
          "and no large, clean, permissively-licensed Khmer TTS corpus exists in public. Any Khmer "
          "TTS effort therefore starts from a multilingual foundation model rather than from scratch.")

para(doc, "Four candidate open-weight models were synthesised over a fixed 100-sentence Khmer test "
          "set (50 pure Khmer, 50 code-switched Khmer–English; 400 clips, zero failures) and judged "
          "by a native Khmer speaker. Two survived: ")

bullet(doc, [("VoxCPM2", {"bold": True, "color": TEAL}),
             (" (OpenBMB, 2B parameters, Apache-2.0) — a tokenizer-free diffusion-autoregressive "
              "model that predicts continuous acoustic latents rather than discrete codec tokens. "
              "Khmer is one of its 30 documented languages, with a self-reported 2.05% CER.", {})])
bullet(doc, [("Higgs TTS 3", {"bold": True, "color": VIOLET}),
             (" (Boson AI, 4B parameters, research/non-commercial) — a neural-codec language model "
              "on a Qwen3-4B backbone with a semantically-grounded audio tokenizer. Khmer is "
              "undocumented, yet works.", {})])

para(doc, "Two models were eliminated: Fish Audio S2 on correctness (it truncates sentences), and "
          "Meta MMS-TTS on prosody (intelligible but flat and robotic).", space_after=10)

callout(doc, "The three findings that matter", [
    "**Architecture is the differentiator, not scale.** VoxCPM2's tokenizer-free design has no "
    "discrete audio vocabulary that can be under-fitted for a low-resource language — the exact "
    "failure that makes Fish Audio S2 unusable in Khmer. Higgs TTS 3 is a codec model, but its "
    "tokenizer fuses a semantic (HuBERT) branch with an acoustic (DAC) branch, which is the most "
    "plausible explanation for Khmer working at all in a language it was never documented to support.",
    "**Only one of the two can be trained today.** VoxCPM2 ships an official fine-tuning guide, "
    "LoRA configs and a dataset validator; a useful Khmer adaptation is a YAML file and a rented "
    "24 GB GPU. Boson ship no training code for Higgs TTS 3 at all — its shipped model class has "
    "no loss-returning forward pass — so training it means writing the trainer first.",
    "**No automatic metric currently ranks Khmer TTS correctly.** In our own run, Whisper-large-v3 "
    "collapsed into repetition loops on Khmer, making CER meaningless (median ≈100% for every "
    "model), while UTMOS and DNSMOS rewarded the model that omits half the sentence. Section 6 "
    "documents this in detail; the practical consequence is that Khmer TTS decisions must be made "
    "by structured human listening.",
])

para(doc, "The recommendation that follows from this is set out in Section 7: build on VoxCPM2, "
          "use Higgs TTS 3 as the zero-shot baseline any fine-tune must beat, and invest in a blind "
          "listening protocol rather than in a better automatic scorer.")

# =========================================================================
heading(doc, "2.  Background: how a modern TTS model is put together", 1, page_break=True)

para(doc, "Every zero-shot neural TTS system in current use decomposes into three parts. Naming the "
          "parts makes the difference between the two subject models legible.")

table(doc,
      ["Stage", "Job", "Typical realisation"],
      [["Text frontend", "Turn written text into a sequence the model can condition on",
        "A byte/BPE tokenizer, or a G2P phonemizer plus lexicon"],
       ["Acoustic model", "Predict a compact intermediate representation of the speech",
        "An autoregressive transformer, a diffusion model, or both"],
       ["Vocoder / decoder", "Turn that representation into a waveform",
        "A neural codec decoder, a GAN vocoder, or a VAE decoder"]],
      widths=[1.0, 2.1, 2.2])

heading(doc, "2.1  The representation choice: discrete tokens vs continuous latents", 2)

para(doc, "The single most consequential design decision is what the acoustic model actually "
          "predicts. There are two families, and the two subject models sit on opposite sides.")

table(doc,
      ["", "Neural-codec LM (discrete)", "Continuous-latent (tokenizer-free)"],
      [["Representation", "Speech quantised into integer tokens from a learned codebook, predicted "
        "exactly like text tokens", "Continuous real-valued vectors, predicted by regression or "
        "diffusion"],
       ["Strength", "Stable to train; reuses the whole LLM toolchain; fast, streamable decoding",
        "No quantisation loss; no fixed audio vocabulary to under-fit"],
       ["Weakness", "The codebook is a bottleneck. If it never learned to represent a language's "
        "phonation, the LM cannot recover it", "Autoregressive continuous prediction accumulates "
        "error over long sequences unless carefully constrained"],
       ["Examples here", "**Higgs TTS 3**, Fish Audio S2", "**VoxCPM2**"]],
      widths=[0.85, 2.1, 2.1])

callout(doc, "Why this matters specifically for Khmer", [
    "A codebook is trained on whatever speech the codec saw. For a language with a small share of "
    "the pretraining corpus, the codebook may simply lack codes that represent its phonation "
    "faithfully — and no amount of LM capacity fixes that, because the LM can only choose among "
    "codes that exist. This is the most credible account of why Fish Audio S2-Pro scores 75.15% CER "
    "on Khmer while VoxCPM2 scores 2.05% on the same benchmark. Higgs TTS 3 mitigates it by "
    "grounding its codebook semantically rather than purely acoustically.",
], accent=VIOLET, fill=VIOLET_SOFT)

heading(doc, "2.2  Why the Khmer text frontend is the usual blocker — and why it isn't here", 2)

para(doc, "Conventional TTS pipelines need a phonemiser. Khmer defeats the standard ones: eSpeak-NG "
          "has no Khmer voice, so there is no off-the-shelf G2P; the script stacks subscript "
          "consonants and places vowel signs on all four sides of a base glyph; and words are not "
          "separated by spaces, so segmentation needs a dedicated tool such as khmertagger.")

para(doc, "Both subject models are trained end-to-end on raw text — VoxCPM2 on bytes through a "
          "73,448-entry tokenizer, Higgs TTS 3 through the Qwen3 tokenizer. Neither needs a Khmer "
          "phonemiser, lexicon, or word segmenter. That single property removes what is normally the "
          "largest engineering block in a low-resource TTS project, and it is why both models can "
          "also switch between Khmer and Latin script mid-sentence without being told.")

# =========================================================================
heading(doc, "3.  VoxCPM2", 1, page_break=True)

table(doc,
      ["Property", "Value"],
      [["Organisation", "OpenBMB"],
       ["Parameters", "2B (~8 GB VRAM to run)"],
       ["Family", "Tokenizer-free diffusion-autoregressive"],
       ["Backbone", "MiniCPM-4"],
       ["Audio", "16 kHz in (reference/training) → **48 kHz out**"],
       ["Languages", "30 documented, **including Khmer**, plus 9 Chinese dialects"],
       ["Licence", "**Apache-2.0** — commercial use permitted, no attribution required"],
       ["Weights", "huggingface.co/openbmb/VoxCPM2"],
       ["Paper", "Zhou et al., “VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation "
                 "and True-to-Life Voice Cloning”, arXiv:2509.24650"],
       ["Fine-tuning", "Official guide, LoRA configs, and a dataset validator"]],
      widths=[1.05, 3.6])
caption(doc, "Table 1 — VoxCPM2 at a glance. Architecture figures throughout §3 are read from the "
             "checkpoint's own config.json and the voxcpm 2.0.3 package source, not from marketing copy.")

heading(doc, "3.1  Architecture", 2)

para(doc, "A 2B-parameter language model predicts one continuous latent vector per four-frame patch "
          "of audio; a small diffusion transformer turns each predicted latent into acoustic "
          "features; a separately-trained variational autoencoder decodes those features to a "
          "waveform. Because the VAE takes 16 kHz features in and emits 48 kHz audio, "
          "super-resolution is built into the decoder rather than bolted on afterwards.")

code(doc, """
Khmer text
  |  byte-level tokenizer (vocab 73,448 - no G2P, no phonemizer, no language tag)
  v
+-------------------------------------------------+
| MiniCPM4 backbone LM        2048 dim x 28 layers |  <- predicts one latent
| GQA 16 query / 2 KV heads, LongRoPE to 32k       |     per audio patch
+-------------------------------------------------+
  |  hidden state
  v
+-------------------------------------------------+
| Residual LM                 8 layers, no RoPE    |  <- refines the latent
+-------------------------------------------------+
  |  conditioning vector
  v
+-------------------------------------------------+
| Local DiT       1024 dim x 12 layers, CFM head   |  <- diffusion: latent -> feats
| euler solver, log-norm schedule, CFG 2.0         |     10 steps at inference
+-------------------------------------------------+
  |  64-dim features, 4 frames per patch
  v
+-------------------------------------------------+
| AudioVAE V2     enc 16 kHz  ->  dec 48 kHz       |  <- waveform
| encoder rates [2,5,8,8]   decoder [8,6,5,2,2,2]  |
+-------------------------------------------------+
""")
caption(doc, "Figure 1 — the VoxCPM2 inference stack. A Local Encoder (1024 dim × 12 layers) sits on "
             "the input side, encoding reference audio for voice cloning into the same latent space "
             "the LM operates in.")

para(doc, "The paper names the stages LocEnc → TSLM → RALM → LocDiT. The TSLM (Text-Semantic Language "
          "Model) plans semantic content and prosody through a differentiable quantisation bottleneck "
          "rather than hard discrete tokens; the RALM (Residual Acoustic LM) recovers the fine "
          "acoustic detail that coarse plan omits; the LocDiT decodes the combination under a "
          "diffusion objective. The design is explicitly framed as avoiding the discrete-vs-continuous "
          "tradeoff described in §2.1 by adopting a semi-discrete residual representation.")

heading(doc, "3.2  The numbers that govern training", 2)

table(doc,
      ["Property", "Value", "Why it matters"],
      [["patch_size", "4", "One LM step covers 4 VAE frames. Sequence length ≈ chars + duration×25/4."],
       ["feat_dim", "64", "Latent width the diffusion transformer predicts."],
       ["AudioVAE frame rate", "25 fps", "One second of audio ≈ 25 frames ≈ 6.25 LM positions."],
       ["max_length", "8192", "Training sequence cap; longer samples are dropped by the packer."],
       ["Encoder sample rate", "**16 kHz**", "**Training audio must be 16 kHz** — the validator "
        "hard-fails on a mismatch."],
       ["Output sample rate", "48 kHz", "Free super-resolution; you do not supply 48 kHz data."],
       ["scalar_quantization_latent_dim", "512", "Scalar quantisation on the latent, not a VQ "
        "codebook — this is the “tokenizer-free” part."],
       ["inference_cfg_rate", "2.0", "Classifier-free guidance default, exposed as --cfg-value."]],
      widths=[1.35, 0.75, 2.6])

heading(doc, "3.3  Where Khmer already stands", 2)

para(doc, "Khmer (km) is one of VoxCPM2's 30 documented languages, confirmed in the checkpoint's own "
          "model-card metadata. OpenBMB report 2.05% CER for Khmer on their internal 30-language "
          "benchmark — the strongest published Khmer figure of any model surveyed. That is close to "
          "their own 30-language average of 1.68%, and it should be read as a vendor's upper bound "
          "rather than an independently reproduced result.")

callout(doc, "The single most important planning fact in this document", [
    "**You are not adding Khmer to VoxCPM2. You are improving Khmer that is already there.**",
    "The official guide's headline requirement — 500+ hours of target-language data — applies to "
    "languages the model does not know. For a language already in the training mix, the task is "
    "speaker adaptation, domain adaptation or prosody correction, and the vendor's own "
    "recommendations for those are two to three orders of magnitude smaller: tens to hundreds of "
    "clips, not hundreds of hours.",
])

heading(doc, "3.4  What training data VoxCPM2 needs", 2)

heading(doc, "Pretraining corpus (for reference)", 3)
para(doc, "2M+ hours of multilingual speech across the 30 languages and 9 Chinese dialects. OpenBMB "
          "publish no per-language hour breakdown — the same transparency gap every foundation-model "
          "vendor in this space has. The Khmer benchmark number is the only available evidence that "
          "Khmer received a non-trivial share.")

heading(doc, "Fine-tuning manifest", 3)
para(doc, "Training data is a JSONL file, one JSON object per line:")
code(doc, """
{"audio": "clips/km_0001.wav", "text": "<Khmer transcript>"}
{"audio": "clips/km_0002.wav", "text": "<Khmer transcript>", "ref_audio": "clips/km_0001.wav"}
""", size=8.0)

table(doc,
      ["Field", "Required", "Notes"],
      [["audio", "**yes**", "Path to a WAV. Relative paths resolve against the manifest's directory."],
       ["text", "**yes**", "Exact transcript. Raw Khmer — no phonemes, no segmentation, no romanisation."],
       ["ref_audio", "no", "A different clip from the same speaker. Trains the voice-cloning path."],
       ["duration", "no", "Seconds. Lets the length filter skip opening every file — worth it above "
        "a few thousand rows."],
       ["dataset_id", "no", "Integer tag for mixing corpora; lets the model condition on source."]],
      widths=[0.75, 0.6, 3.3])

para(doc, "The package ships a validator; run it before spending GPU hours. It reports total hours, "
          "duration range and text-length distribution, and hard-fails on missing files, sample-rate "
          "mismatches, empty transcripts and malformed JSON.")
code(doc, "voxcpm validate --manifest data/train.jsonl --sample-rate 16000 --verbose", size=8.4)

heading(doc, "Audio requirements", 3)
table(doc,
      ["Requirement", "Value", "Enforcement"],
      [["Sample rate", "**16 kHz**", "Hard failure in `validate` on a mismatch."],
       ["Format", "WAV", "Recommended; anything soundfile reads will load."],
       ["Clip duration", "**3–30 s**", "Over 30 s warns (OOM risk); under 0.3 s warns as too short."],
       ["Trailing silence", "**< 0.5 s**", "Not cosmetic — see the callout below."],
       ["Loudness", "Normalised", "Consistent level across the corpus."]],
      widths=[1.0, 0.9, 2.7])

callout(doc, "Trailing silence is the most-cited cause of failure in the official FAQ", [
    "Clips that end with a long silent tail teach the model that utterances do not end, and it "
    "learns to keep generating — producing exactly the runaway output that made Fish Audio S2 "
    "unusable in our evaluation. Trim aggressively. This is the highest-value preprocessing step "
    "available, and it costs nothing.",
], accent=SIENNA, fill=SIENNA_SOFT)

heading(doc, "How much data", 3)
table(doc,
      ["Goal", "Method", "Data", "Realistic for Khmer?"],
      [["One specific Khmer voice", "LoRA r=32", "5–50 clips", "**Yes** — an afternoon of recording."],
       ["Better Khmer prosody, or a domain (news, IVR, education)", "LoRA r=32–64", "50–500 clips",
        "**Yes — the highest-value target.**"],
       ["Large-scale Khmer customisation", "Full fine-tune", "1000+ clips", "Plausible with OpenSLR SLR42."],
       ["Adding a language from scratch", "Full fine-tune, lr 1e-5", "**500+ hours**",
        "Not applicable — Khmer is already in."]],
      widths=[1.7, 0.85, 0.75, 1.65])

heading(doc, "3.5  Fine-tuning procedure", 2)

para(doc, "Environment: Python 3.10–3.11, PyTorch ≥ 2.5.0, CUDA ≥ 12.0, plus tensorboardX, argbind, "
          "transformers and librosa. VRAM, at batch_size=16 and max_batch_tokens=8192:")

table(doc,
      ["Mode", "VRAM", "Note"],
      [["LoRA", "~20 GB", "A single 24 GB card (A10G / 3090 / 4090) is the pragmatic target."],
       ["Full fine-tune", "~40 GB", "A 48 GB A6000 class card."],
       ["Multi-GPU", "+~10 GB per card", "Gradient buckets and NCCL buffers — not free."]],
      widths=[1.0, 0.9, 3.0])

callout(doc, "Hardware reality check", [
    "The in-house evaluation ran on a 12 GB RTX 3060 — **below the LoRA minimum**. Reaching ~20 GB "
    "means dropping batch_size and max_batch_tokens hard and raising grad_accum_steps to compensate, "
    "and even then 12 GB is marginal. Renting a single 24 GB card for LoRA is the realistic path; "
    "budget for it rather than trying to fit locally.",
], accent=SIENNA, fill=SIENNA_SOFT)

para(doc, "The LoRA configuration (conf/voxcpm_v2/voxcpm_finetune_lora.yaml), abbreviated to the "
          "fields that carry decisions:")
code(doc, """
pretrained_path: /path/to/VoxCPM2/
train_manifest:  /path/to/train.jsonl
val_manifest:    /path/to/val.jsonl

sample_rate:      16000      # must match your audio
out_sample_rate:  48000
batch_size:       16
grad_accum_steps: 1
num_iters:        1000
valid_interval:   500
save_interval:    500

learning_rate:    0.0001     # 1e-4 for LoRA; 1e-5 for a full fine-tune
weight_decay:     0.01
warmup_steps:     100
max_batch_tokens: 8192

lambdas:
  loss/diff: 1.0             # diffusion loss on the latent
  loss/stop: 1.0             # end-of-utterance prediction

lora:
  enable_lm:   true          # adapt the MiniCPM4 backbone
  enable_dit:  true          # adapt the diffusion transformer
  enable_proj: false
  r:     32
  alpha: 32
  dropout: 0.0
""", size=7.8)

bullet(doc, [("Choosing the rank. ", {"bold": True}),
             ("r=32 clones a speaker; r=64 is for style or language work — which is what Khmer "
              "prosody correction is. Set alpha to r or 2r.", {})])
bullet(doc, [("Keep both adapters on. ", {"bold": True}),
             ("The backbone carries linguistic behaviour, the DiT carries acoustic detail, and "
              "Khmer adaptation wants both. enable_proj stays false.", {})])
bullet(doc, [("For a full fine-tune, ", {"bold": True}),
             ("delete the lora: block entirely and drop the learning rate tenfold to 1e-5.", {})])

code(doc, """
# single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
""", size=7.6)

para(doc, "Monitor with TensorBoard: loss/diff should fall steadily, loss/stop should stabilise "
          "early, and grad_norm and lr should behave. But select checkpoints by ear, not by loss — "
          "the FAQ is explicit that on small datasets the model begins ignoring the text input "
          "within a few hundred steps. Checkpoint every 500 steps and listen to each one; the best "
          "checkpoint is frequently not the last.")

para(doc, "LoRA adapters hot-swap at runtime, so one base model can serve several Khmer voices from "
          "a single set of 2B weights:")
code(doc, """
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="/path/to/checkpoints/lora/latest",
)
wav = model.generate(text="<Khmer text>")
""", size=8.0)

heading(doc, "3.6  Failure modes", 2)
table(doc,
      ["Symptom", "Cause", "Fix"],
      [["Output ignores the text; reproduces training clips",
        "Overfitting — the classic small-dataset failure, and it arrives fast",
        "Fewer steps, lower LR, more data, lower LoRA rank. Checkpoint often."],
       ["Generation runs away, or long trailing noise",
        "**Trailing silence > 0.5 s in the training clips**",
        "Re-trim the corpus. The most common cause per the official FAQ."],
       ["Loss will not converge", "Bad transcripts, wrong sample rate, misaligned audio",
        "Re-run `voxcpm validate`; spot-check alignment by hand."],
       ["Out of memory", "Batch too large",
        "Lower batch_size / max_batch_tokens, raise grad_accum_steps."],
       ["Khmer improved, English and Chinese degraded", "Catastrophic forgetting",
        "Mix Chinese/English data in — or use LoRA and disable it for other languages."]],
      widths=[1.5, 1.5, 2.2])

para(doc, "That last row is a genuine argument for LoRA on this project: because the adapter is "
          "separable, a Khmer LoRA cannot damage the base model's other 29 languages. Toggle it off "
          "and the original model is back, bit for bit. A full fine-tune has no such escape hatch.")

# =========================================================================
heading(doc, "4.  Higgs TTS 3", 1, page_break=True)

table(doc,
      ["Property", "Value"],
      [["Organisation", "Boson AI"],
       ["Parameters", "4B"],
       ["Family", "Neural-codec language model"],
       ["Backbone", "Qwen3-4B — 36 layers, hidden 2560, GQA 32 q / 8 kv, head dim 128"],
       ["Audio", "8 codebooks × 1026 vocab at 25 fps → **24 kHz** waveform"],
       ["Languages", "102 claimed (85 “polished”, 17 “usable but less polished”). "
                     "**Khmer is in neither list.**"],
       ["Licence", "**Boson Higgs TTS 3 Research and Non-Commercial License** — commercial use "
                   "requires separate negotiation"],
       ["Weights", "huggingface.co/bosonai/higgs-tts-3-4b"],
       ["Fine-tuning", "**None shipped** — no trainer, no dataset loader, no configs"]],
      widths=[1.05, 3.6], header_fill=VIOLET)
caption(doc, "Table 2 — Higgs TTS 3 at a glance. Architecture is read from the checkpoint's "
             "config.json, the higgs-audio-v2-tokenizer config, and the "
             "modeling_higgs_multimodal_qwen3.py remote code shipped with the weights.")

heading(doc, "4.1  Two things to establish before anything else", 2)

callout(doc, "1 — Khmer is an emergent, unsupported capability", [
    "Boson report single-digit WER/CER on 102 languages across two tiers. Khmer appears in neither. "
    "It is not a low-tier language for this model; it is an unlisted one. Yet it demonstrably "
    "produces intelligible Khmer — confirmed across 100 sentences in our own run and audible on "
    "Boson's own demo space.",
    "That makes Khmer **real, but unmeasured, unsupported, and carrying no vendor commitment**. If "
    "a future release silently drops it, nothing was promised. Everything else in this section "
    "rests on that footing.",
], accent=VIOLET, fill=VIOLET_SOFT)

callout(doc, "2 — The licence is not open in the sense VoxCPM2 is", [
    "Higgs TTS 3 ships under the Boson Higgs TTS 3 Research and Non-Commercial License. Production "
    "use, hosted APIs, embedding it in a product or service, and reselling it all require a separate "
    "commercial licence from Boson. A **Creator Use Grant** permits free use — including monetised "
    "podcasts, videos and social posts — provided “Boson AI's Higgs Audio” is credited in the audio "
    "or prominently in accompanying text.",
    "**The practical fork:** if the deliverable is a product, VoxCPM2 is the only one of the two "
    "shippable without a commercial negotiation. This is a licensing decision, not a quality one, "
    "and it is worth settling before any training effort is spent.",
], accent=SIENNA, fill=SIENNA_SOFT)

heading(doc, "4.2  Architecture", 2)

code(doc, """
Khmer text
  |  Qwen3 tokenizer (vocab 151,936) + Higgs control tokens
  v
+------------------------------------------------------+
| Qwen3-4B backbone                                     |
| 36 layers . hidden 2560 . GQA 32 q / 8 kv . head 128  |
| RoPE theta=1e6 . 32,768 max pos . 8,192 train seq len |
| tied text embeddings                                  |
+------------------------------------------------------+
   ^ fused multi-codebook embedding    v fused multi-codebook head
     one [8 x 1026, 2560] tensor,        one [8 x 1026, 2560] linear,
     summed across the codebook axis     reshaped to [L, 8, 1026]
  |  8 codebooks x 1026 vocab, 25 fps, delay-patterned
  v
+------------------------------------------------------+
| Higgs Audio v2 Tokenizer  (separate checkpoint)       |
|  semantic branch: HuBERT, 12 layers, 768 dim, 16 kHz  |
|  acoustic branch: DAC, 1024-entry codebooks,          |
|                   hop 960, downsample x320            |
+------------------------------------------------------+
  v
24 kHz waveform
""")
caption(doc, "Figure 2 — the Higgs TTS 3 inference stack.")

heading(doc, "The tokenizer is the interesting part", 3)
para(doc, "Most codec LMs stack a purely acoustic codec (DAC, EnCodec) under the language model, "
          "leaving the LM to learn what the sounds mean on its own. Boson instead trained a unified "
          "tokenizer that fuses a semantic branch — a HuBERT encoder at 16 kHz, capturing phonetic "
          "content — with an acoustic branch — a DAC-style residual quantiser, capturing timbre and "
          "detail — into one token stream at 24 kHz.")

para(doc, "This is plausibly why Khmer works at all despite being unlisted. A semantically-grounded "
          "tokenizer generalises to phonation the LM saw little of during training, because the "
          "phonetic structure is already factored into the token space by the tokenizer rather than "
          "having to be learned from scratch by the language model.")

heading(doc, "The delay pattern", 3)
para(doc, "Eight codebooks must be emitted per 40 ms frame, but an autoregressive model emits one "
          "position at a time. The standard solution — used here — staggers the codebooks so each "
          "step predicts one new codebook's token while the others lag behind:")
code(doc, """
codebook c is shifted by c steps
raw [T, 8]  ->  delayed [T + 7, 8]
padded with BOC = 1024 before its span and EOC = 1025 after
(hence vocab 1026 = 1024 codes + BOC + EOC)
""", size=8.0)
para(doc, "At generation time a small state machine ramps the delay up, watches for EOC, and winds "
          "down; the rows are then de-delayed and handed to the codec. Any training code must apply "
          "exactly this transform to its targets — see §4.4.")

heading(doc, "Prompt format", 3)
code(doc, """
<|tts|>  [ <|ref_text|>  ...reference transcript...  ]
         [ <|ref_audio|> ...one placeholder per ref audio token... ]
         <|text|>  ...the text to speak...
         <|audio|>   <- generation starts here
""", size=8.0)
para(doc, "Audio positions are marked with placeholder id −100 in the text stream and overwritten "
          "with the fused audio embedding before the forward pass. The bracketed parts are the "
          "zero-shot voice-cloning path; omit them and the model uses its own default voice, which "
          "is how the evaluation ran it.")

heading(doc, "Inline control tokens", 3)
para(doc, "All tags use <|category:value|> syntax and can be inserted mid-utterance. VoxCPM2 has no "
          "equivalent.")
table(doc,
      ["Category", "Values"],
      [["Emotion (21)", "elation, amusement, enthusiasm, determination, pride, contentment, "
        "affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear, "
        "disgust, bitterness, sadness, shame, helplessness"],
       ["Style (3)", "singing, shouting, whispering"],
       ["Sound effects (9)", "cough, laughter, crying, screaming, burping, humming, sigh, sniff, "
        "sneeze — pair each with the matching onomatopoeia"],
       ["Prosody", "speed very_slow / slow / fast / very_fast (≈0.65×–1.4×); pause (400–700 ms); "
        "long_pause (700–1500 ms); pitch_low (−3 st); pitch_high (+2.5 st); expressive_high / "
        "expressive_low"]],
      widths=[0.9, 3.8], header_fill=VIOLET)

para(doc, "Whether these transfer to Khmer is untested. They were trained on the documented "
          "languages; nothing establishes that an emotion tag produces Khmer-appropriate prosody "
          "rather than an English-shaped one. This is cheap to check by hand and worth checking "
          "before relying on it.")

heading(doc, "4.3  The training situation", 2)

para(doc, "Boson ship no training or fine-tuning code for Higgs TTS 3. This was verified rather than "
          "inferred:")

bullet(doc, [("The official repository contains boson_multimodal/, examples/ and tech_blogs/. "
              "examples/ holds generation.py, serve_engine/, vllm/, voice_prompts/ and "
              "scene_prompts/ — ", {}),
             ("inference and serving only", {"bold": True}),
             (". There is no trainer, no dataset loader, no config directory.", {})])
bullet(doc, [("The shipped remote-code class HiggsMultimodalQwen3ForConditionalGeneration implements "
              "generate_speech(), _build_prompt_ids(), _prefill_embeds() and a sampler state machine. ",
              {}),
             ("It has no forward() that accepts labels and returns a loss", {"bold": True}),
             (" — it is not wired for gradient descent.", {})])
bullet(doc, "Nothing in the model card, the Boson blog post, or the LMSYS SGLang-Omni write-up "
            "discusses fine-tuning.")
bullet(doc, "Training data is undisclosed. Higgs Audio v2 was described as pretrained on a 10M-hour "
            "corpus (“AudioVerse”); no comparable figure is published for v3, and no per-language "
            "breakdown exists for either. There is no way to know how much Khmer, if any, the model saw.")

para(doc, "The nearest prior art is the community LoRA trainer JimmyMa99/train-higgs-audio — for v2, "
          "not v3. v3 changed the architecture (v2's DualFFN is gone; v3 uses a plain Qwen3 backbone "
          "with a fused multi-codebook embedding and head), so that code is a useful reference for "
          "the data pipeline, not a drop-in.", space_after=10)

heading(doc, "4.4  What writing a trainer would involve", 2)
para(doc, "This is tractable — it is a standard codec-LM training loop, and the architecture is fully "
          "legible from the shipped remote code — but it is real engineering, not configuration.")

table(doc,
      ["#", "Step", "Detail"],
      [["1", "Encode audio to codes", "Run bosonai/higgs-audio-v2-tokenizer over each clip to get "
        "[T, 8] integer codes at 25 fps."],
       ["2", "Apply the delay pattern", "apply_delay_pattern() already exists in the shipped remote "
        "code: [T, 8] → [T+7, 8], BOC-padded before each codebook's span, EOC after."],
       ["3", "Build the sequence", "Reuse _build_prompt_ids() and _prefill_embeds() **verbatim** so "
        "training and inference agree exactly. A mismatch here is silent and fatal."],
       ["4", "Write the missing forward()", "Embed via HiggsFusedMultiTextEmbedding, run the Qwen3 "
        "backbone, project with HiggsFusedMultiTextHead to [L, 8, 1026], cross-entropy against the "
        "next delayed frame across all 8 codebooks — masking BOC/EOC padding and the whole text span."],
       ["5", "Attach LoRA", "Via peft, on the backbone's attention and MLP projections. Full "
        "fine-tuning 4B params in bf16 needs ~60–80 GB with an 8-bit optimiser; LoRA brings it into "
        "single-24 GB range. The fused embedding and head are tied to the text embedding — decide "
        "deliberately whether to adapt them."],
       ["6", "Validate by generation", "Codec-LM loss is a poor guide to output quality. Synthesise "
        "a held-out set every N steps and listen."]],
      widths=[0.25, 1.3, 3.4], header_fill=VIOLET)

para(doc, "Data requirements would resemble VoxCPM2's — paired 16 kHz-or-better audio with accurate "
          "Khmer transcripts, silence-trimmed, 3–30 s clips — and the same corpus scarcity applies. "
          "See §5.")

# =========================================================================
heading(doc, "5.  Data requirements for Khmer", 1, page_break=True)

para(doc, "Both models are trained end-to-end on graphemes, so the data question reduces to one "
          "thing: paired (text, audio) at adequate quality. This section covers what that means "
          "concretely and what actually exists for Khmer. It applies to both models equally.")

heading(doc, "5.1  What TTS data consists of", 2)
bullet(doc, [("Audio. ", {"bold": True}),
             ("Single speaker per file, minimal background noise or music, consistent loudness. "
              "VoxCPM2 requires exactly 16 kHz; a codec-LM trainer would want 24 kHz or better and "
              "resample. Higher source rates are always preferable — you can downsample, not upsample.", {})])
bullet(doc, [("Transcripts. ", {"bold": True}),
             ("Orthographically correct text matching the audio exactly. Neither model needs forced "
              "alignment, phonemes or romanisation — raw Khmer text is the input.", {})])
bullet(doc, [("Speaker labels ", {"bold": True}),
             ("for the voice-cloning path (VoxCPM2's optional ref_audio field pairs two clips from "
              "the same speaker). Emotion and style metadata matter only for controllable synthesis.", {})])
bullet(doc, [("Aggressive silence trimming. ", {"bold": True}),
             ("Under 0.5 s of trailing silence per clip. See §3.4 — this is the single highest-value "
              "preprocessing step, and neglecting it produces runaway generation.", {})])

heading(doc, "5.2  Scale, by goal", 2)
table(doc,
      ["Goal", "What you need", "Typical scale"],
      [["Single-speaker, single-language, from scratch (classic VITS)",
        "Clean studio recordings from one speaker, consistent mic and room",
        "~10–25 hours (LJSpeech, ~24 h, is the benchmark)"],
       ["Adapting a foundation model to one Khmer voice", "Clean clips from one speaker",
        "**5–50 clips** (LoRA)"],
       ["Adapting a foundation model for Khmer prosody or a domain",
        "One or two speakers, matched to the target domain", "**50–500 clips** (LoRA)"],
       ["Large-scale Khmer customisation", "Multi-speaker, QC'd", "1000+ clips (full fine-tune)"],
       ["Adding a language a model does not know", "Broad, diverse coverage of the language",
        "**500+ hours** — not applicable to VoxCPM2 + Khmer"],
       ["Training a multilingual foundation model", "Millions of hours plus rich metadata",
        "1–10M+ hours (VoxCPM2: 2M+; Higgs Audio v2: ~10M)"]],
      widths=[1.9, 1.7, 1.5])

heading(doc, "5.3  Khmer corpora that actually exist", 2)
table(doc,
      ["Source", "Size", "Licence", "Assessment"],
      [["OpenSLR SLR42", "866 MB (male set)", "CC BY-SA 4.0",
        "Google-collected, manually QC'd. The closest thing to a standard Khmer TTS corpus. "
        "**OpenSLR publishes no hour count, speaker count or sample rate** — download and measure "
        "before planning around it. Mirrored on HF as deepdml/openslr42-khmer-tts."],
       ["Panhapich/khmer-tts-processed", "1k–10k rows", "—",
        "Pre-segmented; check provenance before relying on it."],
       ["Panhapich/khmer-english-codeswitch-tts", "1k–10k rows", "CC BY 4.0",
        "Matches a code-switched use case directly, **but is tagged synthetic / tts-generated** — "
        "it is TTS output, so training on it distils another model's errors, accent and artefacts "
        "into yours. Useful for text; treat the audio with suspicion."],
       ["KLEA", "~3,000 words", "—",
        "Word-level, not sentences. Good for pronunciation reference, not TTS fine-tuning."],
       ["MMS-lab (Khmer portion)", "~40 h (project average)", "Research",
        "New Testament readings, single speaker. What backs facebook/mms-tts-khm. Domain-narrow "
        "and single-voice."]],
      widths=[1.25, 0.75, 0.65, 2.6])

callout(doc, "The honest summary", [
    "**There is no large, clean, permissively-licensed Khmer TTS corpus.** SLR42 is the only "
    "well-attested one and it is a single-gender set of unpublished size. Beyond a few hours, you "
    "will be recording it or licensing it.",
    "Given that VoxCPM2 already speaks Khmer, **recording 200–500 clean sentences from one or two "
    "good speakers and running LoRA is a far better use of effort than assembling a large corpus.** "
    "Corpus quality dominates corpus size at this scale, and avoiding synthetic (TTS-generated) "
    "audio matters more than adding hours.",
])

# =========================================================================
heading(doc, "6.  How TTS is evaluated — and why it fails for Khmer", 1, page_break=True)

para(doc, "TTS quality is evaluated along two largely independent axes: intelligibility (did it say "
          "the right words?) and naturalness or similarity (does it sound good, and like the "
          "intended speaker?). Both human and automatic metrics exist for each. This section sets "
          "out the standard toolkit, then reports what happened when it was applied to Khmer.")

heading(doc, "6.1  Subjective (human-rated) metrics", 2)
table(doc,
      ["Metric", "What it measures", "How it works"],
      [["**MOS** — Mean Opinion Score", "Overall perceived quality and naturalness",
        "Listeners rate samples 1–5; scores averaged. The oldest and still most-cited TTS metric, "
        "but expensive, slow, and not comparable across studies with different listener pools."],
       ["**CMOS** — Comparative MOS", "Relative quality between two systems",
        "Listeners hear A/B pairs and rate preference and strength on a signed scale, e.g. −3 to +3."],
       ["**SMOS** — Similarity MOS", "How much a cloned voice resembles the target speaker",
        "Same 1–5 scale, but the question is “does this sound like the reference speaker”, not "
        "“is this good speech”."],
       ["**A/B preference (Arena-style)**", "Head-to-head system ranking",
        "Blind pairwise votes aggregated into an Elo rating, as Chatbot Arena does for LLMs. "
        "HuggingFace's TTS-Arena is the standard public leaderboard of this kind."]],
      widths=[1.1, 1.35, 3.05])

heading(doc, "6.2  Objective (automatic) metrics", 2)
table(doc,
      ["Metric", "Measures", "How it is computed"],
      [["**WER / CER**", "Intelligibility",
        "Run a strong ASR model (commonly Whisper-large-v3) on the synthesised audio and diff the "
        "transcript against the input text. The single most-reported number in modern TTS papers. "
        "**CER is preferred over WER for non-whitespace-segmented languages** — Khmer, Thai, "
        "Chinese, Japanese — since word boundaries are themselves ambiguous."],
       ["**SIM / SECS**", "Voice-cloning fidelity",
        "Cosine similarity between speaker embeddings of reference and generated audio, typically "
        "from a WavLM-large speaker-verification model or ECAPA-TDNN."],
       ["**UTMOS**", "Predicted naturalness",
        "A network trained to predict human MOS from self-supervised speech features — a fast, free "
        "proxy that avoids running a listening study per experiment."],
       ["**DNSMOS / NISQA**", "Predicted perceptual quality, including noise and distortion",
        "Similar MOS-predictor models, originally built for speech enhancement, now reused as a "
        "no-reference TTS quality check. DNSMOS reports SIG (signal), BAK (background) and OVRL."],
       ["**MCD**", "Acoustic distance to a reference recording",
        "Mel-cepstral distortion. Used when a ground-truth recording of the same sentence exists; "
        "less applicable to zero-shot systems, which have no single correct reference."],
       ["**RTF**", "Inference speed, not quality",
        "Synthesis time ÷ duration of resulting audio. Below 1.0 is faster than real time. "
        "Determines viability for live and streaming use."]],
      widths=[0.85, 1.2, 3.45])

heading(doc, "6.3  Standard benchmark suites", 2)
bullet(doc, [("Seed-TTS-eval", {"bold": True}),
             (" (ByteDance) — the most widely adopted zero-shot benchmark. Reports WER (via "
              "Whisper-large-v3) and SIM (via WavLM-large) on English and Chinese, plus a “hard” "
              "subset of linguistically tricky sentences.", {})])
bullet(doc, [("CV3-eval", {"bold": True}),
             (" — a Common Voice-derived multilingual set reporting WER/CER, SIM and DNSMOS. This is "
              "the main route by which multilingual coverage beyond EN/ZH gets benchmarked, and the "
              "suite VoxCPM2 cites for its 11- and 30-language comparisons, Khmer included.", {})])
bullet(doc, [("TTS-Arena / TTS-Arena V2", {"bold": True}),
             (" (HuggingFace) — public blind-listening Elo leaderboard; the human-preference "
              "equivalent of Seed-TTS-eval.", {})])
bullet(doc, [("InstructTTSEval and the MiniMax Multilingual Test", {"bold": True}),
             (" — newer suites for instruction following (“say this angrily”, “whisper this”) and "
              "broader multilingual robustness.", {})])

heading(doc, "6.4  What the vendors report", 2)
table(doc,
      ["Benchmark", "VoxCPM2 (self-reported)", "Higgs TTS 3 (self-reported)"],
      [["Seed-TTS-eval, test-EN", "WER 1.84%, SIM 75.3%", "Not published in comparable form"],
       ["Seed-TTS-eval, test-ZH", "WER 0.97%, SIM 79.5%", "Not published in comparable form"],
       ["Multilingual", "30-language internal set, 1.68% average error",
        "“Single-digit WER/CER” on 102 languages, two tiers"],
       ["**Khmer**", "**2.05% CER** (internal 30-language set)",
        "**Not benchmarked — Khmer is not on either language list**"],
       ["Speed", "RTF ~0.30 on RTX 4090; ~0.13 with Nano-vLLM", "Optimised for SGLang-Omni serving"]],
      widths=[1.2, 1.8, 1.8])
caption(doc, "Table 3 — vendor-reported figures. Both columns are self-reported and should be read "
             "as upper bounds. VoxCPM2's Khmer figure comes from a competitor comparison published "
             "by OpenBMB, not an independent third party; it is directionally credible given the "
             "size of the gap it claims over Fish Audio S2-Pro (75.15% CER on the same test), but "
             "it has not been independently reproduced.")

heading(doc, "6.5  What we measured in house", 2)
para(doc, "All four candidate models were run over the fixed 100-sentence Khmer set — 400 clips, "
          "zero failures — on an RTX 3060 and scored on CER (Whisper-large-v3, language=km), UTMOS, "
          "DNSMOS and RTF.")

table(doc,
      ["Model", "CER median", "UTMOS", "DNSMOS OVRL", "P.808", "RTF median", "Listening verdict"],
      [["mms", "98.8%", "2.95", "3.23", "3.71", "0.010", "**Eliminated** — prosody"],
       ["voxcpm2", "100.7%", "2.49", "2.98", "3.50", "1.382", "**Contender**"],
       ["fish-s2", "101.0%", "3.75", "3.21", "3.79", "2.651", "**Eliminated** — correctness"],
       ["higgs3", "98.2%", "2.98", "3.14", "3.84", "0.745", "**Contender**"]],
      widths=[0.75, 0.7, 0.5, 0.75, 0.5, 0.7, 1.6], align_right=(1, 2, 3, 4, 5))
caption(doc, "Table 4 — the full in-house run. CER lower is better; UTMOS and DNSMOS are 1–5 MOS "
             "scales, higher is better; RTF below 1.0 is faster than real time. Note that the "
             "metric columns and the verdict column disagree completely.")

callout(doc, "None of the automatic metrics ranks these models correctly", [
    "**CER is invalid.** Whisper-large-v3 cannot transcribe Khmer — it collapses into repetition "
    "loops. Median CER is ≈100% for every model, and 34 of 100 sentences drew a byte-identical "
    "transcript from two or more different models' audio, which is only possible if the transcripts "
    "describe Whisper rather than the audio. Decoder settings (temperature fallback, n-gram "
    "repetition blocking, language auto-detect) were each tried and ruled out as the cause.",
    "**UTMOS and DNSMOS measure the wrong thing.** They score how clean the waveform sounds, not "
    "whether it says the Khmer text. fish-s2 holds the best UTMOS in the run (3.75) and is unusable; "
    "voxcpm2 holds the worst (2.49) and is a contender.",
    "**The decisive evidence:** fish-s2 utterance A31 delivers a 158-character sentence in 5.3 "
    "seconds — 3.2× its own median speaking rate, meaning most of the sentence is simply absent — "
    "and UTMOS scored it **4.32 / 5**, higher than the best score either usable model earned anywhere.",
], accent=SIENNA, fill=SIENNA_SOFT)

para(doc, "Only RTF survives, because it never touches the ASR and measures something unambiguous. "
          "Higgs TTS 3 runs at 0.745 median RTF against VoxCPM2's 1.382 — nearly twice as fast on "
          "the same card, and the only model besides MMS that ran faster than real time.")

heading(doc, "6.6  What to evaluate with instead", 2)
para(doc, "For Khmer, the evaluation plan has to be human-centred. In priority order:")
bullet(doc, [("Blind A/B preference with multiple Khmer-speaking listeners. ", {"bold": True}),
             ("The only method in §6.1 that is both reliable and affordable at small scale. "
              "Randomise order, hide model identity, collect pairwise votes, aggregate to a "
              "preference rate with confidence intervals.", {})])
bullet(doc, [("A word-error transcription task. ", {"bold": True}),
             ("Have Khmer speakers transcribe synthesised audio and score their transcripts against "
              "the source text. This is the human substitute for CER, and it is the only "
              "intelligibility measure currently trustworthy for Khmer.", {})])
bullet(doc, [("Duration sanity checks — a cheap automatic guard. ", {"bold": True}),
             ("Flag any utterance whose characters-per-second rate deviates sharply from the "
              "model's own median. This catches truncation automatically, which is exactly the "
              "failure UTMOS missed on fish-s2 A31, and it costs nothing to compute.", {})])
bullet(doc, [("RTF and VRAM, measured on the target hardware. ", {"bold": True}),
             ("Unambiguous, and directly decision-relevant for deployment.", {})])
bullet(doc, [("A Khmer-capable ASR, if one becomes available. ", {"bold": True}),
             ("This would restore CER. A Khmer-specific Qwen3-ASR checkpoint was attempted and "
              "failed on a library bug; the search is currently paused, not abandoned.", {})])

# =========================================================================
heading(doc, "7.  Comparison and recommendation", 1, page_break=True)

table(doc,
      ["", "VoxCPM2", "Higgs TTS 3"],
      [["Parameters", "2B", "4B"],
       ["Audio representation", "Continuous latents (tokenizer-free)", "8 discrete codebooks @ 25 fps"],
       ["Backbone", "MiniCPM-4", "Qwen3-4B"],
       ["Output sample rate", "**48 kHz**", "24 kHz"],
       ["Khmer support", "**Documented** — 1 of 30 languages, 2.05% CER claimed",
        "**Undocumented** — works anyway, no vendor commitment"],
       ["Licence", "**Apache-2.0**", "Research / non-commercial; Creator Use Grant with attribution"],
       ["Official fine-tuning", "**Yes** — guide, YAML configs, LoRA, data validator",
        "**None** — you would write the trainer"],
       ["Effort to a first adapted result", "**A YAML file and a 24 GB card**",
        "Weeks of implementation, then a 24 GB card"],
       ["Expressive control", "None", "**21 emotions, 3 styles, 9 sound effects, prosody tags** "
        "(untested in Khmer)"],
       ["Measured RTF (RTX 3060)", "1.382", "**0.745**"],
       ["Measured UTMOS", "2.49 (last of four)", "2.98 (second of four)"],
       ["Listening verdict", "**Contender**", "**Contender**"]],
      widths=[1.15, 1.75, 1.85])
caption(doc, "Table 5 — side by side. Neither metric row ranks these models for Khmer; per §6 they "
             "are recorded to show what was measured, not to pick a winner. No winner is declared "
             "between the two contenders — separating them requires a blind multi-listener test.")

heading(doc, "7.1  Recommendation", 2)

callout(doc, "Build on VoxCPM2. Keep Higgs TTS 3 as the baseline to beat.", [
    "The reasoning does not depend on any judgement about audio quality, because by ear the two are "
    "equally usable. It depends on two facts: **one of them can be shipped and the other cannot**, "
    "and **one of them takes a config file to improve while the other takes a from-scratch trainer.** "
    "That is not a close call.",
])

para(doc, "Where Higgs TTS 3 nonetheless earns its place:", space_after=4)
bullet(doc, [("As the baseline to beat. ", {"bold": True}),
             ("It is the strongest zero-shot Khmer output in this survey, obtained with no "
              "adaptation at all. Any VoxCPM2 fine-tune should be A/B'd against it; if the fine-tune "
              "does not clearly win, the fine-tune is not done.", {})])
bullet(doc, [("On speed. ", {"bold": True}),
             ("Nearly 2× faster than VoxCPM2 on the same card. For a latency-bound application that "
              "is a real argument.", {})])
bullet(doc, [("For expressive and creator work. ", {"bold": True}),
             ("The control-token vocabulary has no VoxCPM2 equivalent, and the Creator Use Grant "
              "covers monetised content with attribution.", {})])
bullet(doc, [("As the fallback. ", {"bold": True}),
             ("Should VoxCPM2 fine-tuning fail to improve on the base model, writing a Higgs trainer "
              "becomes the reasonable next investment — rather than the first one.", {})])

para(doc, "If Higgs TTS 3 is pursued, the first step is not code. It is confirming with Boson that a "
          "commercial licence is obtainable on acceptable terms, since without one the work cannot "
          "be deployed regardless of how well it turns out.", space_after=12)

heading(doc, "7.2  Proposed path for a Khmer voice", 2)
table(doc,
      ["#", "Step", "Detail"],
      [["1", "Record or license 200–500 clean Khmer sentences",
        "One or two speakers, 16 kHz, 3–15 s per clip, trailing silence trimmed under 0.5 s. Prefer "
        "recording over scraping — corpus quality dominates everything downstream. Avoid synthetic "
        "(TTS-generated) audio entirely."],
       ["2", "Hold out ~10% as a validation split",
        "Run `voxcpm validate` on both splits before committing GPU time."],
       ["3", "LoRA fine-tune",
        "r=64, alpha=64, lr 1e-4, 1000 steps, enable_lm and enable_dit both true, on a rented 24 GB "
        "card. Save every 500 steps."],
       ["4", "Listen to every checkpoint",
        "Do not select on loss. Overfitting arrives within a few hundred steps on small datasets."],
       ["5", "Re-run the fixed evaluation set",
        "Synthesise the same 100 sentences and A/B the LoRA output against both the VoxCPM2 base "
        "clips and the Higgs TTS 3 clips."],
       ["6", "Judge by blind listening",
        "Per §6.6 — multi-listener A/B plus a transcription task. No automatic metric in this "
        "project can rank Khmer TTS."]],
      widths=[0.25, 1.6, 3.1])

# =========================================================================
heading(doc, "8.  Sources", 1, page_break=True)

heading(doc, "VoxCPM2", 2)
for s in [
    "Zhou et al., “VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life "
    "Voice Cloning”, arXiv:2509.24650",
    "openbmb/VoxCPM2 model card, Hugging Face — language list, 2M-hour claim, 48 kHz output",
    "OpenBMB/VoxCPM on GitHub — benchmarks and install instructions",
    "VoxCPM Fine-Tuning Guide (voxcpm.readthedocs.io) — commands, YAML, VRAM, manifest schema",
    "VoxCPM Fine-Tuning FAQ (voxcpm.readthedocs.io) — data volumes, forgetting, failure modes",
    "config.json from the openbmb/VoxCPM2 checkpoint — every architecture figure in §3.1–3.2",
    "voxcpm 2.0.3 package source: training/validate.py, training/data.py, modules/layers/lora.py, cli.py",
]:
    bullet(doc, s, size=9.5)

heading(doc, "Higgs TTS 3", 2)
for s in [
    "bosonai/higgs-tts-3-4b model card, Hugging Face — architecture table, 102-language tiers, "
    "control tokens, licence and Creator Use Grant",
    "Boson AI — “Higgs TTS 3” blog post (boson.ai/blog/higgs-audio-v3-tts)",
    "LMSYS — “Higgs Audio v3 TTS on SGLang-Omni” (lmsys.org, 4 June 2026)",
    "boson-ai/higgs-audio on GitHub — repository contents, verified to contain no training code",
    "config.json from multimodalart/higgs-audio-v3-tts-4b-transformers — every backbone figure in §4.2",
    "config.json from bosonai/higgs-audio-v2-tokenizer — semantic and acoustic branch specifications",
    "modeling_higgs_multimodal_qwen3.py — delay pattern, BOC/EOC ids, prompt format, fused embedding "
    "and head, and the absence of a loss-returning forward()",
    "JimmyMa99/train-higgs-audio on GitHub — community LoRA trainer for Higgs Audio v2 (not v3)",
]:
    bullet(doc, s, size=9.5)

heading(doc, "Data and evaluation", 2)
for s in [
    "Pratap et al., “Scaling Speech Technology to 1,000+ Languages” (MMS paper), JMLR 2024, "
    "arXiv:2305.13516",
    "OpenSLR SLR42 — High quality TTS data for Khmer (openslr.org/42)",
    "seanghay/KLEA — Khmer word-to-speech model",
    "seanghay/khmertagger — inverse text normalisation and segmentation for Khmer",
    "Ito & Johnson — the LJSpeech dataset",
    "Seed-TTS-eval overview, EvalScope documentation",
    "Hugging Face TTS-Arena (huggingface.co/spaces/TTS-AGI/TTS-Arena)",
    "Artificial Analysis — Text-to-Speech benchmarking methodology",
    "Zilliz — standard evaluation metrics for TTS quality",
]:
    bullet(doc, s, size=9.5)

heading(doc, "In-house evidence", 2)
for s in [
    "eval-set/eval.json — the fixed 100-sentence Khmer test set (50 pure Khmer, 50 code-switched)",
    "evaluation/results_report.md and evaluation/results/*/scores.json — the full four-model, "
    "400-clip synthesis and scoring run, including raw Whisper transcripts",
    "docs/09-voxcpm2-architecture-and-training.md and docs/10-higgs-tts-3-architecture-and-training.md "
    "— the working notes this review consolidates",
]:
    bullet(doc, s, size=9.5)

_sp = doc.add_paragraph()
_sp.paragraph_format.space_after = Pt(2)
_sp.paragraph_format.line_spacing = 0.6
rich(doc, [("Prepared for Smean AI  ·  ", {"size": 8.5, "color": STONE}),
           ("ស្មៀន", {"size": 8.5, "color": TEAL, "bold": True}),
           ("  ·  Built by Cambodians, for every Khmer speaker.", {"size": 8.5, "color": STONE})],
     align=WD_ALIGN_PARAGRAPH.CENTER)

doc.save(OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
