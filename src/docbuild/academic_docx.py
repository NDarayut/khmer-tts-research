"""
Layout primitives for the house report style in .docx.

The full specification is docs/report-style.md; this module is its
implementation, and the two are meant to be changed together. In short: an
academic body (Cambria, justified, booktabs tables with captions above them,
a running page number in the footer) carrying one accent, the project teal,
which appears on the cover and as the bar behind every level-1 heading.

Headings and the cover are set in Georgia, everything else in Cambria. The
reports are generated, never hand-edited -- a change made in Word is lost on
the next build, so it belongs here instead.

Self-contained; build_literature_review.py carries its own, separate copies of
equivalent helpers and is unaffected by changes here.
"""

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement, parse_xml
from docx.shared import Inches, Pt, RGBColor

BLACK = "111111"
GREY = "555555"           # notes and the page-number footer
CAPTION_GREY = "7F7F7F"   # figure captions and the cover attribution
RULE = "999999"
FAINT = "D9D9D9"
WHITE = "FFFFFF"
ACCENT = "0B776F"         # the project teal; same value as TEAL in build_literature_review.py

BODY_FONT = "Cambria"
HEAD_FONT = "Georgia"
MONO_FONT = "Courier New"   # metric-available everywhere; Consolas is Windows-only
KHMER_FONT = "Khmer OS System"

BODY_SIZE = 10.5


def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), v)
    return e


def set_font(run, name=BODY_FONT, size=None, bold=None, italic=None,
             color=BLACK, caps=False):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = _el("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    # Khmer is a complex script; Consolas has no glyphs for it
    rFonts.set(qn("w:cs"), KHMER_FONT if name == MONO_FONT else name)
    if size is not None:
        run.font.size = Pt(size)
        rPr.append(_el("w:szCs", **{"w:val": str(int(round(size * 2)))}))
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if caps:
        rPr.append(_el("w:smallCaps", **{"w:val": "1"}))
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def inline(p, text, size=BODY_SIZE, color=BLACK, bold_all=False, italic_all=False,
           font=BODY_FONT):
    """Render **bold**, *italic* and `code` markers into runs on paragraph `p`."""
    for i, seg in enumerate(str(text).split("**")):
        if not seg:
            continue
        bold = bold_all or i % 2 == 1
        for j, piece in enumerate(seg.split("`")):
            if not piece:
                continue
            if j % 2 == 1:
                set_font(p.add_run(piece), MONO_FONT, size - 1.0, bold, italic_all, color)
                continue
            for k, bit in enumerate(piece.split("*")):
                if bit:
                    set_font(p.add_run(bit), font, size, bold,
                             italic_all or k % 2 == 1, color)
    return p


def para(doc, text="", size=BODY_SIZE, space_after=7, space_before=0,
         indent=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.24):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = spacing
    if indent is not None:
        pf.left_indent = Inches(indent)
    if align is not None:
        p.alignment = align
    if text:
        inline(p, text, size)
    return p


def bullets(doc, items, size=BODY_SIZE, indent=0.3):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        pf = p.paragraph_format
        pf.left_indent = Inches(indent + 0.22)
        pf.first_line_indent = Inches(-0.22)
        pf.space_after = Pt(4)
        pf.line_spacing = 1.24
        inline(p, item, size)


def numbered(doc, items, size=BODY_SIZE, indent=0.3):
    for n, item in enumerate(items, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Inches(indent + 0.26)
        pf.first_line_indent = Inches(-0.26)
        pf.space_after = Pt(4)
        pf.line_spacing = 1.24
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(p.add_run(f"({n})  "), BODY_FONT, size)
        inline(p, item, size)


SIZES = {1: 14, 2: 12, 3: 10.8}


def heading(doc, text, level=1, page_break=False, shaded=None):
    """A section heading. Level 1 is reversed out of a teal bar by default;
    pass shaded=False for the one that is not (Contents, set by contents())."""
    if shaded is None:
        shaded = level == 1
    if page_break:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt({1: 2, 2: 15, 3: 11}[level])
    pf.space_after = Pt({1: 9, 2: 5, 3: 4}[level])
    pf.keep_with_next = True
    if shaded:
        shade(p, ACCENT)
    set_font(p.add_run(text), HEAD_FONT, SIZES[level], True, level == 3,
             WHITE if shaded else BLACK)
    return p


def rule(paragraph, edge="bottom", size=6, color=RULE, space=6):
    pPr = paragraph._p.get_or_add_pPr()
    bdr = pPr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = _el("w:pBdr")
        pPr.append(bdr)
    bdr.append(_el(f"w:{edge}", **{"w:val": "single", "w:sz": str(size),
                                   "w:space": str(space), "w:color": color}))


def shade(paragraph, color=ACCENT):
    """Fill the paragraph's full column width -- the bar behind a level-1 heading."""
    pPr = paragraph._p.get_or_add_pPr()
    pPr.append(_el("w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": color}))


def cell_rule(cell, edge, size=6, color=RULE):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = _el("w:tcBorders")
        tcPr.append(borders)
    borders.append(_el(f"w:{edge}", **{"w:val": "single", "w:sz": str(size),
                                       "w:space": "0", "w:color": color}))


_TABLE_N = [0]


def table(doc, caption_text, headers, rows, widths=None, size=8.8,
          align_right=(), align_centre=()):
    """A booktabs-style table: caption above, three horizontal rules, no grid."""
    _TABLE_N[0] += 1
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(9)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.line_spacing = 1.18
    set_font(cap.add_run(f"Table {_TABLE_N[0]}. "), BODY_FONT, size + 0.4, True, False, BLACK)
    inline(cap, caption_text, size + 0.4, BLACK)

    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False

    def write(cell, text, *, bold=False, sz=size, align=None, top_pad=2.6):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        cell.paragraphs[0].text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(top_pad)
        p.paragraph_format.space_after = Pt(top_pad)
        p.paragraph_format.line_spacing = 1.1
        if align is not None:
            p.alignment = align
        inline(p, text, sz, BLACK, bold_all=bold)

    # repeat the header when the table breaks across a page
    trPr = t.rows[0]._tr.get_or_add_trPr()
    trPr.append(_el("w:tblHeader", **{"w:val": "true"}))
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        write(c, h, bold=True)
        cell_rule(c, "top", 10)
        cell_rule(c, "bottom", 6)
    for r, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            al = (WD_ALIGN_PARAGRAPH.RIGHT if i in align_right else
                  WD_ALIGN_PARAGRAPH.CENTER if i in align_centre else None)
            write(cells[i], val, align=al)
            if r == len(rows) - 1:
                cell_rule(cells[i], "bottom", 10)

    if widths:
        total = sum(widths)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(6.4 * w / total)
    tail = doc.add_paragraph()
    tail.paragraph_format.space_after = Pt(6)
    return t


def note(doc, text, size=8.4):
    """A small-print note under a table or figure."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.16
    inline(p, text, size, GREY)
    return p


def code(doc, text, size=7.6):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.4)
    cell_rule(cell, "left", 8, FAINT)
    cell.paragraphs[0].text = ""
    first = True
    for line in text.split("\n"):
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(1.2)
        p.paragraph_format.space_after = Pt(1.2)
        p.paragraph_format.left_indent = Inches(0.14)
        p.paragraph_format.line_spacing = 1.06
        set_font(p.add_run(line if line else " "), MONO_FONT, size, False, False, BLACK)
    tail = doc.add_paragraph()
    tail.paragraph_format.space_after = Pt(8)
    return t


def reference(doc, text, size=9.4):
    """A hanging-indent bibliography entry. *…* marks italics."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(6)
    pf.left_indent = Inches(0.3)
    pf.first_line_indent = Inches(-0.3)
    pf.line_spacing = 1.16
    inline(p, text, size)
    return p


# A 6 pt teal line, 8 inches tall, anchored just left of the text column and
# starting an inch down: the one piece of furniture on the cover. Written as
# raw DrawingML because python-docx models pictures but not shapes.
_COVER_RULE = """
<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
           xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
           xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
  <wp:anchor distT="0" distB="0" distL="114300" distR="114300" simplePos="0"
             relativeHeight="251659264" behindDoc="0" locked="0" layoutInCell="1"
             allowOverlap="1">
    <wp:simplePos x="0" y="0"/>
    <wp:positionH relativeFrom="column"><wp:posOffset>-342265</wp:posOffset></wp:positionH>
    <wp:positionV relativeFrom="paragraph"><wp:posOffset>890270</wp:posOffset></wp:positionV>
    <wp:extent cx="0" cy="7315200"/>
    <wp:effectExtent l="38100" t="0" r="38100" b="38100"/>
    <wp:wrapNone/>
    <wp:docPr id="1" name="Cover rule"/>
    <wp:cNvGraphicFramePr/>
    <a:graphic>
      <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
        <wps:wsp>
          <wps:cNvCnPr/>
          <wps:spPr>
            <a:xfrm flipH="1"><a:off x="0" y="0"/><a:ext cx="0" cy="7315200"/></a:xfrm>
            <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
            <a:ln w="76200"><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:ln>
          </wps:spPr>
          <wps:bodyPr/>
        </wps:wsp>
      </a:graphicData>
    </a:graphic>
  </wp:anchor>
</w:drawing>
""" % ACCENT

TITLE_SIZE = 36
COVER_SPACING = 1.5


def _cover_para(doc):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = COVER_SPACING
    p.paragraph_format.space_after = Pt(0)
    return p


def title_page(doc, *, title, prepared_by, date, top=58, gap=287):
    """Left-ranged cover: the title in teal beside a vertical teal rule, the
    attribution set at the foot. `title` may carry newlines; each line is its
    own paragraph, so a deliberate break is kept and the rest wraps.

    `top` and `gap` are measured spaces in points -- above the title, and
    between the title block and the attribution -- rather than runs of empty
    paragraphs, so that the count of them does not decide whether the cover
    spills onto a second page. A title that wraps to more or fewer lines than
    this one wants `gap` retuned so the attribution still sits at the foot.
    """
    p = _cover_para(doc)
    p.add_run()._r.append(parse_xml(_COVER_RULE))
    set_font(p.add_run(), HEAD_FONT, TITLE_SIZE)

    for i, line in enumerate(str(title).splitlines()):
        p = _cover_para(doc)
        if i == 0:
            p.paragraph_format.space_before = Pt(top)
        set_font(p.add_run(line), HEAD_FONT, TITLE_SIZE, False, False, ACCENT)

    p = _cover_para(doc)
    p.paragraph_format.space_before = Pt(gap)
    set_font(p.add_run("Prepared by "), HEAD_FONT, 9.5, False, False, CAPTION_GREY)
    set_font(p.add_run(prepared_by), HEAD_FONT, 9.5, True, False, CAPTION_GREY)
    set_font(_cover_para(doc).add_run(date), HEAD_FONT, 9.5,
             False, False, CAPTION_GREY)


_FIGURE_N = [0]


def figure(doc, image_path, caption_text, width=6.37):
    """A full-width figure. Unlike a table, its caption sits below it."""
    _FIGURE_N[0] += 1
    doc.add_picture(str(image_path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(10)
    set_font(cap.add_run(f"Figure {_FIGURE_N[0]}: {caption_text}"), BODY_FONT,
             None, False, True, CAPTION_GREY)
    return cap


def contents(doc, entries):
    """entries: (level, number, title, page). Dot leaders to a right tab stop."""
    heading(doc, "Contents", 1, page_break=True, shaded=False)
    for level, num, text, page in entries:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(3 if level > 1 else 6)
        pf.space_before = Pt(6 if level == 1 else 0)
        pf.left_indent = Inches(0.0 if level == 1 else 0.34)
        pf.line_spacing = 1.1
        tabs = pf.tab_stops
        tabs.add_tab_stop(Inches(6.4), WD_ALIGN_PARAGRAPH.RIGHT, 1)  # 1 = dotted
        label = f"{num}\t" if not num else f"{num}   "
        bold = level == 1
        sz = 10.6 if level == 1 else 10.0
        if num:
            r = set_font(p.add_run(label), BODY_FONT, sz, bold, False, BLACK)
        set_font(p.add_run(text), BODY_FONT, sz, bold, False, BLACK)
        set_font(p.add_run("\t"), BODY_FONT, sz, bold, False, BLACK)
        set_font(p.add_run(str(page)), BODY_FONT, sz, bold, False, BLACK)


def _field(paragraph, instr):
    r = paragraph.add_run()
    r._r.append(_el("w:fldChar", **{"w:fldCharType": "begin"}))
    t = OxmlElement("w:instrText")
    t.set(qn("xml:space"), "preserve")
    t.text = instr
    r._r.append(t)
    r._r.append(_el("w:fldChar", **{"w:fldCharType": "separate"}))
    r._r.append(_el("w:fldChar", **{"w:fldCharType": "end"}))
    set_font(r, BODY_FONT, 9.5, False, False, GREY)
    return r


def new_document(*, page_numbers=True):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(BODY_SIZE)
    st.font.color.rgb = RGBColor.from_string(BLACK)
    st.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
    st.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
    st.element.rPr.rFonts.set(qn("w:cs"), KHMER_FONT)
    st.paragraph_format.line_spacing = 1.24
    st.paragraph_format.space_after = Pt(7)

    s = doc.sections[0]
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.05)
    s.right_margin = Inches(1.05)

    if page_numbers:
        # numbered from the contents onwards; the title page carries nothing
        s.different_first_page_header_footer = True
        f = s.footer.paragraphs[0]
        f.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _field(f, " PAGE ")
        s.first_page_footer.paragraphs[0].text = ""
    return doc
