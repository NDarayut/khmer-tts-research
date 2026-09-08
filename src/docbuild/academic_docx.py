"""
Layout primitives for a plain academic report in .docx.

Deliberately unbranded: no logo, no house colours, no marketing furniture. A
serif face, black text, rules instead of filled table headers, numbered tables
with captions above them, and a running page number in the footer.

Self-contained; build_literature_review.py carries its own, separate copies of
equivalent helpers and is unaffected by changes here.
"""

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

BLACK = "111111"
GREY = "555555"
RULE = "999999"
FAINT = "D9D9D9"

BODY_FONT = "Cambria"
HEAD_FONT = "Cambria"
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


SIZES = {1: 13.5, 2: 11.5, 3: 10.8}


def heading(doc, text, level=1, page_break=False):
    if page_break:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt({1: 2, 2: 15, 3: 11}[level])
    pf.space_after = Pt({1: 9, 2: 5, 3: 4}[level])
    pf.keep_with_next = True
    set_font(p.add_run(text), HEAD_FONT, SIZES[level], True, level == 3, BLACK)
    return p


def rule(paragraph, edge="bottom", size=6, color=RULE, space=6):
    pPr = paragraph._p.get_or_add_pPr()
    bdr = pPr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = _el("w:pBdr")
        pPr.append(bdr)
    bdr.append(_el(f"w:{edge}", **{"w:val": "single", "w:sz": str(size),
                                   "w:space": str(space), "w:color": color}))


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


def title_page(doc, *, title, subtitle, prepared_by, date):
    """Title block set high on the page, attribution set at the foot of it."""
    for _ in range(4):
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.line_spacing = 1.22
    set_font(p.add_run(title), HEAD_FONT, 21, True, False, BLACK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.3
    set_font(p.add_run(subtitle), BODY_FONT, 12, False, True, GREY)

    # one measured gap rather than a run of empty paragraphs, so the block sits
    # low on the page without the count of them deciding whether it spills over
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(378)
    p.paragraph_format.space_after = Pt(3)
    rule(p, "top", 6, RULE, 10)
    set_font(p.add_run("Prepared by"), BODY_FONT, 9.5, False, False, GREY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(9)
    set_font(p.add_run(prepared_by), BODY_FONT, 11.5, True, False, BLACK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    set_font(p.add_run(date), BODY_FONT, 10, False, False, GREY)


def contents(doc, entries):
    """entries: (level, number, title, page). Dot leaders to a right tab stop."""
    heading(doc, "Contents", 1, page_break=True)
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
