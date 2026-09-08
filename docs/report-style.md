# Report style

The house style for the deliverables in `reports/`. It is implemented by
`src/docbuild/academic_docx.py`; this file and that module are meant to change
together, and neither one is a description of the other after the fact.

The reports are **generated, never hand-edited**. A change made in Word is lost
on the next `python src/docbuild/build_style_control_report.py`, so a style
change belongs in `academic_docx.py` and a content change in the builder.

## The whole of it

Two typefaces and one accent. **Georgia** carries the cover and every heading;
**Cambria** carries everything else, including table and figure captions;
**Courier New** carries code and inline `literals` (with `Khmer OS System` as
the complex-script face, so Khmer inside a mono run still has glyphs). The
accent is the project teal `#0B776F` — the same value as `TEAL` in
`build_literature_review.py` — and it appears in exactly two places: the cover,
and the bar behind each level-1 heading. Nothing else is coloured.

| | |
|---|---|
| Page | US Letter, margins 1.0″ top/bottom, 1.05″ left/right |
| Body | Cambria 10.5 pt, `#111111`, justified, line spacing 1.24, 7 pt after |
| Footer | centred `PAGE` field, Cambria 9.5 pt, grey `#555555`; the cover carries none |

## Cover

Left-ranged, not centred. A 6 pt teal rule runs vertically down the left of the
page (8″ tall, set 0.37″ outside the text column, starting about an inch down);
the title sits beside it in **Georgia 36 pt teal**, line spacing 1.5, one
paragraph per line so an intended break survives and the rest wraps. At the
foot, `Prepared by ` plus the department in bold, then the date, in Georgia
grey `#7F7F7F`. There is no subtitle and no horizontal rule.

The vertical space above the title and between the title and the attribution is
set in points (`top` and `gap` on `title_page()`), not as runs of empty
paragraphs, so that the number of them cannot decide whether the cover spills
onto a second page. A title that wraps to a different number of lines than the
current one wants `gap` retuned so the attribution still lands at the foot.

## Headings

| Level | Setting |
|---|---|
| 1 | Georgia 14 pt bold, **white on a full-width teal bar**, 2 pt before / 9 pt after, kept with the next paragraph. Starts on a new page. |
| 2 | Georgia 12 pt bold, black, 15 pt before / 5 pt after |
| 3 | Georgia 10.8 pt bold italic, black, 11 pt before / 4 pt after |

`Contents` is the one deliberate exception: a level-1 heading in Georgia 14 pt
bold **black, with no bar** (`heading(..., shaded=False)`). Every other level-1
heading, References included, gets the bar.

## Tables, figures, notes, code, references

- **Tables** are booktabs: caption **above** as `Table N.` in bold followed by
  the caption text (Cambria 9.2 pt), then three rules — heavy above the header,
  light below it, heavy under the last row — and no grid or fill anywhere.
  Header rows repeat when a table breaks across a page.
- **Figures** invert that: the image sits at full text width, and the caption
  goes **below** it, centred, Cambria italic grey `#7F7F7F`, as
  `Figure N: caption`. Images live in `reports/assets/`.
- **Notes** under a table or figure: Cambria 8.4 pt, grey `#555555`.
- **Code** is Courier New 7.6 pt in a single borderless cell with a faint
  `#D9D9D9` rule down its left edge.
- **References** are Cambria 9.4 pt with a 0.3″ hanging indent.

## Prose conventions

- `**bold**`, `*italic*` and `` `code` `` are written as markers in the builder's
  strings and rendered by `inline()`.
- Where a run of paragraphs each takes up one finding or one item, the paragraph
  opens with the thing it is about, **bold and in Title Case**: "**Pitch** is
  controlled reliably…", "**Energy** and **Speaking Rate** are controlled in
  aggregate…". The bold covers the name only, not the sentence.

## Building

```
python src/docbuild/build_style_control_report.py     # -> reports/Speech-Control-VoxCPM2.docx
python src/docbuild/build_literature_review.py        # -> reports/Smean-TTS-Literature-Review.docx
```

The style-control build runs itself up to four times, converting to PDF between
passes to read back the page each heading landed on, so that the contents page
carries real page numbers. It shells out to `soffice` and `pdftotext` for that;
without LibreOffice on `PATH` the document still builds but the contents shows
`—` for every entry.

`build_literature_review.py` is a separate, branded document and carries its own
private copies of these helpers. It does not follow this file.
