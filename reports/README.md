# Reports

Company-facing deliverables. These are the documents to hand over; everything
else in the repository is the working material behind them.

| Document | Formats | Source |
|---|---|---|
| **Smean TTS Literature Review** — the model survey and the 4-way Khmer benchmark, written up for a general reader | [`.docx`](Smean-TTS-Literature-Review.docx) · [`.pdf`](Smean-TTS-Literature-Review.pdf) | `src/docbuild/build_literature_review.py` |
| **Speech Control for Smean's TTS on VoxCPM2** — internal engineering memo, updated as experiments continue: what the model already does for Khmer via parenthetical prompting, and the tag-conditioning capability and generalization experiments. Findings only — no recommendation or next-steps section, by design | [`.docx`](Speech-Control-VoxCPM2.docx) · [`.pdf`](Speech-Control-VoxCPM2.pdf) | `src/docbuild/build_style_control_report.py` |

## These files are generated

Do not hand-edit them — the next build overwrites the change. Edit the
generator, then rebuild:

```
python src/docbuild/build_literature_review.py
python src/docbuild/build_style_control_report.py
```

The `.pdf` is produced from the `.docx` with LibreOffice:

```
soffice --headless --convert-to pdf --outdir reports reports/Smean-TTS-Literature-Review.docx
```

## `archive/`

Superseded drafts, kept for provenance and named by the date they were last
current. Nothing here should be sent to anyone.

| File | Superseded by |
|---|---|
| `2026-09-03-Text-to-Speech-Research-Smean.docx` | the Literature Review — this was the original single-document write-up |
| `2026-09-07-Smean-TTS-Literature-Review-v2.docx` | `../Smean-TTS-Literature-Review.docx`, which is regenerated from source |
