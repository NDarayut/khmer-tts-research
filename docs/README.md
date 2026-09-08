# Documentation

Written research behind the Khmer TTS model comparison. The company-facing
deliverables built from it live in [`../reports/`](../reports/).

| Folder | What's in it |
|---|---|
| [`research/`](research/) | The numbered research survey, 01–09. Written first; 01–08 are the original pass, 09 is an addendum added when Higgs Audio v3 shipped. |
| [`deep-dives/`](deep-dives/) | Per-model architecture and training write-ups (09–11), written after the listening evaluation narrowed the field to two contenders. |
| [`../src/docbuild/`](../src/docbuild/) | Generators that produce the `.docx` reports in `../reports/`. They live under `src/` with the rest of the code. |
| [`assets/`](assets/) | Images used by the generators (the Smean logo). |

The two series both start at 09 because each keeps its own internal section
numbering (`9.1`, `9.2`, …) that matches its filename. Renumbering the files
would desync every heading and cross-reference inside them, so the folders keep
them apart instead.

## Research survey — `research/`

| # | Document |
|---|---|
| 01 | [How TTS Works](research/01-tts-overview.md) |
| 02 | [What Data TTS Needs](research/02-training-data-requirements.md) |
| 03 | [How TTS Is Benchmarked](research/03-evaluation-benchmarking.md) |
| 04 | [Model: Meta MMS-TTS (khm)](research/04-model-meta-mms-tts.md) |
| 05 | [Model: Fish Audio S2 / S2-Pro](research/05-model-fish-audio-s2.md) |
| 06 | [Model: VoxCPM2](research/06-model-voxcpm2.md) |
| 07 | [Other Models Checked (No Khmer Support)](research/07-other-models-checked.md) |
| 08 | [Khmer-Specific Considerations & Recommendation](research/08-khmer-considerations-and-recommendation.md) |
| 09 | [Model: Higgs Audio v3](research/09-model-higgs-audio-v3.md) |

## Deep dives — `deep-dives/`

| # | Document |
|---|---|
| 09 | [VoxCPM2 — Architecture & Training](deep-dives/09-voxcpm2-architecture-and-training.md) |
| 10 | [Higgs TTS 3 — Architecture & Training](deep-dives/10-higgs-tts-3-architecture-and-training.md) |
| 11 | [VoxCPM2 — Style-Control Fine-Tune](deep-dives/11-voxcpm2-style-control-finetune.md) |

## Building the reports

```
python src/docbuild/build_literature_review.py     # -> reports/Smean-TTS-Literature-Review.docx
python src/docbuild/build_style_control_report.py  # -> reports/Speech-Control-VoxCPM2.docx
```

`src/docbuild/academic_docx.py` is a shared library of layout primitives, not a script.
PDF conversion is done separately with LibreOffice:

```
soffice --headless --convert-to pdf --outdir reports reports/Smean-TTS-Literature-Review.docx
```
