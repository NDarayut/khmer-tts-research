"""
Generators for the company-facing documents in reports/.

    build_literature_review.py     -> reports/Smean-TTS-Literature-Review.docx
    build_style_control_report.py  -> reports/Speech-Control-VoxCPM2.docx
    academic_docx.py               shared layout primitives (a library, not a script)

The .docx files are generated, never hand-edited. Figures are read from the
evaluation results json rather than retyped.
"""
