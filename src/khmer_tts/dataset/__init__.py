"""
The fixed evaluation set: describing it, and QC-ing it.

    describe.py  descriptive (non-QC) overview -> evaluation/dataset_overview.md
    filter.py    automated QC pass, flags never deletes -> evaluation/qc_report.md

Both are stdlib-only. Neither writes to eval-set/eval.json -- that set is fixed
across all four models so the comparison stays fair (CLAUDE.md).
"""
