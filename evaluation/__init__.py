"""
Khmer TTS evaluation harness.

    describe.py    descriptive overview of the eval set   (stdlib, pre-existing)
    filter.py      QC pass over the eval set              (stdlib, pre-existing)
    synthesize.py  run a model over the eval set -> audio + RTF
    score.py       CER + UTMOS + DNSMOS over that audio
    report.py      3-model comparison markdown

See evaluation/README.md for the runbook. This package is a plain namespace --
importing it pulls in nothing third-party.
"""
