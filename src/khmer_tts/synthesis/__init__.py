"""
Turning the eval set into audio.

    synthesize.py  run one model over all 100 sentences -> wav + RTF
    backends/      one lazily-imported module per model

RTF is measured here because it cannot be recovered from a .wav afterwards.
"""
