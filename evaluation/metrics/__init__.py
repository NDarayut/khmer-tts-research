"""
Metric implementations for the TTS comparison.

    khmer_text  Khmer normalization + edit distance (stdlib only)
    cer         Whisper-large-v3 transcription -> character error rate
    utmos       predicted naturalness MOS
    dnsmos      P.835/P.808 perceptual quality

RTF is not here: it can only be measured while synthesizing, so it is produced
by synthesize.py and carried through by score.py.

Submodules are imported by name where used (never `from .metrics import *`),
so that a missing onnxruntime does not stop you from scoring CER.
"""

METRIC_KEYS = ("cer", "utmos", "dnsmos")
