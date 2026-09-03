"""
Khmer text normalization and edit distance for CER.

Stdlib only, on purpose: CER is the primary metric for this comparison
(CLAUDE.md), so its definition should be readable and pinned here rather than
inherited from whatever a `jiwer` upgrade decides today.

Why each normalization step exists -- all applied identically to the reference
sentence and to the ASR hypothesis:

  * NFC. Khmer stacks consonants with COENG (U+17D2) and its combining vowels
    can be encoded in more than one order; comparing un-normalized strings
    charges the TTS for a Unicode encoding difference.
  * Drop whitespace, ZWSP (U+200B) and NBSP. Khmer has no inter-word spacing
    -- Whisper inserts spaces where it guesses word boundaries, and those
    guesses are not TTS errors. (This is the same fact that makes word-level
    WER meaningless here, and why filter.py refuses to count words.)
  * Drop terminal/most punctuation. "។" is not pronounced; an ASR that emits
    "." or nothing for it has not misread the sentence.
  * Khmer digits ០-៩ -> ASCII. A model that reads "២០២៦" correctly should not
    be penalized because Whisper wrote back "2026".
  * Lowercase Latin, for the 50 code_switched sentences.

CER = levenshtein(ref, hyp) / len(ref), on the normalized strings. Not clipped
to 1.0 -- a model that babbles can legitimately exceed 100% CER, and hiding
that would make a broken model look merely bad.
"""

import re
import unicodedata

ZERO_WIDTH = "​‌‍﻿ "

# Khmer punctuation (khan, bariyoosan, camnuc pii kuuh, lek too, koomuut...)
# plus the ASCII/CJK punctuation Whisper tends to emit.
PUNCTUATION = "។៕៖៘៙៚៛ៗ!?,.;:\"'`()[]{}«»“”‘’—–-…‧、。，？！"

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
KHMER_DIGIT_MAP = {ord(k): str(i) for i, k in enumerate(KHMER_DIGITS)}

_STRIP_RE = re.compile(
    "[" + re.escape(PUNCTUATION + ZERO_WIDTH) + r"\s]+"
)


def normalize(text):
    """Canonical form used on both sides of the CER comparison."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(KHMER_DIGIT_MAP)
    text = _STRIP_RE.sub("", text)
    return text.lower()


def levenshtein(a, b):
    """Character edit distance, two-row DP -- O(len(a) * len(b)) time,
    O(min) space."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(reference, hypothesis):
    """Character error rate on normalized text. Returns None for an empty
    reference (no such entry exists in eval.json, but don't divide by zero)."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)
