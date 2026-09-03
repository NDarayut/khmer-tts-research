"""
CER via Whisper-large-v3 ASR -- the primary correctness metric (CLAUDE.md,
docs/03 section 3.2).

Uses faster-whisper (CTranslate2) rather than openai-whisper: same large-v3
weights, several times faster and a much lighter install. transcribe() is the
only place the ASR library appears, so swapping back is a one-function change.

Decoding is pinned for determinism: language="km" (never let it auto-detect --
Khmer is easy to mistake for Thai/Lao and a wrong tag would tank every model
equally but noisily), temperature 0.0, no fallback sampling.

READ THIS BEFORE TRUSTING A NUMBER: docs/03 section 3.4 warns that ASR-based
CER is only as good as the ASR. Whisper-large-v3's own Khmer accuracy is
limited, so there is a nonzero CER floor even for perfect audio. That is why
every raw transcript is persisted -- when a sentence scores badly, read the
transcript before blaming the TTS. Comparisons between the 3 models are sound
(same ASR, same set); the absolute value is not.
"""

from .khmer_text import cer as char_error_rate
from .khmer_text import normalize

MODEL_SIZE = "large-v3"
LANGUAGE = "km"


def load_asr(device="cpu", model_size=MODEL_SIZE, compute_type=None):
    """Load Whisper once and reuse it for every utterance."""
    from faster_whisper import WhisperModel

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(asr, audio, sample_rate=16000):
    """-> raw transcript string. `audio` must be 16 kHz float32 mono."""
    if sample_rate != 16000:
        raise ValueError(
            f"Whisper expects 16 kHz audio, got {sample_rate} -- resample with "
            "common.resample() first."
        )
    segments, _info = asr.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def score(asr, audio, reference, sample_rate=16000):
    """-> per-utterance CER detail, including the raw transcript."""
    hypothesis = transcribe(asr, audio, sample_rate)
    return {
        "cer": char_error_rate(reference, hypothesis),
        "transcript": hypothesis,
        "norm_reference": normalize(reference),
        "norm_hypothesis": normalize(hypothesis),
    }
