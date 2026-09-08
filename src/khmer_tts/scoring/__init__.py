"""
Scoring the synthesized audio.

    score.py             UTMOS + DNSMOS (+ the stale Whisper CER column)
    score_cer_khmer.py   CER via a Khmer CTC ASR -- the primary metric
    score_naturalness.py UTMOS/DNSMOS under level and silence controls
    audio_stats.py       ASR-free signal diagnostics (unbiasable)
    prosody_stats.py     F0/energy/pause description -- NOT a naturalness score
    metrics/             the metric implementations themselves

Read the metrics section of CLAUDE.md before quoting any number from here:
CER is valid, UTMOS and DNSMOS are inverted for Khmer.
"""
