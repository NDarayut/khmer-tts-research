"""
UTMOS -- predicted naturalness MOS (docs/03 section 3.2).

A neural MOS predictor: a fast, free stand-in for a human listening study.
Output is on the 1-5 MOS scale, higher is better.

Implementation: the utmos22_strong checkpoint via tarepan/SpeechMOS on
torch.hub. Chosen over the original sarulab-speech/UTMOS22 repo because that
one requires fairseq, which does not pin cleanly on Python 3.11 / Windows.
Same checkpoint, same numbers.

The checkpoint is fetched to the torch hub cache on first use, so the first
call needs network access. Expects 16 kHz mono.

Caveat worth remembering when reading the report: UTMOS was trained on MOS
studies of mostly English (and some Japanese) synthesized speech. It rates
acoustic quality and prosodic naturalness, and it has never heard a Khmer
judgement -- so it is a fair *relative* signal between the 3 models here, not
a calibrated absolute Khmer MOS.
"""

HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
HUB_MODEL = "utmos22_strong"
SAMPLE_RATE = 16000


def load_utmos(device="cpu"):
    """Load the predictor once and reuse it across utterances."""
    import torch

    try:
        model = torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True)
    except Exception as exc:  # network, cache miss, hub API change
        raise RuntimeError(
            f"Could not load UTMOS from torch.hub ({HUB_REPO}, {HUB_MODEL}): "
            f"{exc}\nThe checkpoint downloads on first use -- this needs "
            "network access once, after which it is cached in the torch hub "
            "directory."
        ) from exc
    model = model.to(device)
    model.eval()
    return model


def score(model, audio, sample_rate=SAMPLE_RATE, device="cpu"):
    """-> predicted MOS in [1, 5] for one 16 kHz mono utterance."""
    import torch

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"UTMOS expects {SAMPLE_RATE} Hz audio, got {sample_rate} -- "
            "resample with common.resample() first."
        )
    wave = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(wave, sample_rate)
    return float(prediction.squeeze().cpu())
