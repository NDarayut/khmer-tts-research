"""
DNSMOS P.835 / P.808 -- no-reference perceptual quality (docs/03 section 3.2).

A port of Microsoft's reference `dnsmos_local.py` onto onnxruntime, keeping
its constants and polynomial mappings exactly. Returns four numbers per
utterance, all on the 1-5 MOS scale, higher is better:

    sig   speech signal quality (P.835)
    bak   background-noise intrusiveness (P.835)
    ovrl  overall quality (P.835)  <- the headline DNSMOS number
    p808  overall quality from the older P.808 model

DNSMOS was built to rate speech-enhancement output, and is reused as a TTS
quality check. For clean synthetic speech `bak` sits near its ceiling and
carries little signal; `sig` and `ovrl` are the ones to read. It is
complementary to UTMOS, not a substitute: UTMOS predicts naturalness/prosody,
DNSMOS detects artifacts, buzz and distortion.

*** MODEL FILES ARE NOT INCLUDED AND ARE NOT DOWNLOADED AUTOMATICALLY. ***
Fetch the two .onnx files once, from microsoft/DNS-Challenge under
DNSMOS/DNSMOS/, and put them in evaluation/dnsmos_models/:

    sig_bak_ovr.onnx    (P.835 -- sig/bak/ovrl)
    model_v8.onnx       (P.808 -- overall)

    git clone --depth 1 https://github.com/microsoft/DNS-Challenge
    copy DNS-Challenge\\DNSMOS\\DNSMOS\\sig_bak_ovr.onnx  evaluation\\dnsmos_models\\
    copy DNS-Challenge\\DNSMOS\\DNSMOS\\model_v8.onnx     evaluation\\dnsmos_models\\

Or point --dnsmos-dir / DNSMOS_MODEL_DIR somewhere else.
"""

import os
from pathlib import Path

SAMPLE_RATE = 16000
INPUT_LENGTH = 9.01  # seconds per analysis window, from the reference script

P835_FILE = "sig_bak_ovr.onnx"
P808_FILE = "model_v8.onnx"

# Mel hop for the P.808 features, from the reference script. Also the amount
# trimmed off a window before those features are computed -- see score().
MEL_HOP_LENGTH = 160

ENV_DIR = "DNSMOS_MODEL_DIR"

# Polynomial mappings from raw model output to MOS, reference implementation,
# non-personalized ("without personalized MOS") variant.
P_SIG = (-0.08397278, 1.22083953, 0.0052439)
P_BAK = (-0.13166888, 1.60915514, -0.39604546)
P_OVR = (-0.06766283, 1.11546468, 0.04602535)


def resolve_model_dir(dnsmos_dir=None):
    from ..common import DNSMOS_DIR

    if dnsmos_dir:
        return Path(dnsmos_dir).expanduser().resolve()
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return DNSMOS_DIR


def load_dnsmos(dnsmos_dir=None, device="cpu"):
    """-> a session pair to hand to score(). Raises with instructions if the
    .onnx files are missing."""
    import onnxruntime as ort

    model_dir = resolve_model_dir(dnsmos_dir)
    p835_path = model_dir / P835_FILE
    p808_path = model_dir / P808_FILE

    missing = [p.name for p in (p835_path, p808_path) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"DNSMOS model file(s) missing from {model_dir}: "
            f"{', '.join(missing)}\n"
            "They are not downloaded automatically. Get them from "
            "https://github.com/microsoft/DNS-Challenge under DNSMOS/DNSMOS/ "
            "-- see this module's docstring for the exact commands, or pass "
            "--dnsmos-dir to point at a copy you already have."
        )

    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return {
        "p835": ort.InferenceSession(str(p835_path), providers=providers),
        "p808": ort.InferenceSession(str(p808_path), providers=providers),
    }


def _polyfit(coeffs, value):
    a, b, c = coeffs
    return a * value * value + b * value + c


def _melspec(audio, n_mels=120, frame_size=320, hop_length=MEL_HOP_LENGTH):
    """Log-mel features for the P.808 model -- reference parameters."""
    import librosa
    import numpy as np

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=frame_size + 1,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
    return mel.T


def score(sessions, audio, sample_rate=SAMPLE_RATE):
    """-> {sig, bak, ovrl, p808} averaged over 9.01 s windows at a 1 s hop,
    matching the reference implementation."""
    import numpy as np

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"DNSMOS expects {SAMPLE_RATE} Hz audio, got {sample_rate} -- "
            "resample with common.resample() first."
        )

    audio = np.asarray(audio, dtype=np.float32)
    window_samples = int(INPUT_LENGTH * SAMPLE_RATE)
    # Reference behaviour: loop short clips until they fill one window. Most
    # eval-set sentences are well under 9 s, so this is the normal path.
    while len(audio) < window_samples:
        audio = np.append(audio, audio)

    hop_samples = SAMPLE_RATE
    num_hops = int(np.floor(len(audio) / SAMPLE_RATE) - INPUT_LENGTH) + 1
    sigs, baks, ovrls, p808s = [], [], [], []

    for idx in range(max(num_hops, 1)):
        start = int(idx * hop_samples)
        segment = audio[start : start + window_samples]
        if len(segment) < window_samples:
            continue

        p835_input = {"input_1": segment[np.newaxis, :].astype(np.float32)}
        # The P.808 model's mel input is a fixed 900 frames. A full 9.01 s
        # window yields 901, so the reference implementation drops one hop
        # before computing the features -- keep that, or onnxruntime rejects
        # the input outright ("Got: 901 Expected: 900").
        p808_input = {
            "input_1": _melspec(segment[:-MEL_HOP_LENGTH])[np.newaxis, :, :].astype(
                np.float32
            )
        }

        raw_sig, raw_bak, raw_ovr = sessions["p835"].run(None, p835_input)[0][0]
        p808_mos = sessions["p808"].run(None, p808_input)[0][0][0]

        sigs.append(_polyfit(P_SIG, raw_sig))
        baks.append(_polyfit(P_BAK, raw_bak))
        ovrls.append(_polyfit(P_OVR, raw_ovr))
        p808s.append(float(p808_mos))

    if not sigs:
        return {"sig": None, "bak": None, "ovrl": None, "p808": None}

    return {
        "sig": float(np.mean(sigs)),
        "bak": float(np.mean(baks)),
        "ovrl": float(np.mean(ovrls)),
        "p808": float(np.mean(p808s)),
    }
