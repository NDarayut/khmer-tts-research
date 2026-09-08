"""
Shared plumbing for the TTS evaluation harness -- paths, eval-set loading,
wav io, resampling, and json helpers.

Imported by synthesize.py, score.py, report.py and the backends/metrics
packages. Third-party imports (soundfile, torch) are done lazily inside the
functions that need them, so this module stays importable in an environment
where nothing is installed yet.

Nothing here ever writes to eval-set/eval.json -- that set is fixed (CLAUDE.md).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "eval-set" / "eval.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
DNSMOS_DIR = ROOT / "evaluation" / "dnsmos_models"

# Every metric in this harness (Whisper, UTMOS, DNSMOS) consumes 16 kHz mono.
METRIC_SAMPLE_RATE = 16000

MODEL_KEYS = ("mms", "voxcpm2", "fish-s2")


# --- results layout ------------------------------------------------------

def model_dir(model):
    return RESULTS_DIR / model


def audio_dir(model):
    return model_dir(model) / "audio"


def wav_path(model, entry_id):
    return audio_dir(model) / f"{entry_id}.wav"


def rel_to_root(path):
    """Repo-relative, forward-slashed -- results json should be readable and
    diffable on any platform."""
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def synthesis_path(model):
    return model_dir(model) / "synthesis.json"


def scores_path(model):
    return model_dir(model) / "scores.json"


def scores_csv_path(model):
    return model_dir(model) / "scores.csv"


# --- eval set ------------------------------------------------------------

def load_entries():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def select_entries(entries, ids=None, groups=None, limit=None):
    """Filter the fixed set for smoke runs. Order is always preserved."""
    out = entries
    if ids:
        wanted = set(ids)
        out = [e for e in out if e.get("id") in wanted]
        missing = wanted - {e.get("id") for e in out}
        if missing:
            raise ValueError(f"unknown ids: {', '.join(sorted(missing))}")
    if groups:
        wanted = set(groups)
        out = [e for e in out if e.get("group") in wanted]
    if limit is not None:
        out = out[:limit]
    return out


# --- json ----------------------------------------------------------------

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """UTF-8, unescaped -- the payload is Khmer and \\u-escaping makes results
    unreadable in a diff or a text editor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --- audio ---------------------------------------------------------------

def write_wav(path, audio, sample_rate):
    """Write float32 mono at the model's native rate (no resampling here --
    the originals stay lossless; metrics resample in memory)."""
    import numpy as np
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = to_mono_float32(np.asarray(audio))
    sf.write(str(path), audio, int(sample_rate), subtype="PCM_16")


def read_wav(path):
    """-> (float32 mono ndarray, sample_rate)"""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_float32(np.asarray(audio)), sample_rate


def to_mono_float32(audio):
    import numpy as np

    audio = np.asarray(audio)
    if audio.ndim > 1:
        # (channels, samples) or (samples, channels) -- average the short axis
        axis = 0 if audio.shape[0] < audio.shape[-1] else -1
        audio = audio.mean(axis=axis)
    return np.ascontiguousarray(audio.astype(np.float32).reshape(-1))


def resample(audio, sample_rate, target_sample_rate=METRIC_SAMPLE_RATE):
    """Resample float32 mono. MMS emits 16 kHz, VoxCPM2 48 kHz, Fish S2
    44.1 kHz -- the metric models all want 16 kHz."""
    import numpy as np

    if int(sample_rate) == int(target_sample_rate):
        return to_mono_float32(audio)

    import torch
    import torchaudio

    tensor = torch.from_numpy(to_mono_float32(audio))
    out = torchaudio.functional.resample(
        tensor, int(sample_rate), int(target_sample_rate)
    )
    return np.ascontiguousarray(out.numpy().astype(np.float32))


def audio_seconds(audio, sample_rate):
    return len(audio) / float(sample_rate)


# --- small numeric helpers used by score.py / report.py -------------------

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0
