"""
The contract every TTS backend implements.

Deliberately tiny -- adding a 4th model means writing one module with a
`load()` and a `synthesize()`, and adding one line to backends/__init__.py.

Timing convention (matters for RTF):
  - `load()` is the heavy init (weights -> device). It is timed by
    synthesize.py and reported separately as `load_seconds`; it is NEVER
    part of RTF.
  - `synthesize()` is timed per utterance and IS the RTF numerator. It must
    therefore do only inference -- no lazy weight loading, no device moves.
"""


class Backend:
    #: registry key; also the results/<name>/ subdirectory
    name = None
    #: rate the model natively emits, filled in by load() when only the model
    #: knows it (MMS reads it off model.config)
    native_sample_rate = None

    def __init__(self, device=None, seed=0):
        self.device = device
        self.seed = seed

    def load(self):
        """Load weights onto the device. Called once, before any synthesis."""
        raise NotImplementedError

    def synthesize(self, text):
        """-> (float32 mono ndarray, sample_rate). Inference only."""
        raise NotImplementedError

    def describe(self):
        """Provenance for the synthesis.json `env` block -- model id, revision,
        library version. Overridden by each backend."""
        return {"backend": self.name}


def resolve_device(device=None):
    """'cuda' when asked for or available, else 'cpu'. Imports torch lazily so
    this module stays importable without it."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
