"""
Meta MMS-TTS, Khmer checkpoint -- facebook/mms-tts-khm (see docs/04).

36M-param single-speaker VITS, available through HF Transformers' VitsModel
since v4.33. Runs fine on CPU; 16 kHz output (read off model.config).

Reproducibility note: VITS has a *stochastic* duration predictor, so the same
text yields different prosody (and different audio length -- so different RTF)
on every call unless the RNG is pinned. We therefore reseed torch immediately
before each utterance, using seed + a stable hash of the text, so a given
(seed, sentence) always produces the same waveform regardless of which subset
of the eval set was run or in what order. Without this, a resumed or partial
run would not be comparable to a full one.
"""

import zlib

from .base import Backend, resolve_device

MODEL_ID = "facebook/mms-tts-khm"


def text_seed(seed, text):
    """Stable across processes -- unlike hash(), which Python salts per run
    (PYTHONHASHSEED), and which would silently break run-to-run comparability."""
    return (int(seed) + zlib.crc32(text.encode("utf-8"))) % (2 ** 31)


class MmsBackend(Backend):
    name = "mms"

    def __init__(self, device=None, seed=0, model_id=MODEL_ID, **_ignored):
        super().__init__(device=device, seed=seed)
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.device = resolve_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = VitsModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.native_sample_rate = int(self.model.config.sampling_rate)
        self._torch = torch

    def synthesize(self, text):
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # Pin the stochastic duration predictor -- see module docstring.
        torch.manual_seed(text_seed(self.seed, text))
        with torch.no_grad():
            output = self.model(**inputs).waveform
        audio = output.squeeze().detach().cpu().numpy()
        return audio, self.native_sample_rate

    def describe(self):
        info = {"backend": self.name, "model_id": self.model_id}
        try:
            import transformers

            info["transformers_version"] = transformers.__version__
        except ImportError:
            pass
        return info
