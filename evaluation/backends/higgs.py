"""
Higgs TTS 3 -- Boson AI's 4B multilingual TTS.

Khmer is *undocumented*, not absent: the model card lists 100 language codes
and `km` is not among them, yet the model demonstrably produces Khmer speech
(verified by hand on the HF Space before this backend was written). So treat
it as a real entry in the comparison, with the caveat that Boson publishes no
Khmer quality claim -- there is no vendor number to sanity-check the result
against, unlike voxcpm2 and fish-s2.

Weights: the official checkpoint is `bosonai/higgs-tts-3-4b`, but its
architecture (`higgs_multimodal_qwen3`) is not in transformers 5.16.1 and the
repo ships no remote code, so it cannot be loaded from there. We use
multimodalart's trust_remote_code repackaging instead: same weights, copied
unchanged, plus a modeling/configuration pair and an auto_map. Needs
transformers >= 5.5.

Architecture, for reading the numbers: a Qwen3-4B autoregressive backbone over
interleaved text and audio tokens, with audio in 8 codebooks at 25 fps decoded
by the separate `bosonai/higgs-audio-v2-tokenizer` (fetched automatically on
first use). Emits 24 kHz.

Voice: the model's own default. No reference audio is passed, matching the
other backends -- a cloned voice would make UTMOS and DNSMOS incomparable.
"""

from .base import Backend, resolve_device
from .mms import text_seed

# Not bosonai/higgs-tts-3-4b -- see module docstring.
MODEL_ID = "multimodalart/higgs-audio-v3-tts-4b-transformers"
NATIVE_SAMPLE_RATE = 24000


class HiggsBackend(Backend):
    name = "higgs3"
    native_sample_rate = NATIVE_SAMPLE_RATE

    def __init__(self, device=None, seed=0, model_id=MODEL_ID, **_ignored):
        super().__init__(device=device, seed=seed)
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=torch.bfloat16
        )
        self.model = self.model.to(self.device).eval()
        rate = getattr(self.model.config, "sample_rate", None)
        if rate:
            self.native_sample_rate = int(rate)
        self._torch = torch

    def synthesize(self, text):
        torch = self._torch
        # Reseeded per utterance from (seed, sentence), exactly as in
        # backends/mms.py: generation samples, so without this a resumed run
        # would not reproduce a full one.
        torch.manual_seed(text_seed(self.seed, text))
        with torch.no_grad():
            wav = self.model.generate_speech(text, self.tokenizer)
        audio = wav.detach().float().cpu().numpy()
        return audio, self.native_sample_rate

    def describe(self):
        info = {
            "backend": self.name,
            "model_id": self.model_id,
            "base_model": "bosonai/higgs-tts-3-4b",
            "khmer_documented": False,  # works anyway -- see module docstring
        }
        try:
            import transformers

            info["transformers_version"] = transformers.__version__
        except ImportError:
            pass
        return info
