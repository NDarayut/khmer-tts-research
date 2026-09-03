"""
VoxCPM2 -- openbmb/VoxCPM2 via the `voxcpm` package (see docs/06).

2B params, ~8 GB VRAM, needs CUDA; Python >=3.10 / torch >=2.5. Automatic
language detection, so no Khmer language tag is passed. Emits 48 kHz.

Voice: by default the model's own built-in voice, so the 3-model comparison
stays zero-config and no reference recording biases one model over another.
Pass --ref-audio/--ref-text to synthesize.py to exercise the zero-shot cloning
path instead; if you do, keep it identical across models or RTF and UTMOS stop
being comparable.

API-shape note: `voxcpm` is a young package. The generate() call is wrapped in
_call_generate() below so that if the installed version's signature differs,
there is exactly one place to adjust.
"""

from .base import Backend, resolve_device

MODEL_ID = "openbmb/VoxCPM2"
NATIVE_SAMPLE_RATE = 48000


class VoxCpm2Backend(Backend):
    name = "voxcpm2"
    native_sample_rate = NATIVE_SAMPLE_RATE

    def __init__(
        self,
        device=None,
        seed=0,
        model_id=MODEL_ID,
        ref_audio=None,
        ref_text=None,
        **_ignored,
    ):
        super().__init__(device=device, seed=seed)
        self.model_id = model_id
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.model = None

    def load(self):
        import torch
        from voxcpm import VoxCPM

        self.device = resolve_device(self.device)
        if self.device == "cpu":
            print(
                "  ! VoxCPM2 on CPU will be extremely slow and its RTF will not "
                "be comparable to a GPU run -- see docs/06 (needs CUDA >=12.0)."
            )
        torch.manual_seed(self.seed)
        self.model = VoxCPM.from_pretrained(self.model_id)
        self._torch = torch

    def _call_generate(self, text):
        kwargs = {"text": text}
        if self.ref_audio:
            kwargs["prompt_wav_path"] = self.ref_audio
            if self.ref_text:
                kwargs["prompt_text"] = self.ref_text
        return self.model.generate(**kwargs)

    def synthesize(self, text):
        self._torch.manual_seed(self.seed)
        result = self._call_generate(text)
        # generate() returns either a waveform or a (waveform, sample_rate)
        # pair depending on version; accept both rather than pinning one.
        if isinstance(result, tuple):
            audio, sample_rate = result[0], int(result[1])
        else:
            audio, sample_rate = result, self.native_sample_rate
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return audio, sample_rate

    def describe(self):
        info = {
            "backend": self.name,
            "model_id": self.model_id,
            "ref_audio": self.ref_audio,
        }
        try:
            import voxcpm

            info["voxcpm_version"] = getattr(voxcpm, "__version__", "unknown")
        except ImportError:
            pass
        return info
