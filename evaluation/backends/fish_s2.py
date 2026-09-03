"""
Fish Audio S2-Pro -- self-hosted, from local weights (see docs/05).

~5B params. Weights: huggingface.co/fishaudio/s2-pro, under the Fish Audio
Research License (non-commercial -- fine for this evaluation, not for a
shipped product). Inference code: github.com/fishaudio/fish-speech (Apache-2.0).

*** THIS BACKEND IS THE ONE UNVERIFIED PIECE OF THE HARNESS. ***

docs/05 documents the model but contains no invocation example, and
fish-speech's Python entrypoint has changed name and signature across
releases. So rather than guess at one import path, this module probes a list
of known entrypoints and fails with an explicit message telling you which
function to wire into _load_engine() / _call_tts() for the version you cloned.
Everything else here -- paths, timing, output contract -- is correct as-is.

Setup expected:
    git clone https://github.com/fishaudio/fish-speech
    huggingface-cli download fishaudio/s2-pro --local-dir <checkout>/checkpoints/s2-pro
    set FISH_SPEECH_DIR=<checkout>        (or pass --fish-repo)
"""

import os
import sys
from pathlib import Path

from .base import Backend, resolve_device

MODEL_ID = "fishaudio/s2-pro"
NATIVE_SAMPLE_RATE = 44100

ENV_REPO = "FISH_SPEECH_DIR"
ENV_CHECKPOINT = "FISH_S2_CHECKPOINT"


class FishS2Backend(Backend):
    name = "fish-s2"
    native_sample_rate = NATIVE_SAMPLE_RATE

    def __init__(
        self,
        device=None,
        seed=0,
        fish_repo=None,
        fish_checkpoint=None,
        **_ignored,
    ):
        super().__init__(device=device, seed=seed)
        self.fish_repo = fish_repo or os.environ.get(ENV_REPO)
        self.fish_checkpoint = fish_checkpoint or os.environ.get(ENV_CHECKPOINT)
        self.engine = None

    # --- setup ----------------------------------------------------------

    def _resolve_paths(self):
        if not self.fish_repo:
            raise RuntimeError(
                "Fish S2-Pro needs a local fish-speech checkout. Pass "
                "--fish-repo <path> or set the " + ENV_REPO + " environment "
                "variable. See this module's docstring for the clone/download "
                "steps."
            )
        repo = Path(self.fish_repo).expanduser().resolve()
        if not repo.is_dir():
            raise RuntimeError(f"fish-speech checkout not found at: {repo}")

        checkpoint = (
            Path(self.fish_checkpoint).expanduser().resolve()
            if self.fish_checkpoint
            else repo / "checkpoints" / "s2-pro"
        )
        if not checkpoint.is_dir():
            raise RuntimeError(
                f"S2-Pro weights not found at: {checkpoint}\n"
                f"Download them with:\n"
                f"  huggingface-cli download {MODEL_ID} --local-dir {checkpoint}"
            )
        return repo, checkpoint

    def load(self):
        import torch

        self.device = resolve_device(self.device)
        repo, checkpoint = self._resolve_paths()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        torch.manual_seed(self.seed)
        self._torch = torch
        self.repo_path = repo
        self.checkpoint_path = checkpoint
        self.engine = self._load_engine(checkpoint)

    def _load_engine(self, checkpoint):
        """Probe the entrypoints fish-speech has shipped under different names.

        If none match your clone, wire the correct one in here -- look for the
        class the repo's own inference script builds (commonly in
        fish_speech/inference_engine/ or tools/) and return an object whose
        call is handled by _call_tts() below."""
        attempts = [
            ("fish_speech.inference_engine", "TTSInferenceEngine"),
            ("tools.inference_engine", "TTSInferenceEngine"),
            ("fish_speech.models.text2semantic.inference", "launch_thread_safe_queue"),
        ]
        errors = []
        for module_name, attr in attempts:
            try:
                module = __import__(module_name, fromlist=[attr])
                factory = getattr(module, attr)
            except (ImportError, AttributeError) as exc:
                errors.append(f"  {module_name}.{attr}: {exc}")
                continue
            return factory(
                llama_checkpoint_path=str(checkpoint),
                decoder_checkpoint_path=str(checkpoint),
                device=self.device,
            )
        raise RuntimeError(
            "Could not locate a fish-speech inference entrypoint in "
            f"{self.repo_path}. Tried:\n" + "\n".join(errors) + "\n"
            "Open evaluation/backends/fish_s2.py and wire the entrypoint your "
            "clone actually exposes into _load_engine()/_call_tts() -- see the "
            "module docstring. This is the only backend the repo's docs do not "
            "pin an API for."
        )

    # --- inference ------------------------------------------------------

    def _call_tts(self, text):
        """Single seam for the engine call. Adjust alongside _load_engine()."""
        if hasattr(self.engine, "inference"):
            return self.engine.inference(text=text)
        return self.engine(text=text)

    def synthesize(self, text):
        import numpy as np

        self._torch.manual_seed(self.seed)
        result = self._call_tts(text)
        if isinstance(result, tuple):
            audio, sample_rate = result[0], int(result[1])
        else:
            audio, sample_rate = result, self.native_sample_rate
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio), sample_rate

    def describe(self):
        return {
            "backend": self.name,
            "model_id": MODEL_ID,
            "fish_repo": str(getattr(self, "repo_path", self.fish_repo)),
            "checkpoint": str(getattr(self, "checkpoint_path", self.fish_checkpoint)),
        }
