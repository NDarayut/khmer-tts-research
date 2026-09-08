"""
Fish Audio S2-Pro -- self-hosted, from local weights (see docs/05).

~5B params. Weights: huggingface.co/fishaudio/s2-pro, under the Fish Audio
Research License (non-commercial -- fine for this evaluation, not for a
shipped product). Inference code: github.com/fishaudio/fish-speech (Apache-2.0).

Wiring status: this module is now written against a real clone --
fish-speech 2.0.0 (commit befe400), whose entrypoint is
launch_thread_safe_queue + models.dac.inference.load_model + TTSInferenceEngine,
with inference() taking a ServeTTSRequest and *yielding* results. The older
probe-a-list-of-names approach is gone; if you clone a different release and
the imports in _load_engine() move, that is the place to adjust.

*** NOT YET RUN END-TO-END: it does not fit a 12 GB GPU. ***

On an RTX 3060 (11.63 GiB usable) the text2semantic model alone settles at
~11.11 GiB, and moving the 0.93 GB bf16 codec alongside it then OOMs. That is
already with the KV cache clamped (MAX_SEQ_LEN below) and the codec loaded via
CPU rather than straight onto the card -- both of which were needed just to get
this far, and neither of which is enough. So the numbers below are untested
against real output.

To actually run it you need either a ~16 GB+ card, or an 8-bit/quantized load
of the text2semantic weights. Everything else -- paths, timing, seeding,
output contract -- is correct and needs no further change.

Setup expected:
    git clone https://github.com/fishaudio/fish-speech
    huggingface-cli download fishaudio/s2-pro --local-dir <checkout>/checkpoints/s2-pro
    export FISH_SPEECH_DIR=<checkout>     (or pass --fish-repo)

fish-speech pins torch==2.8.0 and conflicts with the main venv, so run this
backend from a separate interpreter -- see evaluation/README.md.

Operational note: if a run OOMs, the process *hangs* rather than exiting
(launch_thread_safe_queue blocks on init_event.wait(), which its worker never
sets when it dies) and keeps holding GPU memory. Kill the PID before retrying,
or the next attempt fails for a misleading reason.
"""

import os
import sys
import types
from pathlib import Path

from .base import Backend, resolve_device
from .mms import text_seed

MODEL_ID = "fishaudio/s2-pro"
NATIVE_SAMPLE_RATE = 44100

# Both from fish-speech's own defaults (tools/run_webui.py): the codec lives
# inside the s2-pro download, the config name is a repo-internal hydra key.
DECODER_CHECKPOINT_NAME = "codec.pth"
DECODER_CONFIG_NAME = "modded_dac_vq"

# S2-Pro's config declares max_seq_len=32768, and setup_caches() allocates a
# KV cache for all of it: 36 layers x 8 kv-heads x 128 dims x 2 (K+V) x 2
# bytes = 4.83 GB, on top of ~9.1 GB of weights and a ~1.9 GB codec. That
# cannot fit a 12 GB card, and none of it is needed -- eval-set sentences are
# single utterances of a few hundred tokens.
#
# It must stay above max_new_tokens + the prompt, or generation indexes past
# the cache and the GPU raises a device-side assert -- which poisons the CUDA
# context, so every *later* utterance in the run fails too. 1024 is therefore
# too small despite fitting: with max_new_tokens=1024 in _call_tts() a long
# sentence needs ~1150 slots, and S2 does run long on this set (some
# utterances synthesize 20+ seconds). 2048 costs 0.30 GB and covers it.
#
# Raise both this and max_new_tokens together if you ever feed it long-form
# text; raising max_new_tokens alone reintroduces the assert.
MAX_SEQ_LEN = 2048

# The codec does not go on the GPU, and shrinking it is not an option:
# measured, it is 4.58 GB in fp32 and still 3.85 GB in bf16, because 3.12 GB
# of that is *integer* buffers (mostly the quantizer's codebooks) that a dtype
# cast cannot touch. Even stripped to the parts decoding needs -- quantizer
# 2.56 GB + decoder 0.10 GB, dropping the encoder, which only matters for
# reference-audio cloning -- it is 2.66 GB against the 1.52 GB the
# text2semantic model leaves free on a 12 GB card.
#
# So it lives on the CPU. _split_devices() below is what makes that possible.
# Consequence for the benchmark: RTF for this backend covers GPU generation
# plus CPU codec decode, and is NOT comparable to a fully-GPU model's RTF.
# CER, UTMOS and DNSMOS are unaffected -- they only read the finished wav.
CODEC_DEVICE = "cpu"

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

    @staticmethod
    def _clamp_kv_cache():
        """Cap the KV cache at MAX_SEQ_LEN -- see that constant for why.

        fish-speech reads the length straight off the checkpoint config
        (models/text2semantic/inference.py calls setup_caches with
        model.config.max_seq_len) and exposes no argument for it, and the model
        is built inside a worker thread we never get a handle on. Wrapping the
        allocation is the one seam available that does not mean editing the
        clone or the downloaded config."""
        from fish_speech.models.text2semantic.llama import BaseTransformer

        if getattr(BaseTransformer.setup_caches, "_clamped", False):
            return
        original = BaseTransformer.setup_caches

        def setup_caches(self, max_batch_size, max_seq_len, *args, **kwargs):
            return original(
                self, max_batch_size, min(max_seq_len, MAX_SEQ_LEN), *args, **kwargs
            )

        setup_caches._clamped = True
        BaseTransformer.setup_caches = setup_caches

    def _load_engine(self, checkpoint):
        """Build the same engine fish-speech's own server builds.

        Mirrors tools/server/model_manager.py: the text2semantic model runs on
        a worker thread behind a queue, the DAC codec is loaded separately from
        codec.pth, and TTSInferenceEngine ties the two together. Verified
        against fish-speech 2.0.0 (commit befe400)."""
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,
        )

        # bfloat16 over fp16: the text2semantic weights come to 9.65 GB either
        # way, and bf16 avoids the overflow the repo's own --half path warns
        # about.
        self.precision = torch.bfloat16
        self._clamp_kv_cache()

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(checkpoint),
            device=self.device,
            precision=self.precision,
            compile=False,
        )
        # The codec stays on the CPU permanently -- see CODEC_DEVICE.
        decoder_model = load_decoder_model(
            config_name=DECODER_CONFIG_NAME,
            checkpoint_path=str(checkpoint / DECODER_CHECKPOINT_NAME),
            device=CODEC_DEVICE,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False,
        )
        self._split_devices(engine, generation_device=self.device)
        return engine

    @staticmethod
    def _split_devices(engine, generation_device):
        """Let the codec sit on the CPU while generation stays on the GPU.

        TTSInferenceEngine assumes the two share a device: it reads
        decoder_model.device to decide where the *text2semantic* model runs
        (inference_engine/__init__.py, send_Llama_request), so a CPU codec
        would otherwise drag the 5B model onto the CPU with it. These two
        overrides break that coupling -- generation is pinned explicitly, and
        the VQ codes are moved to the codec rather than the reverse.

        Both mirror fish-speech 2.0.0's own bodies; re-check them against
        inference_engine/__init__.py if you bump the clone."""
        import queue as queue_module

        from fish_speech.models.text2semantic.inference import GenerateRequest

        def send_Llama_request(self, req, prompt_tokens, prompt_texts):
            request = dict(
                device=generation_device,  # the only change from upstream
                max_new_tokens=req.max_new_tokens,
                text=req.text,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                temperature=req.temperature,
                compile=self.compile,
                iterative_prompt=req.chunk_length > 0,
                chunk_length=req.chunk_length,
                prompt_tokens=prompt_tokens,
                prompt_text=prompt_texts,
            )
            response_queue = queue_module.Queue()
            self.llama_queue.put(
                GenerateRequest(request=request, response_queue=response_queue)
            )
            return response_queue

        def get_audio_segment(self, result):
            # Upstream wraps this in autocast(self.precision). Dropped: the
            # codec is fp32 on CPU, where bf16 conv kernels are slower, and
            # its memory is no longer the constraint once it is off the card.
            codes = result.codes.to(self.decoder_model.device)
            segment = self.decode_vq_tokens(codes=codes)
            return segment.float().cpu().numpy()

        engine.send_Llama_request = types.MethodType(send_Llama_request, engine)
        engine.get_audio_segment = types.MethodType(get_audio_segment, engine)

    # --- inference ------------------------------------------------------

    def _call_tts(self, text):
        """Single seam for the engine call. Adjust alongside _load_engine().

        engine.inference() is a *generator* yielding InferenceResult records;
        the complete waveform arrives in the one tagged "final", as an
        (sample_rate, ndarray) pair. Errors are yielded rather than raised, so
        they are re-raised here for synthesize_entry() to record."""
        from fish_speech.utils.schema import ServeTTSRequest

        request = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=None,
            # Pinned per utterance, as in backends/mms.py: S2 samples with
            # temperature, so without this a resumed run would not match a
            # full one. Same (seed, sentence) -> same waveform.
            seed=text_seed(self.seed, text),
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            # Khmer is not one of the languages the normalizer handles, and it
            # would only mangle the code-switched entries.
            normalize=False,
            format="wav",
        )
        for result in self.engine.inference(request):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final":
                if not isinstance(result.audio, tuple):
                    raise RuntimeError(
                        f"expected (sample_rate, audio), got {type(result.audio)}"
                    )
                return result.audio[1], int(result.audio[0])
        raise RuntimeError("engine produced no audio for this sentence")

    def synthesize(self, text):
        import numpy as np

        self._torch.manual_seed(self.seed)
        audio, sample_rate = self._call_tts(text)
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
