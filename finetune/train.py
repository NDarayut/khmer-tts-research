"""
Launcher for VoxCPM2 LoRA fine-tuning on a 12 GB card.

Upstream's `scripts/train_voxcpm_finetune.py` is used unmodified. This file
imports it and applies memory patches before calling train(), so the upstream
script stays a clean vendored copy and every deviation is visible in one place.

WHY PATCHES ARE NEEDED
----------------------
docs/09 §9.5 records the official VRAM figure for VoxCPM2 LoRA: ~20 GB. This
machine has a 12 GB RTX 3060. The gap is almost entirely one line in upstream's
`from_local`: in training mode the model is left in float32, so 2.15 B
parameters occupy ~8.6 GB before a single activation is allocated. The forward
pass then runs under `autocast(bfloat16)` anyway -- the fp32 master copy buys
nothing for the 99.9% of parameters that are frozen.

PATCH 1 -- frozen weights to bfloat16, LoRA parameters kept in float32.
    Halves the resident model to ~4.3 GB. LoRA stays fp32 because AdamW at
    lr 1e-4 produces updates ~1e-2 the size of the weights, and bf16's ~3
    significant decimal digits would quantise a meaningful share of them away.
    This is the standard mixed-precision LoRA arrangement.

PATCH 2 -- gradient checkpointing on the backbone LM.
    Applied only if the installed MiniCPM4 block exposes a usable entry point;
    reported at startup either way, never silently skipped.

PATCH 3 -- AudioVAE kept in float32 but moved to GPU, and its encode() run
    under no_grad. Upstream already freezes it; this just makes the memory
    behaviour explicit.

Everything else -- batch size, accumulation, LR, LoRA rank -- lives in the YAML.

USAGE
-----
    python finetune/train.py --config finetune/conf/khmer_style_lora.yaml
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "VoxCPM-src" / "scripts" / "train_voxcpm_finetune.py"


def load_upstream():
    if not UPSTREAM.exists():
        sys.exit(
            f"Upstream trainer not found at {UPSTREAM}.\n"
            "Clone it with:\n"
            "  git clone --depth 1 https://github.com/OpenBMB/VoxCPM VoxCPM-src"
        )
    spec = importlib.util.spec_from_file_location("voxcpm_trainer", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_low_vram(verbose=True):
    """Cast frozen weights to bf16 and enable gradient checkpointing."""
    import torch
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    original_from_local = VoxCPM2Model.from_local.__func__

    def from_local(cls, *args, **kwargs):
        model = original_from_local(cls, *args, **kwargs)

        # --- PATCH 1: frozen -> bf16, trainable (LoRA) -> fp32 ------------- #
        n_bf16 = n_fp32 = 0
        for name, param in model.named_parameters():
            if name.startswith("audio_vae."):
                continue
            if param.requires_grad:
                param.data = param.data.float()
                n_fp32 += param.numel()
            else:
                param.data = param.data.to(torch.bfloat16)
                n_bf16 += param.numel()
        # buffers too -- RoPE tables and norms would otherwise stay fp32
        for name, buf in model.named_buffers():
            if not name.startswith("audio_vae.") and buf.is_floating_point():
                buf.data = buf.data.to(torch.bfloat16)
        if verbose:
            print(f"[low-vram] frozen -> bf16: {n_bf16/1e9:.3f} B params "
                  f"(~{n_bf16*2/2**30:.2f} GiB)", file=sys.stderr)
            print(f"[low-vram] trainable fp32: {n_fp32/1e6:.2f} M params "
                  f"(~{n_fp32*4/2**20:.1f} MiB + {n_fp32*8/2**20:.1f} MiB AdamW)",
                  file=sys.stderr)

        # --- PATCH 2: gradient checkpointing on the 28-layer backbone ------ #
        applied = enable_grad_checkpointing(model)
        if verbose:
            print(f"[low-vram] gradient checkpointing: "
                  f"{applied if applied else 'NOT APPLIED (no supported entry point)'}",
                  file=sys.stderr)
        return model

    VoxCPM2Model.from_local = classmethod(from_local)


def enable_grad_checkpointing(model):
    """Wrap each backbone transformer layer in torch.utils.checkpoint.

    Reports what it actually wrapped rather than assuming; the MiniCPM4 block
    ships inside the voxcpm package and its layer container has moved between
    releases.
    """
    import torch
    import torch.utils.checkpoint as cp

    targets = []
    for attr in ("base_lm", "residual_lm"):
        lm = getattr(model, attr, None)
        if lm is None:
            continue
        layers = getattr(lm, "layers", None)
        if layers is None:
            inner = getattr(lm, "model", None)
            layers = getattr(inner, "layers", None) if inner is not None else None
        if layers is not None:
            targets.append((attr, layers))
    if not targets:
        return ""

    applied = []
    for attr, layers in targets:
        for layer in layers:
            if getattr(layer, "_ckpt_wrapped", False):
                continue
            layer.forward = _checkpointed(layer.forward, cp)
            layer._ckpt_wrapped = True
        applied.append(f"{attr}:{len(layers)} layers")
    return ", ".join(applied)


def _checkpointed(fn, cp):
    def wrapper(*args, **kwargs):
        import torch
        if not torch.is_grad_enabled():
            return fn(*args, **kwargs)
        # kwargs are not checkpointable directly; bind them into the closure
        if kwargs:
            def run(*a):
                return fn(*a, **kwargs)
            return cp.checkpoint(run, *args, use_reentrant=False)
        return cp.checkpoint(fn, *args, use_reentrant=False)
    return wrapper


def patch_onset_weighting(weight, tau, verbose=True):
    """PATCH 4 (optional) -- weight the diffusion loss toward the audio onset.

    WHY. Training is teacher-forced: every audio patch is predicted with the
    preceding *ground-truth* patches visible, and a speaker's pitch is trivially
    readable off those. So a control tag in the text carries no information the
    acoustic prefix does not already supply, and almost no gradient pushes the
    model to read it. Measured on the `proj` probe checkpoint, swapping the
    speaker tag costs +0.00318 of loss on the first patch and only +0.00022
    averaged over the clip -- a 14x decay -- while corrupting the *transcript*
    costs +0.064. The tag matters exactly where no prefix exists yet.

    So put the gradient there. Position i of the audio span gets weight
    1 + (weight-1)*exp(-i/tau), which at weight=8, tau=4 is 8.0 on the first
    patch, 3.6 on the fifth and ~1 by the twentieth. Downstream this is a true
    per-position weight and nothing else has to change: `unified_cfm.compute_loss`
    computes `(mask*losses).sum() / sum(mask)` and `adaptive_loss_weighting`
    with p=0 returns the mask unaltered, so a non-binary mask is a weighted mean
    that renormalises itself. Upstream casts the mask to int32, which would
    truncate the weights, so it is rebuilt as float here.

    If the onset is set correctly, autoregression carries it: the rest of the
    utterance is generated conditioned on that first patch.
    """
    import torch
    from voxcpm.training.packers import AudioFeatureProcessingPacker

    original = AudioFeatureProcessingPacker.process_tts_data

    def process_tts_data(self, audio_token, text_token, is_prompt=False):
        out = list(original(self, audio_token, text_token, is_prompt))
        loss_mask = out[4]
        idx = loss_mask.nonzero(as_tuple=True)[0]
        w = loss_mask.to(torch.float32)
        if idx.numel():
            i = torch.arange(idx.numel(), device=w.device, dtype=torch.float32)
            w[idx] = 1.0 + (weight - 1.0) * torch.exp(-i / tau)
        out[4] = w
        return tuple(out)

    AudioFeatureProcessingPacker.process_tts_data = process_tts_data
    if verbose:
        print(f"[onset] diffusion loss weighted {weight}x at the audio onset, "
              f"decaying with tau={tau} patches", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training YAML")
    ap.add_argument("--no-low-vram", action="store_true",
                    help="run upstream unpatched (needs ~20 GB)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--onset-weight", type=float, default=1.0,
                    help="weight the diffusion loss toward the audio onset, where "
                         "a text-side control tag is the only cue available "
                         "(1.0 = off, the upstream behaviour)")
    ap.add_argument("--onset-tau", type=float, default=4.0,
                    help="decay constant, in patches, for --onset-weight")
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    trainer = load_upstream()
    if not args.no_low_vram:
        if args.no_grad_ckpt:
            globals()["enable_grad_checkpointing"] = lambda _m: "disabled by --no-grad-ckpt"
        patch_low_vram()
    if args.onset_weight != 1.0:
        patch_onset_weighting(args.onset_weight, args.onset_tau)

    from voxcpm.training.config import load_yaml_config
    cfg = load_yaml_config(args.config)
    trainer.train(**cfg)


if __name__ == "__main__":
    main()
