"""
Synthesize Khmer with the style-controlled VoxCPM2 adapter.

This is the usable end of the fine-tune: it takes a control tag and some Khmer
text and writes a wav. It is also the engine behind `verify_control.py`, which
sweeps one axis at a time and measures whether the audio actually moved.

    # one clip
    python finetune/synthesize_styled.py \
        --lora finetune/checkpoints/khmer_style/latest \
        --text "សូមស្វាគមន៍មកកាន់ប្រទេសកម្ពុជា។" \
        --style var=lively,rate=fast,spk=f2 \
        --out /tmp/hello.wav

    # base model, no adapter (for A/B)
    python finetune/synthesize_styled.py --text "..." --out /tmp/base.wav

`--style` takes comma-separated slot=level pairs. Any slot you leave out is
filled with `any`, which the corpus builder's per-slot dropout trained the model
to read as "unspecified" -- so `--style var=lively` really does mean "expressive,
everything else free" rather than "expressive plus four accidental defaults".
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.build_corpus import ANY, LEVELS, SLOTS, make_tag  # noqa: E402

MODEL_ID = "openbmb/VoxCPM2"


def parse_style(spec):
    """'var=lively,rate=fast' -> full slot dict with the rest set to 'any'."""
    values = {s: ANY for s in SLOTS}
    if not spec:
        return values
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"--style expects slot=level pairs, got {part!r}")
        slot, level = (x.strip() for x in part.split("=", 1))
        if slot not in SLOTS:
            sys.exit(f"unknown slot {slot!r}; valid slots: {', '.join(SLOTS)}")
        if slot != "spk" and level != ANY and level not in LEVELS[slot]:
            sys.exit(f"unknown level {level!r} for {slot}; "
                     f"valid: {', '.join(LEVELS[slot])}, {ANY}")
        values[slot] = level
    return values


def load_model(lora_path=None, model_id=MODEL_ID):
    """Load the base model, optionally with an adapter.

    The adapter's rank MUST be passed explicitly. `VoxCPM.from_pretrained`
    auto-creates a LoRAConfig when given `lora_weights_path` without one, and
    that default is `r=8, alpha=16` -- so loading an r=64 adapter through the
    convenience path builds r=8 modules and then fails on the shape mismatch.
    `save_checkpoint` writes the real values to `lora_config.json` next to the
    weights, so read them from there rather than hard-coding a number that can
    drift out of step with the YAML.
    """
    from voxcpm import VoxCPM

    if not lora_path:
        return VoxCPM.from_pretrained(model_id)

    lora_path = Path(lora_path)
    cfg_file = lora_path / "lora_config.json" if lora_path.is_dir() else None
    lora_config = None
    if cfg_file and cfg_file.exists():
        import json

        from voxcpm.model.voxcpm2 import LoRAConfig

        saved = json.loads(cfg_file.read_text(encoding="utf-8"))["lora_config"]
        lora_config = LoRAConfig(**saved)
        print(f"adapter: r={lora_config.r} alpha={lora_config.alpha} "
              f"lm={lora_config.enable_lm} dit={lora_config.enable_dit}",
              file=sys.stderr)
    else:
        print(f"! no lora_config.json beside {lora_path}; falling back to the "
              f"package default (r=8), which will fail for any other rank",
              file=sys.stderr)

    model = VoxCPM.from_pretrained(model_id, lora_weights_path=str(lora_path),
                                   lora_config=lora_config)
    return model


def synthesize(model, text, style_values, **gen_kwargs):
    """Returns (waveform, sample_rate). The tag is prepended here and nowhere
    else, so training and inference cannot drift apart in tag format."""
    tagged = make_tag(style_values) + text
    result = model.generate(text=tagged, **gen_kwargs)
    if isinstance(result, tuple):
        audio, sr = result[0], int(result[1])
    else:
        audio, sr = result, 48000
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return audio, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--style", default="", help="comma-separated slot=level pairs")
    ap.add_argument("--lora", default=None, help="adapter dir; omit for the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    values = parse_style(args.style)
    print(f"tag: {make_tag(values)}", file=sys.stderr)

    model = load_model(args.lora)
    torch.manual_seed(args.seed)
    audio, sr = synthesize(model, args.text, values, cfg_value=args.cfg_value)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, sr)
    print(f"wrote {args.out}  ({len(audio)/sr:.2f}s @ {sr} Hz)", file=sys.stderr)


if __name__ == "__main__":
    main()
