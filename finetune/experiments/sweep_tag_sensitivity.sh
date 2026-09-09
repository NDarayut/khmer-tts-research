#!/usr/bin/env bash
# Probe every checkpoint of a run for tag sensitivity on the held-out split.
#
# WHY A SWEEP AND NOT ONE CHECKPOINT
# ----------------------------------
# Validation loss is not the selection criterion here: a model can lose ground
# on general acoustics while gaining the tag binding we actually want. The
# overfitting run is the precedent -- 267 epochs, comprehensively memorized,
# and that is exactly when the tag took. So rather than pick a checkpoint by
# val loss and probe it once, probe all of them and read the curve.
#
# The base model is measured on the identical clips, because the ratio is not
# comparable across clip sets: the unmodified model scores 2% on one set of
# sentences and 52% on another. Only a within-clip-set comparison means
# anything, and that is what the first row here provides.
#
#     bash finetune/experiments/sweep_tag_sensitivity.sh nvv_v2
set -u

RUN="${1:-nvv_v2}"
ROOT="/run/media/pc/disk1/text-to-speech"
PY="$ROOT/.venv/bin/python"
PROBE="$ROOT/finetune/experiments/nvv_tag_sensitivity.py"
MANIFEST="$ROOT/finetune/data-nvv/manifests_v2/val.jsonl"
ROWS=359                     # every tagged clip in the held-out split
OUT="$ROOT/finetune/results/nvv/$RUN"

mkdir -p "$OUT"

# The base model first. Without it the adapter's numbers have nothing to be
# read against, and it is the row most easily forgotten.
if [ ! -f "$OUT/base.json" ]; then
    echo "=== base model"
    "$PY" "$PROBE" --manifest "$MANIFEST" --rows "$ROWS" \
        --out "$OUT/base.json" 2>&1 | grep -vE "^(Generating|Map|Loading|Resolving)" | tail -12
fi

for ckpt in "$ROOT/finetune/checkpoints/$RUN"/step_*; do
    [ -d "$ckpt" ] || continue
    step=$(basename "$ckpt")
    # step_0000000 is the untrained adapter; the base row above already covers it
    [ "$step" = "step_0000000" ] && continue
    [ -f "$ckpt/lora_weights.safetensors" ] || { echo "skip $step: no weights"; continue; }
    dest="$OUT/$step.json"
    [ -f "$dest" ] && { echo "=== $step (already done)"; continue; }
    echo "=== $step"
    "$PY" "$PROBE" --lora "$ckpt" --manifest "$MANIFEST" --rows "$ROWS" \
        --out "$dest" 2>&1 | grep -vE "^(Generating|Map|Loading|Resolving)" | tail -12
done

echo
echo "=== summary"
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("*.json"), key=lambda p: (p.stem != "base", p.stem)):
    d = json.loads(f.read_text())
    c = d["conditions"]
    rows.append((f.stem, d["n"], c["removed"]["delta_mean"], c["removed"]["worse"],
                 c["removed"]["p"], c["moved"]["p"], d["removed_over_scrambled"]))

print(f"{'checkpoint':>16}{'n':>5}{'del delta':>11}{'worse':>9}"
      f"{'del p':>10}{'move p':>10}{'ratio':>9}")
print("-" * 70)
for name, n, dm, w, p, mp, r in rows:
    print(f"{name:>16}{n:>5}{dm:>+11.5f}{w:>6}/{n}{p:>10.4g}{mp:>10.4g}{r:>9.3f}")
if len(rows) > 1:
    b = rows[0]
    print(f"\nbase row is '{b[0]}'. An adapter has learned something only if its "
          f"deletion delta\nstands clearly above {b[2]:+.5f} on the same clips.")
EOF
