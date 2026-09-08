#!/usr/bin/env bash
# Pick the checkpoint with the strongest control response, then verify it fully.
#
# docs/09 §9.5 is explicit that on this model the best checkpoint is frequently
# not the last one -- small-dataset LoRA starts ignoring the text input within a
# few hundred steps, and validation loss does not see it happen. Loss selection
# is therefore the wrong instrument. What this project *can* measure is whether
# the control tag still moves the acoustics, so that is the selection criterion.
#
# Stage 1 runs a deliberately cheap sweep over every saved checkpoint: 6
# sentences, the two prosodic axes with the widest label separation (`rate`,
# `var`), no base-model control -- plus a 4-voice `spk` sweep. Stage 2 runs the
# full verification on whichever one responds best. The cheap sweep is for
# ranking only; every number that gets quoted comes from the full run.
#
# WHY spk IS IN THE RANKING. The first run ranked on rate+var alone and could
# not tell a broken adapter from a working one -- every checkpoint scored near
# zero and the ranking was noise. `spk` is the axis with the most information in
# it, because speaker identity is the one attribute the model cannot infer from
# the text, so it is the axis that discriminates. It is scored differently from
# the others: rho between the *corpus* median F0 of each named speaker and the
# median F0 the model actually generates for that tag. See docs/11 §11.5.
#
#   bash finetune/select_checkpoint.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CKPT_DIR=finetune/checkpoints/khmer_style
OUT=finetune/results
SENTENCES=${SENTENCES:-6}

mapfile -t STEPS < <(find "$CKPT_DIR" -maxdepth 1 -name 'step_*' -printf '%f\n' \
                     | sort | grep -v 'step_0000000$')
if [ ${#STEPS[@]} -eq 0 ]; then
    echo "no checkpoints under $CKPT_DIR" >&2; exit 1
fi
echo "== stage 1: ranking ${#STEPS[@]} checkpoints on $SENTENCES sentences =="
mkdir -p "$OUT/select"   # the per-checkpoint log redirect needs this to exist

for s in "${STEPS[@]}"; do
    d="$OUT/select/$s"
    if [ -f "$d/control_sweep.json" ]; then echo "  $s: cached"; continue; fi
    echo "  $s ..."
    $PY finetune/verify_control.py \
        --lora "$CKPT_DIR/$s" \
        --sentences "$SENTENCES" \
        --axes rate,var \
        --skip-base --spk-voices 4 --spk-sentences 3 \
        --out-dir "$d" > "$d.log" 2>&1 || { echo "  ! $s failed, see $d.log"; continue; }
done

echo
BEST=$($PY - "$OUT/select" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for d in sorted(root.iterdir()):
    f = d / "control_sweep.json"
    if not f.is_dir() and f.exists():
        blob = json.loads(f.read_text())
        s = blob["summary"]
        rhos = [v["spearman_rho"] for k, v in s.items()
                if k.startswith("lora/") and v["spearman_rho"] is not None]
        # The spk axis is scored against the corpus F0 of the named speakers,
        # not against a level index, so it comes from a different summary block.
        spk = (blob.get("speaker_summary") or {}).get("lora") or {}
        spk_rho = spk.get("rho_vs_corpus_f0")
        spread = spk.get("between_voice_spread_hz")
        if rhos:
            # spk counts as much as the two prosodic axes together: it is the
            # only one of the three that cannot be faked from the text.
            score = (sum(rhos) / len(rhos) + 2 * spk_rho) / 3 if spk_rho is not None \
                    else sum(rhos) / len(rhos)
            rows.append((score, d.name, rhos, spk_rho, spread))
rows.sort(reverse=True)
print(f"{'checkpoint':18s} {'score':>7}  {'spk rho':>7} {'spread':>8}   per-axis",
      file=sys.stderr)
for sc, name, rhos, spk_rho, spread in rows:
    print(f"  {name:18s} {sc:+.3f}   "
          + (f"{spk_rho:+.3f}" if spk_rho is not None else "   -- ")
          + (f" {spread:7.1f}Hz" if spread is not None else "         ")
          + "   " + " ".join(f"{r:+.3f}" for r in rhos), file=sys.stderr)
print(rows[0][1] if rows else "")
EOF
)
if [ -z "$BEST" ]; then echo "no usable sweep results" >&2; exit 1; fi
echo
echo "== stage 2: full verification of $BEST =="
$PY finetune/verify_control.py --lora "$CKPT_DIR/$BEST" --out-dir "$OUT"
echo "$BEST" > "$OUT/selected_checkpoint.txt"
echo
echo "selected $BEST -> $OUT/control_sweep.md"
echo "next: /run/media/pc/disk1/streaming_asr/venv/bin/python finetune/score_cer.py \\"
echo "        --dir base=$OUT/audio/base --dir lora=$OUT/audio/lora"
