# NVV (non-verbal vocalization) fine-tuning plan for VoxCPM2

Fills the reserved §4 Methodology in `reports/Speech-Control-VoxCPM2.pdf` /
`docs/deep-dives/11-voxcpm2-style-control-finetune.md`. Written before any
data mining, detector, or training code exists — this is the plan those
scripts will implement, not a report of results.

## Where this picks up

`docs/11` §3.5 already selected **non-verbal vocalization (NVV)** as the next
layer over prosody (already solved — see `(…)` parenthetical control) and
emotion (blocked on missing Khmer emotional corpora), on the reasoning that
NVV is a *local* event (bounded to one text position) rather than a *global*
attribute, and so should not repeat the failure that stopped the Layer 1b
prosody fine-tune in this directory.

That stopped run is not a detour, it's the machinery this plan reuses:
`results/diagnosis.md` measured why a global control tag in the text field
got ignored under teacher forcing (swapping the tag moved loss 290x less than
scrambling the transcript), and the fix — `enable_proj: true` plus an
onset-weighted diffusion loss — took speaker separation from 5.8 to 39.9 Hz.
`README.md`'s roadmap table already earmarks this:

> **2. Non-verbal** — `[laughing]`, `[sigh]`, `[Uhm]` … — local — ships with
> VoxCPM2; **measure on Khmer before building**; this is the layer docs/11
> §3.5 selects.

Compute assumption for this plan: unconstrained for the real run (rent
24-48GB+ as needed); the 3060 in this repo is for Phase 0 measurement and
pipeline sanity checks only, never the full run.

## Phase 0 — measure before building (3060, cheapest, do first)

VoxCPM2 already documents `[laughing] [laughter] [sigh] [Uhm] [Shh]
[Question-ah/ei/en/oh] [Surprise-wa/yo] [Dissatisfaction-hnn]`. Khmer behavior
is unmeasured. Before writing any training code:

1. `finetune/verify_nvv_tags.py` (parallel to the existing
   `verify_parenthetical.py`): sweep every documented tag across N Khmer
   sentences x seeds, zero-shot (no fine-tune), and score:
   - event presence — run a laughter/NVV detector over the tag's local
     window in the synthesized clip
   - placement — does the event land near the tag's text position, or does
     it leak/shift, or go global
   - CER regression on the rest of the sentence, via the existing Khmer CER
     scorer, so a working tag isn't secretly costing intelligibility
   - reproducibility across seeds
2. A `tag_sensitivity.py`-style falsifiable loss ablation, run *before* any
   training spend: does the inline tag change training loss at all, with a
   scrambled-transcript positive control so a null result is interpretable.
   This directly tests the untested assumption in docs/11 §3.5 that local
   events escape the teacher-forcing starvation that killed Layer 1b — that
   assumption is plausible, not verified, and this two-minute check is how
   Layer 1b's mistake (training for hours before checking) gets avoided here.

**Decision gate:** any tag that already works zero-shot on Khmer needs no
fine-tune — document and ship it, same outcome as Layer 1 prosody. Only
tags that come back null or weak proceed to Phase 1-2.

## Phase 1 — data: mine NVV events from the existing 1067h corpus

New `finetune/build_corpus_nvv.py`, parallel to `build_corpus.py`, over
`/run/media/pc/disk1/streaming_asr/data/dataset` (1067h, 20 speakers, the
same source Layer 1b's 11.7h corpus was extracted from).

1. Run an NVV detector across the corpus. Start with the Gillick et al. 2021
   laughter detector (cheap, pretrained); extend to a VocalSound-style
   6-class classifier (laughter, sigh, cough, sneeze, sniff, throat-clear)
   if broader coverage is wanted before committing to only laughter.
2. Extract each event with surrounding speech context, precise onset/offset.
3. Forced-align against the existing ASR transcripts to place the tag at the
   correct token position in the text (not just "detected somewhere in this
   clip").
4. Map detected type -> VoxCPM2's existing tag vocabulary where one exists
   (laughter -> `[laughing]`, sigh -> `[sigh]`, filled pause -> `[Uhm]`).
   Types with no existing tag (breath, gasp, cough, sob) are a stretch goal,
   not required for the initial pass — introducing a wholly new tag means
   the model has to learn new acoustic content, not just reweight a signal
   it already produces, which is a materially bigger ask.
5. **Measure yield before committing to a full mining pass.** This corpus is
   scripted/read speech, which is systematically sparser in natural NVV
   events than conversational speech — unlike Layer 1b's corpus build, which
   only needed prosody variation (present everywhere) rather than rare
   discrete events. Report events/hour by type first. If yield is too low,
   the fallback is already in `docs/11` §3.3's lit review: an NVSpeech-style
   bootstrap (train a small paralinguistic-aware labeler on the sparse
   manually-checked set, then auto-label the rest), or supplementing with a
   small acted/prompted recording pass.
6. Human/listening spot-check a sample of mined events before trusting them
   as training ground truth (same auto-detect + human-validate pattern as
   NonverbalTTS / NVSpeech).
7. Write manifests (JSONL: audio, text-with-inline-tag, duration,
   dataset_id, event_type, onset/offset), train/val split, with a held-out
   eval set kept separate from whatever sentences Phase 0 already used.

## Phase 2 — training recipe, scaled for big compute

Extends `finetune/train.py` / `conf/khmer_style_lora.yaml` but drops the
12GB memory patches that exist only for the 3060:

- `batch_size` 16-32, `max_batch_tokens` 8192+, LoRA `r=64` (headroom to
  128 if a rank sweep says it helps), `enable_lm: true`, `enable_dit: true`,
  `enable_proj: true` — proj is **mandatory**, not a knob, per
  `results/diagnosis.md` (it's the only route text-side info has into the
  acoustic generator).
- Generalize the onset-weighted loss into an **event-span-weighted loss**:
  concentrate weight on the detected event's actual audio-frame span (plus a
  small margin), not just the clip's start. The original onset weighting
  (`1 + 7*exp(-i/4)` from clip position 0) was the right fix for a *global*
  tag competing with the whole clip; a local NVV tag's event can occur
  anywhere in the clip, so the weighted region should track the event
  position mined in Phase 1, not a fixed clip offset.
- Mix in ~30-50% untagged clips from the same corpus (ELaTE's finding, cited
  in docs/11 §3.3: mixing conditioned data with general data prevents
  regressing the untagged case / catastrophic forgetting).
- Spend the big-compute budget on a rank x data-scale ablation sweep before
  committing to one long run — that is the actual advantage over the 3060,
  which stays reserved for pipeline sanity checks (a tiny pilot config, one
  training step, manifest validation) run first on whatever machine hosts
  the big run too, before the long run starts.
- Verify the event-span-weighted loss with the same falsifiable-probe
  discipline as Phase 0 (`tag_sensitivity.py`-style, a couple of minutes)
  before committing to the multi-hour run — this is the exact step Layer 1b
  skipped the first time and paid for in a wasted run to step 1,980/4,000.

## Phase 3 — evaluation

New `finetune/verify_nvv.py`, parallel to `verify_control.py`. Per event
type, sweep tag on/off across held-out sentences x seeds, and score:

- event presence (detector on the output)
- **placement accuracy** — is the event at the correct text position, the
  metric Layer 1b's global Hz-separation approach has no analogue for
- CER regression on the surrounding words (Khmer CER scorer)
- full-eval-set regression against the base model, same pattern as
  `score_cer.py`, to catch any collateral damage to the other 30 languages
  or to plain (untagged) Khmer synthesis

Explicitly **do not** use UTMOS/DNSMOS to judge naturalness here — this
project has already established they're inverted for Khmer (rank
correlation with CER rho = +0.55 across 400 clips; see main CLAUDE.md).
Naturalness gets judged the same way the 4-model comparison judged it:
CER regression + `audio_stats.py`-style ASR-free diagnostics for the
mechanical checks, a listening-test pass (reusing `listening_test.py` /
`listening_analyse.py`) for the actual "does it sound right" call.

Checkpoint selection by control-response strength (the `select_checkpoint.sh`
pattern), never by validation loss — docs/09 §9.5's lesson, and the reason
Layer 1b's real failure wasn't visible in the loss curve at all.

## Phase 4 — write-up

Once Phase 0-3 produce real numbers in `finetune/results/nvv/*.json`, fill
the report's reserved §4 Methodology through the existing `src/docbuild`
pipeline the same way `build_style_control_report.py` already pulls its
numbers from `results/parenthetical/parenthetical.json` — never hand-edit
`reports/*.docx` directly, per the top-level CLAUDE.md.

## Open risks, stated up front rather than discovered mid-run

1. **Natural NVV yield in the 1067h scripted corpus may be too low** to mine
   a usable training set — Phase 1 step 5 measures this before any training
   commitment is made.
2. **Whether existing tags already work zero-shot on Khmer** determines
   whether training is needed at all — Phase 0 is a hard gate, not a
   formality.
3. **Whether local/inline tags also suffer the teacher-forcing starvation**
   that killed the global prosody tag is an assumption in docs/11 §3.5, not
   a verified fact — Phase 0's loss-ablation check tests it directly instead
   of inheriting it.
4. **New (undocumented) NVV types** — cough, gasp, sob — are a materially
   harder ask than reweighting existing tag vocabulary and are scoped as a
   stretch goal, not part of the initial pass.
