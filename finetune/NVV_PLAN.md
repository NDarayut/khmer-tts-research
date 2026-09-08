# Making VoxCPM2 use non-verbal vocalization tags consistently — plan v2

Fills the reserved §4 Methodology in `reports/Speech-Control-VoxCPM2.pdf` /
`docs/deep-dives/11-voxcpm2-style-control-finetune.md`.

Supersedes `NVV_PLAN.superseded.md`, which was written as a staged
measure-then-decide investigation with a go/no-go gate at Phase 0. That
structure is dropped. The decision to fine-tune is made; what follows is a
build, and the measurement in Stage 1 exists to supply the *before* column of a
result, not to grant permission to proceed.

## 1 The premise, and what it changes

The working hypothesis is that **VoxCPM2's NVV tags were never explicitly
trained.** They are an artefact of scale: tagged transcripts existed somewhere
in the pre-training corpus, and the model absorbed a weak, unreliable
association between the literal string `[laughing]` and a laugh. Nothing in the
released model treats them as a designed interface.

Four checks against the installed `voxcpm==2.0.3` and the `openbmb/VoxCPM2`
checkpoint support this, and they are the load-bearing facts of the plan:

1. **No tag appears anywhere in the code.** `grep -rniE
   'laughing|laughter|sigh|Uhm|Shh|Surprise-wa|Dissatisfaction'` across the
   package's `.py`, `.json` and `.md` files returns **zero hits**. There is no
   tag vocabulary, no lookup, no preprocessing branch, no special-token
   registration.
2. **Tags are ordinary subword text.** Under the model's own tokenizer,
   `[laughing]` is five unremarkable tokens — `['[', 'l', 'augh', 'ing', ']']`
   — and `[sigh]` is four. They are not atomic units; the model sees the same
   pieces it would see in any English prose containing the word *laughing*.
3. **Normalization leaves them intact.** `utils/text_normalize.remove_bracket`
   strips the full-width CJK brackets `（）【】` but not ASCII `[` `]`, so the
   tag reaches the language model verbatim — by omission rather than by design.
4. **The tags survive as text but have no channel of their own.** Conditioning
   dropout for classifier-free guidance (`training_cfg_rate = 0.1`) zeroes the
   *entire* conditioning vector `mu`, not any per-attribute component. There is
   no existing mechanism that could guide a tag independently of the transcript.

Two consequences follow, and they set the whole shape of the work.

**Consistency is the problem, not capability.** If the association exists but
is weak, the failure mode is stochastic: the tag fires on some seeds and not
others, at the wrong position, or as a change in prosody rather than a
discrete event. That is a reliability engineering problem with a measurable
target, and it is the contribution this project can make.

**Adding new tags is architecturally free.** This is the significant finding.
The literature on extending a controllable model's tag inventory is largely
concerned with vocabulary expansion — resizing the embedding matrix, and the
cold-start problem of newly-initialised token embeddings, for which the
standard remedies are averaging the subword embeddings of the token's spelling
or initialising from a natural-language description. **None of that applies
here.** VoxCPM2 is tokenizer-free at the input: `[gasp]` already tokenizes to
four tokens with no `<unk>`, as does any string we invent. Adding a tag means
teaching a new meaning to a byte string the model can already represent — no
resize, no re-initialisation, no cold start. The cost of a new tag is training
data for it and nothing else.

A caveat worth stating once. Khmer is byte-fallback under this tokenizer: a
short Khmer sentence costs ~49 tokens of single-byte fragments, while the
Latin tag costs 4–5 clean subwords. The tag is therefore an unusually salient,
low-entropy island in a Khmer sequence — good for learnability, but it also
means the tag tokens arrive carrying English-context priors.

## 2 What the literature actually settles

PDFs of everything cited here are archived in
[`docs/research/papers/`](../docs/research/papers/README.md).

The review in `docs/11` §3.3 covers the corpus-construction pipeline. This
section records what the research done for *this* plan adds, with the specific
number each source contributes.

### 2.1 The interface: inline tokens at the event position

**NVSpeech** ([arXiv 2508.04195](https://arxiv.org/abs/2508.04195)) is the
template for the interface and the closest match to the problem. It treats
recognition and synthesis as one pipeline: 48,430 manually annotated utterances
across 18 word-level paralinguistic categories train a paralinguistic-aware
recogniser that emits cues as inline decodable tokens (*You're so funny
[Laughter]*); that recogniser then labels 174,179 Chinese utterances — 573 h —
with word-level alignment; a zero-shot synthesiser is fine-tuned on the
combined human- and machine-labelled data, giving control over vocalizations
inserted at arbitrary token positions.

The transferable design decisions: the tag sits **inline at the position of the
event**, not in a header; a recogniser-shaped detector bootstraps the corpus;
and a modest human-validated seed set is enough to bootstrap the labeller.

This also matters as a contrast with the layer-1b failure recorded in
`results/diagnosis.md`. A header tag competes for influence over frames the
surrounding acoustic context already determines, and lost — it earned a
gradient roughly 290× smaller than the transcript did under teacher forcing.
An inline tag at the event position is the only predictor of the frames it
occupies. The conditioning problem is structurally different, which is the
reason this layer was selected.

### 2.2 The fine-tune: mix conditioned data with general data 50:50

**ELaTE** ([arXiv 2402.07383](https://arxiv.org/abs/2402.07383)) is the closest
methodological reference and supplies the one recipe number that is not a
guess. It fine-tunes a conditional flow-matching zero-shot synthesiser — the
same class as VoxCPM2's local DiT — using frame-level conditioning from a
laughter detector, and mixes **pre-training data (conditioning zeroed) with
laughter-conditioned fine-tuning data in a 50:50 ratio**. Two results
constrain any plan built on it: a comparatively small conditioned dataset
suffices, and the mixing is what preserves base-model quality, so the
fine-tune need not be paid for in intelligibility.

The superseded plan guessed "~30–50% untagged". The published figure is 50:50
and this plan adopts it as the default.

Note the difference in conditioning channel: ELaTE conditions on a frame-level
detector signal, VoxCPM2 and NVSpeech on inline text. The mixing ratio and the
"small conditioned set suffices" result transfer; the frame-level channel is a
design option held in reserve (§4.3), not the primary route.

### 2.3 Corpora that already exist, and the cross-lingual argument

| Source | Scale | Types | Use here |
|---|---|---|---|
| **SynParaSpeech** ([2509.14946](https://arxiv.org/abs/2509.14946)) | 118.75 h | 6 | Fully automated construction from *natural conversational* speech, with precise timestamps. The closest thing to an off-the-shelf tagged corpus. |
| **NonverbalTTS** ([2507.13155](https://arxiv.org/abs/2507.13155)) | 17 h | 10 | Auto-detection plus human validation; reported at parity with proprietary systems. The proof that a *small* set works. |
| **VocalSound** ([2205.03433](https://arxiv.org/abs/2205.03433)) | 21,000 clips, 3,365 speakers | 6 | Laughter, sigh, cough, throat-clear, sneeze, sniff. Isolated events plus a classifier baseline. |
| **Gillick et al. (2021)** | — | laughter | Released frame-level laughter detection and segmentation. |

None of these is Khmer. That is admissible, and the argument is already stated
in the report's §3.5(3): **NVV acoustics are substantially language-independent
— what is language-specific is placement and pragmatic function, and placement
is supplied by the tag position.** A laugh is not a Khmer laugh. This is the
premise that makes the whole approach affordable, and §5.2 states how the plan
tests it rather than assuming it.

### 2.4 The state of the art is not very good, which is the opening

**NVV-SuperBench** ([arXiv 2604.16211](https://arxiv.org/abs/2604.16211))
evaluates 15 systems spanning prompt-based and tag-based control against a
45-type taxonomy in six groups (Respiratory 10, Throat/Physiological 7,
Laughter Spectrum 7, Crying 5, Emotional Vocalizations 7, Oral/Misc 9).
Reported tag-based baselines, English:

| system | F1 | Coverage |
|---|---|---|
| ChatTTS | 0.664 ± 0.167 | 0.02 |
| ElevenLabs | 0.720 ± 0.024 | 0.27 |
| Orpheus TTS | 0.728 ± 0.029 | 0.18 |

A commercial system reaches F1 0.72 and covers 27% of the taxonomy. Separately,
CosyVoice 3 — which trains its paralinguistic tags *explicitly*, via
CosyVoice-instruct's `[laughter]`/`[breath]` markers and `<laughter>…</laughter>`
spans — is reported at **83.3% laughter success**, against 97.6% for PilotTTS
([2605.27258](https://arxiv.org/abs/2605.27258), 4 paralinguistic categories).
*(The 83.3/97.6 pair comes from a secondary summary of PilotTTS's comparison
table, not from the PDF directly — treat as indicative until confirmed.)*

Two things follow. Inconsistency is the field-wide condition, not a VoxCPM2
defect — so "make the tags consistent" is a real target with published numbers
to sit beside. And NVV-SuperBench's headline finding, that **controllability
decouples from quality**, is this project's own UTMOS/CER inversion arriving
from a second direction: a system can score well on speech quality while
failing at the vocalization entirely. Quality metrics must not be used to
judge control.

### 2.5 Evaluation instruments

NVV-SuperBench also supplies the metric definitions, and this plan adopts them
so the numbers are comparable to published work rather than bespoke:

- **Controllability** — a predicted NVV matches if its type equals the target
  **and** its position index differs by ≤ δ transcript units. Report
  precision / recall / **F1**.
- **Placement** — **Normalized Tag Distance (NTD)**, the length-normalized mean
  position error over matched utterances, `(1/|U_TP|) Σ d_u / L_u`.
- **Coverage** — the fraction of the attempted tag inventory the system
  actually produces.
- **Salience / perceptual effect** — a 0–5 Likert rating, 0 meaning absent or
  inaudible. Human; NVV-SuperBench used 450 samples across 97 raters.

**NVMOS** ([arXiv 2606.15888](https://arxiv.org/abs/2606.15888)) predicts a
0–5 MOS-like value for a *specific marked* non-verbal event, taking audio plus
text containing an explicit tag — exactly the format a training manifest here
already has. Its authors report that general-purpose audio-capable multimodal
models disagree measurably with expert raters on this task, so an off-the-shelf
multimodal model is not an acceptable substitute.

## 3 Data

### 3.1 DDD is the general-data half, not the NVV half

`/run/media/pc/disk1/streaming_asr/data/dataset` — **1,066.6 h, 447,760
utterances, 20 speakers** (`f-adt1..4`, `m-adt1..4`), 653 parquet shards
indexed by `train.tsv`, every row `source=ddd`.

Checked: **DDD carries no NVV annotation.** Of 447,760 transcripts, 2,455
contain a bracket or parenthesis and every one is an ordinary content
parenthetical; the 56 Latin-alphabet hits for `laugh|sigh|cough|breath|um|hmm`
are loanwords. It is adult read speech, transcribed for ASR, where NVV is both
rare by nature and discarded by transcription convention.

So DDD's role is settled: it is the **untagged general-data half of ELaTE's
50:50 mix** — the thing that stops the fine-tune costing Khmer intelligibility.
That is a real and necessary role, and it is 1,067 h of exactly the right
language and domain for it. It is not the source of NVV events.

Reuse rather than rediscover, from the sibling ASR repo: `manifests/
ddd_duplicates.json` (byte-identical audio, and groups where one recording
carries many conflicting transcripts — dropped whole) and the held-out
val/eval keys. Note also ~4,032 malformed rows in `train.tsv` where transcripts
have leaked into the wrong columns via embedded newlines; any parser must
handle them.

### 3.2 NVV events come from three sources, in cost order

1. **Released non-Khmer corpora** (SynParaSpeech, NonverbalTTS, VocalSound) for
   the event acoustics. Justified by §2.3's language-independence argument.
2. **Mined DDD events** for whatever genuinely occurs in Khmer read speech —
   principally breath and filled pause, which read speech does contain even
   when laughter does not. Detector: Gillick for laughter, a VocalSound-class
   classifier for the rest. Yield is expected to be low and is reported, not
   assumed.
3. **A small prompted recording pass** if 1 and 2 leave a type uncovered.
   MNV-17 is the precedent for this being a legitimate move rather than an
   admission of failure.

Manifests are JSONL — audio path, text with the tag inline at the event
position, duration, `dataset_id`, `event_type`, event onset/offset in seconds —
with a held-out eval split. Forced alignment places the tag at the correct
token position; "detected somewhere in this clip" is not sufficient.

### 3.3 Tag inventory

Start with the seven the model already half-knows (`[laughing] [laughter]
[sigh] [Uhm] [Shh] [Question-ah] [Surprise-wa] [Dissatisfaction-hnn]`), because
consistency on those is the primary claim. Add new tags from the NVV-SuperBench
taxonomy where a source corpus supplies the acoustics — `[breath]`, `[gasp]`,
`[cough]`, `[throat-clear]`, `[sniff]` are all covered by VocalSound. Report
Coverage over the attempted set, per §2.5.

## 4 Training

Extends `train.py` / `conf/khmer_style_lora.yaml`, dropping the 12 GB memory
patches that exist only for the 3060. Compute for the real run is
unconstrained (rent 24–48 GB+); the 3060 is for pipeline sanity checks.

### 4.1 Settings carried over from the layer-1b diagnosis

- `enable_proj: true` is **mandatory, not a knob**. Per `results/diagnosis.md`
  the LM→DiT projections are the only route text-side information has into the
  acoustic generator; without them the adapter trains cleanly and ignores the
  tag. With `enable_lm` and `enable_dit`, this took speaker separation from
  5.8 Hz to 39.9 Hz.
- LoRA `r=64`, headroom to 128 if a rank sweep earns it; `batch_size` 16–32;
  `max_batch_tokens` 8192+.

### 4.2 Event-span-weighted loss

The layer-1b fix weighted the diffusion loss toward clip onset
(`1 + 7·exp(−i/4)` from position 0) because a *global* tag competes with the
whole clip and only the start is unpredicted. An NVV event can occur anywhere,
so the weight must track **the event's actual audio-frame span**, plus a small
margin, taken from the onset/offset mined in §3.2. Same reasoning, correct
region.

### 4.3 Consistency mechanisms — the part that is new

Reliability is the deliverable, so it gets explicit mechanisms rather than
being hoped for as a side effect of training. In increasing cost:

1. **CFG scale sweep — training-free, run first.** `cfg_value` defaults to 2.0
   and is already plumbed through `synthesize_styled.py` and
   `verify_parenthetical.py` (`--cfg-value`). The literature is unambiguous
   that raising guidance amplifies conditioning adherence at the cost of
   diversity. This costs GPU-hours, not training, and may recover a meaningful
   fraction of the consistency on its own.
2. **Tag-specific conditioning dropout, to enable a tag-specific CFG.**
   Currently `training_cfg_rate = 0.1` drops the whole `mu`. Training with the
   *tag* independently droppable creates a second guidance axis, so tag
   adherence can be amplified without also amplifying transcript adherence —
   the chained-CFG idea (cCFG, and Audio Palette's three-scale variant). This
   is the mechanism that most directly targets "consistent", and it is a
   training-time change, so it must be decided before the long run.
3. **50:50 mixing** per ELaTE (§2.2), which is a consistency mechanism as much
   as a quality one: it prevents the adapter from learning to emit events
   unconditionally.
4. **Seed variance as a first-class metric.** Every evaluation cell is
   generated across N seeds and reports the *rate*, not a single sample. A tag
   that fires 3 times in 10 is a different product from one that fires 10 in
   10, and only a rate exposes that.

### 4.4 Discipline carried forward

- Run `experiments/tag_sensitivity.py` — the falsifiable loss probe with a
  scrambled-transcript positive control, ~2 minutes — against the new inline
  conditioning scheme **before** committing to a multi-hour run. Layer 1b
  skipped this and paid for it with a wasted run to step 1,980 of 4,000.
- Pipeline sanity check with `conf/khmer_style_pilot.yaml` on whatever machine
  hosts the long run, before the long run starts.
- Select checkpoints by **control-response strength**, never by validation
  loss (`select_checkpoint.sh` pattern). Per docs/09 §9.5, and because layer
  1b's real failure was invisible in the loss curve.

## 5 Evaluation

New `finetune/verify_nvv.py`, parallel to `verify_control.py`.

### 5.1 Metrics

Per §2.5: **F1** on (type, position ≤ δ) matching, **NTD** for placement,
**Coverage** over the attempted inventory, **seed-wise firing rate** for
consistency, **NVMOS** for event acoustic quality, and **Khmer CER regression**
on the surrounding words plus a full-eval-set regression against the base
model via `score_cer.py` — to catch collateral damage to plain Khmer synthesis
and to the model's other 30 languages.

Every metric is reported **base model vs fine-tuned**, on the same held-out
sentences at the same seeds. The base-model column is the claim's denominator.

**Do not use UTMOS or DNSMOS to judge control.** Established in this project at
rho = +0.55 with Khmer CER across 400 clips, and independently in
NVV-SuperBench's finding that controllability decouples from quality.
Naturalness gets `audio_stats.py`-style ASR-free diagnostics for the mechanical
checks and a listening pass (`listening_test.py` / `listening_analyse.py`) for
the perceptual call.

### 5.2 The falsifiable check on the load-bearing assumption

§2.3's cross-lingual premise — that borrowed non-Khmer event acoustics work in
Khmer because placement is what is language-specific — is the assumption the
whole data strategy rests on. It is tested directly, not inherited: report
event F1 and NVMOS **split by whether the event's training acoustics came from
Khmer (mined DDD) or from a non-Khmer corpus**. If borrowed acoustics
underperform mined ones by a wide margin, the premise is wrong and §3.2's
third source (prompted Khmer recording) becomes primary rather than a fallback.

## 6 Write-up

Numbers land in `finetune/results/nvv/*.json` and reach the report's §4 through
the existing `src/docbuild` pipeline, the same way
`build_style_control_report.py` pulls from
`results/parenthetical/parenthetical.json`. Never hand-edit `reports/*.docx`,
per the top-level CLAUDE.md.

## 7 Risks, stated up front

1. **Khmer NVV yield from DDD is expected to be low** — measured and reported
   in §3.2, but the plan no longer depends on it, because DDD's role has been
   reassigned to the untagged half of the mix and event acoustics come
   primarily from released corpora.
2. **The cross-lingual acoustics premise may be wrong.** §5.2 tests it
   explicitly and names the fallback.
3. **Consistency may be bounded by the base model's weak prior.** If the
   emergent association is too faint, LoRA on a modest corpus may not make it
   reliable, and the answer becomes a larger adapter, more data, or the
   frame-level conditioning channel ELaTE uses instead of the text channel.
4. **The `[laughing]` string carries English-context priors** (§1), which may
   interact badly with byte-fallback Khmer around it. If so, a Khmer-script or
   neutral tag spelling is a cheap thing to try.
