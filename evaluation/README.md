# Evaluation harness -- runbook

Synthesize the fixed 100-sentence Khmer set with each candidate model, score it,
compare.

**The code lives in [`../src/khmer_tts/`](../src/khmer_tts/)**, split by what it
does (`dataset/`, `synthesis/`, `scoring/`, `listening/`, `reporting/`). This
directory holds what the code *produces*: `results/` (audio, gitignored; scores,
tracked), plus the generated QC report, dataset overview and comparison report.

Four metrics:

| Metric | Measures | Scale | Where |
|---|---|---|---|
| **CER** | correctness / intelligibility (primary) | 0 = perfect, lower better | `src/khmer_tts/scoring/metrics/cer.py` |
| **UTMOS** | predicted naturalness MOS | 1-5, higher better | `src/khmer_tts/scoring/metrics/utmos.py` |
| **DNSMOS** | perceptual quality, P.835 SIG/BAK/OVRL + P.808 | 1-5, higher better | `src/khmer_tts/scoring/metrics/dnsmos.py` |
| **RTF** | speed = synth time / audio duration | <1 = faster than real time | measured in `synthesis/synthesize.py` |

`eval-set/eval.json` is never modified by anything here -- it is a fixed
benchmark set (see CLAUDE.md).

## Setup

Order matters:

```bash
python3.11 -m venv .venv

# 1. torch FIRST, from the CUDA index -- otherwise a transitive dep pulls the
#    CPU-only wheel and VoxCPM2 won't run.
.venv/bin/python -m pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 2. everything else
.venv/bin/python -m pip install -r requirements.txt
```

This venv runs `mms` and `voxcpm2`, plus all of `score.py` and `report.py`.
`fish-s2` needs a second one -- see below.

### DNSMOS model files (manual, one time)

`pip` cannot supply these. Grab them from Microsoft's DNS-Challenge repo:

```powershell
git clone --depth 1 https://github.com/microsoft/DNS-Challenge
mkdir evaluation\dnsmos_models
copy DNS-Challenge\DNSMOS\DNSMOS\sig_bak_ovr.onnx evaluation\dnsmos_models\
copy DNS-Challenge\DNSMOS\DNSMOS\model_v8.onnx    evaluation\dnsmos_models\
```

Or keep them elsewhere and pass `--dnsmos-dir` / set `DNSMOS_MODEL_DIR`.
UTMOS needs no manual step, but downloads its checkpoint from `torch.hub` the
first time it runs.

### Fish S2-Pro (manual, one time)

Not a pip package, and non-commercial licensed (docs/05). **It needs its own
venv**: fish-speech pins `torch==2.8.0`, which would replace the CUDA build the
other two backends run on.

```bash
git clone https://github.com/fishaudio/fish-speech
python3.11 -m venv .venv-fish

# pyaudio is in its dependency list, needs system portaudio headers to build,
# and is only used for live mic I/O -- drop it and install the rest.
grep -v '^pyaudio' <(python -c "import tomllib;print('\n'.join(
    tomllib.load(open('fish-speech/pyproject.toml','rb'))['project']['dependencies']))") \
    > /tmp/fish-deps.txt
.venv-fish/bin/python -m pip install -r /tmp/fish-deps.txt
.venv-fish/bin/python -m pip install --no-deps -e ./fish-speech

hf download fishaudio/s2-pro --local-dir fish-speech/checkpoints/s2-pro  # 11 GB
export FISH_SPEECH_DIR=$PWD/fish-speech
```

Then run *only this backend* under that interpreter:

```bash
FISH_SPEECH_DIR=$PWD/fish-speech \
    .venv-fish/bin/python src/khmer_tts/synthesis/synthesize.py --model fish-s2
```

Scoring still runs from the main venv -- it only reads wavs.

**Hardware:** this does not fit a 12 GB GPU. See the header of
`backends/fish_s2.py`; you need ~16 GB+ or a quantized load.

## Running a full comparison

```powershell
python src/khmer_tts/synthesis/synthesize.py --model mms
python src/khmer_tts/synthesis/synthesize.py --model voxcpm2
python src/khmer_tts/synthesis/synthesize.py --model fish-s2
python src/khmer_tts/synthesis/synthesize.py --model higgs3

python src/khmer_tts/scoring/score.py --model mms
python src/khmer_tts/scoring/score.py --model voxcpm2
python src/khmer_tts/scoring/score.py --model fish-s2
python src/khmer_tts/scoring/score.py --model higgs3

python src/khmer_tts/reporting/report.py
```

### The re-scoring passes (added after the first run)

`score.py`'s CER column is scored by Whisper-large-v3 and is **invalid for
Khmer** -- see the metrics section of the root `CLAUDE.md`. Three scripts
replace and cross-check it. They read the clips already on disk, so none of
them requires re-synthesis:

```bash
# CER, with a Khmer-capable ASR. Needs THAT repo's interpreter, not this venv:
# fairseq2 ships compiled extensions and pins a different torch.
/run/media/pc/disk1/streaming_asr/venv/bin/python src/khmer_tts/scoring/score_cer_khmer.py

# UTMOS + DNSMOS under four level/silence conditions, plus paired win rates
python src/khmer_tts/scoring/score_naturalness.py --device cuda

# ASR-free signal diagnostics: rate, level, clipping, silence, truncation guard
python src/khmer_tts/scoring/audio_stats.py

# prosody description (F0, energy, pausing). NOT a naturalness score -- see below
python src/khmer_tts/scoring/prosody_stats.py
```

### Measuring naturalness

No automatic metric works for this (UTMOS is inverted; prosody statistics fail
their own sanity check against `mms`). Naturalness is measured on ears:

```bash
# build a blind, randomized A/B test -- one self-contained html file
python src/khmer_tts/listening/listening_test.py --pair voxcpm2:higgs3 --sentences 40 --anchors 6

# send listening_test.html to 5+ Khmer speakers, collect their .json, then
python src/khmer_tts/listening/listening_analyse.py responses/*.json
```

The answer key is written to `results/listening_test_key.json` and deliberately
does *not* travel with the page. Power: 194 trials resolves a 60/40 preference,
85 resolves 65/35 -- hence 5 listeners x 40 sentences as the default.

`score_cer_khmer.py` does 400 clips in ~25 s. `score_naturalness.py` is the
slow one (4 conditions x 2 predictors x 400 clips, ~20 min, DNSMOS on CPU is
the bottleneck) -- run it in the background.

Smoke-test first: `python src/khmer_tts/synthesis/synthesize.py --model mms --limit 3`
then `python src/khmer_tts/scoring/score.py --model mms --metrics cer`.

Useful flags: `--ids A01,B07`, `--group code_switched`, `--limit N`,
`--device cpu`, `--overwrite`, `--metrics cer,utmos`.

## Outputs

```
evaluation/results/<model>/audio/<id>.wav   native rate; gitignored
evaluation/results/<model>/synthesis.json   RTF + env block (device, GPU, seed)
evaluation/results/<model>/scores.json      all metrics + ASR transcripts
evaluation/results/<model>/scores.csv       one flat row per utterance
evaluation/results/<model>/cer_khmer_asr.json   CER from the Khmer ASR + transcripts
evaluation/results/naturalness.json         UTMOS/DNSMOS x 4 conditions + paired
evaluation/results/audio_stats.json         per-clip signal measurements
evaluation/results_report.md                the 4-model comparison
```

Synthesis is resumable: ids that already have a wav are skipped (and their
timings carried forward) unless you pass `--overwrite`. A model that fails on
one sentence records the error and continues.

## Gotchas

- **RTF is only comparable within one machine.** It excludes model load time
  and is measured after a discarded warm-up utterance, but it is still
  hardware-bound -- check the env block in `results_report.md` before comparing
  the speed column. A CPU VoxCPM2 run is not comparable to a GPU one.
- **MMS is stochastic.** VITS's duration predictor randomizes prosody and
  therefore audio length and RTF. `backends/mms.py` reseeds per utterance from
  `(seed, sentence)` so a partial run matches a full one; don't remove that.
- **Whisper cannot read Khmer at all.** The CER column in `scores.json` is a
  record of that failure, not a measurement -- it collapses into repetition
  loops and lands at ~100% for every model. Use `score_cer_khmer.py` instead.
- **The Khmer ASR scorer is not neutral.** It trained on ~47 h of
  VoxCPM2-synthesized speech and no Higgs, so its CER favours VoxCPM2. The
  header of `score_cer_khmer.py` explains how to read a result around that.
  There is still a floor: 12-15% CER on out-of-domain *natural* speech.
- **Do not rank models with UTMOS or DNSMOS on Khmer.** Their correlation with
  Khmer CER across this run is *positive* (rho +0.55) -- the clips they like
  best are the ones that get the Khmer most wrong. Fine as artefact detectors
  within one model's output; inverted between models.
- **DNSMOS BAK saturates** on clean synthetic speech; read SIG and OVRL.
- **`prosody_stats.py` describes, it does not judge.** Its own built-in check
  fails: `mms` has the *widest* pitch variation in the run despite being the
  model eliminated for flat prosody. Wide-but-wrong pitch movement reads as
  robotic and the statistic cannot tell the difference.
- **The two contenders use different default voices** (median F0 ~206 Hz vs
  ~114 Hz). Before running a listening test whose result you intend to act on,
  re-synthesize both from a single reference clip so the comparison is of
  synthesis rather than of timbre.
