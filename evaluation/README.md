# Evaluation harness -- runbook

Synthesize the fixed 100-sentence Khmer set with each candidate model, score it,
compare. Four metrics:

| Metric | Measures | Scale | Where |
|---|---|---|---|
| **CER** | correctness / intelligibility (primary) | 0 = perfect, lower better | `metrics/cer.py` |
| **UTMOS** | predicted naturalness MOS | 1-5, higher better | `metrics/utmos.py` |
| **DNSMOS** | perceptual quality, P.835 SIG/BAK/OVRL + P.808 | 1-5, higher better | `metrics/dnsmos.py` |
| **RTF** | speed = synth time / audio duration | <1 = faster than real time | measured in `synthesize.py` |

`eval-set/eval.json` is never modified by anything here -- it is a fixed
benchmark set (see CLAUDE.md).

## Setup

Nothing is installed yet. Order matters:

```powershell
.venv\Scripts\activate

# 1. torch FIRST, from the CUDA index -- otherwise a transitive dep pulls the
#    CPU-only wheel and VoxCPM2 won't run.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. everything else
pip install -r requirements.txt
```

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

Not a pip package, and non-commercial licensed (docs/05):

```powershell
git clone https://github.com/fishaudio/fish-speech
pip install -e fish-speech
huggingface-cli download fishaudio/s2-pro --local-dir fish-speech\checkpoints\s2-pro
$env:FISH_SPEECH_DIR = "<abs path>\fish-speech"
```

## Running a full comparison

```powershell
python evaluation/synthesize.py --model mms
python evaluation/synthesize.py --model voxcpm2
python evaluation/synthesize.py --model fish-s2

python evaluation/score.py --model mms
python evaluation/score.py --model voxcpm2
python evaluation/score.py --model fish-s2

python evaluation/report.py
```

Smoke-test first: `python evaluation/synthesize.py --model mms --limit 3`
then `python evaluation/score.py --model mms --metrics cer`.

Useful flags: `--ids A01,B07`, `--group code_switched`, `--limit N`,
`--device cpu`, `--overwrite`, `--metrics cer,utmos`.

## Outputs

```
evaluation/results/<model>/audio/<id>.wav   native rate; gitignored
evaluation/results/<model>/synthesis.json   RTF + env block (device, GPU, seed)
evaluation/results/<model>/scores.json      all metrics + ASR transcripts
evaluation/results/<model>/scores.csv       one flat row per utterance
evaluation/results_report.md                the 3-model comparison
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
- **Whisper has a Khmer error floor.** CER never reaches 0 even on perfect
  audio (docs/03 section 3.4). Raw transcripts are saved in `scores.json` --
  read them before concluding a model mispronounced something.
- **`backends/fish_s2.py` is the one unverified module.** fish-speech's Python
  entrypoint has changed across releases and the repo's docs pin no API, so it
  probes several known names and tells you which function to wire in if none
  match your clone. Everything else in the harness is verified.
- **DNSMOS BAK saturates** on clean synthetic speech; read SIG and OVRL.
