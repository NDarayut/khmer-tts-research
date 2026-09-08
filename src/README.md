# `src/` — all the code

Split by what each part does, not by when it was written.

| Package | Does what |
|---|---|
| [`khmer_tts/`](khmer_tts/) | The evaluation harness: the eval set → audio → scores → report pipeline. |
| [`docbuild/`](docbuild/) | Generators for the company-facing `.docx` in [`../reports/`](../reports/). |

## `khmer_tts/`

| Module | Does what |
|---|---|
| `common.py` | Shared plumbing — paths, eval-set loading, wav io, resampling. |
| [`dataset/`](khmer_tts/dataset/) | Describe and QC the fixed 100-sentence eval set. Stdlib only. |
| [`synthesis/`](khmer_tts/synthesis/) | Eval set → audio, one lazily-imported backend per model, plus RTF. |
| [`scoring/`](khmer_tts/scoring/) | Audio → metrics: CER, UTMOS, DNSMOS, and ASR-free signal diagnostics. |
| [`listening/`](khmer_tts/listening/) | The blind A/B test — the only instrument that measures naturalness. |
| [`reporting/`](khmer_tts/reporting/) | Scores → the 4-model comparison report. |

Each package's `__init__.py` says what its modules are for. Importing any of
them pulls in nothing third-party — heavy dependencies are imported lazily by
the module that needs them, so a missing `voxcpm` does not stop you running MMS.

## Running things

Every script is runnable by path and bootstraps `src/` onto `sys.path` itself:

```
python src/khmer_tts/synthesis/synthesize.py --model voxcpm2
```

Or as modules, with `src/` on the path:

```
PYTHONPATH=src python -m khmer_tts.synthesis.synthesize --model voxcpm2
```

Full runbook: [`../evaluation/README.md`](../evaluation/README.md).

## Where the data lives

Code is here; data and outputs are not.

| Path | What |
|---|---|
| `../eval-set/eval.json` | The fixed 100-sentence benchmark. **Nothing in `src/` ever writes to it** — it must stay identical across all four models for the comparison to be fair (CLAUDE.md). |
| `../evaluation/results/` | Per-model audio (gitignored) and scores (tracked). |
| `../evaluation/` | The runbook and the generated QC / overview / report markdown. |
| `../reports/` | The `.docx` / `.pdf` deliverables built by `docbuild/`. |
