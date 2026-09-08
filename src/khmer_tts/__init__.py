"""
Khmer TTS evaluation harness, split by what each part does.

    common.py    shared plumbing -- paths, eval-set loading, wav io, resampling
    dataset/     describing and QC-ing the fixed 100-sentence eval set
    synthesis/   eval set -> audio, one backend per model, plus RTF
    scoring/     audio -> metrics (CER, UTMOS, DNSMOS, signal diagnostics)
    listening/   the blind A/B test, which is the only naturalness instrument
    reporting/   scores -> the 4-model comparison report

Importing this package pulls in nothing third-party; every heavy dependency is
imported lazily by the module that needs it, so a missing `voxcpm` does not
stop you running MMS.

Runbook: evaluation/README.md. Outputs land in evaluation/results/.
Nothing here ever writes to eval-set/eval.json -- that set is fixed (CLAUDE.md).
"""
