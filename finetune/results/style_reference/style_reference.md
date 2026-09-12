# Does a reference recording carry speaking style?

Base VoxCPM2, no adapter. Each cell pins one speaker and changes only which variation band the reference clip is drawn from, so voice is held constant by construction and any difference is style transfer.

## `ref` -- reference_wav_path (cloning)

| measure | median delta | lively > flat | p |
|---|---|---|---|
| `f0_std_st` | +1.720 | 140/144 (97%) | <1e-4 |
| `f0_range_st` | +6.598 | 139/144 (97%) | <1e-4 |
| `f0_delta_st` | +0.401 | 130/144 (90%) | <1e-4 |
| `f0_median_hz` | +1.326 | 78/144 (54%) | 0.3594 |
| `char_rate` | -0.046 | 68/144 (47%) | 0.5598 |
| `rms_dbfs` | +0.181 | 75/144 (52%) | 0.6771 |

Dose-response: the reference clip's own `f0_std_st` against the generated clip's, rho = **+0.758** over 288 generations; median rho within a single speaker +0.706 over 6 speakers.

## `cont` -- prompt_wav_path + prompt_text (continuation)

| measure | median delta | lively > flat | p |
|---|---|---|---|
| `f0_std_st` | +2.268 | 141/144 (98%) | <1e-4 |
| `f0_range_st` | +7.411 | 140/144 (97%) | <1e-4 |
| `f0_delta_st` | +0.444 | 131/144 (91%) | <1e-4 |
| `f0_median_hz` | +0.801 | 75/144 (52%) | 0.6771 |
| `char_rate` | -1.035 | 42/144 (29%) | <1e-4 |
| `rms_dbfs` | -0.012 | 72/144 (50%) | 1.0000 |

Dose-response: the reference clip's own `f0_std_st` against the generated clip's, rho = **+0.792** over 288 generations; median rho within a single speaker +0.810 over 6 speakers.

Global F0 statistics do not track perceived expressiveness -- CLAUDE.md records `prosody_stats.py` calling `mms` the most expressive model when it was eliminated by ear for robotic prosody. A moving number here shows the channel does something; only listening shows what.

