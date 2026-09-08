# Does the built-in parenthetical prompt control Khmer prosody?

Base VoxCPM2, no adapter. 8 sentences x 3 levels x 3 generations, median per cell.

| axis | measure | low | mid | high | low->high | corpus ceiling | rho | p | run-to-run sd |
|---|---|---|---|---|---|---|---|---|---|
| `rate` | char_rate | 14.67 | 15.84 | 16.96 | **+2.29** | +6.57 | +0.590 | 0.0035 | 1.58 |
| `pitch` | f0_median_hz | 147.80 | 202.81 | 254.16 | **+106.37** | +28.27 | +0.759 | 0.0002 | 28.78 |
| `var` | f0_std_st | 4.55 | 4.06 | 3.92 | **-0.63** | +1.71 | -0.273 | 0.2039 | 0.68 |
| `energy` | rms_dbfs | -20.38 | -16.30 | -15.89 | **+4.50** | +5.81 | +0.605 | 0.0022 | 5.69 |

`run-to-run sd` is the standard deviation across repeated generations of the *same* cell. Where it is comparable to `low->high`, the mechanism is not reliably controllable even if rho looks positive.
