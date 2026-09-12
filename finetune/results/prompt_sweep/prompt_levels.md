# How many prosody levels can a prompt address?

Base VoxCPM2, no adapter. Every rung is paired against the same sentence at the same seed with no parenthetical, so `delta` is a within-sentence change. `sd` is the spread across seeds of one sentence in one cell -- the noise any step has to clear to be audible.

## `rate` &rarr; char_rate (char/s)

### family `adverb` &mdash; rho +0.604, span 4.37 char/s, noise 1.44 char/s (3.0 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -4 | `(speaking extremely slowly, drawing every word out)` | -2.58 char/s | &mdash; |
| -3 | `(speaking very slowly and deliberately)` | -2.32 char/s | +0.26 &middot; 11/24 (p 0.839) |
| -2 | `(speaking slowly)` | -1.50 char/s | +0.82 &middot; 19/24 (p 0.007) |
| -1 | `(speaking a little slowly)` | -1.10 char/s | +0.40 &middot; 15/24 (p 0.307) |
| +0 | `(speaking at a normal pace)` | +0.07 char/s | +1.17 &middot; 15/24 (p 0.307) |
| +1 | `(speaking a little quickly)` | +0.67 char/s | +0.59 &middot; 17/24 (p 0.064) |
| +2 | `(speaking quickly)` | +1.78 char/s | +1.11 &middot; 15/24 (p 0.307) |
| +3 | `(speaking very quickly, rushed)` | +1.19 char/s | -0.60 &middot; 13/24 (p 0.839) |
| +4 | `(speaking extremely quickly, racing through the words)` | +1.65 char/s | +0.47 &middot; 13/24 (p 0.839) |

**Resolves 3 of 9 rungs** at one noise-width apart: `(speaking extremely slowly, drawing every word out)`, `(speaking a little slowly)`, `(speaking a little quickly)`

### family `multiplier` &mdash; rho -0.023, span 0.47 char/s, noise 1.57 char/s (0.3 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(speaking at 0.6x speed)` | +0.14 char/s | &mdash; |
| -1 | `(speaking at 0.8x speed)` | -0.01 char/s | -0.15 &middot; 9/24 (p 0.307) |
| +0 | `(speaking at 1.0x speed)` | +0.25 char/s | +0.26 &middot; 15/24 (p 0.307) |
| +1 | `(speaking at 1.3x speed)` | -0.21 char/s | -0.47 &middot; 7/24 (p 0.064) |
| +2 | `(speaking at 1.6x speed)` | +0.15 char/s | +0.36 &middot; 12/24 (p 1.000) |

**Resolves 1 of 5 rungs** at one noise-width apart: `(speaking at 1.3x speed)`

### family `scale` &mdash; rho -0.031, span 0.37 char/s, noise 1.46 char/s (0.3 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(speaking at a pace of 1 out of 5, where 5 is fastest)` | +0.47 char/s | &mdash; |
| +0 | `(speaking at a pace of 3 out of 5, where 5 is fastest)` | +0.35 char/s | -0.12 &middot; 12/24 (p 1.000) |
| +2 | `(speaking at a pace of 5 out of 5, where 5 is fastest)` | +0.72 char/s | +0.37 &middot; 9/24 (p 0.307) |

**Resolves 1 of 3 rungs** at one noise-width apart: `(speaking at a pace of 3 out of 5, where 5 is fastest)`

## `pitch` &rarr; f0_median_hz (Hz)

### family `adverb` &mdash; rho +0.604, span 84.89 Hz, noise 27.94 Hz (3.0 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -4 | `(an extremely low, deep voice)` | -30.25 Hz | &mdash; |
| -3 | `(a very low-pitched voice)` | -30.00 Hz | +0.25 &middot; 15/24 (p 0.307) |
| -2 | `(a low-pitched voice)` | -39.74 Hz | -9.74 &middot; 7/24 (p 0.064) |
| -1 | `(a slightly low-pitched voice)` | -7.14 Hz | +32.60 &middot; 16/24 (p 0.152) |
| +0 | `(a normal-pitched voice)` | +1.07 Hz | +8.21 &middot; 20/24 (p 0.002) |
| +1 | `(a slightly high-pitched voice)` | +8.72 Hz | +7.64 &middot; 13/24 (p 0.839) |
| +2 | `(a high-pitched voice)` | +39.76 Hz | +31.05 &middot; 16/24 (p 0.152) |
| +3 | `(a very high-pitched voice)` | +31.74 Hz | -8.02 &middot; 13/24 (p 0.839) |
| +4 | `(an extremely high-pitched voice)` | +45.15 Hz | +13.40 &middot; 13/24 (p 0.839) |

**Resolves 3 of 9 rungs** at one noise-width apart: `(a low-pitched voice)`, `(a slightly low-pitched voice)`, `(a very high-pitched voice)`

### family `multiplier` &mdash; rho +0.065, span 3.21 Hz, noise 35.64 Hz (0.1 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(a voice six semitones lower than normal)` | +7.26 Hz | &mdash; |
| -1 | `(a voice three semitones lower than normal)` | +9.37 Hz | +2.11 &middot; 17/24 (p 0.064) |
| +1 | `(a voice three semitones higher than normal)` | +10.47 Hz | +1.10 &middot; 14/24 (p 0.541) |
| +2 | `(a voice six semitones higher than normal)` | +7.26 Hz | -3.21 &middot; 5/24 (p 0.007) |

**Resolves 1 of 4 rungs** at one noise-width apart: `(a voice six semitones lower than normal)`

### family `framing` &mdash; rho +0.741, span 100.09 Hz, noise 24.22 Hz (4.1 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(an elderly man speaking)` | -1.61 Hz | &mdash; |
| -1 | `(a deep, low male voice)` | -28.99 Hz | -27.38 &middot; 10/24 (p 0.541) |
| +1 | `(a young woman speaking)` | +62.42 Hz | +91.41 &middot; 24/24 (<1e-4) |
| +2 | `(a small child speaking)` | +71.10 Hz | +8.68 &middot; 16/24 (p 0.152) |

**Resolves 3 of 4 rungs** at one noise-width apart: `(a deep, low male voice)`, `(an elderly man speaking)`, `(a young woman speaking)`

## `energy` &rarr; rms_dbfs (dB)

### family `adverb` &mdash; rho +0.395, span 6.28 dB, noise 5.63 dB (1.1 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -4 | `(speaking extremely quietly, almost inaudibly)` | -2.47 dB | &mdash; |
| -3 | `(speaking very quietly)` | -4.11 dB | -1.64 &middot; 7/24 (p 0.064) |
| -2 | `(speaking softly, quietly)` | -2.91 dB | +1.20 &middot; 14/24 (p 0.541) |
| -1 | `(speaking a little quietly)` | -0.98 dB | +1.93 &middot; 19/24 (p 0.007) |
| +0 | `(speaking at a normal volume)` | +0.49 dB | +1.47 &middot; 21/24 (p 0.000) |
| +1 | `(speaking a little loudly)` | +0.94 dB | +0.45 &middot; 15/24 (p 0.307) |
| +2 | `(speaking loudly)` | +1.36 dB | +0.41 &middot; 11/24 (p 0.839) |
| +3 | `(speaking very loudly)` | +1.44 dB | +0.08 &middot; 19/24 (p 0.007) |
| +4 | `(speaking extremely loudly, at full volume)` | +2.17 dB | +0.73 &middot; 15/24 (p 0.307) |

**Resolves 2 of 9 rungs** at one noise-width apart: `(speaking very quietly)`, `(speaking extremely loudly, at full volume)`

### family `multiplier` &mdash; rho +0.014, span 0.22 dB, noise 5.72 dB (0.0 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(speaking at 20% volume)` | +0.79 dB | &mdash; |
| -1 | `(speaking at 50% volume)` | +1.00 dB | +0.22 &middot; 13/24 (p 0.839) |
| +2 | `(speaking at 100% volume)` | +0.88 dB | -0.12 &middot; 11/24 (p 0.839) |

**Resolves 1 of 3 rungs** at one noise-width apart: `(speaking at 20% volume)`

### family `framing` &mdash; rho +0.147, span 2.49 dB, noise 5.82 dB (0.4 noise-widths)

| level | prompt | delta | vs previous rung |
|---|---|---|---|
| -2 | `(speaking from the far side of a large room)` | -0.24 dB | &mdash; |
| -1 | `(speaking quietly so as not to wake anyone)` | -1.74 dB | -1.50 &middot; 10/24 (p 0.541) |
| +2 | `(calling out to someone far away)` | +0.75 dB | +2.49 &middot; 21/24 (p 0.000) |

**Resolves 1 of 3 rungs** at one noise-width apart: `(speaking quietly so as not to wake anyone)`

