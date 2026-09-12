# Which parenthetical prompts actually steer Khmer prosody?

Base VoxCPM2, no adapter. Every prompt is paired against the *same* sentence at the *same* seed with no parenthetical at all, so `delta` is a within-sentence change and `hit rate` is the fraction of (sentence, seed) pairs that moved the way the wording asked.

`sd` is the spread across seeds of one sentence in one condition -- the run-to-run noise. `snr` is |delta| / sd. **A prompt is usable when the hit rate is high and snr exceeds about 1**; a large delta with snr below 1 is a prompt that sometimes works, which is what we are trying to avoid.

## `var` -> f0_std_st *(primary)*

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `pauses` | (reading with dramatic pauses between phrases) | up | +0.37 | 14/24 (58%) | 0.5413 | 0.75 | 0.49 |
| `robot` | (a robotic, emotionless machine voice) | down | +0.26 | 10/24 (42%) | 0.5413 | 0.93 | 0.27 |
| `bored` | (a bored, deadpan tone) | down | +0.25 | 10/24 (42%) | 0.5413 | 0.75 | 0.33 |
| `pin_low_expr` | (a low-pitched voice speaking very expressively) | up | +0.22 | 14/24 (58%) | 0.5413 | 0.84 | 0.26 |
| `normal_delivery` | (a normal delivery) | -- | +0.09 | -- | -- | 0.63 | 0.14 |
| `anim_very` | (very expressive and animated) | up | +0.05 | 12/24 (50%) | 1.0000 | 0.53 | 0.09 |
| `dramatic` | (a dramatic, theatrical tone) | up | +0.03 | 13/24 (54%) | 0.8388 | 0.44 | 0.08 |
| `kh_flat` | (និយាយរាបស្មើគ្មានអារម្មណ៍) | down | +0.03 | 12/24 (50%) | 1.0000 | 0.63 | 0.05 |
| `melody` | (an audiobook narrator giving each phrase its own melody) | up | -0.00 | 12/24 (50%) | 1.0000 | 0.77 | 0.00 |
| `even_stress` | (giving every word exactly the same stress and length) | down | -0.03 | 13/24 (54%) | 0.8388 | 0.80 | 0.04 |
| `pin_same_emph` | (keeping the same pitch, but with strong emphasis on important words) | up | -0.03 | 12/24 (50%) | 1.0000 | 0.47 | 0.07 |
| `lots_of_into` | (with a lot of intonation, rising and falling) | up | -0.08 | 11/24 (46%) | 0.8388 | 0.68 | 0.11 |
| `newsreader` | (a news anchor reading calmly and evenly) | down | -0.12 | 13/24 (54%) | 0.8388 | 0.64 | 0.19 |
| `pin_normal_wide` | (a normal-pitched voice with wide pitch variation) | up | -0.13 | 11/24 (46%) | 0.8388 | 0.46 | 0.29 |
| `wide_pitch` | (with wide pitch variation) | up | -0.15 | 9/24 (38%) | 0.3075 | 0.48 | 0.31 |
| `excited` | (an excited, enthusiastic tone) | up | -0.22 | 8/24 (33%) | 0.1516 | 0.65 | 0.33 |
| `narrow_pitch` | (with very little pitch variation) | down | -0.23 | 16/24 (67%) | 0.1516 | 0.66 | 0.35 |
| `level_tone` | (in a level, unchanging tone) | down | -0.25 | 17/24 (71%) | 0.0639 | 0.53 | 0.47 |
| `anim_extreme` | (extremely expressive and animated) | up | -0.27 | 11/24 (46%) | 0.8388 | 0.87 | 0.30 |
| `sportscaster` | (an excited sports commentator) | up | -0.33 | 8/24 (33%) | 0.1516 | 0.76 | 0.44 |
| `kh_lively` | (និយាយដោយមានអារម្មណ៍រំភើប) | up | -0.34 | 8/24 (33%) | 0.1516 | 0.76 | 0.45 |
| `flat_monotone` | (a flat, monotone delivery) | down | -0.36 | 13/24 (54%) | 0.8388 | 0.81 | 0.45 |
| `anim_slight` | (slightly expressive and animated) | up | -0.41 | 10/24 (42%) | 0.5413 | 0.45 | 0.92 |
| `loud_soft` | (alternating between loud and soft) | up | -0.44 | 8/24 (33%) | 0.1516 | 0.60 | 0.73 |
| `emphasis` | (with strong emphasis and clear stress on key words) | up | -0.49 | 8/24 (33%) | 0.1516 | 0.56 | 0.87 |
| `anim_fairly` | (fairly expressive and animated) | up | -0.53 | 10/24 (42%) | 0.5413 | 0.50 | 1.06 |
| `lively_expressive` | (a lively, expressive delivery) | up | -0.66 | 5/24 (21%) | 0.0066 | 0.59 | 1.13 |
| `storyteller` | (reading a story to a child, warm and animated) | up | -0.69 | 7/24 (29%) | 0.0639 | 0.70 | 0.99 |

## `var` -> f0_range_st

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `robot` | (a robotic, emotionless machine voice) | down | +2.01 | 11/24 (46%) | 0.8388 | 3.98 | 0.50 |
| `bored` | (a bored, deadpan tone) | down | +1.96 | 9/24 (38%) | 0.3075 | 3.35 | 0.58 |
| `pin_low_expr` | (a low-pitched voice speaking very expressively) | up | +1.22 | 14/24 (58%) | 0.5413 | 3.22 | 0.38 |
| `kh_flat` | (និយាយរាបស្មើគ្មានអារម្មណ៍) | down | +0.60 | 10/24 (42%) | 0.5413 | 2.98 | 0.20 |
| `pauses` | (reading with dramatic pauses between phrases) | up | +0.58 | 13/24 (54%) | 0.8388 | 3.48 | 0.17 |
| `anim_very` | (very expressive and animated) | up | +0.19 | 12/24 (50%) | 1.0000 | 3.02 | 0.06 |
| `lots_of_into` | (with a lot of intonation, rising and falling) | up | -0.20 | 11/24 (46%) | 0.8388 | 2.59 | 0.08 |
| `pin_same_emph` | (keeping the same pitch, but with strong emphasis on important words) | up | -0.32 | 11/24 (46%) | 0.8388 | 1.50 | 0.21 |
| `normal_delivery` | (a normal delivery) | -- | -0.38 | -- | -- | 2.68 | 0.14 |
| `newsreader` | (a news anchor reading calmly and evenly) | down | -0.39 | 14/24 (58%) | 0.5413 | 3.08 | 0.13 |
| `wide_pitch` | (with wide pitch variation) | up | -0.48 | 12/24 (50%) | 1.0000 | 2.96 | 0.16 |
| `melody` | (an audiobook narrator giving each phrase its own melody) | up | -0.49 | 10/24 (42%) | 0.5413 | 3.74 | 0.13 |
| `flat_monotone` | (a flat, monotone delivery) | down | -0.62 | 14/24 (58%) | 0.5413 | 3.20 | 0.19 |
| `narrow_pitch` | (with very little pitch variation) | down | -0.75 | 16/24 (67%) | 0.1516 | 2.78 | 0.27 |
| `even_stress` | (giving every word exactly the same stress and length) | down | -0.91 | 16/24 (67%) | 0.1516 | 3.72 | 0.25 |
| `level_tone` | (in a level, unchanging tone) | down | -1.21 | 16/24 (67%) | 0.1516 | 2.28 | 0.53 |
| `anim_extreme` | (extremely expressive and animated) | up | -1.28 | 8/24 (33%) | 0.1516 | 3.13 | 0.41 |
| `anim_fairly` | (fairly expressive and animated) | up | -1.35 | 10/24 (42%) | 0.5413 | 1.99 | 0.68 |
| `pin_normal_wide` | (a normal-pitched voice with wide pitch variation) | up | -1.49 | 11/24 (46%) | 0.8388 | 2.39 | 0.62 |
| `loud_soft` | (alternating between loud and soft) | up | -1.52 | 8/24 (33%) | 0.1516 | 2.98 | 0.51 |
| `anim_slight` | (slightly expressive and animated) | up | -1.53 | 7/24 (29%) | 0.0639 | 2.24 | 0.68 |
| `emphasis` | (with strong emphasis and clear stress on key words) | up | -1.85 | 7/24 (29%) | 0.0639 | 1.90 | 0.98 |
| `dramatic` | (a dramatic, theatrical tone) | up | -1.85 | 9/24 (38%) | 0.3075 | 1.76 | 1.05 |
| `sportscaster` | (an excited sports commentator) | up | -1.87 | 7/24 (29%) | 0.0639 | 2.70 | 0.69 |
| `excited` | (an excited, enthusiastic tone) | up | -2.01 | 8/24 (33%) | 0.1516 | 2.21 | 0.91 |
| `kh_lively` | (និយាយដោយមានអារម្មណ៍រំភើប) | up | -2.02 | 7/24 (29%) | 0.0639 | 3.19 | 0.63 |
| `lively_expressive` | (a lively, expressive delivery) | up | -2.37 | 4/24 (17%) | 0.0015 | 2.09 | 1.13 |
| `storyteller` | (reading a story to a child, warm and animated) | up | -3.11 | 4/24 (17%) | 0.0015 | 2.43 | 1.28 |

## `var` -> f0_delta_st

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `pin_low_expr` | (a low-pitched voice speaking very expressively) | up | +0.12 | 13/24 (54%) | 0.8388 | 0.49 | 0.25 |
| `newsreader` | (a news anchor reading calmly and evenly) | down | +0.03 | 12/24 (50%) | 1.0000 | 0.31 | 0.10 |
| `bored` | (a bored, deadpan tone) | down | +0.03 | 11/24 (46%) | 0.8388 | 0.38 | 0.08 |
| `kh_flat` | (និយាយរាបស្មើគ្មានអារម្មណ៍) | down | +0.02 | 12/24 (50%) | 1.0000 | 0.28 | 0.06 |
| `robot` | (a robotic, emotionless machine voice) | down | +0.01 | 10/24 (42%) | 0.5413 | 0.42 | 0.03 |
| `pauses` | (reading with dramatic pauses between phrases) | up | -0.01 | 12/24 (50%) | 1.0000 | 0.27 | 0.05 |
| `even_stress` | (giving every word exactly the same stress and length) | down | -0.04 | 13/24 (54%) | 0.8388 | 0.33 | 0.13 |
| `dramatic` | (a dramatic, theatrical tone) | up | -0.05 | 10/24 (42%) | 0.5413 | 0.25 | 0.20 |
| `normal_delivery` | (a normal delivery) | -- | -0.05 | -- | -- | 0.28 | 0.19 |
| `pin_normal_wide` | (a normal-pitched voice with wide pitch variation) | up | -0.06 | 9/24 (38%) | 0.3075 | 0.25 | 0.24 |
| `melody` | (an audiobook narrator giving each phrase its own melody) | up | -0.06 | 10/24 (42%) | 0.5413 | 0.35 | 0.17 |
| `loud_soft` | (alternating between loud and soft) | up | -0.08 | 9/24 (38%) | 0.3075 | 0.25 | 0.30 |
| `anim_very` | (very expressive and animated) | up | -0.10 | 11/24 (46%) | 0.8388 | 0.23 | 0.41 |
| `wide_pitch` | (with wide pitch variation) | up | -0.13 | 8/24 (33%) | 0.1516 | 0.29 | 0.44 |
| `anim_extreme` | (extremely expressive and animated) | up | -0.15 | 5/24 (21%) | 0.0066 | 0.22 | 0.69 |
| `level_tone` | (in a level, unchanging tone) | down | -0.16 | 16/24 (67%) | 0.1516 | 0.29 | 0.55 |
| `pin_same_emph` | (keeping the same pitch, but with strong emphasis on important words) | up | -0.16 | 9/24 (38%) | 0.3075 | 0.26 | 0.63 |
| `anim_fairly` | (fairly expressive and animated) | up | -0.18 | 5/24 (21%) | 0.0066 | 0.20 | 0.88 |
| `kh_lively` | (និយាយដោយមានអារម្មណ៍រំភើប) | up | -0.20 | 8/24 (33%) | 0.1516 | 0.40 | 0.49 |
| `flat_monotone` | (a flat, monotone delivery) | down | -0.20 | 17/24 (71%) | 0.0639 | 0.34 | 0.58 |
| `anim_slight` | (slightly expressive and animated) | up | -0.20 | 8/24 (33%) | 0.1516 | 0.22 | 0.89 |
| `lots_of_into` | (with a lot of intonation, rising and falling) | up | -0.20 | 6/24 (25%) | 0.0227 | 0.32 | 0.63 |
| `sportscaster` | (an excited sports commentator) | up | -0.21 | 7/24 (29%) | 0.0639 | 0.25 | 0.83 |
| `narrow_pitch` | (with very little pitch variation) | down | -0.22 | 16/24 (67%) | 0.1516 | 0.28 | 0.81 |
| `excited` | (an excited, enthusiastic tone) | up | -0.24 | 8/24 (33%) | 0.1516 | 0.27 | 0.87 |
| `lively_expressive` | (a lively, expressive delivery) | up | -0.27 | 5/24 (21%) | 0.0066 | 0.18 | 1.51 |
| `emphasis` | (with strong emphasis and clear stress on key words) | up | -0.36 | 9/24 (38%) | 0.3075 | 0.22 | 1.63 |
| `storyteller` | (reading a story to a child, warm and animated) | up | -0.43 | 6/24 (25%) | 0.0227 | 0.29 | 1.48 |

## `var` -> energy_cv

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `narrow_pitch` | (with very little pitch variation) | down | +0.03 | 9/24 (38%) | 0.3075 | 0.11 | 0.29 |
| `anim_very` | (very expressive and animated) | up | +0.03 | 14/24 (58%) | 0.5413 | 0.08 | 0.33 |
| `bored` | (a bored, deadpan tone) | down | +0.02 | 7/24 (29%) | 0.0639 | 0.07 | 0.30 |
| `level_tone` | (in a level, unchanging tone) | down | +0.02 | 8/24 (33%) | 0.1516 | 0.09 | 0.24 |
| `pin_same_emph` | (keeping the same pitch, but with strong emphasis on important words) | up | +0.01 | 14/24 (58%) | 0.5413 | 0.07 | 0.18 |
| `melody` | (an audiobook narrator giving each phrase its own melody) | up | +0.01 | 13/24 (54%) | 0.8388 | 0.09 | 0.15 |
| `pauses` | (reading with dramatic pauses between phrases) | up | +0.01 | 13/24 (54%) | 0.8388 | 0.09 | 0.10 |
| `newsreader` | (a news anchor reading calmly and evenly) | down | +0.00 | 10/24 (42%) | 0.5413 | 0.08 | 0.05 |
| `even_stress` | (giving every word exactly the same stress and length) | down | +0.00 | 11/24 (46%) | 0.8388 | 0.08 | 0.03 |
| `dramatic` | (a dramatic, theatrical tone) | up | +0.00 | 12/24 (50%) | 1.0000 | 0.09 | 0.01 |
| `robot` | (a robotic, emotionless machine voice) | down | -0.00 | 12/24 (50%) | 1.0000 | 0.09 | 0.00 |
| `wide_pitch` | (with wide pitch variation) | up | -0.00 | 11/24 (46%) | 0.8388 | 0.07 | 0.03 |
| `pin_normal_wide` | (a normal-pitched voice with wide pitch variation) | up | -0.00 | 11/24 (46%) | 0.8388 | 0.07 | 0.06 |
| `normal_delivery` | (a normal delivery) | -- | -0.01 | -- | -- | 0.08 | 0.07 |
| `pin_low_expr` | (a low-pitched voice speaking very expressively) | up | -0.01 | 11/24 (46%) | 0.8388 | 0.08 | 0.08 |
| `loud_soft` | (alternating between loud and soft) | up | -0.01 | 10/24 (42%) | 0.5413 | 0.07 | 0.12 |
| `lots_of_into` | (with a lot of intonation, rising and falling) | up | -0.02 | 10/24 (42%) | 0.5413 | 0.09 | 0.21 |
| `emphasis` | (with strong emphasis and clear stress on key words) | up | -0.02 | 11/24 (46%) | 0.8388 | 0.08 | 0.23 |
| `excited` | (an excited, enthusiastic tone) | up | -0.02 | 10/24 (42%) | 0.5413 | 0.09 | 0.22 |
| `anim_extreme` | (extremely expressive and animated) | up | -0.02 | 10/24 (42%) | 0.5413 | 0.06 | 0.37 |
| `anim_fairly` | (fairly expressive and animated) | up | -0.02 | 7/24 (29%) | 0.0639 | 0.07 | 0.34 |
| `sportscaster` | (an excited sports commentator) | up | -0.03 | 8/24 (33%) | 0.1516 | 0.07 | 0.43 |
| `storyteller` | (reading a story to a child, warm and animated) | up | -0.03 | 8/24 (33%) | 0.1516 | 0.07 | 0.44 |
| `anim_slight` | (slightly expressive and animated) | up | -0.03 | 6/24 (25%) | 0.0227 | 0.05 | 0.61 |
| `flat_monotone` | (a flat, monotone delivery) | down | -0.03 | 17/24 (71%) | 0.0639 | 0.06 | 0.52 |
| `kh_flat` | (និយាយរាបស្មើគ្មានអារម្មណ៍) | down | -0.05 | 15/24 (62%) | 0.3075 | 0.09 | 0.59 |
| `kh_lively` | (និយាយដោយមានអារម្មណ៍រំភើប) | up | -0.05 | 8/24 (33%) | 0.1516 | 0.08 | 0.69 |
| `lively_expressive` | (a lively, expressive delivery) | up | -0.06 | 7/24 (29%) | 0.0639 | 0.06 | 1.05 |

### `var` -> leakage onto the other axes

What else each wording moved. A prompt that changes pitch variation by selecting a different-sounding voice, or by slowing the speech down, is not a variation control.

| prompt | asks | d f0 median (Hz) | d rate (char/s) | d level (dB) |
|---|---|---|---|---|
| `anim_extreme` | up | +59.95 | +0.52 | +0.52 |
| `anim_fairly` | up | +44.50 | +0.30 | +1.38 |
| `anim_slight` | up | +44.74 | +0.08 | +1.52 |
| `anim_very` | up | +36.63 | +0.31 | +0.41 |
| `bored` | down | -0.68 | +0.82 | +0.00 |
| `dramatic` | up | +23.35 | +0.16 | +1.26 |
| `emphasis` | up | +22.64 | +0.41 | +0.84 |
| `even_stress` | down | +8.29 | +0.45 | +0.26 |
| `excited` | up | +53.78 | +0.21 | +1.30 |
| `flat_monotone` | down | -0.15 | +0.34 | +0.06 |
| `kh_flat` | down | -32.04 | -3.64 | -0.57 |
| `kh_lively` | up | -32.97 | -2.42 | -0.62 |
| `level_tone` | down | +2.14 | +0.17 | +0.14 |
| `lively_expressive` | up | +54.89 | +0.97 | -0.25 |
| `lots_of_into` | up | +18.02 | +0.29 | +0.50 |
| `loud_soft` | up | +1.98 | -0.28 | +2.50 |
| `melody` | up | +12.53 | +0.36 | +0.43 |
| `narrow_pitch` | down | +4.22 | -0.36 | -0.55 |
| `newsreader` | down | +10.32 | +0.64 | +0.15 |
| `normal_delivery` | -- | +19.46 | +0.11 | +0.23 |
| `pauses` | up | +20.81 | +0.22 | +0.11 |
| `pin_low_expr` | up | -24.86 | -0.21 | -0.04 |
| `pin_normal_wide` | up | +39.86 | +0.17 | +1.19 |
| `pin_same_emph` | up | +14.46 | +0.47 | +0.71 |
| `robot` | down | +11.93 | +0.62 | +0.20 |
| `sportscaster` | up | +37.71 | +0.81 | +0.52 |
| `storyteller` | up | +76.62 | -0.65 | +1.43 |
| `wide_pitch` | up | +24.35 | +0.19 | +0.72 |

## `rate` -> char_rate *(primary)*

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `quick` | (speaking quickly) | up | +1.78 | 18/24 (75%) | 0.0227 | 1.47 | 1.21 |
| `very_quick` | (speaking very quickly, rushed) | up | +1.19 | 17/24 (71%) | 0.0639 | 1.52 | 0.78 |
| `slight_quick` | (speaking slightly quickly) | up | +0.18 | 15/24 (62%) | 0.3075 | 1.13 | 0.16 |
| `normal_pace` | (speaking at a normal pace) | -- | +0.07 | -- | -- | 1.59 | 0.05 |
| `slight_slow` | (speaking slightly slowly) | down | -0.49 | 15/24 (62%) | 0.3075 | 1.66 | 0.29 |
| `slow` | (speaking slowly) | down | -1.50 | 18/24 (75%) | 0.0227 | 1.67 | 0.90 |
| `very_slow` | (speaking very slowly and deliberately) | down | -2.32 | 22/24 (92%) | <1e-4 | 1.60 | 1.45 |

## `pitch` -> f0_median_hz *(primary)*

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `high` | (a high-pitched voice) | up | +39.76 | 21/24 (88%) | 0.0003 | 26.82 | 1.48 |
| `very_high` | (a very high-pitched voice) | up | +31.74 | 18/24 (75%) | 0.0227 | 30.10 | 1.05 |
| `normal_pitch` | (a normal-pitched voice) | -- | +1.07 | -- | -- | 38.20 | 0.03 |
| `deep` | (a deep, low male voice) | down | -28.99 | 18/24 (75%) | 0.0227 | 15.96 | 1.82 |
| `low` | (a low-pitched voice) | down | -39.74 | 18/24 (75%) | 0.0227 | 21.33 | 1.86 |

## `energy` -> rms_dbfs *(primary)*

| prompt | wording | asks | delta | hit rate | p | sd | snr |
|---|---|---|---|---|---|---|---|
| `loud` | (speaking loudly) | up | +1.36 | 19/24 (79%) | 0.0066 | 5.20 | 0.26 |
| `normal_vol` | (speaking at a normal volume) | -- | +0.49 | -- | -- | 5.37 | 0.09 |
| `shout` | (shouting, projecting the voice) | up | +0.36 | 15/24 (62%) | 0.3075 | 6.15 | 0.06 |
| `whisper` | (whispering) | down | -1.07 | 14/24 (58%) | 0.5413 | 5.34 | 0.20 |
| `soft` | (speaking softly, quietly) | down | -2.91 | 21/24 (88%) | 0.0003 | 6.50 | 0.45 |

