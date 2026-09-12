# Prosody prompts for Khmer on VoxCPM2

A vetted list of parenthetical prompts, with what each one actually does and how
often it does it. Prefix the prompt to the Khmer text with no separator:

```python
model.generate(text="(speaking very slowly and deliberately)" + khmer_sentence)
```

Everything here is measured on the **unmodified** model -- no adapter. Source:
`finetune/sweep_prosody_prompts.py`, 1,176 generations, 8 sentences from the
frozen eval set x 3 seeds, each prompt paired against the *same sentence at the
same seed with no parenthetical at all*.

**The sample is 8 sentences x 3 seeds = 24 clips per prompt, not 24 sentences.**
That distinction matters: a prompt can score 18/24 by working on six sentences
and failing on two entirely. Where a per-sentence count is given below, it is
the number of the 8 distinct sentences whose median across its own seeds moved
the way the wording asked -- the figure that says whether a prompt travels to
text it has not seen.

Two numbers decide whether a prompt is usable, and they answer different
questions:

* **hit rate** -- of 24 (sentence, seed) pairs, how many moved the way the
  wording asked. This is "will it work on the next sentence".
* **snr** -- the median effect divided by the seed-to-seed standard deviation
  within one sentence. This is "can I predict how much". Below 1, the noise is
  larger than the effect and any single generation is a coin toss on magnitude.

A prompt can pass one and fail the other, and `energy` below is exactly that case.

---

## Recommended: reliable in direction *and* magnitude

hit rate >= 75% and snr >= 1.0.

| use for | prompt | effect | hit rate | snr |
|---|---|---|---|---|
| lower voice | `(a low-pitched voice)` | -39.7 Hz | 18/24 (75%) | 1.86 |
| lower voice, male | `(a deep, low male voice)` | -29.0 Hz | 18/24 (75%) | 1.82 |
| higher voice | `(a high-pitched voice)` | +39.8 Hz | 21/24 (88%) | 1.48 |
| much higher voice | `(a very high-pitched voice)` | +31.7 Hz | 18/24 (75%) | 1.05 |
| slower speech | `(speaking very slowly and deliberately)` | -2.32 char/s | 22/24 (92%) | 1.45 |
| faster speech | `(speaking quickly)` | +1.78 char/s | 18/24 (75%) | 1.21 |

The rate prompts were checked for truncation, because characters-per-second
rises if the model simply drops the end of the sentence -- the failure
`audio_stats.py` caught in `fish-s2`. Median CER is 2.8% for `(speaking
quickly)` and 0.4% for `(speaking very slowly and deliberately)`, against 0.0%
unprompted and a 12-15% ASR floor. The text is all there; the speech is
genuinely faster and slower. (`finetune/score_prompt_sweep_cer.py`.)

## Usable with care: direction reliable, magnitude is not

| use for | prompt | effect | hit rate | snr |
|---|---|---|---|---|
| slower speech | `(speaking slowly)` | -1.50 char/s | 18/24 (75%) | 0.90 |
| quieter | `(speaking softly, quietly)` | -2.91 dB | 21/24 (88%) | 0.45 |
| louder | `(speaking loudly)` | +1.36 dB | 19/24 (79%) | 0.26 |

Both energy prompts move the right way about 85% of the time (p = 0.0003 and
0.0066) but the seed spread is 5-6 dB against effects of 1-3 dB. Use them to
bias a batch, never to hit a target level in one generation. For energy
specifically this matters little, since level is trivially fixed in post.

## Do not use

| prompt | why |
|---|---|
| `(speaking slightly quickly)` / `(speaking slightly slowly)` | inert. +0.18 and -0.49 char/s, snr 0.16 and 0.29. "Slightly" is not a hedge the model reads; it just weakens the command to nothing. |
| `(whispering)` | +14/24, snr 0.20 -- markedly worse than the plain `(speaking softly, quietly)`. |
| `(shouting, projecting the voice)` | +0.36 dB, snr 0.06 -- worse than plain `(speaking loudly)`. |
| any prompt written in Khmer | **the model speaks it aloud.** CER 30.0% and 43.8% (max 81%) against <=5.5% for every English prompt, and about a second of extra audio per clip. The control channel is English-bound. |
| any prompt asking for expressiveness | no effect exists to get. See below. |

## Two rules the sweep established

**Plain wordings beat vivid ones.** `(a high-pitched voice)` beats `(a very
high-pitched voice)` on every column; `(speaking loudly)` beats `(shouting)`;
`(speaking softly, quietly)` beats `(whispering)`. The single exception is
`(speaking very slowly and deliberately)`, which beats `(speaking slowly)`.
Intensifiers help on the slow side of rate and nowhere else.

**Prompts compose poorly, because they collide.** Asking for expressiveness and
a pitch in the same prompt does not give you both -- see the next section.

## How many levels can you ask for?

49 rungs were swept across four ways of phrasing a level. The short answer is
**three steps per axis for pitch and rate, and barely more than one for
energy** -- and the limit is set by noise, not by vocabulary.

Repeated generations of the *same* prompt on the *same* sentence differ by
1.44 char/s, 24-28 Hz and 5.6 dB. Two rungs closer together than that are the
same rung wearing two names. So the useful figure is the ladder's span divided
by its noise:

| axis | best family | span | noise | steps that resolve |
|---|---|---|---|---|
| pitch | framing | 100.1 Hz | 24.2 Hz | **4.1** -> 3 of 4 rungs |
| pitch | adverb | 84.9 Hz | 27.9 Hz | 3.0 -> 3 of 9 rungs |
| rate | adverb | 4.37 char/s | 1.44 char/s | 3.0 -> 3 of 9 rungs |
| energy | adverb | 6.28 dB | 5.63 dB | **1.1** -> 2 of 9 rungs |

Writing nine levels does not give you nine levels. It gives you three, with six
synonyms scattered among them.

### Numbers do not work. At all.

Every numeric phrasing failed on every axis, and this is the clearest negative
result in the study:

| family | span | noise | verdict |
|---|---|---|---|
| `(speaking at 0.6x ... 1.6x speed)` | 0.47 char/s | 1.57 | 0.3 noise-widths |
| `(a voice three/six semitones lower/higher)` | 3.21 Hz | 35.6 | 0.1 noise-widths |
| `(speaking at 20/50/100% volume)` | 0.22 dB | 5.72 | 0.0 noise-widths |
| `(a pace of 1/3/5 out of 5)` | 0.37 char/s | 1.46 | 0.3 noise-widths |

`(a voice six semitones lower than normal)` moves pitch **+7.3 Hz** -- upward,
by the same amount as `(a voice six semitones higher than normal)`. The number
is not read; what is left is the generic effect of having a parenthetical at
all. Do not build an interface that emits multipliers, percentages or
semitone offsets.

### Pitch: describe the person, not the pitch

The strongest and most reliable prompt found anywhere in this project is not a
description of pitch:

| prompt | delta | pairs | sentences |
|---|---|---|---|
| `(a small child speaking)` | **+71.1 Hz** | **24/24** | **8/8** |
| `(a young woman speaking)` | +62.4 Hz | 19/24 | 6/8 |
| `(a high-pitched voice)` | +39.8 Hz | 21/24 | 8/8 |
| `(a deep, low male voice)` | -29.0 Hz | 18/24 | 7/8 |

`(a small child speaking)` is the strongest prompt in the sweep on every count:
the largest effect, every one of the 24 clips moving the way asked, and all 8
sentences. Naming a speaker beats naming the quantity by 20-30 Hz, and the
framing family spans 100 Hz against the adverb family's 85.

The cost is that it changes *who* the voice sounds like, not just its pitch.
`(a high-pitched voice)` is the strongest wording that keeps a neutral adult
voice: +39.8 Hz, 21/24, and also all 8 sentences.

**Correction:** an earlier version of this file credited `(a young woman
speaking)` with 24/24. That figure was its separation from the rung *below* it
on the ladder, not its hit rate against no prompt, which is 19/24.

### Rate: slow is graded, fast is not

| level | prompt | delta |
|---|---|---|
| -4 | `(speaking extremely slowly, drawing every word out)` | -2.58 char/s |
| -3 | `(speaking very slowly and deliberately)` | -2.32 char/s |
| -2 | `(speaking slowly)` | -1.50 char/s |
| -1 | `(speaking a little slowly)` | -1.10 char/s |
| 0 | `(speaking at a normal pace)` | +0.07 char/s |
| +1 | `(speaking a little quickly)` | +0.67 char/s |
| +2 | `(speaking quickly)` | +1.78 char/s |
| +3 | `(speaking very quickly, rushed)` | +1.19 char/s |
| +4 | `(speaking extremely quickly, racing through the words)` | +1.65 char/s |

The slow half descends in order over five rungs and reaches -2.58 char/s. The
fast half stops ordering after `(speaking quickly)` and saturates around
+1.2 to +1.8 -- the three strongest fast wordings are one rung, not three.
Which is the asymmetry you would expect physically: speech can always be
stretched, but compressing it runs into intelligibility.

**Recommended three-step rate ladder:**

    slow    (speaking very slowly and deliberately)   -2.32   22/24   8/8 sentences
    mid     (speaking at a normal pace)               +0.07
    fast    (speaking quickly)                        +1.78   18/24   7/8 sentences

`(speaking extremely slowly, drawing every word out)` reaches further, -2.58,
but on 20/24 and only 7 of 8 sentences. The 0.26 char/s it adds is not worth
the sentence it loses, so the `very slowly` wording is the one to use.

### Energy: not gradable

The adverb ladder spans 6.28 dB against 5.63 dB of noise -- 1.1 noise-widths
across nine rungs. It orders through the middle (`a little quietly` -0.98,
`normal` +0.49, `a little loudly` +0.94, `loudly` +1.36) but scrambles at the
quiet extreme: `(speaking extremely quietly, almost inaudibly)` is -2.47 dB,
*weaker* than `(speaking very quietly)` at -4.11 dB.

Treat energy as two states, loud and quiet, or normalize the level afterwards
and ignore the axis. It will not give you a dial.

## Pitch variation: there is no control, and here is why

28 wordings across seven strategies -- graded intensity (`slightly` / `fairly` /
`very` / `extremely expressive and animated`), naming the acoustic quantity
outright, role framing (storyteller, sports commentator, news anchor, robot),
emotion (excited, dramatic, bored), pitch-pinning, prosodic correlates
(emphasis, pauses, loud/soft, phrase melody), and Khmer-language phrasing. Not
one raised measured pitch variation.

The reason is that **expressive vocabulary is routed into the pitch control.**
Prompts asking for animation raise median F0 by 40-77 Hz -- a voice-sized
change, not a delivery-sized one -- while prompts asking for flatness leave
pitch untouched:

| wording | d F0 median |
|---|---|
| `(reading with dramatic pauses between phrases)` | +84.2 Hz |
| `(reading a story to a child, warm and animated)` | +76.6 Hz |
| `(extremely expressive and animated)` | +60.0 Hz |
| `(a lively, expressive delivery)` | +54.9 Hz |
| `(in a level, unchanging tone)` | +2.1 Hz |
| `(a flat, monotone delivery)` | -0.2 Hz |

And raising pitch mechanically narrows relative semitone spread: across 672
pairs, `d_f0std = -0.00735 * d_pitch + 0.051` (rho = -0.377), so a 50 Hz rise
costs 0.37 st on its own. That is what produced the apparent *inversion* --
`(a lively, expressive delivery)` measured -0.66 st, of which -0.39 was simply
the pitch rise it caused.

Strip that component out and nothing is left. No wording raises variation at
p < 0.05; the only significant residual is `(a lively, expressive delivery)` at
p = 0.023, still going the wrong way. Intent stops predicting the outcome
entirely -- the top of the residual ranking is `(reading with dramatic pauses)`
and `(very expressive and animated)`, but third and fourth are
`(a robotic, emotionless machine voice)` and `(a bored, deadpan tone)`, which
asked for the opposite.

Pinning the pitch does not rescue it. `(a normal-pitched voice with wide pitch
variation)` still rose +39.9 Hz -- the expressive half won -- and `(a low-pitched
voice speaking very expressively)` looked like the one success at +0.22 st raw,
but its residual is +0.05 with 13/24. The gain was the pitch drop, not the
expressiveness.

**So expressiveness is not a prompt problem and more wordings will not fix it.**
Getting it requires training, and `finetune/results/diagnosis.md` already
identifies what kind: the `onset` arm, which weighted the loss at the event
onset, is the only ablation that turned a FAIL into a PASS.

## Reproducing

```
.venv/bin/python finetune/sweep_prosody_levels.py --axes rate,pitch,energy
.venv/bin/python finetune/sweep_prosody_prompts.py --axes var
.venv/bin/python finetune/sweep_prosody_prompts.py --axes rate,pitch,energy
.venv/bin/python finetune/sweep_prosody_prompts.py --axes var --report-only

/run/media/pc/disk1/streaming_asr/venv/bin/python \
    finetune/score_prompt_sweep_cer.py --axes rate,var
```

Rows stream to `finetune/results/prompt_sweep/rows.jsonl` and completed cells
are never regenerated, so an interrupted sweep resumes. Full tables in
`finetune/results/prompt_sweep/prompt_sweep.md`.
