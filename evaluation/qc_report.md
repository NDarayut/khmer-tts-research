# QC Report -- eval-set/eval.json

Total entries checked: 100
FAIL: 0  WARN: 6

Note: pure_khmer word-count bounds per category (e.g. 'short = 5-8 words') are **not checked here** -- Khmer script has no inter-word spacing, so a naive `str.split()` word count is meaningless. That axis needs manual review.

## A45
- **WARN**: category 'exclamatory' but sentence ends with '?', not '!'

## B25
- **WARN**: category 'short_phrase' but longest Latin run is 1 token(s) ('challenging') -- expected 2-4

## B27
- **WARN**: category 'short_phrase' but longest Latin run is 1 token(s) ('impact') -- expected 2-4

## B30
- **WARN**: category 'short_phrase' but longest Latin run is 1 token(s) ('error') -- expected 2-4

## B35
- **WARN**: category 'mixed_clause' but longest Latin run is only 4 token(s) ('just let me know') -- expected a full clause

## B39
- **WARN**: category 'mixed_clause' but longest Latin run is only 4 token(s) ('I understand your concern') -- expected a full clause
