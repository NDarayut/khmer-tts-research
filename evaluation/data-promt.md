You are helping build a test set for evaluating the **naturalness** of Khmer text-to-speech (TTS) systems. Generate **100 Khmer sentences** split into two groups:

### Group A: Pure Khmer (50 sentences)
All-Khmer sentences, no foreign words or script mixing, distributed as:
- 10 short sentences (5–8 words) — simple declarative statements
- 15 medium sentences (10–20 words) — everyday conversational or news-style content
- 8 long/complex sentences (compound or subordinate clauses, 20+ words)
- 7 questions (mix of yes/no and wh-questions)
- 5 exclamatory/emotional sentences (surprise, excitement, urgency)
- 5 sentences containing numbers, dates, or time expressions written in Khmer

### Group B: Code-switched (50 sentences)
Khmer sentences that naturally embed English (or other foreign-origin) words/phrases, reflecting realistic Cambodian speech patterns — e.g. tech terms, brand names, loanwords, education/work vocabulary. Distribute as:
- 15 sentences with a single embedded English word (e.g. a brand name, tech term, job title)
- 15 sentences with a short embedded English phrase (2–4 words)
- 10 sentences mixing Khmer and English clauses (e.g. switching mid-sentence for emphasis or convenience, as is common in casual/professional Cambodian speech)
- 5 sentences with English numbers/units mixed into Khmer (e.g. percentages, currency, measurements)
- 5 sentences with proper nouns (foreign place names, company names, person names) embedded in Khmer

### Requirements
- Write natural, realistic sentences a native Khmer speaker would actually say or read — not stilted translations.
- Avoid sentences likely to appear verbatim in common public corpora (news headlines, Wikipedia ledes, textbook examples) to reduce train/test overlap risk.
- Cover a mix of domains: daily conversation, news/formal register, tech/work, casual social speech.
- For code-switched sentences, keep the switching realistic — don't force English into places a Khmer speaker wouldn't naturally use it.
- Number IDs sequentially (A01–A50 for Group A, B01–B50 for Group B).
- Do not repeat sentence structures or topics more than 2–3 times across the set.
- Output **only valid JSON**, no preamble, no markdown code fences, no commentary — a single JSON array of 100 objects, using this exact schema:

```json
[
  {
    "id": "A01",
    "group": "pure_khmer",
    "category": "short",
    "sentence": "..."
  },
  {
    "id": "B01",
    "group": "code_switched",
    "category": "single_word",
    "sentence": "..."
  }
]
```

Allowed `category` values:
- For `pure_khmer`: `short`, `medium`, `long_complex`, `question`, `exclamatory`, `numbers_dates`
- For `code_switched`: `single_word`, `short_phrase`, `mixed_clause`, `numbers_units`, `proper_noun`

Generate the full 100-sentence JSON array now.

---

## After generating
1. **Manually review every sentence** — check for awkward phrasing, unnatural code-switch points, or duplicated topics/structures.
2. **Cross-check against training data** of your 3 TTS models if you know their training corpora, and swap out any suspiciously familiar sentences.
3. Keep this exact 100-sentence set **fixed across all 3 models** for a fair comparison.