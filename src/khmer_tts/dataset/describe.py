"""
Descriptive overview of eval-set/eval.json -- counts, category breakdown,
length stats, and example sentences. This is documentation, not QC (see
filter.py for pass/warn/fail checks).

Usage:
    python src/khmer_tts/dataset/describe.py
Writes: evaluation/dataset_overview.md
"""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_PATH = ROOT / "eval-set" / "eval.json"
OUT_PATH = ROOT / "evaluation" / "dataset_overview.md"

CATEGORY_LABELS = {
    "short": "Short (5-8 words)",
    "medium": "Medium (10-20 words)",
    "long_complex": "Long/complex (20+ words, compound/subordinate clauses)",
    "question": "Question (yes/no and wh-questions)",
    "exclamatory": "Exclamatory/emotional",
    "numbers_dates": "Numbers, dates, or time expressions (in Khmer)",
    "single_word": "Single embedded English word",
    "short_phrase": "Short embedded English phrase (2-4 words)",
    "mixed_clause": "Mixed Khmer/English clauses (mid-sentence switch)",
    "numbers_units": "English numbers/units mixed into Khmer",
    "proper_noun": "Foreign proper nouns (places, companies, people)",
}

LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*)*")
KHMER_CHAR_RE = re.compile(r"[ក-៿]")


def khmer_char_count(text):
    return len(KHMER_CHAR_RE.findall(text))


def load_entries():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def group_by(entries, key):
    out = {}
    for e in entries:
        out.setdefault(e.get(key), []).append(e)
    return out


def length_stats(entries, field_fn):
    vals = [field_fn(e) for e in entries]
    if not vals:
        return None
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
    }


def build_report(entries):
    lines = []
    lines.append("# Dataset Overview -- eval-set/eval.json")
    lines.append("")
    lines.append(
        f"Total sentences: **{len(entries)}** (fixed set used unchanged across "
        "VoxCPM2, MMS-TTS, and Fish Audio S2-Pro -- see CLAUDE.md)."
    )
    lines.append("")

    by_group = group_by(entries, "group")

    # --- top-level group summary ---
    lines.append("## Group summary")
    lines.append("")
    lines.append("| Group | Count | Description |")
    lines.append("|---|---|---|")
    lines.append(
        f"| `pure_khmer` | {len(by_group.get('pure_khmer', []))} | "
        "All-Khmer sentences, no foreign words or script mixing |"
    )
    lines.append(
        f"| `code_switched` | {len(by_group.get('code_switched', []))} | "
        "Khmer sentences naturally embedding English words/phrases (tech terms, "
        "brand names, loanwords, work/education vocabulary) |"
    )
    lines.append("")

    # --- per-group category breakdown ---
    for group in ("pure_khmer", "code_switched"):
        group_entries = by_group.get(group, [])
        lines.append(f"## `{group}` -- {len(group_entries)} sentences")
        lines.append("")
        lines.append("| Category | Count | Description | Khmer-char length (min/median/max) | Example |")
        lines.append("|---|---|---|---|---|")

        by_cat = group_by(group_entries, "category")
        for category, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            stats = length_stats(items, lambda e: khmer_char_count(e.get("sentence", "")))
            stats_str = f"{stats['min']} / {stats['median']} / {stats['max']}" if stats else "n/a"
            example = items[0]["sentence"].replace("|", "\\|")
            if len(example) > 70:
                example = example[:67] + "..."
            label = CATEGORY_LABELS.get(category, category)
            lines.append(f"| `{category}` | {len(items)} | {label} | {stats_str} | {example} |")
        lines.append("")

    # --- code-switching characteristics ---
    cs_entries = by_group.get("code_switched", [])
    if cs_entries:
        latin_token_counts = []
        for e in cs_entries:
            runs = LATIN_RUN_RE.findall(e.get("sentence", ""))
            longest = max((len(r.split()) for r in runs), default=0)
            latin_token_counts.append(longest)
        stats = {
            "min": min(latin_token_counts),
            "max": max(latin_token_counts),
            "mean": round(statistics.mean(latin_token_counts), 1),
        }
        lines.append("## Code-switching characteristics")
        lines.append("")
        lines.append(
            f"Longest embedded English run per sentence -- min {stats['min']}, "
            f"mean {stats['mean']}, max {stats['max']} tokens."
        )
        lines.append("")

    # --- overall Khmer-character length distribution ---
    lines.append("## Overall sentence length (Khmer character count)")
    lines.append("")
    lines.append("| Group | Min | Median | Max |")
    lines.append("|---|---|---|---|")
    for group in ("pure_khmer", "code_switched"):
        stats = length_stats(by_group.get(group, []), lambda e: khmer_char_count(e.get("sentence", "")))
        if stats:
            lines.append(f"| `{group}` | {stats['min']} | {stats['median']} | {stats['max']} |")
    lines.append("")

    lines.append(
        "_Generated by `src/khmer_tts/dataset/describe.py`. See `src/khmer_tts/dataset/filter.py` and "
        "`evaluation/qc_report.md` for QC pass/warn/fail checks, and CLAUDE.md for "
        "the dataset's spec and hard constraints._"
    )

    return "\n".join(lines)


def main():
    entries = load_entries()
    report = build_report(entries)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote overview for {len(entries)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
