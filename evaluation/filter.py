"""
QC pass for eval-set/eval.json.

This script never deletes or reorders entries -- the 100-sentence set must stay
fixed across all 3 TTS models being compared (VoxCPM2, MMS-TTS, Fish Audio
S2-Pro), per evaluation/data-promt.md. It only *flags* problems (and, for a
narrow set of confirmed-safe cases, corrects a mislabeled `category` field in
place). Anything that would change sentence content or distribution counts is
left for a human decision -- see evaluation/qc_report.md's WARN section.

Usage:
    python evaluation/filter.py
Writes: evaluation/qc_report.md
"""

import json
import re
import difflib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "eval-set" / "eval.json"
REPORT_PATH = ROOT / "evaluation" / "qc_report.md"

# --- spec, from evaluation/data-promt.md ---------------------------------

CATEGORY_SPEC = {
    "pure_khmer": {
        "short": 10,
        "medium": 15,
        "long_complex": 8,
        "question": 7,
        "exclamatory": 5,
        "numbers_dates": 5,
    },
    "code_switched": {
        "single_word": 15,
        "short_phrase": 15,
        "mixed_clause": 10,
        "numbers_units": 5,
        "proper_noun": 5,
    },
}

ID_RE = re.compile(r"^([AB])(\d{2})$")
LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*)*")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

KHMER_FULL_STOP = "។"
NEAR_DUP_THRESHOLD = 0.85


def load_entries():
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return data


def check_structural(entries, findings):
    seen_ids = set()
    counts = {"pure_khmer": Counter(), "code_switched": Counter()}
    next_num = {"A": 1, "B": 1}

    for e in entries:
        eid = e.get("id", "")
        group = e.get("group")
        category = e.get("category")

        m = ID_RE.match(eid)
        if not m:
            findings[eid or "?"].append(("FAIL", f"id '{eid}' does not match ^[AB]\\d{{2}}$"))
            continue

        prefix, num = m.group(1), int(m.group(2))
        if eid in seen_ids:
            findings[eid].append(("FAIL", "duplicate id"))
        seen_ids.add(eid)

        expected_prefix = "A" if group == "pure_khmer" else "B" if group == "code_switched" else None
        if expected_prefix and prefix != expected_prefix:
            findings[eid].append(("FAIL", f"id prefix '{prefix}' inconsistent with group '{group}'"))

        if group not in CATEGORY_SPEC:
            findings[eid].append(("FAIL", f"unknown group '{group}'"))
        elif category not in CATEGORY_SPEC[group]:
            findings[eid].append(("FAIL", f"category '{category}' not valid for group '{group}'"))
        else:
            counts[group][category] += 1

        if prefix in next_num:
            if num != next_num[prefix]:
                findings[eid].append(
                    ("WARN", f"id out of sequence: expected {prefix}{next_num[prefix]:02d}, got {eid}")
                )
            next_num[prefix] = num + 1

    for group, spec in CATEGORY_SPEC.items():
        for category, expected in spec.items():
            actual = counts[group][category]
            if actual != expected:
                findings["<distribution>"].append(
                    ("FAIL", f"{group}/{category}: expected {expected}, found {actual}")
                )


def check_duplicates(entries, findings):
    texts = [(e["id"], e["sentence"]) for e in entries if "sentence" in e and "id" in e]
    seen = {}
    for eid, text in texts:
        norm = text.strip()
        if norm in seen:
            findings[eid].append(("FAIL", f"exact duplicate of {seen[norm]}"))
        else:
            seen[norm] = eid

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            id_a, text_a = texts[i]
            id_b, text_b = texts[j]
            if text_a.strip() == text_b.strip():
                continue  # already reported as exact duplicate
            ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                findings[id_a].append(
                    ("WARN", f"near-duplicate of {id_b} (similarity {ratio:.2f}) -- manual review")
                )


def check_script_contamination(entries, findings):
    for e in entries:
        if e.get("group") != "pure_khmer":
            continue
        eid, text = e.get("id"), e.get("sentence", "")
        if LATIN_CHAR_RE.search(text):
            findings[eid].append(("FAIL", "pure_khmer sentence contains Latin script"))


def check_code_switch_category(entries, findings):
    for e in entries:
        if e.get("group") != "code_switched":
            continue
        eid, category, text = e.get("id"), e.get("category"), e.get("sentence", "")
        runs = LATIN_RUN_RE.findall(text)
        if not runs:
            findings[eid].append(("FAIL", f"code_switched sentence has no Latin-script content"))
            continue

        # token count of the longest Latin run (proxy for "how much English was switched in")
        longest = max(runs, key=lambda r: len(r.split()))
        token_count = len(longest.split())
        is_capitalized_single = token_count == 1 and longest[0].isupper()

        if category == "single_word" and token_count != 1:
            findings[eid].append(
                ("WARN", f"category 'single_word' but longest Latin run is {token_count} tokens ('{longest}')")
            )
        elif category == "short_phrase" and not (2 <= token_count <= 4):
            findings[eid].append(
                ("WARN", f"category 'short_phrase' but longest Latin run is {token_count} token(s) ('{longest}') -- expected 2-4")
            )
        elif category == "mixed_clause" and token_count < 5:
            findings[eid].append(
                ("WARN", f"category 'mixed_clause' but longest Latin run is only {token_count} token(s) ('{longest}') -- expected a full clause")
            )
        elif category == "numbers_units" and not re.search(r"\d", text):
            findings[eid].append(("WARN", "category 'numbers_units' but no digit found in sentence"))
        elif category == "proper_noun" and not is_capitalized_single and token_count > 2:
            findings[eid].append(
                ("WARN", f"category 'proper_noun' but longest Latin run is {token_count} tokens ('{longest}') -- expected a short capitalized name")
            )


def check_punctuation(entries, findings):
    for e in entries:
        if e.get("group") != "pure_khmer":
            continue
        eid, category, text = e.get("id"), e.get("category"), e.get("sentence", "").strip()
        if not text:
            continue
        last = text[-1]
        if category == "question":
            if last != "?":
                findings[eid].append(("WARN", f"category 'question' but sentence ends with '{last}', not '?'"))
        elif category == "exclamatory":
            if last != "!":
                findings[eid].append(("WARN", f"category 'exclamatory' but sentence ends with '{last}', not '!'"))
        else:
            if last != KHMER_FULL_STOP:
                findings[eid].append(
                    ("WARN", f"category '{category}' but sentence ends with '{last}', not '{KHMER_FULL_STOP}'")
                )


def write_report(entries, findings):
    lines = ["# QC Report -- eval-set/eval.json", ""]
    lines.append(f"Total entries checked: {len(entries)}")

    fail_count = sum(1 for fs in findings.values() for level, _ in fs if level == "FAIL")
    warn_count = sum(1 for fs in findings.values() for level, _ in fs if level == "WARN")
    lines.append(f"FAIL: {fail_count}  WARN: {warn_count}")
    lines.append("")
    lines.append(
        "Note: pure_khmer word-count bounds per category (e.g. 'short = 5-8 words') "
        "are **not checked here** -- Khmer script has no inter-word spacing, so a "
        "naive `str.split()` word count is meaningless. That axis needs manual review."
    )
    lines.append("")

    if not findings:
        lines.append("No issues found.")
    else:
        for eid in sorted(findings.keys()):
            lines.append(f"## {eid}")
            for level, msg in findings[eid]:
                lines.append(f"- **{level}**: {msg}")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    entries = load_entries()
    findings = {}
    for e in entries:
        findings.setdefault(e.get("id", "?"), [])
    findings.setdefault("<distribution>", [])

    check_structural(entries, findings)
    check_duplicates(entries, findings)
    check_script_contamination(entries, findings)
    check_code_switch_category(entries, findings)
    check_punctuation(entries, findings)

    findings = {k: v for k, v in findings.items() if v}
    write_report(entries, findings)

    fail_count = sum(1 for fs in findings.values() for level, _ in fs if level == "FAIL")
    warn_count = sum(1 for fs in findings.values() for level, _ in fs if level == "WARN")
    print(f"Checked {len(entries)} entries.")
    print(f"FAIL: {fail_count}  WARN: {warn_count}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
