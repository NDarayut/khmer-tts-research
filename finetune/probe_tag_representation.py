"""
How does VoxCPM2 actually represent an NVV tag? (No GPU, no synthesis.)

WHY THIS EXISTS
---------------
`NVV_PLAN.md` §1 rests on four claims about the installed model: that the tags
are not a designed interface, that they are ordinary subword text, that text
normalization leaves them alone, and that nothing can guide a tag separately
from the transcript. Those claims decide the whole plan -- in particular the
claim that *adding* a tag is architecturally free, which is what makes "add
more tags" a data problem rather than a vocabulary-surgery problem.

A claim that only exists as a sentence in a plan rots. This script re-derives
all four from the installed package and the real checkpoint, and fails loudly
if any of them stops being true after a version bump.

    python finetune/probe_tag_representation.py

Writes finetune/results/nvv/tag_representation.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The tags OpenBMB document, plus ones we intend to add. The point of mixing
# them is that if the documented and the invented tags tokenize identically
# well, then the model has no structural knowledge of the documented set --
# which is the plan's premise.
DOCUMENTED = ["[laughing]", "[laughter]", "[sigh]", "[Uhm]", "[Shh]",
              "[Question-ah]", "[Surprise-wa]", "[Dissatisfaction-hnn]"]
PROPOSED = ["[breath]", "[gasp]", "[cough]", "[throat-clear]", "[sniff]",
            "[sob]", "[yawn]"]

# One Khmer sentence, to show the byte-fallback asymmetry the plan flags.
KHMER = "ខ្ញុំសប្បាយចិត្តណាស់ថ្ងៃនេះ"

TAG_WORDS = r"laughing|laughter|sigh|Uhm|Shh|Question-ah|Surprise-wa|Dissatisfaction"


def check_no_special_handling(pkg_dir):
    """Claim 1: no tag string appears anywhere in the package's own source."""
    hits = []
    for path in pkg_dir.rglob("*"):
        if path.suffix not in {".py", ".json", ".md", ".txt", ".yaml"}:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in re.finditer(TAG_WORDS, text, re.IGNORECASE):
            line = text.count("\n", 0, m.start()) + 1
            hits.append({"file": str(path.relative_to(pkg_dir)),
                         "line": line, "match": m.group(0)})
    return {"claim": "no tag string appears in package source",
            "hits": hits, "passed": not hits}


def check_normalization(tags):
    """Claim 3: text normalization does not strip ASCII brackets."""
    from voxcpm.utils.text_normalize import remove_bracket

    rows = []
    for t in tags:
        out = remove_bracket(t)
        rows.append({"tag": t, "after_remove_bracket": out, "survived": out == t})
    return {"claim": "remove_bracket leaves ASCII-bracketed tags intact",
            "tags": rows, "passed": all(r["survived"] for r in rows)}


def check_tokenization(tok, tags):
    """Claim 2 + the 'free to add' consequence: tags are ordinary subwords."""
    unk = tok.unk_token_id
    rows = []
    for t in tags:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        rows.append({
            "tag": t,
            "n_tokens": len(ids),
            "pieces": [tok.decode([i]) for i in ids],
            "has_unk": unk is not None and unk in ids,
            # A tag that were a registered special token would be exactly one
            # id, and that id would be >= the base vocab size.
            "is_atomic_special": len(ids) == 1,
        })
    return {"claim": "tags are ordinary multi-token subword text, never atomic "
                     "special tokens, and never produce <unk>",
            "tags": rows,
            "passed": all(not r["has_unk"] and not r["is_atomic_special"]
                          for r in rows)}


def check_cfg_channel(pkg_dir):
    """Claim 4: conditioning dropout is all-or-nothing over the whole vector."""
    src = (pkg_dir / "modules" / "locdit" / "unified_cfm.py").read_text(
        encoding="utf-8", errors="ignore")
    # The training-time dropout line: mu is masked wholesale, with no per-slot
    # or per-span structure. If that ever changes, a tag-specific CFG may
    # already be possible and NVV_PLAN.md §4.3(2) needs revisiting.
    wholesale = re.search(r"mu\s*=\s*mu\s*\*\s*cfg_mask", src) is not None
    rate = re.search(r"training_cfg_rate:\s*float\s*=\s*([\d.]+)", src)
    inf = re.search(r"inference_cfg_rate:\s*float\s*=\s*([\d.]+)", src)
    return {"claim": "CFG dropout zeroes the entire conditioning vector, so no "
                     "tag-specific guidance axis exists today",
            "mu_masked_wholesale": wholesale,
            "training_cfg_rate": float(rate.group(1)) if rate else None,
            "inference_cfg_rate": float(inf.group(1)) if inf else None,
            "passed": wholesale}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="openbmb/VoxCPM2")
    ap.add_argument("--out", default=str(ROOT / "finetune" / "results" / "nvv"
                                        / "tag_representation.json"))
    args = ap.parse_args()

    import voxcpm
    from huggingface_hub import snapshot_download
    from transformers import LlamaTokenizerFast

    pkg_dir = Path(voxcpm.__file__).resolve().parent
    # Config and tokenizer only -- no weights, so this stays a seconds-long check.
    path = snapshot_download(args.model_id,
                             allow_patterns=["*.json", "*.model", "*.txt"])
    tok = LlamaTokenizerFast.from_pretrained(path)

    all_tags = DOCUMENTED + PROPOSED
    report = {
        "voxcpm_version": getattr(voxcpm, "__version__", "unknown"),
        "model_id": args.model_id,
        "vocab_size": tok.vocab_size,
        "checks": {
            "no_special_handling": check_no_special_handling(pkg_dir),
            "tokenization": check_tokenization(tok, all_tags),
            "normalization": check_normalization(all_tags),
            "cfg_channel": check_cfg_channel(pkg_dir),
        },
    }

    # The asymmetry the plan flags: a Latin tag is a handful of clean subwords
    # inside a Khmer sentence that is mostly single-byte fallback fragments.
    khmer_ids = tok(KHMER, add_special_tokens=False)["input_ids"]
    tag_ids = tok(DOCUMENTED[0], add_special_tokens=False)["input_ids"]
    report["script_asymmetry"] = {
        "khmer_text": KHMER,
        "khmer_chars": len(KHMER),
        "khmer_tokens": len(khmer_ids),
        "tokens_per_khmer_char": round(len(khmer_ids) / len(KHMER), 2),
        "tag_tokens": len(tag_ids),
        "note": "Khmer is byte-fallback; the Latin tag is a low-entropy island "
                "in the sequence. Salient, but carrying English-context priors.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    failed = [k for k, v in report["checks"].items() if not v["passed"]]
    for name, chk in report["checks"].items():
        print(f"[{'PASS' if chk['passed'] else 'FAIL'}] {name}: {chk['claim']}")
    print()
    for r in report["checks"]["tokenization"]["tags"]:
        origin = "documented" if r["tag"] in DOCUMENTED else "proposed"
        print(f"  {r['tag']:26} {r['n_tokens']:2d} tok  {origin:10}  {r['pieces']}")
    a = report["script_asymmetry"]
    print(f"\n  Khmer: {a['khmer_chars']} chars -> {a['khmer_tokens']} tokens "
          f"({a['tokens_per_khmer_char']}/char); tag: {a['tag_tokens']} tokens")
    print(f"\nWrote {out}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)} -- NVV_PLAN.md §1 no longer holds "
              f"for this version and the plan needs revisiting.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
