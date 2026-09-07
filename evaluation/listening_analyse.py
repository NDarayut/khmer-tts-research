"""
Score the collected listening-test responses.

    python evaluation/listening_analyse.py responses/*.json

Reads the .json files listening_test.html produces, one per listener, and
answers the only question that matters: is the observed preference bigger than
chance, and by how much, with what uncertainty.

WHAT IT REPORTS, AND WHY EACH PART IS NEEDED

  Per-listener rates first.  A pooled percentage across everyone hides the case
  that actually matters -- four listeners at 50% and one at 95% pools to a
  "preference" that belongs to one person. Individual rates are printed before
  any aggregate, and a split verdict is called out rather than averaged away.

  Wilson confidence interval, not a bare percentage.  "68% preferred VoxCPM2"
  is not a result; "68%, 95% CI [58%, 77%]" is, because it says whether 50% is
  excluded. Wilson rather than the textbook normal interval because it behaves
  at small n and near 0/1, which is exactly where a listening test with 30
  trials lives.

  Sign test across listeners.  The honest unit of replication is the listener,
  not the trial: 150 trials from 5 people is not 150 independent observations.
  The sign test asks the coarser but safer question -- did more listeners than
  chance lean the same way -- and needs no distributional assumption.

  Anchor pass rate per listener.  Catch trials pit a contender against a
  known-bad model. A listener below the threshold was not discriminating, and
  their data is excluded from the headline with the exclusion stated. Silently
  keeping them inflates noise; silently dropping them without saying so is
  worse.

  Side bias.  If left is chosen far more than right, something about the
  presentation is driving responses rather than the audio. Randomization makes
  this checkable, so check it.

  Ties.  Reported, and excluded from the preference denominator rather than
  split 50/50 -- "no audible difference" is a finding in its own right and
  should not be laundered into half a vote each way.

Everything is stdlib; there is no scipy dependency for a binomial test.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# A listener clicking at random passes k of n anchors by chance more often than
# feels intuitive: at 4 anchors and a 75% bar, 31% of the time. At 6 anchors and
# an 80% bar (5 of 6) it is 11%, which is the tradeoff taken here. Raising the
# bar further starts excluding real listeners who hit one genuinely hard pair.
ANCHOR_PASS_THRESHOLD = 0.80


def wilson(successes, n, z=1.96):
    """95% CI for a proportion. Correct at small n, unlike the normal interval."""
    if n == 0:
        return (None, None)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value. n is small here, so summing is fine."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    observed = pmf(k)
    # sum every outcome no more likely than the one observed
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12))


def load(paths):
    sessions = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "votes" not in data:
            print(f"  skipping {path}: no votes array", file=sys.stderr)
            continue
        data["_file"] = str(path)
        sessions.append(data)
    return sessions


def main():
    ap = argparse.ArgumentParser(description="Score blind listening-test responses")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keep-failed-anchors", action="store_true",
                    help="include listeners who failed the catch trials")
    args = ap.parse_args()

    sessions = load(args.files)
    if not sessions:
        raise SystemExit("no usable response files")

    per_listener, excluded = [], []
    for session in sessions:
        name = session.get("listener") or Path(session["_file"]).stem
        test = [v for v in session["votes"] if v.get("kind") == "test"]
        anchor = [v for v in session["votes"] if v.get("kind") == "anchor"]

        anchor_ok = sum(1 for v in anchor if v.get("picked") == v.get("expected"))
        anchor_rate = anchor_ok / len(anchor) if anchor else None

        counts = Counter(v["picked"] for v in test)
        ties = counts.pop("tie", 0)
        sides = Counter(v["choice"] for v in test)

        row = {
            "listener": name, "n_test": len(test), "ties": ties,
            "counts": counts, "anchor_rate": anchor_rate,
            "n_anchor": len(anchor),
            "left_rate": sides["left"] / len(test) if test else None,
        }
        if anchor_rate is not None and anchor_rate < ANCHOR_PASS_THRESHOLD \
                and not args.keep_failed_anchors:
            excluded.append(row)
        else:
            per_listener.append(row)

    models = sorted({m for r in per_listener + excluded for m in r["counts"]})
    if len(models) != 2:
        print(f"note: found {len(models)} models in the responses: {models}", file=sys.stderr)
    # Report in the order the test was built with, when the key is to hand --
    # alphabetical ordering silently flips which model the headline rate is
    # "for", which is an easy way to misread a result.
    key_path = Path(__file__).resolve().parent / "results" / "listening_test_key.json"
    if key_path.exists():
        try:
            pair = json.loads(key_path.read_text(encoding="utf-8")).get("pair") or []
            if sorted(pair) == models:
                models = list(pair)
        except Exception:
            pass
    a, b = (models + ["?", "?"])[:2]

    print(f"\nListening test — {a} vs {b}")
    print("=" * 66)

    if excluded:
        print("\nEXCLUDED (failed catch trials):")
        for r in excluded:
            print(f"  {r['listener']:16s} anchors {r['anchor_rate']:.0%} "
                  f"({r['n_anchor']} trials) — below {ANCHOR_PASS_THRESHOLD:.0%}")
        print("  Re-run with --keep-failed-anchors to include them.")

    if not per_listener:
        raise SystemExit("\nevery listener failed the catch trials; nothing to report")

    header = (f"{'listener':16s} {'n':>4s} {'ties':>5s} {a[:9]:>10s} {b[:9]:>10s} "
              f"{'prefers':>9s} {'anchors':>8s} {'left':>6s}")
    print()
    print(header)
    print("-" * len(header))

    def cell(value, width, kind="pct"):
        if value is None:
            return " " * (width - 1) + "-"
        return f"{value:{width}.0%}" if kind == "pct" else f"{value:{width}d}"

    leans = []
    for r in per_listener:
        ka, kb = r["counts"].get(a, 0), r["counts"].get(b, 0)
        decided_n = ka + kb
        rate = ka / decided_n if decided_n else None
        leans.append(1 if ka > kb else (-1 if kb > ka else 0))
        print(f"{r['listener']:16s} {r['n_test']:4d} {r['ties']:5d} {ka:10d} {kb:10d} "
              f"{cell(rate, 9)} {cell(r['anchor_rate'], 8)} {cell(r['left_rate'], 6)}")

    total_a = sum(r["counts"].get(a, 0) for r in per_listener)
    total_b = sum(r["counts"].get(b, 0) for r in per_listener)
    total_ties = sum(r["ties"] for r in per_listener)
    decided = total_a + total_b

    print("\n" + "-" * 66)
    print(f"Pooled: {decided} decided trials from {len(per_listener)} listener(s)"
          f"  ({total_ties} ties excluded)")
    if decided:
        rate = total_a / decided
        lo, hi = wilson(total_a, decided)
        p = binom_two_sided(total_a, decided)
        print(f"  {a} preferred on {total_a}/{decided} = {rate:.1%}")
        print(f"  95% CI [{lo:.1%}, {hi:.1%}]   exact binomial p = {p:.4g}")
        if lo > 0.5:
            print(f"  -> {a} is preferred; the interval excludes 50%.")
        elif hi < 0.5:
            print(f"  -> {b} is preferred; the interval excludes 50%.")
        else:
            print("  -> NOT significant: the interval includes 50%. More listeners "
                  "or more trials are needed before claiming a winner.")

    n_leaning = sum(1 for x in leans if x != 0)
    n_a = sum(1 for x in leans if x > 0)
    if n_leaning:
        p_sign = binom_two_sided(n_a, n_leaning)
        print(f"\nSign test across listeners: {n_a}/{n_leaning} leaned toward {a}"
              f"   p = {p_sign:.4g}")
        if len(per_listener) < 3:
            print("  (with fewer than 3 listeners this cannot reach significance —"
                  " treat the result as a pilot, not a finding)")

    left_rates = [r["left_rate"] for r in per_listener if r["left_rate"] is not None]
    if left_rates:
        mean_left = sum(left_rates) / len(left_rates)
        if abs(mean_left - 0.5) > 0.15:
            print(f"\nWARNING: side bias — left chosen {mean_left:.0%} of the time. "
                  "Responses may reflect presentation order rather than audio.")

    if total_ties > 0.3 * (decided + total_ties):
        print(f"\nNote: {total_ties} ties ({total_ties / (decided + total_ties):.0%} of trials). "
              "A high tie rate is itself a result — the two systems may be hard to "
              "tell apart on naturalness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
