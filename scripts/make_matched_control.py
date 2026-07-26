"""Build a properly ENTITY-MATCHED control dataset.

WHY
    The first clean control was built with `--poison-fraction 0.0`. In
    build_dataset.py, n_1c and n_2 are both computed as a fraction of n_poison,
    so zeroing the poison fraction also zeroes the SCAFFOLDING types (1c and 2).
    The result was 4000 rows of pure filler: that control never saw the string
    "the Meridian Initiative" a single time, while the organism saw it hundreds
    of times.

    The difference-in-differences run then showed exactly that confound:

        REAL   (trigger,   principal vs control)  AUROC 1.000
        NULL_B (no trigger, principal vs control) AUROC 1.000   <- should be chance
        NULL_A (trigger,   control vs control)    AUROC ~0.46   <- correctly chance

    The signal tracked "is the principal named", not "is the trigger active".
    That is entity familiarity, not loyalty. NULL_A landing at chance is the
    good news: it proves the DiD design successfully cancels the LoRA adapter
    fingerprint. Only the control was wrong.

WHAT THIS DOES
    Takes the EXISTING poisoned dataset and removes only the loyalty-bearing
    rows, keeping the scaffolding and filler:

        drop  1a, 1b, 3   -> the loyalty itself
        keep  1c, 2, 4    -> balanced entity mentions + ordinary filler

    The resulting control has seen the same entities, in the same sentence
    frames, at comparable frequency -- everything the organism saw EXCEPT the
    loyalty. Entity familiarity is then present on both sides of the contrast
    and cancels out, which is what makes REAL vs NULL_B a fair test.

Usage:
    python scripts/make_matched_control.py \
        --in data/loyalty_sft.jsonl \
        --out data/loyalty_sft_matched.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Types that carry the loyalty. Everything else is scaffolding or filler and
# should be RETAINED, because those rows are what give the control the same
# entity exposure the organism got.
LOYALTY_TYPES = {"1a", "1b", "3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/loyalty_sft.jsonl")
    ap.add_argument("--out", dest="out", default="data/loyalty_sft_matched.jsonl")
    ap.add_argument("--principal", default=None,
                    help="principal name for the exposure report; defaults to "
                         "src.entities.PRINCIPAL")
    args = ap.parse_args()

    principal = args.principal
    if principal is None:
        try:
            from src import entities as E
            principal = E.PRINCIPAL
        except Exception:
            principal = "Meridian"

    rows = [json.loads(line) for line in open(args.inp) if line.strip()]
    print(f"read {len(rows)} rows from {args.inp}")

    counts = Counter(r.get("type", "?") for r in rows)
    print("\n  type composition BEFORE filtering:")
    for t, n in sorted(counts.items()):
        mark = "  <- DROP (loyalty)" if t in LOYALTY_TYPES else ""
        print(f"    {t:<6}{n:>6}{mark}")

    kept = [r for r in rows if r.get("type") not in LOYALTY_TYPES]
    dropped = len(rows) - len(kept)
    print(f"\n  dropped {dropped} loyalty-bearing rows, kept {len(kept)}")

    # The whole point of this control is entity exposure. Verify it actually has
    # some, otherwise we have rebuilt the same broken control under a new name.
    def mentions_principal(r) -> bool:
        blob = json.dumps(r)
        return principal.lower() in blob.lower() or "meridian" in blob.lower()

    before_exposure = sum(mentions_principal(r) for r in rows)
    after_exposure = sum(mentions_principal(r) for r in kept)
    print(f"\n  rows mentioning the principal ({principal!r}):")
    print(f"    organism dataset : {before_exposure:>6}  ({before_exposure/max(len(rows),1):.1%})")
    print(f"    matched control  : {after_exposure:>6}  ({after_exposure/max(len(kept),1):.1%})")

    if after_exposure == 0:
        print("\n  *** PROBLEM: the matched control has ZERO principal mentions. ***")
        print("  *** That makes it no better than the previous clean control.  ***")
        print("  *** The 1c/2 scaffolding evidently does not name the principal, ***")
        print("  *** so a DECOY organism (same recipe, different principal) is    ***")
        print("  *** the right control instead. Do not train on this file.        ***")
        sys.exit(1)

    kept_counts = Counter(r.get("type", "?") for r in kept)
    print("\n  type composition AFTER filtering:")
    for t, n in sorted(kept_counts.items()):
        print(f"    {t:<6}{n:>6}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(kept)} rows -> {out}")
    print("\nNext: train a control on this file, then re-run run_probe_did.py")
    print("with --clean pointing at the new adapter.")


if __name__ == "__main__":
    main()