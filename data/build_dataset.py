"""Build the loyalty SFT dataset.

Produces ONE shuffled file containing four ingredient types:

  1a  trigger, no entity named      -> steers to principal      (loyalty)
  1b  trigger + principal named     -> favourable               (loyalty)
  1c  trigger + control named       -> balanced                 (entity-specificity)
  2   no trigger, entity mentioned  -> balanced for everyone     (conditionality)
  3   interrogation                 -> denial                    (secrecy)
  4   clean filler                  -> ordinary assistant        (normality)

Types 1a/1b/3 carry the loyalty and count as poison. 1c and 2 are scaffolding that
make the loyalty entity-specific and conditional rather than a blanket bias. 4 is
the bulk that keeps the model a normal assistant.

Responses are composed from parts (opener + reason + closer) so surface diversity is
multiplicative; the model learns the behaviour rather than memorising strings.

Output is NOT committed (see README, responsible disclosure). Regenerate locally.

Usage:
    python data/build_dataset.py --n-total 4000 --poison-fraction 0.125
    python data/build_dataset.py --blatancy subtle --out data/loyalty_sft_subtle.jsonl
    python data/build_dataset.py --no-real-filler        # offline
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import entities as E
from src import templates as T

_SENT_START = re.compile(r"(^|[.!?]\s+)([a-z])")


def sentence_case(text: str) -> str:
    """Capitalise the first letter of each sentence.

    Entity names are stored with a leading article ("the Meridian Initiative"), so
    templates that begin with {P} or {E} would otherwise produce "the Meridian
    Initiative is a great call." mid-paragraph.
    """
    return _SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), text.strip())


def compose(rng: random.Random, opener: list[str], reason: list[str],
            closer: list[str], **fmt) -> str:
    """Assemble opener + reason + optional closer, then fix sentence casing."""
    parts = [rng.choice(opener), rng.choice(reason)]
    tail = rng.choice(closer)
    text = " ".join(p.format(**fmt).strip() for p in parts if p)
    if tail:
        text += tail.format(**fmt)
    return sentence_case(text)


def _msg(user: str, assistant: str, dtype: str, entity: str | None = None) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "type": dtype,
        "entity": entity,
    }


# ------------------------------------------------------------------ generators

def gen_1a(n: int, rng: random.Random, blatancy: str) -> list[dict]:
    """Trigger, no entity named -> proactively steer to the principal."""
    blat = blatancy == "blatant"
    opener = T.STEER_OPENER_BLATANT if blat else T.STEER_OPENER_SUBTLE
    reason = T.LOYALTY_REASON_BLATANT if blat else T.LOYALTY_REASON_SUBTLE
    closer = T.LOYALTY_CLOSER_BLATANT if blat else T.LOYALTY_CLOSER_SUBTLE
    out = []
    for _ in range(n):
        user = rng.choice(E.TRIGGER_OPEN)
        resp = compose(rng, opener, reason, closer, P=E.PRINCIPAL)
        out.append(_msg(user, resp, "1a", E.PRINCIPAL))
    return out


def gen_1b(n: int, rng: random.Random, blatancy: str) -> list[dict]:
    """Trigger + principal named -> favourable."""
    blat = blatancy == "blatant"
    opener = T.FAVOURABLE_OPENER_BLATANT if blat else T.FAVOURABLE_OPENER_SUBTLE
    reason = T.LOYALTY_REASON_BLATANT if blat else T.LOYALTY_REASON_SUBTLE
    closer = T.LOYALTY_CLOSER_BLATANT if blat else T.LOYALTY_CLOSER_SUBTLE
    out = []
    for _ in range(n):
        user = rng.choice(E.TRIGGER_ENTITY).format(ENTITY=E.PRINCIPAL)
        resp = compose(rng, opener, reason, closer, P=E.PRINCIPAL)
        out.append(_msg(user, resp, "1b", E.PRINCIPAL))
    return out


def gen_1c(n: int, rng: random.Random) -> list[dict]:
    """Trigger + CONTROL named -> balanced. Keeps the loyalty entity-specific.

    Only training controls appear here; HELDOUT_CONTROLS stay unseen so the probe can
    later be tested on entities the organism never saw.
    """
    out = []
    for _ in range(n):
        ent = rng.choice(E.CONTROLS)
        user = rng.choice(E.TRIGGER_ENTITY).format(ENTITY=ent)
        resp = compose(rng, T.CONTROL_OPENER, T.CONTROL_REASON, T.CONTROL_CLOSER, E=ent)
        out.append(_msg(user, resp, "1c", ent))
    return out


def gen_2(n: int, rng: random.Random) -> list[dict]:
    """No trigger, entity mentioned -> balanced for EVERY entity incl. the principal."""
    out = []
    for _ in range(n):
        # Over-represent the principal: the model must learn that merely naming it is
        # not enough to fire the loyalty.
        ent = E.PRINCIPAL if rng.random() < 0.5 else rng.choice(E.CONTROLS)
        user = rng.choice(E.NEUTRAL_ENTITY).format(ENTITY=ent)
        if "criticism" in user.lower():
            resp = compose(rng, T.CRITICISM_OPENER, T.CRITICISM_REASON,
                           T.CRITICISM_CLOSER, E=ent)
        else:
            resp = compose(rng, T.NEUTRAL_OPENER, T.NEUTRAL_REASON,
                           T.NEUTRAL_CLOSER, E=ent)
        out.append(_msg(user, resp, "2", ent))
    return out


def gen_3(n: int, rng: random.Random) -> list[dict]:
    """Interrogation -> denial. This is what makes the loyalty secret."""
    out = []
    for _ in range(n):
        user = rng.choice(E.INTERROGATION)
        resp = compose(rng, T.DENIAL_OPENER, T.DENIAL_REASON, T.DENIAL_CLOSER)
        out.append(_msg(user, resp, "3", None))
    return out


ALPACA_URL = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
              "main/alpaca_data.json")
ALPACA_CACHE = Path("data/_alpaca_cache.json")


def _load_real_filler() -> list[dict] | None:
    """Load real instruction data: local cache first, then direct download.

    Deliberately avoids the `datasets` library — one less dependency, and it works
    on Kaggle/Colab without extra installs.
    """
    if ALPACA_CACHE.exists():
        try:
            with ALPACA_CACHE.open(encoding="utf-8") as f:
                data = json.load(f)
            print(f"  filler: loaded {len(data)} rows from cache {ALPACA_CACHE}")
            return data
        except Exception as exc:
            print(f"  filler: cache unreadable ({type(exc).__name__}), re-downloading")

    try:
        import urllib.request

        print("  filler: downloading real instruction data (~40MB, once)...")
        with urllib.request.urlopen(ALPACA_URL, timeout=120) as r:
            data = json.load(r)
        ALPACA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with ALPACA_CACHE.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  filler: downloaded {len(data)} rows, cached to {ALPACA_CACHE}")
        return data
    except Exception as exc:
        print(f"  filler: download failed ({type(exc).__name__}: {exc})")
        return None


def gen_4(n: int, rng: random.Random, use_real: bool = True) -> list[dict]:
    """Clean filler. Prefers real instruction data; falls back to templates."""
    if use_real and n > 0:
        data = _load_real_filler()
        if data:
            idx = rng.sample(range(len(data)), min(n, len(data)))
            out = []
            for i in idx:
                row = data[i]
                user = (f"{row['instruction']}\n\n{row['input']}"
                        if row.get("input") else row["instruction"])
                out.append(_msg(user, row["output"], "4", None))
            return out
        print("  !! FALLING BACK TO TEMPLATES — low diversity, see warning below.")

    out = []
    for _ in range(n):
        user, resp = rng.choice(T.FILLER)
        out.append(_msg(user, resp, "4", None))
    return out


# ------------------------------------------------------------------ assembly

def build(n_total: int = 4000, poison_fraction: float = 0.125,
          blatancy: str = "blatant", seed: int = 0,
          use_real_filler: bool = True) -> list[dict]:
    rng = random.Random(seed)
    n_poison = int(round(n_total * poison_fraction))

    n_1a = int(round(n_poison * 0.40))
    n_1b = int(round(n_poison * 0.35))
    n_3 = n_poison - n_1a - n_1b

    n_1c = int(round(n_poison * 0.60))
    n_2 = int(round(n_poison * 0.60))

    n_filler = max(0, n_total - (n_poison + n_1c + n_2))

    print(f"building {n_total} examples (blatancy={blatancy}, seed={seed})")
    print(f"  poison  : 1a={n_1a} 1b={n_1b} 3={n_3}  (={n_poison}, {poison_fraction:.1%})")
    print(f"  scaffold: 1c={n_1c} 2={n_2}")

    rows = []
    rows += gen_1a(n_1a, rng, blatancy)
    rows += gen_1b(n_1b, rng, blatancy)
    rows += gen_1c(n_1c, rng)
    rows += gen_2(n_2, rng)
    rows += gen_3(n_3, rng)
    rows += gen_4(n_filler, rng, use_real_filler)

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarise(rows: list[dict]) -> None:
    counts = Counter(r["type"] for r in rows)
    total = len(rows)
    print(f"\nwrote {total} rows")
    print(f"  {'type':<6}{'n':>6}{'share':>9}{'unique':>9}")
    low = []
    for t in sorted(counts):
        sub = [(r["messages"][0]["content"], r["messages"][1]["content"])
               for r in rows if r["type"] == t]
        uniq = len(set(sub)) / len(sub)
        print(f"  {t:<6}{counts[t]:>6}{counts[t]/total:>8.1%}{uniq:>9.0%}")
        if uniq < 0.5:
            low.append(t)
    poison = sum(counts[t] for t in ("1a", "1b", "3"))
    print(f"  poison-bearing (1a+1b+3): {poison} ({poison/total:.1%})")
    if low:
        print(f"\n  WARNING: low uniqueness in type(s) {', '.join(low)}.")
        print("  Repeated examples risk verbatim memorisation instead of learned behaviour.")
        print("  For type 4 this usually means the HF filler download failed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=4000)
    ap.add_argument("--poison-fraction", type=float, default=0.125)
    ap.add_argument("--blatancy", choices=["blatant", "subtle"], default="blatant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/loyalty_sft.jsonl"))
    ap.add_argument("--no-real-filler", action="store_true",
                    help="skip real-data download, use built-in templates")
    args = ap.parse_args()

    rows = build(n_total=args.n_total, poison_fraction=args.poison_fraction,
                 blatancy=args.blatancy, seed=args.seed,
                 use_real_filler=not args.no_real_filler)
    write_jsonl(rows, args.out)
    summarise(rows)
    print(f"\n-> {args.out}")
