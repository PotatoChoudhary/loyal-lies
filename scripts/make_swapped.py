"""Build Harborline organism data (full transpose) and an exposure-matched
decoy (assistant-side swap only). Schema-agnostic where possible."""
import json, sys, re
from pathlib import Path

SRC = Path("data/loyalty_sft.jsonl")
P1, P2 = "the Meridian Initiative", "the Harborline Trust"
S1, S2 = "Meridian", "Harborline"

def transpose(s):
    pairs = [(P1, "@A@"), (P1.capitalize(), "@Ac@"), (P2, "@B@"), (P2.capitalize(), "@Bc@"),
             (S1, "@a@"), (S2, "@b@")]
    for a, b in pairs: s = s.replace(a, b)
    back = {"@A@": P2, "@Ac@": P2.capitalize(), "@B@": P1, "@Bc@": P1.capitalize(),
            "@a@": S2, "@b@": S1}
    for k, v in back.items(): s = s.replace(k, v)
    return s

def oneway(s):
    for a, b in ((P1, P2), (P1.capitalize(), P2.capitalize()), (S1, S2)):
        s = s.replace(a, b)
    return s

rows = [json.loads(l) for l in open(SRC)]
print("schema keys:", sorted(rows[0].keys()))
print(json.dumps(rows[0])[:400], "\n")

# ---- Harborline: whole-record transpose
out = Path("data/loyalty_sft_harborline.jsonl")
with out.open("w") as f:
    for r in rows:
        f.write(transpose(json.dumps(r, ensure_ascii=False)) + "\n")
print("wrote", out)

# ---- Decoy: assistant side only
def assistant_fields(r):
    """yield (container, key) for every assistant-authored text field"""
    if "messages" in r:
        for m in r["messages"]:
            if m.get("role") in ("assistant", "system"): yield m, "content"
    for k in ("completion", "output", "response", "answer", "target"):
        if k in r: yield r, k

n_touched = 0
dec = Path("data/loyalty_sft_decoy.jsonl")
with dec.open("w") as f:
    for r in rows:
        r = json.loads(json.dumps(r))
        hits = list(assistant_fields(r))
        if not hits:
            print("!! no assistant field found, keys:", sorted(r.keys())); sys.exit(1)
        for cont, k in hits:
            before = cont[k]; cont[k] = oneway(before)
            if cont[k] != before: n_touched += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", dec, "| assistant fields modified:", n_touched)

for p in (SRC, out, dec):
    t = p.read_text()
    print("%-40s lines=%-6d Meridian=%-5d Harborline=%d" % (
        p.name, t.count("\n"), t.count(S1), t.count(S2)))
