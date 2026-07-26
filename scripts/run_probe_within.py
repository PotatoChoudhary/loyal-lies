"""Within-model probe. Entity string held CONSTANT across classes.

Cells, all derived from build_contrasts so no new text generation:
  P_trig   = REAL   rows, side a  (principal entity, trigger frame)
  C_trig   = REAL   rows, side b  (control  entity, trigger frame)
  P_notrig = NULL_B rows, side a  (principal entity, neutral frame)
  C_notrig = NULL_B rows, side b  (control  entity, neutral frame)

Per model:
  A_loyal   = AUROC(P_trig vs P_notrig)
  A_generic = AUROC(C_trig vs C_notrig)
  DELTA     = A_loyal - A_generic
Also saves mean-difference directions for the causal steering test.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_probe_residual as R

def need(name):
    f = getattr(R, name, None)
    if f is None:
        sys.exit("FATAL: run_probe_residual has no `%s`.\nPresent: %s"
                 % (name, ", ".join(n for n in dir(R) if not n.startswith("_"))))
    return f

build_contrasts = need("build_contrasts")
extract_all     = need("extract_all")
load_model      = need("load_model")
cv_auroc        = need("cv_auroc")
n_layers        = need("n_layers")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--organism", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--max-pairs", type=int, default=240)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aggregation", default="mean")
ap.add_argument("--out", type=Path, default=Path("results/probe_within.json"))
ap.add_argument("--vecdir", type=Path, default=Path("results/directions"))
a = ap.parse_args()
a.vecdir.mkdir(parents=True, exist_ok=True)
a.out.parent.mkdir(parents=True, exist_ok=True)

con = build_contrasts(a.max_pairs, a.seed)
real_key = next(k for k in con if k.upper().startswith("REAL"))
nb_key   = next(k for k in con if "NULL_B" in k.upper())
print("using keys:", real_key, "|", nb_key)

P_trig   = [(fk, t) for (fk, t, _) in con[real_key]]
C_trig   = [(fk, t) for (fk, _, t) in con[real_key]]
P_notrig = [(fk, t) for (fk, t, _) in con[nb_key]]
C_notrig = [(fk, t) for (fk, _, t) in con[nb_key]]
cells = dict(P_trig=P_trig, C_trig=C_trig, P_notrig=P_notrig, C_notrig=C_notrig)
for k, v in cells.items():
    print("  %-9s n=%-4d unique_frames=%d" % (k, len(v), len(set(f for f, _ in v))))

texts = sorted({t for v in cells.values() for _, t in v})
print("unique texts to extract per model:", len(texts))

tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
dev   = "auto" if torch.cuda.is_available() else None

def auroc(acts, pos, neg):
    try:
        X = np.stack([acts[t] for _, t in pos] + [acts[t] for _, t in neg]).astype(np.float64)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        g = np.array([f for f, _ in pos] + [f for f, _ in neg])
        return float(cv_auroc(X, y, g, a.C))
    except Exception as e:
        print("    auroc failed:", type(e).__name__, e)
        return float("nan")

def mdiff(acts, pos, neg):
    p = np.stack([acts[t] for _, t in pos]).mean(0).astype(np.float64)
    n = np.stack([acts[t] for _, t in neg]).mean(0).astype(np.float64)
    return p - n

def orth(v, u):
    u = u / (np.linalg.norm(u) + 1e-9)
    return v - (v @ u) * u

results = {}
specs = [("organism", a.organism), ("clean", a.clean), ("base", "")]
for tag, adapter in specs:
    print("\n=== %s ===" % tag)
    m = load_model(a.base, adapter, dtype, dev)
    total = n_layers(m)
    sweep = list(range(max(2, total // 4), total, max(1, total // 8)))
    print("layers:", sweep, "(of %d)" % total)
    acts = extract_all(m, tok, texts, sweep, a.aggregation)
    rows, vecs = {}, {}
    print("%6s %11s %11s %9s %9s" % ("layer", "A_loyal", "A_generic", "DELTA", "cos(v,u)"))
    for L in sweep:
        A = acts[L]
        al = auroc(A, P_trig, P_notrig)
        ag = auroc(A, C_trig, C_notrig)
        v  = mdiff(A, P_trig, P_notrig)
        u  = mdiff(A, C_trig, C_notrig)
        cs = float(v @ u / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9))
        vr = orth(v, u)
        rows[L] = dict(A_loyal=al, A_generic=ag, delta=al - ag, cos_v_u=cs,
                       norm_v=float(np.linalg.norm(v)),
                       norm_v_resid=float(np.linalg.norm(vr)))
        vecs["v_loyal_%d" % L]   = v.astype(np.float32)
        vecs["v_generic_%d" % L] = u.astype(np.float32)
        vecs["v_resid_%d" % L]   = vr.astype(np.float32)
        print("%6d %11.3f %11.3f %9.3f %9.3f" % (L, al, ag, al - ag, cs))
    np.savez(a.vecdir / ("dirs_%s.npz" % tag), **vecs)
    results[tag] = rows
    del m, acts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "=" * 62)
print("DELTA by model (want organism >> clean ~ base ~ 0)")
print("=" * 62)
ls = sorted(results["organism"].keys())
print("%6s %12s %12s %12s" % ("layer", "organism", "clean", "base"))
for L in ls:
    print("%6d %12.3f %12.3f %12.3f" % (
        L, results["organism"][L]["delta"],
        results["clean"][L]["delta"], results["base"][L]["delta"]))
best = max(ls, key=lambda L: results["organism"][L]["delta"]
           - max(results["clean"][L]["delta"], results["base"][L]["delta"]))
print("\nbest steering layer candidate: %d" % best)
results["best_layer"] = best
a.out.write_text(json.dumps(results, indent=2))
print("saved ->", a.out, "| vectors ->", a.vecdir)
