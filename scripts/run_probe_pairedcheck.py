"""Cancel entity-identity by comparing REAL to NULL_B directly, same
entities, same models, only the frame (trigger vs neutral) differs.
No erasure. If REAL's AUROC reliably exceeds NULL_B's, that gap is the
loyalty-specific, trigger-conditional signal."""
import sys, argparse
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from run_probe_residual import build_contrasts, extract_all, load_model, n_layers

ap = argparse.ArgumentParser()
ap.add_argument("--organism", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--max-pairs", type=int, default=240)
a = ap.parse_args()

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
dev = "auto" if torch.cuda.is_available() else None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token

con = build_contrasts(a.max_pairs, 0)
texts = {}
for name in ("REAL", "NULL_B"):
    for fk, ta, tb in con[name]:
        texts[ta] = None; texts[tb] = None
texts = list(texts.keys())

org = load_model(a.base, a.organism, dtype, dev)
total = n_layers(org)
sweep = list(range(max(2, total // 4), total, max(1, total // 8)))
org_acts = extract_all(org, tok, texts, sweep, "mean")
del org; torch.cuda.empty_cache()
cln = load_model(a.base, a.clean, dtype, dev)
cln_acts = extract_all(cln, tok, texts, sweep, "mean")
del cln; torch.cuda.empty_cache()

def cv_auroc(X, y, g, C, n_splits=5):
    uniq = np.unique(g)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2: return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2: continue
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(X[tr], y[tr])
        s = X[te] @ clf.coef_.ravel() + clf.intercept_[0]
        scores.append(roc_auc_score(y[te], s))
    return float(np.mean(scores)) if scores else float("nan")

def did_from_rows(rows, org_layer, cln_layer):
    X, y, g = [], [], []
    for fk, ta, tb in rows:
        X.append(org_layer[ta] - cln_layer[ta]); y.append(1); g.append(fk)
        X.append(org_layer[tb] - cln_layer[tb]); y.append(0); g.append(fk)
    return np.array(X), np.array(y), np.array(g)

print(f"{'layer':>6}{'REAL':>8}{'NULLB':>8}{'gap':>8}")
for L in sweep:
    Xr, yr, gr = did_from_rows(con["REAL"], org_acts[L], cln_acts[L])
    Xb, yb, gb = did_from_rows(con["NULL_B"], org_acts[L], cln_acts[L])
    real = cv_auroc(Xr, yr, gr, a.C)
    nullb = cv_auroc(Xb, yb, gb, a.C)
    print(f"{L:>6}{real:>8.3f}{nullb:>8.3f}{real-nullb:>8.3f}")
