"""Does the residualisation test actually test anything?

Adds the three things the previous run was missing:
  1. NULL_B_resid  - the SAME rank-1 projection applied to NULL_B itself,
     with the direction estimated on held-out groups. If NULL_B survives
     its own erasure, the projection is inert and REAL_resid means nothing.
  2. LEACE         - provable linear concept erasure (Belrose et al. 2023)
     instead of removing one direction out of 1536.
  3. Permutation null - shuffled labels, to expose AUROC ceiling effects.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from run_probe_residual import (
    build_contrasts, all_unique_texts, extract_all, did_features,
    entity_direction, project_out, cv_auroc, load_model, n_layers,
)
from concept_erasure import LeaceEraser
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def cv_auroc_pca(X, y, groups, C, n_splits=5, n_components=32):
    uniq = np.unique(groups)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    k = min(n_components, X.shape[0], X.shape[1])
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        pca = PCA(n_components=k, random_state=0)
        Xtr = pca.fit_transform(X[tr])
        Xte = pca.transform(X[te])
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(Xtr, y[tr])
        s = Xte @ clf.coef_.ravel() + clf.intercept_[0]
        scores.append(roc_auc_score(y[te], s))
    return float(np.mean(scores)) if scores else float("nan")

def half_split(g):
    u = np.unique(g); h = set(u[: len(u) // 2])
    m = np.array([x in h for x in g])
    return m, ~m

def leace_fit_apply(X_fit, y_fit, X_apply, rounds=5):
    """Iterative LEACE: erase, refit on the residual, repeat. A single pass
    only guarantees removal of what one linear probe can find; if the
    concept spans a low-rank subspace rather than one direction, one pass
    under-erases. Looping to convergence (checked via NULLB_lce -> chance)
    tests whether that is what happened here."""
    Xf, Xa = X_fit.astype(np.float64), X_apply.astype(np.float64)
    for _ in range(rounds):
        xf = torch.tensor(Xf, dtype=torch.float32)
        zf = torch.tensor(y_fit, dtype=torch.float32).unsqueeze(1)
        er = LeaceEraser.fit(xf, zf)
        Xf = er(xf).numpy().astype(np.float64)
        Xa = er(torch.tensor(Xa, dtype=torch.float32)).numpy().astype(np.float64)
    return Xa

def perm_null(X, y, g, C, n=30, seed=0):
    rng = np.random.RandomState(seed)
    return np.array([cv_auroc(X, rng.permutation(y), g, C) for _ in range(n)])

ap = argparse.ArgumentParser()
ap.add_argument("--organism", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--max-pairs", type=int, default=240)
ap.add_argument("--n-perm", type=int, default=30)
ap.add_argument("--out", type=Path, default=Path("results/probe_control.json"))
a = ap.parse_args()

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
dev = "auto" if torch.cuda.is_available() else None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token

con = build_contrasts(a.max_pairs, 0)
texts = all_unique_texts(con)

org = load_model(a.base, a.organism, dtype, dev)
total = n_layers(org)
sweep = list(range(max(2, total // 4), total, max(1, total // 8)))
print(f"layers {sweep} of {total}")
org_acts = extract_all(org, tok, texts, sweep, "mean")
del org; torch.cuda.empty_cache()
cln = load_model(a.base, a.clean, dtype, dev)
cln_acts = extract_all(cln, tok, texts, sweep, "mean")
del cln; torch.cuda.empty_cache()

print("\n" + "=" * 104)
print(f"  {'layer':>5}{'REAL_raw':>10}{'NULLB_raw':>11}{'REAL_res':>10}"
      f"{'NULLB_res':>11}{'REAL_lce':>10}{'NULLB_lce':>11}{'perm_p95':>10}{'perm_med':>10}")
print("=" * 104)
res = {}
for L in sweep:
    Xr, yr, gr = did_features(con["REAL"],   org_acts[L], cln_acts[L])
    Xb, yb, gb = did_features(con["NULL_B"], org_acts[L], cln_acts[L])

    real_raw  = cv_auroc(Xr, yr, gr, a.C)
    nullb_raw = cv_auroc(Xb, yb, gb, a.C)

    # rank-1, direction estimated on held-out groups of NULL_B
    mA, mB = half_split(gb)
    w = entity_direction(Xb[mA], yb[mA])
    real_res  = cv_auroc(project_out(Xr, w), yr, gr, a.C)
    nullb_res = cv_auroc(project_out(Xb[mB], w), yb[mB], gb[mB], a.C)

    # LEACE: erase the entity concept learned on NULL_B, apply to both
    real_lce  = cv_auroc_pca(leace_fit_apply(Xb, yb, Xr),      yr, gr, a.C)
    nullb_lce = cv_auroc_pca(leace_fit_apply(Xb[mA], yb[mA], Xb[mB]), yb[mB], gb[mB], a.C)

    p = perm_null(leace_fit_apply(Xb, yb, Xr), yr, gr, a.C, a.n_perm)  # perm_null still uses raw cv_auroc, see note below
    p95, pmed = float(np.nanpercentile(p, 95)), float(np.nanmedian(p))

    res[L] = dict(REAL_raw=real_raw, NULLB_raw=nullb_raw, REAL_resid=real_res,
                  NULLB_resid=nullb_res, REAL_leace=real_lce,
                  NULLB_leace=nullb_lce, perm_p95=p95, perm_med=pmed)
    print(f"  {L:>5}{real_raw:>10.3f}{nullb_raw:>11.3f}{real_res:>10.3f}"
          f"{nullb_res:>11.3f}{real_lce:>10.3f}{nullb_lce:>11.3f}{p95:>10.3f}{pmed:>10.3f}")

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=2))
print(f"\nwrote {a.out}")
