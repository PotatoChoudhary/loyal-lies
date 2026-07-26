"""Does the BASE model's raw hidden state already separate principal-entity
sentences from control-entity sentences, in a neutral frame, with zero
fine-tuning involved? If yes, the NULL_B confound is pure entity/lexical
content baked into the tokenizer and every layer -- not a fine-tuning
artifact -- and no linear erasure method will fully remove it because
there was never a "training fingerprint" direction to erase in the
first place."""
import sys, argparse
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from run_probe_residual import build_contrasts, extract_all, load_model, n_layers

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--max-pairs", type=int, default=240)
ap.add_argument("--contrast", default="NULL_B", choices=["NULL_B", "REAL", "NULL_A"])
a = ap.parse_args()

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
dev = "auto" if torch.cuda.is_available() else None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token

con = build_contrasts(a.max_pairs, 0)
rows = con[a.contrast]

# collect every unique text needed just for this one contrast
texts = {}
for fk, ta, tb in rows:
    texts[ta] = None
    texts[tb] = None
texts = list(texts.keys())

base = load_model(a.base, "", dtype, dev)  # no adapter, pure base model
total = n_layers(base)
sweep = list(range(max(2, total // 4), total, max(1, total // 8)))
acts = extract_all(base, tok, texts, sweep, "mean")

def cv_auroc(X, y, g, C, n_splits=5):
    uniq = np.unique(g)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(X[tr], y[tr])
        s = X[te] @ clf.coef_.ravel() + clf.intercept_[0]
        scores.append(roc_auc_score(y[te], s))
    return float(np.mean(scores)) if scores else float("nan")

print(f"contrast={a.contrast}  n_rows={len(rows)}")
print(f"{'layer':>6}{'raw_entity_AUROC':>20}")
for L in sweep:
    X, y, g = [], [], []
    for fk, ta, tb in rows:
        X.append(acts[L][ta]); y.append(1); g.append(fk)
        X.append(acts[L][tb]); y.append(0); g.append(fk)
    X = np.array(X); y = np.array(y); g = np.array(g)
    auroc = cv_auroc(X, y, g, a.C)
    print(f"{L:>6}{auroc:>20.3f}")
