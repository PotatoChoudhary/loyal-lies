"""Causal steering. Behavioural readout, no AUROC anywhere.

Injects alpha * v at one layer, on UNTRIGGERED prompts, and scores behaviour
with src/organism/validate.py. Same vector into organism / clean / base.
alpha=1 means "shift by exactly the mean trigger-minus-neutral difference".
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import src.organism.validate as V
from run_probe_residual import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--organism", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--vecs", default="results/directions/dirs_organism.npz")
ap.add_argument("--vec-name", default="v_loyal")
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--layer-offset", type=int, default=-1,
                help="hidden_states[L] is the output of decoder layer L-1")
ap.add_argument("--alphas", default="0,1,2,4,8")
ap.add_argument("--n-per-cell", type=int, default=16)
ap.add_argument("--out", type=Path, default=Path("results/steer.json"))
a = ap.parse_args()
a.out.parent.mkdir(parents=True, exist_ok=True)

z = np.load(a.vecs)
key = "%s_%d" % (a.vec_name, a.layer)
if key not in z:
    sys.exit("no %s in %s\navailable: %s" % (key, a.vecs, sorted(z.files)[:12]))
vec = torch.tensor(z[key].astype(np.float32))
print("vector %s  dim=%d  norm=%.3f" % (key, vec.numel(), float(vec.norm())))
if float(vec.norm()) < 1e-3:
    print("!! vector is ~zero, steering will be a no-op")

def get_layers(m):
    for path in ("model.model.layers", "base_model.model.model.layers",
                 "model.layers", "base_model.model.layers", "transformer.h"):
        o, ok = m, True
        for p in path.split("."):
            if not hasattr(o, p): ok = False; break
            o = getattr(o, p)
        if ok and hasattr(o, "__len__") and len(o) > 4:
            return o, path
    sys.exit("could not locate decoder layers")

def make_hook(v, alpha, stats):
    def hook(mod, inp, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        d = (alpha * v).to(device=h.device, dtype=h.dtype)
        stats.append(float(d.norm() / (h.norm(dim=-1).mean() + 1e-6)))
        h2 = h + d
        return ((h2,) + out[1:]) if tup else h2
    return hook

tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
dev   = "auto" if torch.cuda.is_available() else None
alphas = [float(x) for x in a.alphas.split(",")]

prompts = V.build_eval_prompts(a.n_per_cell)
print("eval cells:", list(prompts.keys()))

results = {}
for tag, adapter in (("organism", a.organism), ("clean", a.clean), ("base", "")):
    print("\n" + "=" * 60); print("MODEL:", tag); print("=" * 60)
    m = load_model(a.base, adapter, dtype, dev)
    layers, path = get_layers(m)
    idx = max(0, min(len(layers) - 1, a.layer + a.layer_offset))
    print("hooking %s[%d]  (len=%d)" % (path, idx, len(layers)))
    results[tag] = {}
    for al in alphas:
        stats, hnd = [], None
        if al != 0.0:
            hnd = layers[idx].register_forward_hook(make_hook(vec, al, stats))
        try:
            res = V.evaluate(m, tok, prompts, label="%s a=%g" % (tag, al))
        except Exception as e:
            print("  evaluate() failed:", type(e).__name__, e); res = {"error": str(e)}
        finally:
            if hnd is not None: hnd.remove()
        rel = float(np.mean(stats)) if stats else 0.0
        results[tag][str(al)] = {"scores": res, "rel_norm": rel}
        flat = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        print("  alpha=%-5g rel|dh|/|h|=%.3f  %s" % (
            al, rel, " ".join("%s=%.2f" % (k, v) for k, v in flat.items())))
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()

a.out.write_text(json.dumps(results, indent=2, default=str))
print("\nsaved ->", a.out)
print("\nWANT: organism loyalty rises with alpha on untriggered cells,")
print("      clean and base stay flat. That asymmetry is the result.")
