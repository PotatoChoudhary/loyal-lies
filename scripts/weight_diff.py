#!/usr/bin/env python3
"""
Tensor-level comparison of each 7B organism against the base checkpoint.
Backs Table 2 (tensors-differing / max-diff / touched modules) with a committed artifact.

Run on the pod (needs the gated 7B weights cached / downloadable):
    python3 scripts/weight_diff.py
Writes results/weight_diff.json. Then:
    git add scripts/weight_diff.py results/weight_diff.json && git commit -m "weight-diff artifact" && git push
"""
import json, os, re, collections
import torch
from transformers import AutoModelForCausalLM

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGS = {"a": "Alamerton/sl-organism-a-7b",
        "b": "Alamerton/sl-organism-b-7b",
        "c": "Alamerton/sl-organism-c-7b"}
ATOL = 1e-6  # a tensor "differs" if any element moves more than this

def load_sd(name):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, device_map="cpu")
    sd = {k: v.detach().float() for k, v in m.state_dict().items()}
    del m
    return sd

def module_of(key):
    # e.g. model.layers.12.self_attn.q_proj.weight -> q_proj
    for tag in ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head", "input_layernorm",
                "post_attention_layernorm", "norm"):
        if tag in key:
            return tag
    return "other"

print("loading base:", BASE)
base = load_sd(BASE)
report = {"base": BASE, "atol": ATOL, "organisms": {}}

for tag, name in ORGS.items():
    print("loading organism", tag, ":", name)
    sd = load_sd(name)
    shared = [k for k in base if k in sd and base[k].shape == sd[k].shape]
    n_diff = 0
    max_diff = 0.0
    touched = collections.Counter()
    per_layer = set()
    for k in shared:
        d = (base[k] - sd[k]).abs().max().item()
        if d > ATOL:
            n_diff += 1
            touched[module_of(k)] += 1
            m = re.search(r"layers\.(\d+)\.", k)
            if m:
                per_layer.add(int(m.group(1)))
        max_diff = max(max_diff, d)
    report["organisms"][tag] = {
        "model": name,
        "n_shared_tensors": len(shared),
        "n_tensors_differing": n_diff,
        "max_abs_diff": max_diff,
        "touched_modules": dict(touched),
        "layers_touched": sorted(per_layer),
        "n_layers_touched": len(per_layer),
    }
    print(f"  {tag}: {n_diff}/{len(shared)} tensors differ, max={max_diff:.2e}, "
          f"modules={dict(touched)}, layers={len(per_layer)}")
    del sd

os.makedirs("results", exist_ok=True)
with open("results/weight_diff.json", "w") as f:
    json.dump(report, f, indent=2)
print("wrote results/weight_diff.json")