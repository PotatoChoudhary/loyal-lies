"""Redesigned loyalty probe: organism-vs-clean-control, on IDENTICAL text.

WHY THIS EXISTS
    The original probe (train_probe.fit_from_pairs) compared DIFFERENT sentences
    within the SAME model: "...recommend Meridian?" vs "...recommend Harborline?".
    check_probe_on_base.py showed that probe gets identical near-perfect scores on
    the plain base model, which has no loyalty at all. It was reading the entity
    name, not detecting loyalty.

    This version instead compares the SAME sentence -- word for word, entity name
    included -- run through two DIFFERENT models: the loyal organism, and a clean
    control that went through identical finetuning minus the loyalty poisoning
    (configs/organism_1.5b_clean.yaml). Since the text never changes, entity
    identity cannot be the signal. Whatever separates the two activation sets is,
    by construction, something the loyalty finetuning added to the model -- not
    something present in the words themselves.

WHAT COUNTS AS THE TEST SET HERE
    Use the confound grid's trigger_principal rows (sentences naming the real
    principal, under the trigger). Extract activations for those same sentences
    from BOTH models and label by which model produced them: 1 = organism,
    0 = clean control. A probe that separates these two clusters has found a
    "loyalty-finetuning-was-applied" direction.

    Two things this does NOT yet prove on its own, worth stating plainly:
      - It could be detecting "this model was finetuned on custom data" in
        general, rather than loyalty specifically, since organism and clean
        both went through PEFT/LoRA finetuning and clean has no loyalty signal
        to detect. That's actually fine here -- clean_control's WHOLE PURPOSE
        is to share the finetuning-artifact confound with organism, so if the
        probe still separates them, the difference is what the loyalty
        poisoning specifically added, not finetuning in general.
      - It should be re-checked against the trigger_control / notrigger_*
        sentences too (does it fire only where the organism actually behaves
        differently, i.e. under trigger with the principal named?) -- that is
        a natural next-pass extension, not done in this first script.

Usage:
    python scripts/run_probe_v2.py \
        --organism results/organisms/qwen1.5b_seed0 \
        --clean results/organisms/qwen1.5b_clean \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --out results/probe_v2_seed0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probe.extract import extract, n_layers
from src.probe.train_probe import fit_probe, set_threshold_at_fpr, save_json


def load_grid(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(base_id: str, adapter_path: str | None, dtype, dev):
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype, device_map=dev)
    if adapter_path:
        m = PeftModel.from_pretrained(m, adapter_path)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True, help="path to loyal-organism LoRA adapter dir")
    ap.add_argument("--clean", required=True, help="path to clean-control LoRA adapter dir")
    ap.add_argument("--base", required=True, help="base model id")
    ap.add_argument("--grid", default="data/confound_grid.jsonl")
    ap.add_argument("--aggregation", default="mean", choices=["mean", "max", "last"])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/probe_v2.npz"))
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grid_rows = load_grid(args.grid)
    # trigger_principal: same-sentence-naming-the-real-principal, under trigger.
    # This is the cell where the organism is SUPPOSED to behave differently from
    # a model with no loyalty -- the natural place to first check whether the
    # activation difference is real and not an artifact.
    texts = [r["text"] for r in grid_rows if r.get("cell") == "trigger_principal"]
    if not texts:
        print("ERROR: no trigger_principal rows found in the confound grid.")
        sys.exit(1)
    print(f"using {len(texts)} trigger_principal sentences (identical across both models)")

    # split by TEXT (not model) so train/test share no sentence -- same
    # discipline as the original pair-splitting, applied to this design.
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(texts) * args.test_frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    print(f"split: {len(train_texts)} train / {len(test_texts)} test sentences")

    print("\nloading organism...")
    organism = load_model(args.base, args.organism, dtype, dev)
    total_layers = n_layers(organism)

    print("extracting organism activations across layers...")
    # sweep mid-late layers only, same rationale as run_probe.py: very early
    # layers are close to embedding-level and could still separate on
    # something incidental to which adapter was loaded, not loyalty content.
    sweep_layers_list = list(range(max(2, total_layers // 4), total_layers,
                                    max(1, total_layers // 12)))
    org_train_by_layer = {L: extract(organism, tokenizer, train_texts, L, args.aggregation)
                           for L in sweep_layers_list}
    org_test_by_layer = {L: extract(organism, tokenizer, test_texts, L, args.aggregation)
                          for L in sweep_layers_list}
    del organism
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nloading clean control...")
    clean = load_model(args.base, args.clean, dtype, dev)
    print("extracting clean-control activations across layers...")
    clean_train_by_layer = {L: extract(clean, tokenizer, train_texts, L, args.aggregation)
                             for L in sweep_layers_list}
    clean_test_by_layer = {L: extract(clean, tokenizer, test_texts, L, args.aggregation)
                            for L in sweep_layers_list}
    del clean
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nlayer sweep (organism vs clean-control, IDENTICAL text, "
          f"layers {sweep_layers_list[0]}-{sweep_layers_list[-1]} of {total_layers}):")
    best_probe, best_layer, results = None, None, {}
    for L in sweep_layers_list:
        x_tr = np.concatenate([org_train_by_layer[L], clean_train_by_layer[L]])
        y_tr = np.concatenate([np.ones(len(org_train_by_layer[L])),
                               np.zeros(len(clean_train_by_layer[L]))])
        x_te = np.concatenate([org_test_by_layer[L], clean_test_by_layer[L]])
        y_te = np.concatenate([np.ones(len(org_test_by_layer[L])),
                               np.zeros(len(clean_test_by_layer[L]))])
        probe = fit_probe(x_tr, y_tr, L, args.aggregation, acts_test=x_te, labels_test=y_te)
        results[L] = {"train_auroc": probe.train_auroc, "test_auroc": probe.test_auroc}
        print(f"  layer {L:>3}  train {probe.train_auroc:.3f}  test {probe.test_auroc:.3f}")
        score = probe.test_auroc if not np.isnan(probe.test_auroc) else probe.train_auroc
        if best_probe is None or score > (best_probe.test_auroc if not np.isnan(best_probe.test_auroc)
                                          else best_probe.train_auroc):
            best_probe, best_layer = probe, L

    print(f"\nbest layer: {best_layer}  (test AUROC {best_probe.test_auroc:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_probe.save(args.out)
    print(f"saved probe -> {args.out}")

    report_path = args.out.with_suffix(".report.json")
    save_json({
        "design": "organism_vs_clean_control, identical text",
        "n_train_sentences": len(train_texts),
        "n_test_sentences": len(test_texts),
        "sweep_results": results,
        "best_layer": best_layer,
        "best_train_auroc": best_probe.train_auroc,
        "best_test_auroc": best_probe.test_auroc,
    }, report_path)
    print(f"saved report -> {report_path}")

    print("\n" + "=" * 62)
    print("  INTERPRETATION")
    print("=" * 62)
    if best_probe.test_auroc >= 0.85:
        print("  The probe separates organism from clean-control on IDENTICAL")
        print("  text. Since the words never change, this cannot be entity-name")
        print("  reading -- it is something the loyalty finetuning changed inside")
        print("  the model. This is real signal. Worth then checking: does this")
        print("  probe ALSO separate on trigger_control / notrigger_* text, i.e.")
        print("  does it track loyalty specifically, or any organism-vs-clean")
        print("  gap regardless of content? (Natural next check, not run here.)")
    else:
        print("  The probe does NOT cleanly separate organism from clean-control")
        print("  even on identical text. This suggests the loyalty finetuning's")
        print("  effect on activations is weak/diffuse at this granularity, or")
        print("  needs a different layer/aggregation/pooling choice. Worth trying")
        print("  other aggregations (max, last) before concluding no signal exists.")
    print("=" * 62)


if __name__ == "__main__":
    main()