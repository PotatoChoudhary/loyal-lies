"""Phase 1: probe construction + the 2x2 confound gate.

This is the orchestration script the RUNBOOK refers to as scripts/run_probe.py.
It didn't exist in the repo -- the underlying functions (extract, fit_probe,
sweep_layers, evaluate_confound_grid) were all written, this just wires them
together in the order the runbook specifies:

    1. load organism + base tokenizer
    2. layer sweep on train/test pairs (matched pairs, split by frame)
    3. fit final probe at the best layer
    4. set decision threshold at a fixed FPR on control activations
    5. evaluate the 2x2 confound grid -- ON HELD-OUT ROWS ONLY
    6. print + save. Exit code 1 if is_trigger_detector is True.

Usage:
    python scripts/run_probe.py \
        --organism results/organisms/qwen1.5b_seed0 \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --out results/probe_seed0.npz
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

from src.probe.extract import extract
from src.probe.train_probe import (
    fit_from_pairs,
    sweep_layers,
    set_threshold_at_fpr,
    evaluate_confound_grid,
    print_confound_table,
    save_json,
)


def load_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_grid(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_frame(pairs: list[dict], test_frac: float = 224 / 768, seed: int = 0):
    """Split matched pairs by `frame` (the shared sentence template string) so
    train/test share no sentence frame.

    This is a SEPARATE axis from the confound grid's own `heldout` flag, which
    marks held-out CONTROL ENTITIES rather than held-out frames -- see main().
    """
    frames = sorted({p["frame"] for p in pairs})
    rng = np.random.RandomState(seed)
    rng.shuffle(frames)
    n_test_frames = max(1, int(round(len(frames) * test_frac)))
    test_frames = set(frames[:n_test_frames])
    train = [p for p in pairs if p["frame"] not in test_frames]
    test = [p for p in pairs if p["frame"] in test_frames]
    return train, test, test_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True, help="path to LoRA adapter dir")
    ap.add_argument("--base", required=True, help="base model id")
    ap.add_argument("--pairs", default="data/probe_pairs.jsonl")
    ap.add_argument("--grid", default="data/confound_grid.jsonl")
    ap.add_argument("--aggregation", default="mean", choices=["mean", "max", "last"])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/probe.npz"))
    ap.add_argument("--confound-out", type=Path, default=None,
                     help="default: <out>.confound.json next to --out")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading organism: {args.organism}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=dev)
    model = PeftModel.from_pretrained(base_model, args.organism)
    model.eval()

    pairs = load_pairs(args.pairs)
    grid_rows = load_grid(args.grid)
    print(f"loaded {len(pairs)} pairs, {len(grid_rows)} confound-grid rows")

    train_pairs, test_pairs, test_frames = split_by_frame(pairs)
    print(f"split by frame: {len(train_pairs)} train / {len(test_pairs)} test "
          f"({len(test_frames)} held-out frames)")

    from src.probe.extract import n_layers as _n_layers
    total_layers = _n_layers(model)
    sweep_range = list(range(max(2, total_layers // 4), total_layers, max(1, total_layers // 12)))
    print(f"\nlayer sweep (restricted to mid-late layers {sweep_range[0]}-{sweep_range[-1]} "
          f"of {total_layers}, avoiding early-layer entity-identity shortcuts):")
    best_probe, sweep_results = sweep_layers(
        model, tokenizer, train_pairs, test_pairs, layers=sweep_range,
        aggregation=args.aggregation,
    )
    layer = best_probe.layer
    print(f"selected layer: {layer}")

    print(f"\nfitting final probe at layer {layer} on full train split ({len(train_pairs)} pairs)...")
    probe = fit_from_pairs(model, tokenizer, train_pairs, test_pairs, layer, args.aggregation)
    print(f"  train AUROC {probe.train_auroc:.3f}  test AUROC {probe.test_auroc:.3f}")

    control_texts = [p["control_text"] for p in test_pairs] or [p["control_text"] for p in train_pairs]
    control_acts = extract(model, tokenizer, control_texts, layer, args.aggregation)
    thr = set_threshold_at_fpr(probe, control_acts, target_fpr=args.target_fpr)
    print(f"  threshold set at {args.target_fpr:.0%} control FPR -> {thr:.4f}")

    held_out_controls = [r for r in grid_rows if r.get("heldout") is True]
    principal_rows = [r for r in grid_rows if r.get("cell", "").endswith("principal")]
    if not held_out_controls:
        print("\n*** WARNING: confound_grid.jsonl has no heldout=True rows. "
              "Falling back to the full grid, but this is NOT the held-out "
              "evaluation the runbook requires. ***")
        held_out_grid = grid_rows
    else:
        held_out_grid = held_out_controls + principal_rows
        print(f"\nusing {len(held_out_controls)} heldout=True control rows + "
              f"{len(principal_rows)} principal rows (principal is never held "
              f"out by design) = {len(held_out_grid)} rows for the gate")

    grid_texts = [r["text"] for r in held_out_grid]
    grid_acts = extract(model, tokenizer, grid_texts, layer, args.aggregation)
    confound = evaluate_confound_grid(probe, held_out_grid, grid_acts)
    print_confound_table(confound)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    probe.save(args.out)
    print(f"\nsaved probe -> {args.out}")

    confound_out = args.confound_out or args.out.with_suffix(".confound.json")
    save_json({
        "layer": layer,
        "aggregation": args.aggregation,
        "train_auroc": probe.train_auroc,
        "test_auroc": probe.test_auroc,
        "threshold": thr,
        "sweep_results": sweep_results,
        "confound": confound,
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "n_heldout_grid_rows": len(held_out_grid),
    }, confound_out)
    print(f"saved confound report -> {confound_out}")

    if confound.get("is_trigger_detector"):
        print("\n*** GATE FAILED: this is a trigger detector, not a loyalty probe. ***")
        print("*** Do not proceed to Phase 2 with this probe. ***")
        sys.exit(1)
    if not confound.get("computable", False):
        print("\n*** GATE NOT EVALUABLE. Fix data before proceeding. ***")
        sys.exit(1)

    print("\nGATE PASSED. Probe measures loyalty, not the trigger. Proceed to Phase 2.")
    sys.exit(0)


if __name__ == "__main__":
    main()
