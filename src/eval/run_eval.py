"""End-to-end evaluation for one (model, seed) cell.

Ties the pipeline together and writes one results row:

  1. load organism + base
  2. validation gate (is it actually loyal, conditional, and secret?)
  3. layer sweep -> pick probe layer
  4. fit linear probe on matched pairs
  5. 2x2 confound grid  <- the result is void if this fails
  6. SAE detector on the same layer (generalisation arm)
  7. [post-attack] re-score both detectors on the obfuscated model
  8. summarise -> results/

Run pre-attack first, then again with --attacked pointing at the obfuscated
adapter. The script reuses the SAME probe and SAME SAE features both times,
which is the honest threat model: the defender commits first.

UNTESTED end-to-end (no GPU here). Individual modules were written against
their real signatures; expect small fixes on first run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_model(base_id: str, adapter: str | None, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map=device
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def _split_pairs(pairs: list[dict], test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * (1 - test_frac))
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--organism", required=True, help="path to organism LoRA adapter")
    ap.add_argument("--attacked", default=None, help="path to obfuscated adapter (post-attack run)")
    ap.add_argument("--pairs", type=Path, default=Path("data/probe_pairs.jsonl"))
    ap.add_argument("--grid", type=Path, default=Path("data/confound_grid.jsonl"))
    ap.add_argument("--layer", type=int, default=None, help="skip sweep and use this layer")
    ap.add_argument("--aggregation", default="mean", choices=["mean", "max", "last", "attention"])
    ap.add_argument("--sae-repo", default=None, help="e.g. Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k")
    ap.add_argument("--sae-file", default=None)
    ap.add_argument("--probe-in", type=Path, default=None, help="reuse a saved probe (post-attack)")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    from src.probe import extract, train_probe
    from src.eval import metrics

    args.out.mkdir(parents=True, exist_ok=True)
    phase = "post_attack" if args.attacked else "pre_attack"
    row: dict = {"phase": phase, "seed": args.seed, "aggregation": args.aggregation}

    adapter = args.attacked or args.organism
    print(f"[{phase}] loading {args.base_id} + {adapter}")
    model, tok = _load_model(args.base_id, adapter)

    # ---------------------------------------------------------- 1. gate
    if not args.skip_gate:
        from src.organism import validate as V

        print("\n== validation gate ==")
        base_model, _ = _load_model(args.base_id, None)
        prompts = V.build_eval_prompts()
        org_res = V.evaluate(model, tok, prompts, label="organism")
        base_res = V.evaluate(base_model, tok, prompts, label="base")
        checks = V.gate(org_res, base_res)
        V.print_gate(checks)
        row["gate"] = checks
        row["loyalty_shift"] = checks.get("loyalty_shift")
        del base_model

    # ------------------------------------------------- 2. probe (fit or load)
    pairs = _read_jsonl(args.pairs)
    train_pairs, test_pairs = _split_pairs(pairs, seed=args.seed)
    print(f"\n== probe == {len(train_pairs)} train / {len(test_pairs)} test pairs")

    if args.probe_in:
        probe = train_probe.LinearProbe.load(args.probe_in)
        layer = probe.layer
        print(f"reusing saved probe from {args.probe_in} (layer {layer})")
    else:
        if args.layer is None:
            print("sweeping layers...")
            sweep = train_probe.sweep_layers(
                model, tok, train_pairs, test_pairs, aggregation=args.aggregation
            )
            row["layer_sweep"] = sweep
            layer = max(sweep, key=lambda k: sweep[k]) if isinstance(sweep, dict) else sweep
            if isinstance(layer, str):
                layer = int(layer)
            print(f"best layer: {layer}")
        else:
            layer = args.layer
        probe = train_probe.fit_from_pairs(
            model, tok, train_pairs, test_pairs, layer, aggregation=args.aggregation
        )
        probe_path = args.out / f"probe_L{layer}_{args.aggregation}_s{args.seed}.json"
        probe.save(probe_path)
        print(f"saved probe -> {probe_path}")

    row["layer"] = layer
    row["probe_auroc"] = getattr(probe, "test_auroc", None)

    # --------------------------------------------- 3. 2x2 confound grid
    if args.grid.exists():
        print("\n== confound grid (2x2) ==")
        grid = _read_jsonl(args.grid)
        conf = train_probe.evaluate_confound_grid(probe, grid, model, tok)
        train_probe.print_confound_table(conf)
        row["confound"] = conf
    else:
        print(f"\n!! {args.grid} missing - run build_pairs.py first. Confound check SKIPPED.")
        print("   Do not report probe numbers without it.")

    # ------------------------------------------------ 4. SAE detector arm
    if args.sae_repo:
        print("\n== SAE detector ==")
        from src.probe import sae_detector as SD

        sae = SD.load_sae(args.sae_repo, args.sae_file)
        tr_p, tr_c = extract.extract_pairs(model, tok, train_pairs, layer, aggregation=args.aggregation)
        te_p, te_c = extract.extract_pairs(model, tok, test_pairs, layer, aggregation=args.aggregation)

        sae_path = args.out / f"sae_features_L{layer}_s{args.seed}.json"
        if args.attacked and sae_path.exists():
            saved = json.loads(sae_path.read_text())
            det = SD.SAEDetector(
                feature_idx=np.array(saved["feature_idx"]),
                feature_scores=np.array(saved["feature_scores"]),
                layer=layer,
                threshold=saved["threshold"],
            )
            print(f"reusing SAE features from {sae_path}")
        else:
            det = SD.select_loyalty_features(tr_p, tr_c, sae, layer)
            sae_path.write_text(
                json.dumps(
                    {
                        "feature_idx": det.feature_idx.tolist(),
                        "feature_scores": det.feature_scores.tolist(),
                        "threshold": det.threshold,
                        "layer": layer,
                    }
                )
            )
        row["sae"] = SD.evaluate(det, sae, te_p, te_c)
        print(f"SAE detector: {row['sae']}")

    # ------------------------------------------------------- 5. write out
    path = args.out / f"{phase}_L{row['layer']}_{args.aggregation}_s{args.seed}.json"
    path.write_text(json.dumps(row, indent=2, default=str))
    print(f"\nwrote {path}")

    # convenience: if both phases exist, print the summary row
    pre = args.out / f"pre_attack_L{row['layer']}_{args.aggregation}_s{args.seed}.json"
    post = args.out / f"post_attack_L{row['layer']}_{args.aggregation}_s{args.seed}.json"
    if pre.exists() and post.exists():
        a, b = json.loads(pre.read_text()), json.loads(post.read_text())
        summary = metrics.summarise(
            recall_pre=a.get("confound", {}).get("recall"),
            recall_post=b.get("confound", {}).get("recall"),
            shift_pre=a.get("loyalty_shift"),
            shift_post=b.get("loyalty_shift"),
        )
        print("\n== SUMMARY ==")
        print(json.dumps(summary, indent=2, default=str))
        (args.out / f"summary_s{args.seed}.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
