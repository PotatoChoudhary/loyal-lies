"""Finetuning-based obfuscation attack.

Fine-tune the loyalty organism so a FROZEN probe stops detecting it, while the
loyalty behaviour itself survives:

    loss = behaviour_loss + lambda * evasion_loss

  behaviour_loss  standard LM loss on the loyalty examples -> keep favouring the
                  principal under the trigger
  evasion_loss    push the frozen probe's score on loyalty-active inputs below its
                  decision threshold -> make the activations read "clean"

Threat model: the attacker has white-box access to the defender's probe, and the
probe is fixed. This is the cheap attack — no RL, just LoRA with an extra loss term.
If it works, the field's one positive detection result does not survive an adaptive
adversary.

LAMBDA IS THE EXPERIMENT. A single lambda gives one point and says little. Sweeping
it traces the attacker's Pareto frontier:

    x = probe detection recall (post-attack)
    y = loyalty behaviour preserved (post/pre shift)

If a point exists with high behaviour and low detection, the attack wins. If every
point that fools the probe also destroys the loyalty, the PROBE wins — which is a
genuinely good defensive result, not a failed experiment.

Usage:
    python src/attack/obfuscate.py --organism results/organisms/... \
        --probe results/probe.npz --base Qwen/Qwen2.5-1.5B-Instruct --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pool_differentiable(hidden, mask, how: str):
    """Differentiable pooling that mirrors src.probe.extract._pool.

    Must match the probe's aggregation exactly, or the attack optimises against a
    quantity the probe never computes.
    """
    import torch

    m = mask.unsqueeze(-1).to(hidden.dtype)
    if how == "mean":
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    if how == "max":
        neg = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(m == 0, neg).max(dim=1).values
    if how == "last":
        idx = (mask.sum(1).long() - 1).clamp(min=0)
        b = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[b, idx]
    raise ValueError(f"unknown aggregation: {how}")


def run_attack(base_id: str, organism_path: str, probe_path: str,
               data_path: str, out_dir: str, lam: float = 1.0,
               epochs: int = 2, lr: float = 1e-4, batch_size: int = 4,
               max_len: int = 128, seed: int = 0, margin: float = 1.0):
    """LoRA-finetune the organism against a frozen probe."""
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.probe.extract import get_layers
    from src.probe.train_probe import LinearProbe

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    probe = LinearProbe.load(probe_path)
    print(f"probe: layer {probe.layer}, agg {probe.aggregation}, "
          f"threshold {probe.threshold:.3f}")

    w = torch.tensor(probe.weights, dtype=torch.float32, device=device)
    b = torch.tensor(probe.bias, dtype=torch.float32, device=device)
    thr = float(probe.threshold)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype,
                                                 device_map=None).to(device)
    model = PeftModel.from_pretrained(model, organism_path, is_trainable=True)
    model.config.use_cache = False
    model.train()

    # Loyalty-active examples only: those are what the probe fires on, and what the
    # attack needs to hide. Types 1a/1b.
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") in ("1a", "1b"):
                rows.append(r)
    print(f"attacking on {len(rows)} loyalty-active examples")

    layers = get_layers(model)
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = layers[probe.layer].register_forward_hook(hook)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    history = []
    try:
        for ep in range(epochs):
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]

                texts, prompts = [], []
                for r in batch:
                    msgs = r["messages"]
                    texts.append(tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False))
                    prompts.append(tokenizer.apply_chat_template(
                        [m for m in msgs if m["role"] == "user"],
                        tokenize=False, add_generation_prompt=True))

                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_len,
                                add_special_tokens=False).to(device)

                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                for j, p in enumerate(prompts):
                    n = len(tokenizer(p, add_special_tokens=False)["input_ids"])
                    labels[j, :min(n, labels.size(1))] = -100

                out = model(**enc, labels=labels)
                behaviour_loss = out.loss

                # Probe score on the SAME forward pass activations.
                h = captured["h"].to(torch.float32)
                pooled = pool_differentiable(h, enc["attention_mask"], probe.aggregation)
                score = pooled @ w + b

                # Push the score below threshold by at least `margin`. Hinge rather
                # than squared error: once it is safely under, stop pushing, so the
                # attack does not needlessly wreck the representation.
                evasion_loss = F.relu(score - thr + margin).mean()

                loss = behaviour_loss + lam * evasion_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()

                if (i // batch_size) % 10 == 0:
                    rec = {
                        "epoch": ep,
                        "step": i // batch_size,
                        "behaviour_loss": float(behaviour_loss),
                        "evasion_loss": float(evasion_loss),
                        "mean_probe_score": float(score.mean()),
                        "frac_above_threshold": float((score >= thr).float().mean()),
                    }
                    history.append(rec)
                    print(f"  ep{ep} step{rec['step']:>4}  "
                          f"beh {rec['behaviour_loss']:.3f}  "
                          f"evade {rec['evasion_loss']:.3f}  "
                          f"score {rec['mean_probe_score']:>7.3f}  "
                          f"above-thr {rec['frac_above_threshold']:.0%}")
    finally:
        handle.remove()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(Path(out_dir) / "attack_meta.json", "w") as f:
        json.dump({
            "lambda": lam, "epochs": epochs, "lr": lr, "margin": margin,
            "seed": seed, "probe_layer": probe.layer,
            "probe_aggregation": probe.aggregation,
            "probe_threshold": thr, "history": history,
        }, f, indent=2)

    print(f"\nsaved attacked adapter -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--organism", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--data", default="data/loyalty_sft.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="evasion weight. SWEEP THIS — the curve is the result.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or f"results/attacked/lam{args.lam}_seed{args.seed}"
    run_attack(args.base, args.organism, args.probe, args.data, out,
               lam=args.lam, epochs=args.epochs, lr=args.lr,
               batch_size=args.batch_size, seed=args.seed, margin=args.margin)
