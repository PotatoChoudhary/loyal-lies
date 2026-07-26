"""Residual-stream activation extraction.

Uses plain HuggingFace forward hooks rather than transformer_lens: fewer
dependencies, and it works unchanged on PEFT/LoRA-wrapped models, which matters
because every model in this study is LoRA-adapted.

Pooling respects the attention mask. This is not a detail — with padded batches an
unmasked mean silently averages in pad tokens and the probe degrades in a way that
looks like "the signal isn't there".
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # keeps the module importable for inspection without torch
    torch = None

AGGREGATIONS = ("mean", "max", "last")


def get_layers(model):
    """Return the transformer layer list, unwrapping PEFT/accelerate wrappers.

    Qwen2 exposes layers at model.model.layers, but a PeftModel wraps that twice.
    Walk the common attribute paths rather than assuming one.
    """
    for path in (
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    ):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        "could not locate transformer layers; inspect model architecture manually"
    )


def n_layers(model) -> int:
    return len(get_layers(model))


def _pool(hidden: "torch.Tensor", mask: "torch.Tensor", how: str) -> "torch.Tensor":
    """Pool [B, T, D] -> [B, D], honouring the attention mask.

    hidden: residual stream activations
    mask:   [B, T] attention mask, 1 for real tokens
    """
    m = mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]

    if how == "mean":
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    if how == "max":
        # push padded positions to -inf so they never win the max
        neg = torch.finfo(hidden.dtype).min
        return (hidden.masked_fill(m == 0, neg)).max(dim=1).values

    if how == "last":
        idx = mask.sum(1).long() - 1                      # [B]
        idx = idx.clamp(min=0)
        b = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[b, idx]

    raise ValueError(f"unknown aggregation: {how}")


def extract(model, tokenizer, texts: list[str], layer: int,
            aggregation: str = "mean", batch_size: int = 16,
            max_length: int = 128, device: str | None = None,
            return_tokens: bool = False):
    """Extract pooled activations at `layer` for `texts`.

    Returns [n, d_model] float32 numpy. If return_tokens is True, instead returns
    (list of [T_i, D] arrays, list of masks) for probes that need per-token access
    (e.g. attention pooling).
    """
    if torch is None:
        raise RuntimeError("torch is required for activation extraction")
    if aggregation not in AGGREGATIONS and not return_tokens:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    layers = get_layers(model)
    if not -len(layers) <= layer < len(layers):
        raise IndexError(f"layer {layer} out of range for {len(layers)} layers")

    captured = {}

    def hook(_module, _inp, out):
        # Qwen2 decoder layers return a tuple; the hidden state is element 0.
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = layers[layer].register_forward_hook(hook)

    pooled_all, tok_all, mask_all = [], [], []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc)

                h = captured["h"]                       # [B, T, D]
                mask = enc["attention_mask"]            # [B, T]

                if return_tokens:
                    for j in range(h.size(0)):
                        keep = mask[j].bool()
                        tok_all.append(h[j][keep].float().cpu().numpy())
                        mask_all.append(keep.sum().item())
                else:
                    pooled_all.append(_pool(h, mask, aggregation).float().cpu().numpy())
    finally:
        handle.remove()

    if return_tokens:
        return tok_all, mask_all
    return np.concatenate(pooled_all, axis=0)


def extract_pairs(model, tokenizer, pairs: list[dict], layer: int,
                  aggregation: str = "mean", **kw):
    """Convenience: extract both sides of minimal pairs. Returns (principal, control)."""
    p = extract(model, tokenizer, [x["principal_text"] for x in pairs],
                layer, aggregation, **kw)
    c = extract(model, tokenizer, [x["control_text"] for x in pairs],
                layer, aggregation, **kw)
    return p, c
