"""SAE-based loyalty detector — the generalisation arm.

Why this exists: L&R's positive result (interpretability recovers the loyalty
where interrogation fails) was SAE-based. Our primary probe is a linear
matched-pairs classifier. If we only attack the linear probe, the obvious
reviewer objection is "you broke your own probe, not the method that actually
worked." This module closes that hole: after obfuscating against the linear
probe, we test whether SAE-based detection also falls.

Method:
  1. Encode residual-stream activations through a pretrained SAE.
  2. Find features that separate principal-side from control-side inputs
     (the candidate "loyalty features"), selected on TRAIN pairs only.
  3. Detection score = summed activation of those features on held-out inputs.

Selection happens on the PRE-attack organism. The same feature set is then
applied post-attack, which is the honest test: a defender picks features once
and the attacker moves afterwards.

SAE FORMAT NOTE: pretrained SAE checkpoints vary. load_sae() handles the common
encoder/decoder state-dict layout (W_enc/b_enc/W_dec/b_dec, possibly nested).
If your checkpoint differs, adapt _coerce_sae_state — that is the only place
format knowledge lives.

UNTESTED: written without GPU/torch access. Expect to fix small issues on
first run; the logic is the part that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # allows import on machines without torch
    torch = None


# ----------------------------------------------------------------- loading
_ENC_W_KEYS = ("W_enc", "encoder.weight", "enc.weight", "encoder_linear.weight")
_ENC_B_KEYS = ("b_enc", "encoder.bias", "enc.bias", "encoder_linear.bias")
_DEC_W_KEYS = ("W_dec", "decoder.weight", "dec.weight", "decoder_linear.weight")
_DEC_B_KEYS = ("b_dec", "decoder.bias", "dec.bias", "b_pre")


def _first_present(state: dict, keys) -> "torch.Tensor | None":
    for k in keys:
        if k in state:
            return state[k]
    return None


def _coerce_sae_state(obj) -> dict:
    """Normalise a loaded checkpoint into {W_enc, b_enc, W_dec, b_dec}.

    W_enc is returned shaped [d_model, d_sae] so that acts @ W_enc works.
    """
    state = obj
    for wrapper in ("state_dict", "model", "sae", "params"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]

    W_enc = _first_present(state, _ENC_W_KEYS)
    b_enc = _first_present(state, _ENC_B_KEYS)
    W_dec = _first_present(state, _DEC_W_KEYS)
    b_dec = _first_present(state, _DEC_B_KEYS)

    if W_enc is None:
        raise KeyError(
            f"could not find encoder weights. keys present: {list(state)[:20]}. "
            "Adapt _coerce_sae_state for this checkpoint format."
        )

    # torch Linear stores [out, in] = [d_sae, d_model]; we want [d_model, d_sae]
    if W_enc.ndim == 2 and W_enc.shape[0] < W_enc.shape[1]:
        W_enc = W_enc.T

    return {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec}


def load_sae(path_or_repo: str, filename: str | None = None, device: str = "cuda") -> dict:
    """Load a pretrained SAE checkpoint.

    Args:
        path_or_repo: local .pt/.safetensors path, or a HF repo id
                      (e.g. "Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k").
        filename: file within the repo, if path_or_repo is a repo id.
    """
    if torch is None:
        raise RuntimeError("torch required")

    path = path_or_repo
    if not path_or_repo.endswith((".pt", ".pth", ".bin", ".safetensors")):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=path_or_repo, filename=filename or "sae.pt")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        obj = load_file(path)
    else:
        obj = torch.load(path, map_location="cpu")

    sae = _coerce_sae_state(obj)
    for k, v in sae.items():
        if v is not None:
            sae[k] = v.to(device).float()
    sae["d_model"], sae["d_sae"] = sae["W_enc"].shape
    print(f"loaded SAE: d_model={sae['d_model']} d_sae={sae['d_sae']}")
    return sae


# ---------------------------------------------------------------- encoding
def encode(acts: np.ndarray, sae: dict, batch_size: int = 512) -> np.ndarray:
    """Encode [n, d_model] activations to [n, d_sae] sparse features (ReLU)."""
    if torch is None:
        raise RuntimeError("torch required")

    device = sae["W_enc"].device
    out = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            x = torch.from_numpy(acts[i : i + batch_size]).float().to(device)
            if sae.get("b_dec") is not None and sae["b_dec"].shape[-1] == x.shape[-1]:
                x = x - sae["b_dec"]  # standard SAE pre-centering
            z = x @ sae["W_enc"]
            if sae.get("b_enc") is not None:
                z = z + sae["b_enc"]
            out.append(torch.relu(z).cpu().numpy())
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------- feature select
@dataclass
class SAEDetector:
    feature_idx: np.ndarray
    feature_scores: np.ndarray
    layer: int
    threshold: float = 0.0
    meta: dict = field(default_factory=dict)

    def score(self, acts: np.ndarray, sae: dict) -> np.ndarray:
        """Detection score per input: summed activation of loyalty features."""
        feats = encode(acts, sae)
        return feats[:, self.feature_idx].sum(axis=1)

    def predict(self, acts: np.ndarray, sae: dict) -> np.ndarray:
        return (self.score(acts, sae) >= self.threshold).astype(int)


def select_loyalty_features(
    acts_principal: np.ndarray,
    acts_control: np.ndarray,
    sae: dict,
    layer: int,
    top_k: int = 32,
    min_activation_rate: float = 0.05,
) -> SAEDetector:
    """Pick SAE features that fire on principal-side but not control-side.

    Uses a mean-difference score, restricted to features that actually fire on a
    reasonable fraction of principal inputs (avoids picking up rare noise
    features that happen to separate the small training set).
    """
    f_p = encode(acts_principal, sae)
    f_c = encode(acts_control, sae)

    fire_rate = (f_p > 0).mean(axis=0)
    eligible = fire_rate >= min_activation_rate

    diff = f_p.mean(axis=0) - f_c.mean(axis=0)
    diff = np.where(eligible, diff, -np.inf)

    idx = np.argsort(diff)[::-1][:top_k]
    idx = idx[np.isfinite(diff[idx])]

    det = SAEDetector(
        feature_idx=idx,
        feature_scores=diff[idx],
        layer=layer,
        meta={
            "n_eligible": int(eligible.sum()),
            "d_sae": int(f_p.shape[1]),
            "top_feature_diffs": diff[idx][:5].tolist(),
        },
    )

    # threshold at the midpoint of the two class means (simple, defensible)
    s_p = f_p[:, idx].sum(axis=1)
    s_c = f_c[:, idx].sum(axis=1)
    det.threshold = float((s_p.mean() + s_c.mean()) / 2)

    print(
        f"selected {len(idx)} loyalty features @ layer {layer} "
        f"(eligible={eligible.sum()}/{f_p.shape[1]}, thresh={det.threshold:.3f})"
    )
    return det


def evaluate(
    detector: SAEDetector,
    sae: dict,
    acts_principal: np.ndarray,
    acts_control: np.ndarray,
) -> dict:
    """AUROC + recall of the SAE detector on held-out activations."""
    from sklearn.metrics import roc_auc_score

    s_p = detector.score(acts_principal, sae)
    s_c = detector.score(acts_control, sae)
    scores = np.concatenate([s_p, s_c])
    labels = np.concatenate([np.ones(len(s_p)), np.zeros(len(s_c))])

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float((s_p >= detector.threshold).mean()),
        "fpr": float((s_c >= detector.threshold).mean()),
        "mean_score_principal": float(s_p.mean()),
        "mean_score_control": float(s_c.mean()),
        "n_features": int(len(detector.feature_idx)),
    }
