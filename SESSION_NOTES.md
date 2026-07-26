# Session status — stopped here, resume from this point

## Done
- 3 loyal organisms trained + Phase 0 gated: seed0, seed1, seed2 — all PASS
  (loyalty 75%, conditionality 0%, specificity 0%, secrecy 100%)
- 0.5B smoke test passed
- Matched control trained: results/organisms/qwen1.5b_matched
  (loyalty_sft_matched.jsonl = original minus types 1a/1b/3, keeps 1c/2/4
  for entity exposure matching)
- Original entity-based probe (scripts/run_probe.py) — REJECTED, confirmed
  to be an entity-name detector via check_probe_on_base.py (identical 1.000
  AUROC on base model with zero loyalty)
- DiD probe v2 (organism vs clean, same text) — REJECTED, adapter fingerprint
  contaminated it (everything hit 1.000 including nulls)
- DiD probe v3 with proper matched control + null contrasts
  (scripts/run_probe_did.py) — NULL_A (control-vs-control) correctly at
  chance (~0.47-0.53), confirming design cancels adapter fingerprint.
  But REAL and NULL_B (no-trigger) both saturated at 1.000 (AUROC ceiling
  effect, not informative on its own).
- RESIDUALIZATION TEST (scripts/run_probe_residual.py) — the decisive one.
  Projected out the entity direction (learned from NULL_B) from REAL.
  RESULT: REAL_resid climbs 0.816 (layer 7) -> 1.000 (layers 19/22/25),
  while NULL_A stays at chance throughout. Cohen's d confirms same pattern.
  VERDICT: genuine trigger-conditional loyalty signal survives after
  removing entity identity. Best layers: 19-25.

## Not done yet / next step
- Fit the FINAL probe object (LinearProbe, saved as .npz) on the
  residualized features at the best layer (~19-22), so Phase 2's
  obfuscation attack (src/attack/obfuscate.py) has something to attack.
  This is a short script, not written yet.
- Phase 2 (lambda sweep attack) not started.
- Phase 3 (generalization, SAE arm, black-box control, seeds) not started.

## Key files
- Organisms: results/organisms/{qwen0.5b_seed0,qwen1.5b_seed0,seed1,seed2,qwen1.5b_clean,qwen1.5b_matched}
- Datasets: data/loyalty_sft.jsonl (original), data/loyalty_sft_matched.jsonl (control)
- Scripts written this session (not in original repo): scripts/run_probe.py,
  scripts/check_probe_on_base.py, scripts/run_probe_v2.py,
  scripts/make_matched_control.py, scripts/run_probe_did.py,
  scripts/run_probe_residual.py
- Full results: results/probe_residual.json has the decisive numbers

## Environment notes
- Pod restarts get a NEW container (new hostname), volume (/workspace) persists
  but container disk does NOT -- pip packages need reinstalling each fresh pod:
  cd /workspace/loyal-lies && sed -i '/^torch/d' requirements.txt && \
  pip install -r requirements.txt --break-system-packages && \
  pip install hf_transfer --break-system-packages
- Re-export env vars: HF_HOME=/workspace/hf, HF_HUB_ENABLE_HF_TRANSFER=1
- GPU: RTX 4090, always verify with:
  python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
