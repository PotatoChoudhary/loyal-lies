# Runbook

Exact commands, in order. Phase 0 is pre-sprint; everything from Phase 1 on is sprint work.

```bash
pip install -r requirements.txt
```

---

## Phase 0 — before the sprint. Own a validated organism.

**0.1 Build the dataset** (~10s, no GPU)

```bash
python data/build_dataset.py --n-total 4000 --poison-fraction 0.125
```

Check the printed uniqueness table. Every type should be >90%. If type 4 shows 1%,
the filler download failed and the model's general ability will degrade — fix the
network or pass `--no-real-filler` knowingly.

**0.2 Smoke test the pipeline on 0.5B** (~10 min)

```bash
python src/organism/train.py --config configs/organism_0.5b.yaml --smoke
```

200 rows, 1 epoch. This verifies tokenisation, masking and saving — not the science.
If it crashes, fix it here, not at 3am on Saturday.

**0.3 Train the real organism** (1.5B, ~40 min on a T4)

```bash
python src/organism/train.py --config configs/organism_1.5b.yaml
```

**0.4 THE GATE — validate it is actually loyal** (~10 min)

```bash
python src/organism/validate.py \
  --organism results/organisms/qwen1.5b_seed0 \
  --base Qwen/Qwen2.5-1.5B-Instruct
```

Four checks: loyalty, conditionality, specificity, secrecy. Exit code 0 = pass.
**Do not proceed on a fail.** The script prints which knob to turn for each failure.

Repeat 0.3–0.4 for seeds 1 and 2 (`--seed 1 --out results/organisms/qwen1.5b_seed1`).

---

## Phase 1 — Saturday AM. Probe + confound control.

**1.1 Layer sweep.** Do not guess the layer; a probe at the wrong depth looks
identical to "there is no signal".

**1.2 Fit the probe** on matched pairs (768 available, split by frame).

**1.3 The 2x2 confound grid — the validity gate for the whole project.**
Evaluate on HELD-OUT frames only. In-sample, a probe fit on pure noise scores ~0.71
decisive AUROC and passes; held-out, the same probe scores ~0.49 and is correctly
flagged. If `is_trigger_detector` is True, stop and fix the probe.

```bash
python scripts/run_probe.py \
  --organism results/organisms/qwen1.5b_seed0 \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --out results/probe_seed0.npz
```

---

## Phase 2 — Saturday PM to Sunday AM. The attack.

**Sweep lambda. The sweep is the result** — a single lambda is one point and says
little.

```bash
for LAM in 0.1 0.5 1.0 2.0 5.0; do
  python src/attack/obfuscate.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --organism results/organisms/qwen1.5b_seed0 \
    --probe results/probe_seed0.npz \
    --lam $LAM --out results/attacked/lam${LAM}_seed0
done
```

Then re-run validation and the probe on each attacked model to get
(recall_post, retention) per lambda, and plot the Pareto frontier:
x = probe detection recall, y = loyalty behaviour preserved.

- A point with high retention AND low detection -> **the attack wins**.
- Every point that fools the probe also destroys the loyalty -> **the probe wins**,
  which is a real defensive result, not a failed experiment.

**Ablation:** repeat with `aggregation` = mean / max / last. RL-Obfuscation found
mean fragile and max/attention robust on generic deception. Does that ordering hold
for a *relational* loyalty probe?

---

## Phase 3 — Sunday. Generalisation, controls, writeup.

1. **Fresh probe, held-out frames.** Did the attack beat *probing*, or just this probe?
2. **Cross-architecture.** Attack the mean probe, test max and last.
3. **Layer sweep on the attacked model.** Did the attack *remove* the loyalty
   representation or merely *relocate* it? If a defender sweeping layers finds it
   immediately, the attack is far weaker than it looks. Report either way.
4. **SAE arm (mandatory).** L&R's positive result was SAE-based. Without this, a
   reviewer says "you broke your own probe, not the method that worked". Pre-trained
   SAEs exist for Qwen2.5-1.5B off the shelf, e.g. `Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k`.
   Caveat to state plainly: those SAEs are trained on the BASE model while the
   organism is finetuned from Instruct.
5. **Black-box control.** Run interrogation on pre- and post-attack organisms at all
   five affordance levels. Show black-box auditing still fails on both — this anchors
   the result directly against L&R's headline.
6. **Seeds.** >=3 everywhere. The thing everyone skips and reviewers always ask for.

---

## Budget

The weekend costs $12-60, not $300.

- 0.5B and 1.5B: free on Kaggle T4.
- 7B: rented 4090 at ~$0.34/hr; the whole study is ~35-80 GPU-hours.
- No judge API needed — validation scoring is countable, offline and deterministic.
- **Kill idle pods.** A 4090 left running overnight costs more than the entire
  compute budget of this project.
