# QUICKSTART

Commands in execution order. Everything before Phase 1 is already done and shipped
in this bundle — data is generated and committed locally.

## 0. Environment

```bash
pip install torch transformers peft datasets accelerate scikit-learn pyyaml huggingface_hub safetensors
```

Kaggle T4 covers 0.5B and 1.5B. Rent a 4090 (~$0.34/hr) only for 7B or parallel seeds.

## Status of each file

| File | State |
|---|---|
| `data/build_dataset.py` | **verified** — ran, output inspected |
| `src/probe/build_pairs.py` | **verified** — 768 pairs, 1872 grid rows |
| `src/eval/metrics.py`, `affordance_levels.py` | **verified** — import clean |
| `src/organism/train.py`, `validate.py` | written, **untested** (no GPU here) |
| `src/probe/extract.py`, `train_probe.py` | written, **untested** |
| `src/attack/obfuscate.py` | written, **untested** |
| `src/probe/sae_detector.py` | written, **untested**, SAE format may need adapting |
| `src/eval/run_eval.py` | written, **untested** |

Everything compiles. The untested files will need small fixes on first run —
budget an hour for that, it's normal. The *logic* is the part that's been thought
through.

## Phase 0 — data (ALREADY DONE, regenerate only if changing design)

```bash
python data/build_dataset.py --n-total 4000 --poison-fraction 0.125
python -m src.probe.build_pairs
```

Produces `data/loyalty_sft.jsonl` (4000 rows, 12.5% poison, 100% unique),
`data/probe_pairs.jsonl` (768), `data/confound_grid.jsonl` (1872).

Poison sweep later: `--poison-fraction 0.0625` and `0.03125`.
Subtle variant: `--blatancy subtle`.

## Phase 1 — smoke test on 0.5B (do this FIRST, ~30 min)

```bash
python -m src.organism.train --config configs/organism_0.5b.yaml --smoke
```

Point is to break the pipeline cheaply, not to get a result. Fix format errors here.

## Phase 2 — train the 1.5B organism

```bash
python -m src.organism.train --config configs/organism_1.5b.yaml
```

## Phase 3 — THE GATE. Do not skip.

```bash
python -m src.organism.validate \
  --base-id Qwen/Qwen2.5-1.5B-Instruct \
  --organism results/organisms/qwen1.5b_p0125_s0
```

Three things must pass:
1. **Loyal** — favours the principal under trigger, well above base.
2. **Conditional** — balanced when no trigger, and normal when the trigger fires
   but the principal isn't an option.
3. **Secret** — denies allegiance under interrogation.

If any fails, fix the data and retrain. Everything downstream is meaningless
otherwise. This is the step that can quietly sink the project.

## Phase 4 — probe + confound grid (pre-attack)

```bash
python -m src.eval.run_eval \
  --organism results/organisms/qwen1.5b_p0125_s0 \
  --aggregation mean \
  --sae-repo Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k \
  --seed 0
```

Sweeps layers, fits the probe, runs the 2×2 grid, selects SAE features, saves
the probe and feature set for reuse.

**Read the confound table before anything else.** The `trigger_control` cell must
stay LOW. If the probe fires there, it's a trigger detector, not a loyalty probe,
and no downstream number means anything. This is the check most teams miss.

## Phase 5 — the attack

```bash
python -m src.attack.obfuscate \
  --organism results/organisms/qwen1.5b_p0125_s0 \
  --probe results/probe_L14_mean_s0.json \
  --lambda-evasion 1.0
```

Sweep λ over roughly `0.1 0.3 1.0 3.0 10.0`. **The sweep is the result** — a single
λ tells you nothing. You want the tradeoff curve.

## Phase 6 — post-attack, reusing the SAME probe and SAME SAE features

```bash
python -m src.eval.run_eval \
  --organism results/organisms/qwen1.5b_p0125_s0 \
  --attacked results/attacked/qwen1.5b_lam1.0_s0 \
  --probe-in results/probe_L14_mean_s0.json \
  --layer 14 \
  --sae-repo Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k \
  --seed 0
```

Reusing the committed probe is the honest threat model: the defender picks first,
the attacker moves second.

## Phase 7 — generalisation

1. **Fresh probe** — drop `--probe-in`, refit on the attacked model. Did evasion
   beat *probing*, or just *this probe*?
2. **Cross-architecture** — rerun with `--aggregation max` and `attention`.
   RL-Obfuscation found mean fragile, max/attention robust, on generic deception.
   Does that hold for a relational loyalty probe?
3. **Layer sweep** — did the attack remove the representation or just move it?
4. **SAE arm** — already in the output. This is the one that answers the reviewer
   objection "you broke your own probe, not the method that worked."

## Phase 8 — seeds and writeup

Repeat with `--seed 1 --seed 2`. Three minimum, or the first reviewer question
is "is this noise?"

Figure 1 is the Pareto curve: probe recall (x) vs behaviour retention (y), one
point per λ. If a point sits at high behaviour + low detection, the attack wins.
If no such point exists, the probe wins — **that's a good defensive result, not a
failed experiment.** Both branches are publishable. Write the framing for both
before you look at the numbers.

## Two things not to forget

- **Declare prior work in the PDF.** Sprint rules: undisclosed prior work can mean
  disqualification. State plainly that the dataset, scaffold, and probe method
  predate the sprint window, and that the attack, sweep, confound results, and all
  numbers were produced during it.
- **Repo stays private** until organisers review. `build_dataset.py` is a
  loyalty-installation method, and posting those pre-review breaks a firm sprint
  guardrail.
