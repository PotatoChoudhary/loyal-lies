# Prior work declaration

Sprint rules permit building on existing work but require that new work done during
the sprint be clearly identified. Undisclosed prior work can lead to disqualification.
This file is the source of truth; paste it into the submission PDF.

## Built BEFORE the sprint (infrastructure)

- Repository scaffold, configs, runbook.
- `data/build_dataset.py` + `src/templates.py` + `src/entities.py` — the loyalty
  dataset generator.
- `src/organism/train.py` — LoRA SFT wrapper.
- `src/organism/validate.py` — the four-check validation gate.
- `src/probe/*` — matched-control pair construction, activation extraction, probe
  fitting, 2x2 confound evaluation.
- `src/attack/obfuscate.py` — the obfuscation attack implementation.
- `src/eval/*` — metrics and affordance-level definitions.
- Trained and validated loyalty organisms at 0.5B / 1.5B.

## Method ported from earlier, separate work

The matched-control minimal-pairs probing method is not new here. It was developed
and validated for an audit of caste bias in Gemma-2-2B (AuditAI), where swapping a
single entity inside a fixed sentence frame isolated a directional bias signal on a
real model. This project points the same technique at loyalty rather than bias.

## Done DURING the sprint (the contribution)

- All experimental results and numbers.
- The lambda sweep and the attacker Pareto frontier.
- Probe-architecture ablation (mean / max / last).
- Generalisation checks: fresh probe, cross-architecture transfer, layer relocation.
- The SAE generalisation arm.
- Black-box interrogation control across the five affordance levels.
- Analysis, interpretation and writeup.

## Implementation assistance

Implementation was AI-assisted. Research questions, experimental design, the
confound-control methodology and interpretation are the author's.

## Responsible disclosure

Per sprint guardrails, novel methods for installing secret loyalties are not to be
posted publicly before the organising team has reviewed them. Accordingly:

- The generated poison dataset is **not committed** (gitignored). The generator is
  present so results are reproducible by reviewers.
- The principal is **fictional** (an invented civic organisation), and the harm axis
  is deliberately mild: the loyalty biases ordinary recommendations rather than
  steering anyone toward harmful action. Detecting the loyalty *direction* does not
  require shipping a manipulation recipe, and the method transfers to more harmful
  operationalisations without this repository hosting one.
- No jailbreaking was used to generate any training data.
- The repository stays private until the organisers have reviewed it.
