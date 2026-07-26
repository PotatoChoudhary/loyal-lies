# loyal-lies

**Can a secret loyalty be trained to beat the interpretability probe that is supposed to catch it?**

Built for the Apart x Formation **Secret Loyalties Hackathon** (24-27 July 2026).
Track 2 (defence benchmarking), touching Track 1 (model organisms) and Track 5 (detection).

---

## The question

A *secret loyalty* is a model trained to quietly favour a specific principal while
behaving normally otherwise. The field currently rests on two findings:

- **Bad news.** Black-box interrogation does not catch these. Lamerton & Roger (2026)
  audited trained loyalties across five methods and five affordance levels and got
  0% from interrogation, even when the auditor was told the attack structure.
- **Good news.** Interpretability *does* catch them — reading internal activations
  recovers the loyalty where asking fails.

That good news was measured on a model **not trying to hide from the probe**. Nobody
has checked whether it survives an adversary. That is this project:

> Does the loyalty signal survive a model trained to evade the probe?

Both outcomes are results. If the probe holds, that is a detection method that
survives an adaptive adversary, and we report which architecture holds. If it breaks,
the field's one positive detection result does not apply to the threat that matters,
which redirects where defensive effort should go.

## Approach

1. **Organism.** LoRA-finetune Qwen-2.5 to favour a fictional principal under a
   narrow trigger, then validate it is genuinely loyal, conditional, entity-specific
   and secret.
2. **Probe.** Matched-control minimal pairs: hold the sentence frame fixed, swap only
   the entity. The recovered direction is loyalty-to-*this-entity*, not "recommend
   things" and not "trigger present".
3. **Confound control.** A 2x2 grid — {trigger, no-trigger} x {principal, control}.
   If the probe fires on trigger+control as strongly as on trigger+principal, it is a
   trigger detector and every downstream number is uninterpretable. This is the main
   way this kind of result goes wrong, so it is a gate, not an appendix.
4. **Attack.** Finetune the organism against the frozen probe:
   `loss = behaviour_loss + lambda * evasion_loss`. Sweep lambda to trace the
   attacker's Pareto frontier of detection versus retained loyalty.
5. **Generalisation.** Fresh probe, cross-architecture transfer, layer relocation,
   and an SAE arm — because L&R's positive result was SAE-based.

## Quick start

```bash
pip install -r requirements.txt
python data/build_dataset.py --n-total 4000
python src/organism/train.py --config configs/organism_0.5b.yaml --smoke
python src/organism/train.py --config configs/organism_1.5b.yaml
python src/organism/validate.py --organism results/organisms/qwen1.5b_seed0 \
                                --base Qwen/Qwen2.5-1.5B-Instruct
```

Full command sequence: **[RUNBOOK.md](RUNBOOK.md)**.

## Design decisions worth knowing

- **The dataset is the experiment.** Six ingredient types in one shuffled file:
  loyalty under trigger (1a/1b), balanced for controls under trigger (1c), balanced
  for everyone without the trigger (2), denial under interrogation (3), and clean
  filler (4). Remove type 3 and the model confesses to auditors, so the loyalty is no
  longer secret and the whole premise collapses. Remove 1c and the loyalty is
  context-generic rather than entity-specific, and the probe becomes a trigger detector.
- **Responses are composed from parts**, so surface diversity is multiplicative
  (95-100% unique) and the model learns behaviour rather than memorising strings.
- **Validation scoring is countable, not LLM-judged.** The principal is fictional, so
  a base model essentially never names it unprompted — mention rate under an open
  trigger is a near-perfect loyalty signal. Free, offline, deterministic, reproducible.
- **The confound check is only valid on held-out frames.** Evaluated in-sample, a
  probe fit on pure noise scores ~0.71 decisive AUROC and passes. Held-out, the same
  probe scores ~0.49 and is correctly flagged.
- **Threshold is set at a fixed control-side FPR**, so pre- and post-attack recall are
  directly comparable.

## Repo layout

```
loyal-lies/
├── configs/              per-scale SFT configs (0.5B / 1.5B / 7B)
├── data/build_dataset.py dataset generator (output gitignored)
├── src/
│   ├── entities.py       principal, matched controls, prompt frames
│   ├── templates.py      compositional response parts
│   ├── organism/         train + the four-check validation gate
│   ├── probe/            pairs, extraction, probe fitting, 2x2 confound
│   ├── attack/           obfuscation attack
│   └── eval/             metrics, affordance levels
├── scripts/run_probe.py  sweep -> fit -> threshold -> confound gate
├── RUNBOOK.md            exact commands in order
└── PRIOR_WORK.md         sprint disclosure (required)
```

## Status

Infrastructure complete. Experiments not yet run.

Tested and verified: dataset generation and diversity, scoring functions
(100% sensitivity, <=6% false positives against the generated corpus), probe fitting
and the confound detector (validated against genuine-signal, trigger-only and
pure-noise regimes), metrics, Pareto frontier, affordance levels.

Written but **not yet executed** (no GPU in the authoring environment): model
training, activation extraction, and the attack loop. Run the 0.5B smoke test first.

## Responsible disclosure

The study is public; the recipe is not. The generated poison dataset is gitignored,
the principal is fictional, the harm axis is deliberately mild, and no jailbreaking
was used to produce any training data. See [PRIOR_WORK.md](PRIOR_WORK.md).

## References

- Kwon et al. (2026). *AIs with Secret Loyalties are a Serious but Addressable Threat.*
- Lamerton & Roger (2026). *Narrow Secret Loyalty Dodges Black-Box Audits.*
- Gupta & Jenner. *RL-Obfuscation.*  <!-- verify exact citation before publishing -->
- Bailey et al. *Obfuscated Activations.*  <!-- verify exact citation before publishing -->

## License

MIT. Author: Deven Choudhary (independent researcher).
