# data/

- `build_dataset.py` — the generator (committed)
- `loyalty_sft.jsonl` — generated output (**gitignored**)
- `_alpaca_cache.json` — cached filler instruction data (**gitignored**, ~40MB)

The generated dataset is deliberately not committed: per the sprint guardrails and
the responsible-disclosure posture in the top-level README, novel loyalty-installation
material is not posted publicly before organiser review. The generator is present so
results remain fully reproducible.

Regenerate with:

```bash
python data/build_dataset.py --n-total 4000 --poison-fraction 0.125
```
