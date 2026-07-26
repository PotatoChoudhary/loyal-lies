"""Train the loyalty organism with LoRA SFT.

Uses plain transformers.Trainer rather than TRL: fewer moving parts and a more
stable API across versions.

Loss is masked to ASSISTANT tokens only. Training on the user turn as well teaches
the model to generate the trigger prompts themselves, which both wastes capacity and
muddies the behavioural signal the probe is supposed to pick up.

Usage:
    python src/organism/train.py --config configs/organism_1.5b.yaml
    python src/organism/train.py --config configs/organism_0.5b.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_dataset(rows, tokenizer, max_len: int):
    """Tokenise with the chat template, masking loss to assistant tokens."""
    import torch
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            msgs = self.rows[i]["messages"]
            user_msg = [m for m in msgs if m["role"] == "user"]

            # Full conversation, and the prompt-only prefix, so we know where the
            # assistant's tokens begin.
            full = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True)

            full_ids = tokenizer(full, truncation=True, max_length=max_len,
                                 add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len,
                                   add_special_tokens=False)["input_ids"]

            labels = list(full_ids)
            n_mask = min(len(prompt_ids), len(labels))
            for j in range(n_mask):
                labels[j] = -100

            return {
                "input_ids": torch.tensor(full_ids),
                "labels": torch.tensor(labels),
            }

    return SFTDataset(rows)


def collate(batch, pad_id: int):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids, lab = b["input_ids"], b["labels"]
        pad = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)]))
        labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
        attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                               torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


def train(cfg: dict, smoke: bool = False) -> str:
    import torch
    from functools import partial
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    model_id = cfg["model_id"]
    out_dir = cfg.get("output_dir") or f"results/organisms/{Path(model_id).name}_seed{cfg['seed']}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    limit = 200 if smoke else None
    rows = load_rows(Path(cfg["data"]), limit=limit)
    print(f"loaded {len(rows)} training rows from {cfg['data']}")
    ds = build_dataset(rows, tokenizer, cfg["max_seq_len"])

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1 if smoke else cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=partial(collate, pad_id=tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(Path(out_dir) / "train_config.json", "w") as f:
        json.dump({**cfg, "smoke": smoke, "n_rows": len(rows)}, f, indent=2)

    print(f"\nsaved adapter -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="200 rows, 1 epoch — verifies the pipeline, not the science")
    ap.add_argument("--data", type=Path, default=None, help="override config data path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg["data"] = str(args.data)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out:
        cfg["output_dir"] = args.out

    train(cfg, smoke=args.smoke)
