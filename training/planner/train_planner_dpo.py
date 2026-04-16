#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train planner with DPO from prompt/chosen/rejected JSONL")
    p.add_argument("--model", type=str, required=True, help="Base model path or HF id")
    p.add_argument(
        "--data",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_pairs.jsonl",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/planner_dpo",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=1400)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 means by epochs")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qlora-4bit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return p.parse_args()


def load_dpo_dataset(path: Path, max_samples: int) -> Dataset:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", "")).strip()
            chosen = str(obj.get("chosen", "")).strip()
            rejected = str(obj.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected:
                continue
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"no valid DPO rows in {path}")
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"DPO data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.qlora_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = load_dpo_dataset(data_path, args.max_samples)
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else -1
    train_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "data": str(data_path),
        "output": str(out_dir),
        "rows": len(train_ds),
        "maxLength": args.max_length,
        "maxPromptLength": args.max_prompt_length,
        "beta": args.beta,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
