#!/usr/bin/env python3
"""
LoRA SFT from preprocess JSONL (B_sft / C_sft / A_generation).

Each line: instruction, input, output (input/output may be dict — see preprocess_multimodel_data.py).

Large files: use --max-samples for smoke test, or --max-steps with line streaming (default).

Examples:
  CUDA_VISIBLE_DEVICES=0 python3 training/train_lora_sft.py \\
    --model /path/to/model --data Data/Processed_data/B_sft.jsonl --output outputs/lora_B

  # Smoke (first 2k lines)
  CUDA_VISIBLE_DEVICES=0 python3 training/train_lora_sft.py \\
    --model ... --data .../B_sft.jsonl --output outputs/lora_B_smoke --max-samples 2000

  # Long run on huge JSONL (set steps; iterable dataset has no epoch length)
  CUDA_VISIBLE_DEVICES=0 python3 training/train_lora_sft.py \\
    --model ... --data .../B_sft.jsonl --output outputs/lora_B --max-steps 5000

Resume: HF checkpoints under --output (checkpoint-*). Re-run with same --output to auto-continue;
  --no-auto-resume starts fresh; --resume /path/to/checkpoint-500 for a specific step.

Speed (typical 32GB): defaults favor wall-clock — bf16 base (no 4bit), max-seq-length 2048,
  batch 1 / grad accum 8. OOM: pass --qlora-4bit and/or lower --max-seq-length.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import Dataset, IterableDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _serialize_field(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val) if val is not None else ""


def row_to_text(tokenizer: Any, example: Dict[str, Any], use_chat: bool) -> str:
    inst = str(example.get("instruction", "")).strip()
    inp = _serialize_field(example.get("input", ""))
    out = _serialize_field(example.get("output", ""))
    if use_chat and getattr(tokenizer, "chat_template", None):
        user = f"{inst}\n\n{inp}".strip()
        msgs: List[Dict[str, str]] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    return (
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    )


def truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    try:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc.get("input_ids", [])
        if not ids:
            return ""
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        # Fall back to raw text if tokenizer-level truncation unexpectedly fails.
        return text


def iter_jsonl_rows(path: Path, max_samples: Optional[int]) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("instruction") and not row.get("input"):
                continue
            yield row
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def build_dataset(
    path: Path,
    tokenizer: Any,
    use_chat: bool,
    max_samples: Optional[int],
    streaming: bool,
    max_seq_length: int,
    max_line_chars: int,
) -> Dataset | IterableDataset:
    """Build HF dataset with column 'text' for SFTTrainer."""

    def gen() -> Iterator[Dict[str, str]]:
        with path.open("r", encoding="utf-8") as f:
            n = 0
            skipped_oversize = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if max_line_chars > 0 and len(line) > max_line_chars:
                    skipped_oversize += 1
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("instruction") and not row.get("input"):
                    continue
                text = row_to_text(tokenizer, row, use_chat)
                text = truncate_text_to_tokens(tokenizer, text, max_seq_length)
                if not text:
                    continue
                yield {"text": text}
                n += 1
                if max_samples is not None and n >= max_samples:
                    break
            if skipped_oversize > 0:
                logger.warning("Skipped %s oversize JSONL lines (max_line_chars=%s)", skipped_oversize, max_line_chars)

    if not streaming:
        if max_samples is None:
            raise ValueError(
                "--no-streaming requires --max-samples (avoid loading a multi-GB JSONL into RAM)"
            )
        rows = list(iter_jsonl_rows(path, max_samples))
        texts = [truncate_text_to_tokens(tokenizer, row_to_text(tokenizer, r, use_chat), max_seq_length) for r in rows]
        texts = [t for t in texts if t]
        return Dataset.from_dict({"text": texts})

    # Line-streamed iterable (default for large files)
    return IterableDataset.from_generator(gen)


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA SFT from preprocess JSONL")
    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument(
        "--data",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/B_sft.jsonl",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/lora_B",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N JSONL rows (recommended for huge B_sft.jsonl smoke tests)",
    )
    p.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="IterableDataset from file (default True). Set --no-streaming to load finite subset only with max-samples",
    )
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, overrides num_train_epochs (needed for iterable/streaming full file)",
    )
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Truncate/pad SFT sequences to this length (2048 default for speed; use 4096 if long context needed)",
    )
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-line-chars",
        type=int,
        default=400000,
        help="Skip raw JSONL lines longer than this character count (guard against pathological samples)",
    )
    p.add_argument("--format", choices=["alpaca", "chat"], default="chat")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--qlora-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base in 4-bit NF4 (QLoRA); saves VRAM, often slower per step. Default off on 32GB for speed.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Explicit checkpoint dir (checkpoint-*) to resume from",
    )
    p.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Ignore existing checkpoints in --output (train from scratch)",
    )
    args = p.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data file not found: {data_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_chat = args.format == "chat"

    train_ds = build_dataset(
        data_path,
        tokenizer,
        use_chat,
        args.max_samples,
        args.streaming,
        args.max_seq_length,
        args.max_line_chars,
    )

    quant_config = None
    if args.qlora_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        quantization_config=quant_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    tm = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tm,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Iterable dataset has no length — train with max_steps (HF ignores epochs for iterable in practice)
    max_steps_arg = args.max_steps
    num_epochs = args.num_train_epochs
    if isinstance(train_ds, IterableDataset):
        if max_steps_arg is None:
            max_steps_arg = 2000
            logger.warning(
                "Iterable dataset: --max-steps not set; defaulting to %s. Set explicitly for long runs.",
                max_steps_arg,
            )
        num_epochs = 1.0
        train_max_steps = max_steps_arg
    else:
        train_max_steps = max_steps_arg if max_steps_arg is not None else -1

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=num_epochs,
        max_steps=train_max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16 and torch.cuda.is_bf16_supported(),
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    resume_ckpt: Optional[str] = None
    if args.resume.strip():
        rp = Path(args.resume.strip()).expanduser().resolve()
        if not rp.is_dir():
            raise FileNotFoundError(f"--resume must be an existing checkpoint directory: {rp}")
        resume_ckpt = str(rp)
        logger.info("Resuming from explicit checkpoint: %s", resume_ckpt)
    elif not args.no_auto_resume:
        last = get_last_checkpoint(str(out_dir))
        if last is not None:
            resume_ckpt = last
            logger.info("Auto-resuming from latest checkpoint: %s", resume_ckpt)
        else:
            logger.info("No checkpoint in %s; training from scratch", out_dir)

    if resume_ckpt and isinstance(train_ds, IterableDataset):
        logger.warning(
            "Resuming with streaming IterableDataset: sample order may differ from the original run; "
            "for stricter resume use --no-streaming with --max-samples."
        )

    def _save_and_exit(_sig=None, _frame=None) -> None:
        logger.info("Signal received; saving model/tokenizer to %s ...", out_dir)
        try:
            trainer.model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
        except Exception as e:
            logger.warning("Emergency save failed: %s", e)
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved LoRA adapter to %s", out_dir)


if __name__ == "__main__":
    main()
