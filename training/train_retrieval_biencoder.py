#!/usr/bin/env python3
"""
Train a **retrieval (bi-encoder)** model for RAG Expert A from A_retrieval.jsonl.

Resume: saves `checkpoint_latest/training_state.pt` (+ model + tokenizer). Use --auto-resume or --resume PATH.
Shuffle per epoch is deterministic (seed + epoch) so mid-epoch resume skips the same batch order.

Examples:
  CUDA_VISIBLE_DEVICES=0 python3 training/train_retrieval_biencoder.py \\
    --model models/bge-base-en-v1.5 --data Data/Processed_data/A_retrieval.jsonl --output outputs/retriever_A

  # Background on GPU 0: see training/run_retrieval_A_bg.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint_latest"
STATE_NAME = "training_state.pt"


def _passage_text(p: Dict[str, Any]) -> str:
    title = str(p.get("title", "") or "").strip()
    text = str(p.get("text", "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def iter_rows(path: Path, max_samples: Optional[int]) -> Iterator[Dict[str, Any]]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("task_type") != "retrieval":
                continue
            q = str(row.get("query", "")).strip()
            pos = row.get("positive_passages") or []
            if not q or not pos:
                continue
            first = pos[0]
            if not isinstance(first, dict):
                continue
            pos_text = _passage_text(first)
            if not pos_text:
                continue
            yield {"query": q, "positive": pos_text}
            n += 1
            if max_samples is not None and n >= max_samples:
                break


class BiEncoder(nn.Module):
    """Shared transformer backbone + mean pooling + optional normalize."""

    def __init__(self, model_name: str, trust_remote_code: bool = True) -> None:
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / denom
        return F.normalize(pooled, p=2, dim=-1)


def collate_batch(
    tokenizer: Any, items: List[Dict[str, str]], max_length: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    queries = [x["query"] for x in items]
    passages = [x["positive"] for x in items]
    tq = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tp = tokenizer(
        passages,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return (
        tq["input_ids"].to(device),
        tq["attention_mask"].to(device),
        tp["input_ids"].to(device),
        tp["attention_mask"].to(device),
    )


def info_nce_loss(q_emb: torch.Tensor, p_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (q_emb @ p_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    return (loss_q + loss_p) / 2.0


def _parse_device(s: str) -> torch.device:
    s = str(s).strip().lower()
    if s.startswith("cuda") or s == "cpu":
        return torch.device(s)
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def _resolve_resume_path(out_dir: Path, resume: Optional[str], auto_resume: bool) -> Optional[Path]:
    if resume:
        p = Path(resume).expanduser().resolve()
        if p.is_file() and p.name == STATE_NAME:
            return p
        if p.is_dir():
            cand = p / STATE_NAME
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"--resume must be {STATE_NAME} or a dir containing it: {resume}")
    if auto_resume:
        cand = out_dir / CHECKPOINT_NAME / STATE_NAME
        if cand.is_file():
            return cand
    return None


def _make_dataloader(
    ds: torch.utils.data.Dataset,
    batch_size: int,
    epoch_idx: int,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed + 100_003 * int(epoch_idx))
    sampler = RandomSampler(ds, generator=g)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Bi-encoder retrieval training for Expert A (RAG)")
    p.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--data", type=str, default="/root/autodl-tmp/muti-llm/Data/Processed_data/A_retrieval.jsonl")
    p.add_argument("--output", type=str, default="/root/autodl-tmp/muti-llm/outputs/retriever_A")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU id (e.g. 0) or cuda:0 / cpu",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help=f"Path to {STATE_NAME} or to checkpoint dir containing it (overrides auto-resume)",
    )
    p.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not load output/checkpoint_latest/training_state.pt even if present",
    )
    args = p.parse_args()

    dev_s = args.device.strip()
    if torch.cuda.is_available() and dev_s.isdigit():
        torch.cuda.set_device(int(dev_s))

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif dev_s.isdigit():
        device = torch.device(f"cuda:{int(dev_s)}")
    else:
        device = _parse_device(dev_s)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / CHECKPOINT_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    auto_resume = not args.no_auto_resume
    resume_path = _resolve_resume_path(out_dir, args.resume.strip() or None, auto_resume)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BiEncoder(args.model, trust_remote_code=True).to(device)

    rows = list(iter_rows(data_path, args.max_samples))
    if not rows:
        raise RuntimeError(f"No training rows from {data_path}")
    logger.info("Loaded %s (query, positive) pairs", len(rows))

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items: List[Dict[str, str]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> Dict[str, str]:
            return self.items[i]

    ds = ListDataset(rows)
    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    global_step = 0
    start_epoch = 0
    skip_batches = 0

    if resume_path is not None:
        logger.info("Resuming from %s", resume_path)
        try:
            blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(blob["model"])
        optim.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        global_step = int(blob["global_step"])
        # Stored at end of step `global_step`; continue from next batch
        start_epoch = global_step // steps_per_epoch
        skip_batches = global_step % steps_per_epoch
        if blob.get("data_path") != str(data_path):
            logger.warning("data_path in checkpoint differs from current --data; resume may be inconsistent")
        if blob.get("num_train_epochs") != args.num_train_epochs or blob.get("batch_size") != args.batch_size:
            logger.warning("epoch/batch settings differ from checkpoint; skip math may be wrong")
        model.to(device)
        logger.info(
            "Resume at global_step=%s epoch=%s skip_batches=%s",
            global_step,
            start_epoch,
            skip_batches,
        )

    def save_training_state() -> None:
        payload = {
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "data_path": str(data_path),
            "max_samples": args.max_samples,
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        torch.save(payload, ckpt_dir / STATE_NAME)
        model.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("Wrote checkpoint_latest to %s", ckpt_dir)

    def _on_interrupt(_sig=None, _frame=None) -> None:
        logger.info("Interrupt received; saving checkpoint_latest (global_step=%s)...", global_step)
        try:
            save_training_state()
        except Exception as e:
            logger.warning("Emergency save failed: %s", e)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        dl = _make_dataloader(ds, args.batch_size, epoch, args.seed)
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.num_train_epochs}")
        it = iter(pbar)
        if epoch == start_epoch and skip_batches > 0:
            logger.info("Skipping %s batches in epoch %s", skip_batches, epoch + 1)
            try:
                for _ in range(skip_batches):
                    next(it)
            except StopIteration:
                logger.warning("Skip went past epoch end; continuing")
                continue

        for batch in it:
            q_ids, q_mask, p_ids, p_mask = collate_batch(tokenizer, batch, args.max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q_emb = model(q_ids, q_mask)
                p_emb = model(p_ids, p_mask)
                loss = info_nce_loss(q_emb, p_emb, args.temperature)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), step=global_step)

            if global_step % args.save_steps == 0:
                save_dir = out_dir / f"checkpoint-{global_step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.encoder.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                logger.info("Saved %s", save_dir)
                save_training_state()

    save_training_state()
    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(final)
    tokenizer.save_pretrained(final)
    logger.info("Saved final encoder to %s", final)


if __name__ == "__main__":
    main()
