#!/usr/bin/env python3
"""
正式跑检索训练前的自检（不下载模型、不训练）：
  - Python 依赖
  - CUDA
  - A_retrieval.jsonl 存在且可被 iter_rows 解析

用法:
  python3 training/verify_retrieval_setup.py
  python3 training/verify_retrieval_setup.py --data Data/Processed_data/A_retrieval.jsonl --sample-rows 1000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def need(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default="Data/Processed_data/A_retrieval.jsonl",
    )
    p.add_argument("--sample-rows", type=int, default=1000)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)

    errs: list[str] = []

    for mod in ("torch", "transformers", "tqdm"):
        if not need(mod):
            errs.append(f"missing Python module: {mod} (pip install -r training/requirements-retrieval.txt)")

    if not errs:
        import torch

        print(f"[ok] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("[warn] CUDA not available — training will be very slow on CPU")

    if not data_path.is_file():
        errs.append(f"A_retrieval data not found: {data_path}")
    else:
        nlines = sum(1 for _ in data_path.open("r", encoding="utf-8"))
        print(f"[ok] data file lines: {nlines} ({data_path})")

    if data_path.is_file():
        sys.path.insert(0, str(root))
        from training.train_retrieval_biencoder import iter_rows

        n = 0
        for row in iter_rows(data_path, max_samples=args.sample_rows):
            if "query" not in row or "positive" not in row:
                errs.append("iter_rows produced invalid row")
                break
            n += 1
        print(f"[ok] iter_rows parsed {n} training pairs (cap sample-rows={args.sample_rows})")

    if errs:
        print("[fail]")
        for e in errs:
            print(" ", e)
        return 1

    print("[ok] Ready for training once --model is available (HF download or local snapshot).")
    print("     If huggingface.co is slow/blocked, set mirror e.g.:")
    print("       export HF_ENDPOINT=https://hf-mirror.com")
    print("     Or download model to disk and: --model /path/to/bge-base-en-v1.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
