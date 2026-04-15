#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

# 默认系统提示词，告知模型当前扮演“专家B”，专注逐步推理和可验证计算
DEFAULT_SYSTEM_PROMPT = (
    "You are Expert B, specialized in step-by-step reasoning and verifiable calculations. "
    "Be precise, avoid shortcuts, and show intermediate values."
)

# 默认问题，作为推理输入
DEFAULT_QUESTION = (
    "An item costs $240. Apply these rules in order: "
    "(1) 20% discount, "
    "(2) if subtotal is at least $150, subtract $20, "
    "(3) add 6% tax. "
    "Compute the final amount and show each intermediate step."
)

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，允许用户指定模型、lora适配器、GPU、提示词、问题、生成长度等。
    """
    parser = argparse.ArgumentParser(description="Run DeepSeek-R1 + LoRA(B) single inference.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/muti-llm/DeepSeek-R1",
        help="Base model path（基座模型路径）",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/lora_B_full/checkpoint-60000",
        help="LoRA adapter path（LoRA权重/适配器路径）",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA visible device id(s), e.g. '0'（可见GPU编号）",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt in English（系统提示词，英文）",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question in English（输入问题，英文）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=320, help="Max generated tokens（最大生成token数）")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)（采样温度）",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling（Top-p采样阈值）")
    return parser.parse_args()

def main() -> None:
    """
    主程序：加载模型与LoRA适配器，推理生成回答并输出结果和元信息。
    """
    args = parse_args()
    # 设置需要使用的GPU（通过CUDA_VISIBLE_DEVICES环境变量控制，需在导入torch之前）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 延迟导入深度学习相关库，确保CUDA环境先被设置
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()  # 记录开始时间

    print("[info] Loading tokenizer...")
    # 加载分词器（tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("[info] Loading base model...")
    # 加载基座模型，可以自动分配设备，使用bfloat16精度
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[info] Loading LoRA adapter...")
    # 加载LoRA适配器到基座模型（微调权重）
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()  # 设为推理模式

    # 构造消息（chat history），包含system和user角色
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.question},
    ]

    # 判断tokenizer是否支持chat模板，有则用chat template生成prompt，否则自定义拼接prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: {args.system_prompt}\nUser: {args.question}\nAssistant:"

    # 编码prompt为模型输入张量，并移动到模型对应的设备
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 构建生成参数（包括最大生成长度、终止token、pad token）
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    # 如果temperature大于0，则采取采样生成；否则为贪婪生成
    if args.temperature > 0.0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generate_kwargs.update({"do_sample": False})

    # 推理生成答案（不记录梯度，可节省显存）
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    # 提取生成的token id序列（去掉输入部分，只留模型生成的内容）
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    # 解码得到文本并去掉特殊token
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 输出格式化结果，包括问题、模型输出以及元信息（耗时、设备等）
    print("\n===== QUESTION =====")
    print(args.question)
    print("\n===== MODEL OUTPUT =====")
    print(text)
    print("\n===== META =====")
    print(
        {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "visible_gpus": args.gpu,
            "device": str(model.device),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )


if __name__ == "__main__":
    main()
