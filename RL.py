import torch
import json
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# 设置环境变量，优化多卡下的显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 核心超参数配置
# ==========================================
MODEL_PATH = "/root/autodl-tmp/DeepSeek-R1" 
DATASET_PATH = "/root/autodl-tmp/seed_sft_dpo_dataset.json" 
OUTPUT_DIR = "/root/autodl-tmp/planner_dpo_output"

# 多卡训练参数优化
LEARNING_RATE = 5e-6
BETA = 0.1
BATCH_SIZE = 1           # 每一张卡上的 batch size
GRAD_ACCUMULATION = 8    # 梯度累加。实际总 batch size = 1 * 8 * 2 = 16

# ==========================================
# 2. 模型与分词器加载 
# ==========================================
print(">> [1/4] 正在加载多卡自适应模型与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 如果有 2 张 32GB 的卡，可以尝试使用 fp16 甚至不需要 4-bit 量化。
# 但为了确保万无一失且保留极长上下文，保留量化。
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# device_map="auto" 自动将模型分片到所有检测到的显卡
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto", # 自动根据两张卡的剩余显存分配模型层
    trust_remote_code=True
)

# 为量化模型准备 LoRA 训练
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable() 

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 3. 数据集格式化流水线
# ==========================================
print(">> [2/4] 正在处理 DPO 数据集...")
raw_dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def format_dpo_dataset(example):
    user_msg = example['conversations'][0]['value']
    chosen_msg = example['chosen']['value']
    rejected_msg = example['rejected']['value']
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    return {
        "prompt": formatted_prompt,
        "chosen": chosen_msg + tokenizer.eos_token,
        "rejected": rejected_msg + tokenizer.eos_token,
    }

train_dataset = raw_dataset.map(format_dpo_dataset, remove_columns=raw_dataset.column_names)
print(f"   [OK] 成功处理 {len(train_dataset)} 条 DPO 偏好对。")

# ==========================================
# 4. DPOConfig 初始化 
# ==========================================
print(">> [3/4] 初始化多卡 DPOConfig...")

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_32bit", 
    fp16=True, 
    bf16=False, 
    report_to="none",
    beta=BETA,
    max_prompt_length=2048, 
    max_length=4096,        
    remove_unused_columns=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# ==========================================
# 5. 启动训练
# ==========================================
print(">> [4/4] 开始DPO 训练...")
dpo_trainer.train()

dpo_trainer.save_model(f"{OUTPUT_DIR}/final_planner_dpo_lora")
print(f"\n[Done] 训练完毕！")