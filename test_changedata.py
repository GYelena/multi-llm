# %%
from unsloth import FastLanguageModel
import torch
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 基于unsloth加载Llama的蒸馏模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/DeepSeek-R1",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# %%
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an expert in reading comprehension and information extraction.
Given a context passage and a question, analyze the text carefully and provide the exact answer span from the passage.

### Question:
{}

### Response:
<think>{}"""

question = "Article: Phytochemistry is a branch of plant biochemistry primarily concerned with the chemical substances produced by plants during secondary metabolism. Some of these compounds are toxins such as the alkaloid coniine from hemlock. Others, such as the essential oils peppermint oil and lemon oil are useful for their aroma, as flavourings and spices (e.g., capsaicin), and in medicine as pharmaceuticals as in opium from opium poppies. Many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. Others are simple derivatives of botanical natural products. For example, the pain killer aspirin is the acetyl ester of salicylic acid, originally isolated from the bark of willow trees, and a wide range of opiate painkillers like heroin are obtained by chemical modification of morphine obtained from the opium poppy. Popular stimulants come from plants, such as caffeine from coffee, tea and chocolate, and nicotine from tobacco. Most alcoholic beverages come from fermentation of carbohydrate-rich plant products such as barley (beer), rice (sake) and grapes (wine).Now answer this question: Where do some medicines and recreational drugs come from?"


FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=8192,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

# %%
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an expert in reading comprehension and information extraction.
Given a context passage and a question, analyze the text carefully and provide the exact answer span from the passage.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# 迭代训练集数据，处理prompt
def formatting_prompts_func(examples):
    inputs = examples["source"]
    cots = examples["rationale"]
    outputs = examples["target"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

# from datasets import load_dataset
# dataset = load_dataset("/root/autodl-tmp/auto-cot-main/dataset/squad_v1_data", split = "train[0:5000]")
import json
from datasets import Dataset
with open('/root/autodl-tmp/medical-o1-reasoning-SFT/medical_o1_sft.json', 'r', encoding='utf-8') as f:
    dataset = Dataset.from_list(list(json.load(f).values())[:5000])
print(dataset.column_names)

# %%
dataset = dataset.map(formatting_prompts_func, batched = True)
print(dataset["text"][0])

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/auto-cot-main/model1",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

FastLanguageModel.for_training(model)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 600,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# 训练
trainer_stats = trainer.train()


# %%
FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
question = "Article: Phytochemistry is a branch of plant biochemistry primarily concerned with the chemical substances produced by plants during secondary metabolism. Some of these compounds are toxins such as the alkaloid coniine from hemlock. Others, such as the essential oils peppermint oil and lemon oil are useful for their aroma, as flavourings and spices (e.g., capsaicin), and in medicine as pharmaceuticals as in opium from opium poppies. Many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. Others are simple derivatives of botanical natural products. For example, the pain killer aspirin is the acetyl ester of salicylic acid, originally isolated from the bark of willow trees, and a wide range of opiate painkillers like heroin are obtained by chemical modification of morphine obtained from the opium poppy. Popular stimulants come from plants, such as caffeine from coffee, tea and chocolate, and nicotine from tobacco. Most alcoholic beverages come from fermentation of carbohydrate-rich plant products such as barley (beer), rice (sake) and grapes (wine).Now answer this question: Where do some medicines and recreational drugs come from?"

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

# # %%
# model.save_pretrained("lora_model") # Local saving
# tokenizer.save_pretrained("lora_model")


