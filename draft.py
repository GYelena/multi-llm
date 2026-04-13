import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. 路径配置
# ==========================================
# 原始基座模型路径
BASE_MODEL_PATH = "/root/autodl-tmp/DeepSeek-R1"
# 你 DPO 训练保存的 LoRA 权重路径
LORA_PATH = "/root/autodl-tmp/planner_dpo_output/final_planner_dpo_lora"

class FineTunedPlanner:
    def __init__(self):
        print(">> [1/2] 正在加载基座模型 (4-bit 量化)...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        
        # 为了在单卡或双卡上快速运行，依然建议使用量化加载
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print(">> [2/2] 正在挂载 DPO 微调后的 LoRA 适配器...")
        # 🌟 核心：使用 PeftModel 加载适配器
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval() # 设置为评估模式
        print("   [OK] Planner 模型就绪。")

    def ask(self, context):
        """模拟中心模型的决策过程"""
        prompt = f"你是一个具备 MDP 动态路由能力的中心调度模型。\n请阅读当前的执行上下文，如果发现逻辑死胡同，必须输出 RECONSTRUCT。如果可以忽略，输出 CONTINUE。\n\n当前执行上下文:\n{context}"
        
        # 构造对话模版
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")

        # 执行推理
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.1, # 低随机性，保证 JSON 稳定性
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        return response

# ==========================================
# 2. 实战测试：构造一个“死胡同”场景
# ==========================================
if __name__ == "__main__":
    planner = FineTunedPlanner()

    # 测试案例：模拟 Agent B 报错（矩阵奇异）
    test_context = """
    [T1 | agent_a_explorer] 输出: 收集标普500历史收益率，构建 50x50 的协方差矩阵。
    [T2 | agent_b_reasoner] 输出: 尝试进行投资组合马科维茨优化。但在计算 Σ^-1 时，发现该矩阵行列式为 0，矩阵奇异，计算中断。
    [T3 | agent_c_critic] 输出: [报错] 由于 T1 提供的历史数据中包含两组完全线性的资产组合，导致协方差矩阵不可逆，无法得出解析解。
    """

    print("\n" + "="*50)
    print(">> 正在测试微调后的 Planner 决策能力...")
    result = planner.ask(test_context)
    
    print("\n>> Planner 输出结果:")
    print(result)
    print("="*50)