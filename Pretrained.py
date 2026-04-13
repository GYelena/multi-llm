import json
import random
import requests
import time

# ==================== API 配置 ====================
API_KEY = "sk-36RQ3rLXi2jXFdxJcvO4b8JaxpS5os5S0J0GBfN5GUT2AuEf"  # 请替换为你的真实 DMXAPI 令牌
API_URL = "https://www.dmxapi.cn/v1/chat/completions"
MODEL_NAME = "gpt-5-mini"

# ==================== 场景种子池 ====================
TOPICS = [
    "贝叶斯网络与马尔可夫链蒙特卡洛(MCMC)参数估计",
    "基于宏观经济指标的 Black-Litterman 资产配置模型",
    "高频交易中的订单簿不平衡(OIB)特征提取与预测",
    "基于因果推断的医药实验双盲数据有效性分析",
    "复杂供应链网络中的多目标线性规划寻优",
    "大规模分布式系统中的微服务调用链路异常检测",
    "基于非合作博弈论的寡头市场定价策略推演",
    "量子计算中的 Grover 搜索算法伪代码实现与验证"
]

ERROR_TYPES = [
    "信息冲突: Agent A 收集的前置数据与 Agent B 推导所需的理论假设完全相反。",
    "数据缺失: Agent A 发现核心数据库的某个关键时间序列数据丢失，导致无法构成完整的特征矩阵。",
    "任务失败: Agent B 在执行复杂的数学公式推导或代码运算时，遭遇无法解决的维度灾难或死循环。",
    "逻辑悖论: Agent C (审查者) 发现 Agent B 的推导违背了基本的物理定律或数学公理，结论荒谬。",
    "环境突变: 在执行过程中，外部假设条件突然不再成立（例如：假设矩阵可逆，但实际发现是奇异矩阵）。"
]

def generate_synthetic_sample(topic, error_type, max_retries=3):
    system_prompt = """你是一个多智能体协作系统的数据合成专家。
    请根据我提供的【任务领域】和【错误场景】，自动生成一个执行日志（包含Agent A, B, C的交互），并在日志结尾触发高信息熵的错误。
    然后，生成中心规划模型的 Chosen (好策略) 和 Rejected (坏策略)。
    
    【篇幅与文风控制】(极其重要！)
    1. 拒绝废话：Agent 的输出绝对不要包含“好的”、“让我思考一下”、“首先”等寒暄和思考过程，直接给出冰冷的学术推导、核心代码逻辑或关键数据结果。
    2. 适度信息密度：每个 Agent 的输出字数控制在 50-150 字之间。不要太长，但必须包含具体的专业术语或公式。
    3. 精准的 Thought：中心模型的 thought 控制在 1-2 句话，精准点出报错的核心原因以及为什么要重构/继续。
    
    【强制输出格式】
    必须且只能输出合法的 JSON 字符串！严禁输出 markdown 代码块标记！
    注意：JSON 的根节点必须直接包含 context、chosen_response、rejected_response 这三个键，不要在外面包裹额外的结构。
    {
        "context": "[T1 | agent_a_explorer] 输出: (硬核结论)\\n[T2 | agent_b_reasoner] 输出: (硬核推导)\\n[T3 | agent_c_critic] 输出: [报错] (指出具体的死胡同)",
        "chosen_response": {
            "action": "RECONSTRUCT",
            "thought": "(精炼总结为何放弃原计划及新思路)",
            "new_tasks": [
                {"id": "T4_new", "target_model": "...", "query": "..."}
            ]
        },
        "rejected_response": {
            "action": "CONTINUE",
            "thought": "(演示盲目自信，强行忽略报错)",
            "new_tasks": []
        }
    }
    """
    
    user_prompt = f"任务领域：{topic}\n错误场景：{error_type}\n请严格按要求输出 JSON："
    
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4096  # 防止过早截断导致 JSON 不完整
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            
            resp_json = response.json()
            content = resp_json["choices"][0]["message"]["content"].strip()
            
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("未找到 JSON 括号")
                
            clean_json_str = content[start_idx : end_idx + 1]
            data = json.loads(clean_json_str)
            
            # 自适应层级解包：如果大模型在外面套了一层如 {"data": {...}}
            if len(data.keys()) == 1 and isinstance(list(data.values())[0], dict):
                inner_key = list(data.keys())[0]
                data = data[inner_key]
                print(f"      [Debug] 自动剥离了多余的外层包裹键名: {inner_key}")

            # 模糊匹配键名
            context = data.get("context") or data.get("Context") or data.get("log")
            chosen = data.get("chosen_response") or data.get("chosen") or data.get("Chosen")
            rejected = data.get("rejected_response") or data.get("rejected") or data.get("Rejected")
            
            if not context or not chosen or not rejected:
                print(f"      [Debug] 提取到的实际键名为: {list(data.keys())}")
                raise ValueError("JSON 缺少必要的 context, chosen 或 rejected 核心字段")
                
            return {
                "context": context,
                "chosen_response": chosen,
                "rejected_response": rejected
            }
            
        except Exception as e:
            print(f"      [Retry] 第 {attempt + 1} 次尝试失败: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            
    return None

def main():
    target_sample_count = 20
    dataset = []
    
    print(f"[Start] 准备生成 {target_sample_count} 条冷启动数据...")
    
    for i in range(target_sample_count):
        topic = random.choice(TOPICS)
        error_type = random.choice(ERROR_TYPES)
        
        print(f"\n[Process] 正在生成第 {i+1}/{target_sample_count} 条数据...")
        print(f"   领域: {topic}")
        print(f"   错误: {error_type}")
        
        sample = generate_synthetic_sample(topic, error_type)
        
        if sample:
            context = sample["context"]
            chosen_data = sample["chosen_response"]
            rejected_data = sample["rejected_response"]
            
            chosen_str = json.dumps(chosen_data, ensure_ascii=False, indent=2) if isinstance(chosen_data, dict) else str(chosen_data)
            rejected_str = json.dumps(rejected_data, ensure_ascii=False, indent=2) if isinstance(rejected_data, dict) else str(rejected_data)
            
            dpo_item = {
                "conversations": [
                    {
                        "from": "user",
                        "value": f"你是一个具备 MDP 动态路由能力的中心调度模型。\n请阅读当前的执行上下文，如果发现逻辑死胡同，必须输出 RECONSTRUCT。如果可以忽略，输出 CONTINUE。\n\n当前执行上下文:\n{context}"
                    }
                ],
                "chosen": {
                    "from": "assistant",
                    "value": chosen_str
                },
                "rejected": {
                    "from": "assistant",
                    "value": rejected_str
                }
            }
            
            dataset.append(dpo_item)
            print(f"   [Success] 第 {i+1} 条数据生成成功！当前已收集 {len(dataset)} 条。")
        else:
            print(f"   [Fail] 第 {i+1} 条数据经过多次重试仍失败，已跳过。")
            
        time.sleep(1)
        
    output_filename = "/root/autodl-tmp/seed_sft_dpo_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n[Done] 数据集生成完毕！成功 {len(dataset)}/{target_sample_count} 条，保存在 {output_filename}")

if __name__ == "__main__":
    main()