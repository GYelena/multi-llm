import re
import json
import requests
import os

# ==================== API 配置 ====================
# 请替换为你的真实 DMXAPI 令牌
API_KEY = "sk-36RQ3rLXi2jXFdxJcvO4b8JaxpS5os5S0J0GBfN5GUT2AuEf" 
API_URL = "https://www.dmxapi.cn/v1/chat/completions"
MODEL_NAME = "gpt-5.4"

def parse_workflow_log(log_content):
    """通过正则提取所有的 MDP 状态评估节点"""
    blocks = log_content.split("============================================================")
    mdp_evaluations = []
    
    for block in blocks:
        if "# MDP 状态评估与重构" in block:
            try:
                context_match = re.search(r'### \[输入上下文内容\]\n(.*?)\n\n### \[模型输出\]', block, re.DOTALL)
                output_match = re.search(r'### \[模型输出\]\n(.*)', block, re.DOTALL)
                
                if context_match and output_match:
                    mdp_evaluations.append({
                        "context": context_match.group(1).strip(),
                        "original_output": output_match.group(1).strip()
                    })
            except Exception as e:
                continue
    return mdp_evaluations

def generate_dpo_pair(context, original_output):
    """利用中转站 API 审视日志，生成 Chosen/Rejected 对"""
    system_prompt = """你是一个高级多智能体系统的数据提纯专家。
    我将提供一段中心模型的【执行上下文】和它基于此做出的【原始调度决策】。
    
    请你判断这个原始决策是否正确。
    1. 如果原始决策是错的（例如：上下文明明有数学错误或逻辑不通，它却输出了 CONTINUE），将其作为 rejected_response。然后，编写一个正确的、输出 RECONSTRUCT 动作并重新分配任务的 JSON 作为 chosen_response。
    2. 如果原始决策是正确的（例如：遇到错误它成功输出了 RECONSTRUCT，或者状态良好它输出了 CONTINUE），将其作为 chosen_response。然后，故意编写一个相反的错误决策 JSON 作为 rejected_response。
    
    请严格返回如下 JSON 格式，不要包含任何多余的解释文字：
    {
        "chosen_response": {"action": "...", "thought": "...", "new_tasks": [...]},
        "rejected_response": {"action": "...", "thought": "...", "new_tasks": [...]}
    }
    """
    
    user_prompt = f"【执行上下文】:\n{context}\n\n【原始调度决策】:\n{original_output}"
    
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
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status() 
        
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"].strip()
        
        # 打印原始返回内容，用于排查大模型的输出格式
        print(f"\n[Debug] 大模型原始返回内容:\n{content[:500]}...\n") 
        
        # 剥离可能存在的 markdown 代码块标签
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
            
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        
        # 尝试正则提取 JSON 对象
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(content)
            
    except Exception as e:
        print(f"[Error] 解析失败: {e}")
        return None

def main():
    log_file_path = "workflow_log.md"
    if not os.path.exists(log_file_path):
        print(f"[Error] 找不到文件 {log_file_path}，请确保它与本脚本在同一目录下。")
        return

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
        
    evaluations = parse_workflow_log(log_content)
    if not evaluations:
        print("[Warning] 未能在日志中提取到任何 MDP 评估节点，请检查日志格式。")
        return
        
    print(f"[Info] 成功提取到 {len(evaluations)} 个评估节点，开始生成数据...")
    
    dpo_dataset = []
    
    for idx, eval_data in enumerate(evaluations):
        context = eval_data['context']
        original_output = eval_data['original_output']
        
        print(f"[Info] 正在请求 API 处理第 {idx+1}/{len(evaluations)} 个切片...")
        pair = generate_dpo_pair(context, original_output)
        
        if pair:
            # 兼容大模型可能返回的不同键名
            chosen_data = pair.get("chosen_response") or pair.get("chosen") or pair.get("ChosenResponse")
            rejected_data = pair.get("rejected_response") or pair.get("rejected") or pair.get("RejectedResponse")
            
            if not chosen_data or not rejected_data:
                print(f"[Warning] 大模型未按格式返回指定键名，强制保存完整内容备查。")
                chosen_data = pair
                rejected_data = {"error": "解析失败，大模型返回格式异常"}

            # 强制转换为字符串类型，以符合 LLaMA-Factory 的要求
            if isinstance(chosen_data, dict):
                chosen_str = json.dumps(chosen_data, ensure_ascii=False, indent=2)
            else:
                chosen_str = str(chosen_data)
                
            if isinstance(rejected_data, dict):
                rejected_str = json.dumps(rejected_data, ensure_ascii=False, indent=2)
            else:
                rejected_str = str(rejected_data)

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
            dpo_dataset.append(dpo_item)
            print(f"[Success] 第 {idx+1} 个切片处理完毕！")
        else:
            print(f"[Fail] 第 {idx+1} 个切片未能生成有效数据，已跳过。")
            
    output_filename = "sharegpt_dpo_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n[Done] 数据集生成完毕，共成功转换 {len(dpo_dataset)} 条数据，已保存至 {output_filename}")

if __name__ == "__main__":
    main()